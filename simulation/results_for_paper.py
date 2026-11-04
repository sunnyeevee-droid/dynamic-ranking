"""Tables of the simulation study (main settings and missing games)."""
import json

import numpy as np
import pandas as pd

from run_simulations import N_REPLICATES, OUT as SIM_OUT, PARTS
from summarize import SUMMARIES

OUT = SIM_OUT / 'tables'
MAIN = ['K2_T20_P1', 'K2_T40_P1', 'K2_T20_P2', 'K2_T40_P2',
        'K3_T20_P1', 'K3_T40_P1', 'K3_T20_P2', 'K3_T40_P2']
MISSING = [(1.00, 'K2_T20_P1'), (0.90, 'K2_T20_miss_p90'), (0.80, 'K2_T20_miss_p80'),
           (0.70, 'K2_T20_miss_p70'), (0.60, 'K2_T20_miss_p60')]
METRICS = ['state_ARI', 'theta_MSE', 'class_ARI_mean', 'eta_MSE']


def load(key):
    frame = pd.read_csv(SUMMARIES / f'{key}.csv')
    assert frame.replicate.nunique() == N_REPLICATES, f'{key} is not finished'
    return frame.sort_values(['replicate', 'year']).reset_index(drop=True)


def radius_by_replicate(frame):
    return (frame.assign(true_r=frame.true_radius.round(6))
            .groupby(['replicate', 'true_r']).estimated_radius.median().unstack())


def main_table():
    table, misfits, radius = [], [], []
    for key in MAIN:
        frame = load(key)
        rep = frame.drop_duplicates('replicate')
        row = dict(setting=key, K=int(rep.K_TRUE.iloc[0]), T=int(rep['T'].iloc[0]),
                   P=2 if key.endswith('P2') else 1, K_recovery=rep.K_correct.mean())
        for m in METRICS:
            row[m] = rep[m].mean()
            row[m + '_se'] = rep[m].std(ddof=1) / np.sqrt(len(rep))
        table.append(row)

        for r in rep[rep.K_correct == 0].itertuples():
            sizes = frame[frame.replicate == r.replicate].estimated_state.value_counts()
            info = json.loads((PARTS / key / f'replicate_{r.replicate:03d}.json').read_text())
            k_dist = {int(k): v for k, v in info['K_distribution'].items()}
            misfits.append(dict(setting=key, replicate=r.replicate, K_true=r.K_TRUE, K_hat=r.K_hat,
                                state_ARI=r.state_ARI, smallest_state_size=int(sizes.min()),
                                P_K_hat=k_dist[r.K_hat] / sum(k_dist.values()),
                                P_K_true=k_dist.get(r.K_TRUE, 0) / sum(k_dist.values())))

        per_rep = radius_by_replicate(frame)
        for truth in per_rep.columns:
            est = per_rep[truth]
            radius.append(dict(setting=key, true_radius=truth, mean_estimate=est.mean(),
                               mean_relative_error=(est / truth - 1).mean()))

    table = pd.DataFrame(table)
    table.to_csv(OUT / 'main_table.csv', index=False)
    pd.DataFrame(misfits).to_csv(OUT / 'misfits.csv', index=False)
    pd.DataFrame(radius).to_csv(OUT / 'radius_recovery.csv', index=False)
    lines = [f'{r.K} & {r.T} & {r.P}\n& {r.K_recovery:.2f}\n'
             f'& {r.state_ARI:.4f} ({r.state_ARI_se:.4f})\n'
             f'& {r.theta_MSE:.4f} ({r.theta_MSE_se:.4f})\n'
             f'& {r.class_ARI_mean:.4f} ({r.class_ARI_mean_se:.4f})\n'
             f'& {r.eta_MSE:.4f} ({r.eta_MSE_se:.4f}) \\\\\n' for r in table.itertuples()]
    (OUT / 'table_simulation_results.tex').write_text('\n'.join(lines))
    print(table.round(4).to_string(index=False))


def missing_table():
    rows = []
    for p, key in MISSING:
        frame = load(key)
        rep = frame.drop_duplicates('replicate')
        per_rep = radius_by_replicate(frame)
        rows.append(dict(retain=p, setting=key, K_recovery=rep.K_correct.mean(), state_ARI=rep.state_ARI.mean(),
                         theta_MSE=rep.theta_MSE.mean(), eta_MSE=rep.eta_MSE.mean(),
                         class_ARI=rep.class_ARI_mean.mean(),
                         observed_games=float(frame.observed_games.mean()),
                         radius200_mean=per_rep[200.0].mean(), radius400_mean=per_rep[400.0].mean(),
                         radius200_spread=per_rep[200.0].quantile(.975) - per_rep[200.0].quantile(.025),
                         radius400_spread=per_rep[400.0].quantile(.975) - per_rep[400.0].quantile(.025)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'missing_table.csv', index=False)
    base = table.iloc[0]
    pct, end = '\\%', ' \\\\\n'

    def stack(value, change):
        return '\\shortstack{' + value + '\\\\(' + change + ')}'

    lines = [f'1.00\n& 0{pct}\n& {base.K_recovery:.3f}\n& {base.state_ARI:.4f}\n& {base.theta_MSE:.4f}\n'
             f'& {base.eta_MSE:.4f}\n& {base.class_ARI:.4f}{end}']
    for r in table.iloc[1:].itertuples():
        dk = r.K_recovery - base.K_recovery
        cells = [stack(f'{r.K_recovery:.3f}', '0.000' if abs(dk) < 5e-4 else f'{dk:+.3f}'),
                 stack(f'{r.state_ARI:.4f}', f'{r.state_ARI - base.state_ARI:+.4f}'),
                 stack(f'{r.theta_MSE:.4f}', f'{100 * (r.theta_MSE / base.theta_MSE - 1):+.1f}' + pct),
                 stack(f'{r.eta_MSE:.4f}', f'{100 * (r.eta_MSE / base.eta_MSE - 1):+.1f}' + pct),
                 stack(f'{r.class_ARI:.4f}', f'{r.class_ARI - base.class_ARI:+.4f}')]
        lines.append(f'{r.retain:.2f}\n& {round(100 * (1 - r.retain)):.0f}{pct}\n& ' + '\n& '.join(cells) + end)
    (OUT / 'table_missing_data.tex').write_text('\n'.join(lines))
    print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    main_table()
    missing_table()
