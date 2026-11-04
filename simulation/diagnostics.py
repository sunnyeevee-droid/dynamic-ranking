"""Calibration and sampling diagnostics of the simulation study."""
import numpy as np
import pandas as pd

from results_for_paper import MAIN, MISSING, OUT
from run_simulations import PARTS

LABEL = {'K2_T20_P1': '2 & 20 & 1', 'K2_T40_P1': '2 & 40 & 1', 'K2_T20_P2': '2 & 20 & 2',
         'K2_T40_P2': '2 & 40 & 2', 'K3_T20_P1': '3 & 20 & 1', 'K3_T40_P1': '3 & 40 & 1',
         'K3_T20_P2': '3 & 20 & 2', 'K3_T40_P2': '3 & 40 & 2',
         **{key: f'\\multicolumn{{3}}{{c}}{{$p={p:.1f}$}}' for p, key in MISSING[1:]}}


def covers(draws, truth, level):
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return (truth >= lo) & (truth <= hi)


def credible_set(values, counts, level):
    p = counts / counts.sum()
    order = np.argsort(-p)
    return set(values[order][:np.searchsorted(np.cumsum(p[order]), level) + 1])


def replicate_metrics(path):
    z = np.load(path)
    s = z['draws_states'].astype(int)
    theta = z['draws_theta'].astype(float)
    true_s, true_theta = z['true_state'], z['true_theta']
    k_true = len(np.unique(true_s))
    ks = z['draws_K']
    values, counts = np.unique(ks, return_counts=True)
    row = dict(eta_95=covers(z['draws_eta'], z['true_eta'], 0.95).mean(),
               theta_95=covers(theta, true_theta, 0.95).mean(),
               Kset_95=float(k_true in credible_set(values, counts, 0.95)),
               P_true_K=np.mean(ks == k_true))

    iu = np.triu_indices(len(true_s), 1)
    co_post = (s[:, :, None] == s[:, None, :]).mean(axis=0)[iu]
    co_true = (true_s[:, None] == true_s[None, :])[iu]
    row['co_same'], row['co_diff'] = co_post[co_true].mean(), co_post[~co_true].mean()

    tie = np.stack([(theta[:, t, :, None] == theta[:, t, None, :]).mean(axis=0) for t in range(len(true_s))])
    tie_true = true_theta[:, :, None] == true_theta[:, None, :]
    off_diagonal = ~np.eye(true_theta.shape[1], dtype=bool)[None]
    row['tie_same'] = tie[tie_true & off_diagonal].mean()
    row['tie_diff'] = tie[~tie_true].mean()

    half = len(ks) // 2
    row['r_accept'], row['eta_accept'] = float(z['r_accept']), float(z['eta_accept'])
    row['K_changes'] = int(np.count_nonzero(np.diff(ks)))
    row['P_true_K_first_half'] = np.mean(ks[:half] == k_true)
    row['P_true_K_second_half'] = np.mean(ks[half:] == k_true)
    reached = np.flatnonzero(z['trace_state_ARI'] >= 0.99)
    row['first_iter_ARI_099'] = int(reached[0]) + 1 if len(reached) else np.nan
    return row, pd.DataFrame(dict(p=co_post, truth=co_true))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, reliability = [], []
    for key in MAIN + [key for _, key in MISSING[1:]]:
        per_rep = []
        for path in sorted((PARTS / key).glob('replicate_???.npz')):
            row, rel = replicate_metrics(path)
            per_rep.append(row)
            if key in MAIN:
                reliability.append(rel)
        frame = pd.DataFrame(per_rep)
        rows.append(dict(setting=key, R=len(frame), **frame.mean().to_dict(),
                         first_iter_ARI_099_max=frame.first_iter_ARI_099.max()))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'diagnostics_by_setting.csv', index=False)

    rel = pd.concat(reliability)
    rel['bin'] = pd.cut(rel.p, [-0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 1.01])
    rel.groupby('bin', observed=True).agg(pairs=('p', 'size'), predicted=('p', 'mean'),
                                          observed=('truth', 'mean')).to_csv(OUT / 'coclustering_reliability.csv')

    calibration = [f"{LABEL[r.setting]} & {r.eta_95:.2f} & {r.theta_95:.2f} & {r.Kset_95:.2f} & {r.P_true_K:.2f} "
                   f"& {r.co_same:.3f} & {r.co_diff:.3f} \\\\" for r in table.itertuples()]
    sampling = [f"{LABEL[r.setting]} & {r.r_accept:.2f} & {r.eta_accept:.2f} & {r.K_changes:.0f} "
                f"& {r.P_true_K_first_half:.2f} / {r.P_true_K_second_half:.2f} "
                f"& {r.first_iter_ARI_099:.0f} ({r.first_iter_ARI_099_max:.0f}) "
                f"& {r.tie_same:.2f} & {r.tie_diff:.3f} \\\\" for r in table.itertuples()]
    for lines in (calibration, sampling):
        lines.insert(len(MAIN), '\\midrule')
    (OUT / 'table_calibration.tex').write_text('\n'.join(calibration) + '\n')
    (OUT / 'table_sampling.tex').write_text('\n'.join(sampling) + '\n')
    print(table.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
