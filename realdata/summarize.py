"""Posterior summaries of the two leagues from the pooled chains.

    python summarize.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import load
from fit import LEAGUES, OUT, chain_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from summary import posterior_summary  # noqa: E402

CI_LEVEL = 0.90


def load_chains(league):
    return [dict(np.load(chain_path(league, seed))) for seed in LEAGUES[league]['seeds']]


def pooled(chains):
    width = max(c['radii'].shape[1] for c in chains)
    pad = lambda r: np.pad(r, ((0, 0), (0, width - r.shape[1])), constant_values=np.nan)
    return dict(ss=np.concatenate([c['states'] for c in chains]).astype(int),
                r_arr=np.concatenate([pad(c['radii']) for c in chains]),
                Ks=np.concatenate([c['K'] for c in chains]).astype(int),
                ths=np.concatenate([c['theta'] for c in chains]),
                etas=np.concatenate([c['eta'] for c in chains]))


def transition_frequencies(states, K):
    """Posterior mean of the observed state-to-state transition frequencies."""
    freq = np.full((len(states), K, K), np.nan)
    for n, z in enumerate(states):
        counts = np.zeros((K, K))
        np.add.at(counts, (z[:-1], z[1:]), 1)
        rows = counts.sum(axis=1, keepdims=True)
        np.divide(counts, rows, out=freq[n], where=rows > 0)
    return np.nanmean(freq, axis=0)


def summarize(league):
    data, chains = load(league), load_chains(league)
    fit = pooled(chains)
    s = posterior_summary(fit)
    K_hat, z_hat = s['K_hat'], s['z_hat']
    q = [(1 - CI_LEVEL) / 2, (1 + CI_LEVEL) / 2]
    season_r = np.take_along_axis(fit['r_arr'], fit['ss'], axis=1)
    modal = fit['Ks'] == K_hat

    season_rows, team_rows = [], []
    for t, season in enumerate(data['seasons']):
        same = modal & (fit['ss'][:, t] == z_hat[t])
        theta = fit['ths'][same, t]
        lo, hi = np.quantile(theta, q, axis=0)
        n_tiers = int(s['class_hat'][t].max())
        season_rows.append(dict(
            season=season, state=int(z_hat[t]) + 1, dahl_radius=s['r_hat'][t],
            radius_median=np.median(season_r[same, t]), radius_low=np.quantile(season_r[same, t], q[0]),
            radius_high=np.quantile(season_r[same, t], q[1]), state_probability=same.sum() / modal.sum(),
            n_tiers=n_tiers, eta_mean=s['eta_hat'][t], eta_low=np.quantile(fit['etas'][:, t], q[0]),
            eta_high=np.quantile(fit['etas'][:, t], q[1])))
        team_rows += [dict(season=season, team=team, tier=int(c), n_tiers=n_tiers, theta_mean=m, theta_low=a,
                           theta_high=b, tier_theta=th)
                      for team, c, m, a, b, th in zip(data['teams'][t], s['class_hat'][t], theta.mean(axis=0),
                                                      lo, hi, s['theta_hat'][t])]

    out = OUT / league
    pd.DataFrame(season_rows).to_csv(out / 'season_summary.csv', index=False)
    pd.DataFrame(team_rows).sort_values(['season', 'tier', 'theta_mean'], ascending=[True, True, False]) \
        .to_csv(out / 'team_summary.csv', index=False)
    K_rows = [dict(chain=name, K=k, probability=np.mean(Ks == k))
              for name, Ks in [*((f'chain {i + 1}', c['K']) for i, c in enumerate(chains)), ('pooled', fit['Ks'])]
              for k in np.unique(fit['Ks'])]
    pd.DataFrame(K_rows).to_csv(out / 'K_posterior.csv', index=False)
    labels = [f'State {k + 1}' for k in range(K_hat)]
    pd.DataFrame(transition_frequencies(fit['ss'][modal], K_hat), index=labels, columns=labels) \
        .to_csv(out / 'transition_frequencies.csv')

    print(f"{league.upper()}: K_hat = {K_hat}, P(K = K_hat) = {modal.mean():.3f}; "
          f"Dahl path {''.join(str(z + 1) for z in z_hat)}; radii {np.unique(np.round(s['r_hat'], 1))}")
    print(pd.DataFrame(season_rows)[['season', 'state', 'dahl_radius', 'n_tiers', 'eta_mean']].round(3)
          .to_string(index=False))


if __name__ == '__main__':
    for name in sys.argv[1:] or LEAGUES:
        summarize(name)
