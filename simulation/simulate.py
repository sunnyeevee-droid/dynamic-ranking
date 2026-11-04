"""Simulation designs, data generation and evaluation."""
import numpy as np
import pandas as pd

import model as M
from sampler import make_A, roughness, sigmoid
from summary import adjusted_rand, labels_from_theta, posterior_summary, reorder_labels_by_theta

N = 20
GAMES_PER_PAIR = 2
ETA_TRUE = 0.3
DELTA_TRUE = 0.3
MISSING_MASK_SEED_OFFSET = 500_000_000
A0 = make_A(N)


def centered(x):
    return x - x.mean()


def two_tier(a):
    return centered(np.r_[np.repeat(a, 10), np.repeat(-a, 10)])


THREE_TIER = centered(np.r_[np.repeat(4.0 / 3.0, 5), np.repeat(0.0, 10), np.repeat(-4.0 / 3.0, 5)])


def blocks(*parts):
    """blocks((state, ability vector, length), ...) -> state path and abilities by season."""
    z = np.concatenate([np.full(n, s) for s, _, n in parts]).astype(int)
    theta = np.array([th for _, th, n in parts for _ in range(n)])
    return z, theta


def design(key):
    u_half, u1, u2 = two_tier(0.5), two_tier(1.0), two_tier(2.0)
    if key.startswith('K2_T20_miss') or key == 'K2_T20_P1':
        z, theta = blocks((0, u1, 5), (1, u2, 5), (0, u1, 5), (1, u2, 5))
        seed = 2026
    elif key == 'K2_T40_P1':
        z, theta = blocks((0, u1, 10), (1, u2, 10), (0, u1, 10), (1, u2, 10))
        seed = 2026
    elif key in ('K2_T20_P2', 'K2_T40_P2'):
        z, theta = blocks(*[(0, u1, 5), (1, u2, 5), (0, THREE_TIER, 5), (1, u2, 5)] * (1 if 'T20' in key else 2))
        seed = 2026
    elif key in ('K3_T20_P1', 'K3_T40_P1'):
        z, theta = blocks(*[(0, u_half, 5), (1, u1, 5), (2, u2, 5), (1, u1, 5)] * (1 if 'T20' in key else 2))
        seed = 20261001
    else:
        z, theta = blocks(*[(0, u_half, 5), (1, u1, 5), (2, u2, 5), (1, THREE_TIER, 5)] * (1 if 'T20' in key else 2))
        seed = 20261001 if key == 'K3_T20_P2' else 20261201
    retain = int(key[-2:]) / 100 if 'miss' in key else 1.0
    return dict(key=key, z_true=z, theta=theta, T=len(z), K_true=len(np.unique(z)), seed=seed, retain=retain)


def simulate_dataset(d, rep_seed):
    rng = np.random.default_rng(rep_seed)
    pairs = [(i, j) for i in range(N - 1) for j in range(i + 1, N)]
    Xl, hl, yl = [], [], []
    for t in range(d['T']):
        theta = d['theta'][t]
        G = len(pairs) * GAMES_PER_PAIR
        X, h, y = np.zeros((G, N)), np.ones(G, dtype=float), np.zeros(G, dtype=int)
        g = 0
        for i, j in pairs:
            for home, away in [(i, j), (j, i)] * (GAMES_PER_PAIR // 2):
                X[g, home], X[g, away] = 1.0, -1.0
                mu = theta[home] - theta[away] + ETA_TRUE
                p_home = sigmoid(mu - DELTA_TRUE)
                p_draw = sigmoid(mu + DELTA_TRUE) - sigmoid(mu - DELTA_TRUE)
                p_away = 1.0 - sigmoid(mu + DELTA_TRUE)
                probs = np.clip(np.array([p_away, p_draw, p_home]), 1e-12, 1.0)
                y[g] = rng.choice([-1, 0, 1], p=probs / probs.sum())
                g += 1
        Xl.append(X), hl.append(h), yl.append(y)
    data = dict(Xl=Xl, hl=hl, yl=yl, Ns=[N] * d['T'], true_theta_by_season=d['theta'].copy(),
                true_radius_by_season=np.array([roughness(th, A0) for th in d['theta']]),
                observed_games_by_season=np.array([len(y) for y in yl]))
    if d['retain'] < 1.0:
        # Each match is kept independently with probability retain.
        mask_rng = np.random.default_rng(int(rep_seed) + MISSING_MASK_SEED_OFFSET + int(round(1000 * d['retain'])))
        for t in range(d['T']):
            keep = mask_rng.random(len(yl[t])) < d['retain']
            Xl[t], hl[t], yl[t] = Xl[t][keep].copy(), hl[t][keep].copy(), yl[t][keep].copy()
        data['observed_games_by_season'] = np.array([len(y) for y in yl])
    return data


def seeds(d, replicate):
    rep_seed = d['seed'] + 10_000 * int(replicate)
    mcmc_seed = rep_seed + 777
    if d['retain'] < 1.0:
        mcmc_seed += 1_000_000 * int(round(100 * d['retain']))
    return rep_seed, mcmc_seed


def evaluate(summary, data, d, replicate):
    theta_true, r_true = data['true_theta_by_season'], data['true_radius_by_season']
    z_hat = summary['z_hat']
    theta_mse_year = np.mean((summary['theta_hat'] - theta_true) ** 2, axis=1)
    class_ari_year = [adjusted_rand(reorder_labels_by_theta(labels_from_theta(th), th), c)
                      for th, c in zip(theta_true, summary['class_hat'])]
    common = dict(setting=d['key'], replicate=int(replicate), T=d['T'], K_TRUE=d['K_true'],
                  K_hat=int(summary['K_hat']), K_correct=int(summary['K_hat'] == d['K_true']),
                  state_ARI=adjusted_rand(d['z_true'], z_hat), theta_MSE=float(theta_mse_year.mean()),
                  class_ARI_mean=float(np.mean(class_ari_year)),
                  eta_MSE=float(np.mean((summary['eta_hat'] - ETA_TRUE) ** 2)))
    return pd.DataFrame([dict(common, year=t + 1, true_state=int(d['z_true'][t]), estimated_state=int(z_hat[t]),
                              true_radius=float(r_true[t]), estimated_radius=float(summary['r_hat'][t]),
                              observed_games=int(data['observed_games_by_season'][t]),
                              theta_MSE_year=float(theta_mse_year[t]), class_ARI_year=float(class_ari_year[t]),
                              eta_hat=float(summary['eta_hat'][t]))
                         for t in range(d['T'])])


def run_replicate(key, replicate, n_mcmc, burn_in, on_sweep=None):
    d = design(key)
    rep_seed, mcmc_seed = seeds(d, replicate)
    data = simulate_dataset(d, rep_seed)
    return d, data, M.run_mcmc(data, n_mcmc, burn_in, mcmc_seed, on_sweep=on_sweep)


def evaluate_saved(key, replicate, chain):
    """Posterior summary and evaluation of one saved chain."""
    d = design(key)
    data = simulate_dataset(d, seeds(d, replicate)[0])
    fit = dict(ss=chain['draws_states'].astype(int), r_arr=chain['draws_radii'].astype(float),
               Ks=chain['draws_K'].astype(int), ths=chain['draws_theta'].astype(float),
               etas=chain['draws_eta'].astype(float))
    return evaluate(posterior_summary(fit), data, d, replicate)
