"""Two chains per league, 50,000 iterations each with a 5,000-iteration burn-in.

    python fit.py epl
    python fit.py nba
"""
import os
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_var] = '1'

import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from sampler import mcmc_step, relabel  # noqa: E402
from data import load  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results' / 'realdata'
N_ITER = 50_000
BURN_IN = 5_000
CHECKPOINT_EVERY = 1_000

COMMON = dict(alpha=0.5, kappa=2.0, gam=0.5, lam=0.5, d_f=0.65, sb=2.0, a_r=10.0,
              n_grid=15, psd_r=0.20, psd_eta=0.16, eta_prior_sd=1.0)
LEAGUES = {'epl': dict(COMMON, b_r=200.0, seeds=(2026092601, 2026092602)),
           'nba': dict(COMMON, b_r=350.0, seeds=(2026, 3026))}


def chain_path(league, seed):
    return OUT / league / f'chain_{seed}.npz'


def run_chain(league, seed):
    cfg, data = LEAGUES[league], load(league)
    T = len(data['yl'])
    checkpoint = chain_path(league, seed).with_suffix('.ckpt.pkl')
    rng = np.random.default_rng(seed)
    if checkpoint.exists():
        saved = pickle.loads(checkpoint.read_bytes())
        rng.bit_generator.state = saved['rng']
        start, s, draws, counts = saved['iteration'], saved['state'], saved['draws'], saved['counts']
    else:
        alpha_r = float(rng.gamma(cfg['a_r'], 1.0 / cfg['b_r']))
        s = dict(states=np.zeros(T, dtype=int), log_r=np.array([-np.log(alpha_r)]),
                 beta=[rng.normal(0., .1, n) for n in data['Ns']], eta=np.zeros(T), alpha_r=alpha_r)
        start, draws, counts = 0, {k: [] for k in ('states', 'radii', 'K', 'theta', 'eta', 'alpha_r')}, np.zeros(4)

    for it in range(start, N_ITER):
        theta, Pi, (a_r, n_r), (a_eta, n_eta) = mcmc_step(s, data, cfg, rng)
        counts += (a_r, n_r, a_eta, n_eta)
        if it >= BURN_IN:
            states, log_r, _ = relabel(s['states'].copy(), s['log_r'].copy(), Pi)
            for key, value in zip(draws, (states, np.exp(log_r), len(log_r), np.array(theta), s['eta'].copy(),
                                          s['alpha_r'])):
                draws[key].append(value)
        if (it + 1) % CHECKPOINT_EVERY == 0 and it + 1 < N_ITER:
            tmp = checkpoint.with_suffix('.tmp')
            tmp.write_bytes(pickle.dumps(dict(iteration=it + 1, rng=rng.bit_generator.state, state=s,
                                              draws=draws, counts=counts)))
            tmp.replace(checkpoint)
            print(f'{league} seed {seed}: {it + 1}/{N_ITER}, K={len(s["log_r"])}', flush=True)

    K = np.array(draws['K'])
    radii = np.full((len(K), K.max()), np.nan)
    for i, r in enumerate(draws['radii']):
        radii[i, :len(r)] = r
    np.savez(chain_path(league, seed), states=np.array(draws['states'], dtype=np.int16), radii=radii,
             K=K.astype(np.int16), theta=np.array(draws['theta']), eta=np.array(draws['eta']),
             alpha_r=np.array(draws['alpha_r']), r_accept=counts[0] / counts[1], eta_accept=counts[2] / counts[3])
    checkpoint.unlink(missing_ok=True)


def main(league):
    (OUT / league).mkdir(parents=True, exist_ok=True)
    seeds = [seed for seed in LEAGUES[league]['seeds'] if not chain_path(league, seed).exists()]
    with mp.get_context('fork').Pool(len(seeds) or 1) as pool:
        pool.starmap(run_chain, [(league, seed) for seed in seeds])


if __name__ == '__main__':
    main(sys.argv[1])
