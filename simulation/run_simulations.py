"""Simulation study: 12 settings x 50 replicates.

    python run_simulations.py run
    python run_simulations.py status
"""
import os
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from simulate import adjusted_rand, design, run_replicate

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results' / 'simulation'
PARTS = OUT / 'parts'

N_MCMC = 10_000
BURN_IN = 5_000
N_REPLICATES = 50
MAX_WORKERS = 8
SETTINGS = ['K2_T20_P1', 'K2_T40_P1', 'K2_T20_P2', 'K2_T40_P2',
            'K3_T20_P1', 'K3_T40_P1', 'K3_T20_P2', 'K3_T40_P2',
            'K2_T20_miss_p90', 'K2_T20_miss_p80', 'K2_T20_miss_p70', 'K2_T20_miss_p60']


def part_path(key, replicate):
    return PARTS / key / f'replicate_{replicate:03d}.npz'


def worker(task):
    key, replicate = task
    path = part_path(key, replicate)
    if path.exists():
        return key, replicate, 'skipped', 0.0
    start = time.time()
    z_true = design(key)['z_true']
    trace = {'state_ARI': [], 'K': [], 'radius': []}

    def record(states, log_r):
        trace['state_ARI'].append(adjusted_rand(z_true, states))
        trace['K'].append(len(log_r))
        trace['radius'].append(np.exp(np.asarray(log_r))[states])

    d, data, fit = run_replicate(key, replicate, N_MCMC, BURN_IN, on_sweep=record)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(
        tmp,
        true_theta=data['true_theta_by_season'], true_radius=data['true_radius_by_season'],
        true_state=d['z_true'].astype(np.int16), true_eta=np.full(d['T'], 0.3),
        trace_state_ARI=np.asarray(trace['state_ARI'], dtype=np.float32),
        trace_K=np.asarray(trace['K'], dtype=np.int16),
        trace_radius=np.asarray(trace['radius'], dtype=np.float32),
        draws_states=fit['ss'].astype(np.int16), draws_radii=fit['r_arr'].astype(np.float32),
        draws_K=fit['Ks'].astype(np.int16), draws_theta=fit['ths'].astype(np.float32),
        draws_eta=fit['etas'].astype(np.float32), draws_alpha_r=fit['alphas_r'],
        r_accept=fit['r_accept'], eta_accept=fit['eta_accept'])
    tmp.replace(path)
    K_values, K_counts = np.unique(fit['Ks'], return_counts=True)
    path.with_suffix('.json').write_text(json.dumps(dict(
        setting=key, replicate=replicate, seconds=round(time.time() - start, 1),
        K_distribution={int(k): int(v) for k, v in zip(K_values, K_counts)},
        r_accept=float(fit['r_accept']), eta_accept=float(fit['eta_accept'])), indent=2))
    return key, replicate, 'done', time.time() - start


def command_run():
    order = sorted(SETTINGS, key=lambda k: 'T40' not in k)
    tasks = [(k, r) for k in order for r in range(1, N_REPLICATES + 1) if not part_path(k, r).exists()]
    print(f'{len(tasks)} replicates pending', flush=True)
    start = time.time()
    with mp.get_context('fork').Pool(MAX_WORKERS) as pool:
        for done, (key, replicate, status, seconds) in enumerate(pool.imap_unordered(worker, tasks), 1):
            hours = (time.time() - start) / 3600
            print(f'{status} {key} rep {replicate:03d} ({seconds / 60:.1f} min) [{done}/{len(tasks)}, {hours:.1f} h]',
                  flush=True)


def command_status():
    total = 0
    for key in SETTINGS:
        n = len(list((PARTS / key).glob('replicate_???.npz'))) if (PARTS / key).exists() else 0
        total += n
        print(f'{key:18s} {n:3d}/{N_REPLICATES}')
    print(f'{"total":18s} {total:3d}/{N_REPLICATES * len(SETTINGS)}')


if __name__ == '__main__':
    {'run': command_run, 'status': command_status}[sys.argv[1] if len(sys.argv) > 1 else 'status']()
