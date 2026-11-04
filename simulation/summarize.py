"""Posterior summaries and evaluation of every saved chain (one CSV per setting)."""
import multiprocessing as mp

import numpy as np
import pandas as pd

from run_simulations import N_REPLICATES, OUT, PARTS, SETTINGS
from simulate import evaluate_saved

SUMMARIES = OUT / 'summaries'


def one(task):
    key, replicate = task
    return evaluate_saved(key, replicate, np.load(PARTS / key / f'replicate_{replicate:03d}.npz'))


def main():
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    tasks = [(key, r) for key in SETTINGS for r in range(1, N_REPLICATES + 1)]
    with mp.get_context('fork').Pool(8) as pool:
        frames = pool.map(one, tasks, chunksize=4)
    results = pd.concat(frames, ignore_index=True)
    for key, frame in results.groupby('setting'):
        frame.to_csv(SUMMARIES / f'{key}.csv', index=False)


if __name__ == '__main__':
    main()
