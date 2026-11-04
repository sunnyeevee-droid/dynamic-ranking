import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

R_SCRIPT = Path(__file__).resolve().parent / 'salso_batch.R'
SEED = 2026


def exact_labels(theta):
    return np.array([np.unique(row, return_inverse=True)[1] + 1 for row in theta])


def salso_tiers(theta_sets):
    """SALSO partition of the exact-tie partitions of each (draws x teams) array.

    The number of tiers is capped at the most frequent number of tiers in a draw.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        manifest = []
        for k, theta in enumerate(theta_sets):
            labels = exact_labels(theta)
            values, freq = np.unique(labels.max(axis=1), return_counts=True)
            np.savetxt(tmp / f's{k}.csv', labels, fmt='%d', delimiter=',')
            manifest.append(dict(file=f's{k}.csv', cap=int(values[np.argmax(freq)])))
        pd.DataFrame(manifest).to_csv(tmp / 'manifest.csv', index=False)
        subprocess.run(['Rscript', str(R_SCRIPT), str(tmp), str(SEED)], check=True, capture_output=True)
        return [pd.read_csv(tmp / f'est_s{k}.csv').tier.to_numpy() for k in range(len(theta_sets))]


def ranked(labels, values):
    """Tier 1 = highest mean of `values`."""
    labs = np.unique(labels)
    order = labs[np.argsort([-values[labels == lab].mean() for lab in labs])]
    new = {lab: k + 1 for k, lab in enumerate(order)}
    return np.array([new[lab] for lab in labels])
