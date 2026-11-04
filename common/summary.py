"""Posterior summaries: modal K, Dahl state path and radii, SALSO tiers, home effects."""
import numpy as np

from salso import salso_tiers


def labels_from_theta(theta):
    """Tiers of an ability vector: groups of exactly equal abilities."""
    return np.unique(np.asarray(theta), return_inverse=True)[1]


def reorder_labels_by_theta(labels, theta):
    labels = np.asarray(labels)
    theta = np.asarray(theta)
    labs = np.unique(labels)
    means = np.array([theta[labels == lab].mean() for lab in labs])
    out = np.zeros_like(labels, dtype=int)
    for new_id, lab in enumerate(labs[np.argsort(-means)], start=1):
        out[labels == lab] = new_id
    return out


def theta_by_tier_average(theta, labels):
    theta = np.asarray(theta)
    labels = np.asarray(labels)
    out = theta.copy()
    for lab in np.unique(labels):
        out[labels == lab] = theta[labels == lab].mean()
    return out


def adjusted_rand(labels_a, labels_b):
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    n = len(labels_a)
    if n < 2:
        return 1.0
    _, a_inv = np.unique(labels_a, return_inverse=True)
    _, b_inv = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((a_inv.max() + 1, b_inv.max() + 1), dtype=int)
    for i in range(n):
        contingency[a_inv[i], b_inv[i]] += 1
    comb2 = lambda x: x * (x - 1) / 2.0
    sum_comb = np.sum(comb2(contingency))
    sum_a = np.sum(comb2(contingency.sum(axis=1)))
    sum_b = np.sum(comb2(contingency.sum(axis=0)))
    total = comb2(n)
    expected = sum_a * sum_b / total if total > 0 else 0.0
    denom = 0.5 * (sum_a + sum_b) - expected
    return 0.0 if denom == 0 else float((sum_comb - expected) / denom)


def posterior_summary(fit):
    """Modal K, Dahl state path and radii, season tiers by SALSO, and posterior-mean home effects.

    For season t the draws with K = K_hat in the Dahl state of t give a sample of
    exact-tie partitions; the tiers are their SALSO estimate (VI loss) with at most as
    many tiers as the most frequent count in a draw, and each team's ability is the
    posterior mean averaged over its tier.
    """
    ss, Ks, ths, etas, r_arr = fit["ss"], fit["Ks"], fit["ths"], fit["etas"], fit["r_arr"]
    T = ss.shape[1]
    K_values, K_counts = np.unique(Ks, return_counts=True)
    K_hat = int(K_values[np.argmax(K_counts)])
    idx_K = np.where(Ks == K_hat)[0]
    memberships = np.array([(s[:, None] == s[None, :]).astype(float) for s in ss[idx_K]])
    losses = np.sum((memberships - memberships.mean(axis=0)[None]) ** 2, axis=(1, 2))
    dahl_idx = int(idx_K[int(np.argmin(losses))])
    z_hat = ss[dahl_idx].copy()
    r_hat = r_arr[dahl_idx, z_hat].astype(float)

    sets = [ths[(Ks == K_hat) & (ss[:, t] == z_hat[t]), t] for t in range(T)]
    theta_hat, class_hat = [], []
    for theta, labels in zip(sets, salso_tiers(sets)):
        averaged = theta_by_tier_average(theta.mean(axis=0), labels)
        theta_hat.append(averaged)
        class_hat.append(reorder_labels_by_theta(labels, averaged))

    return dict(K_hat=K_hat, z_hat=z_hat, r_hat=r_hat, theta_hat=np.array(theta_hat), class_hat=class_hat,
                eta_hat=etas.mean(axis=0), K_distribution=dict(zip(K_values.astype(int), K_counts.astype(int))),
                dahl_index=dahl_idx)
