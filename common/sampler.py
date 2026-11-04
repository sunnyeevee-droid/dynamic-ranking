"""MCMC updates of the dynamic projection model, shared by the simulation study and the applications."""
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, isotonic_regression


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_A(N):
    m = N * (N - 1) // 2
    A = np.zeros((m, N))
    k = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            A[k, i] = 1.0
            A[k, j] = -1.0
            k += 1
    return A


def roughness(theta, A):
    return float(np.sum(np.abs(A @ theta)))


def proj_complete_graph_exact(beta, radius):
    """Projection onto {sum_i theta_i = 0, sum_{i<j} |theta_i - theta_j| <= radius}."""
    beta = np.asarray(beta, dtype=float)
    centered = beta - beta.mean()
    if len(beta) <= 1 or radius == 0:
        return np.zeros_like(centered)
    order = np.argsort(centered)
    values = centered[order]
    weights = 2 * np.arange(len(beta)) - len(beta) + 1
    if float(weights @ values) <= radius:
        return centered

    def fitted(multiplier):
        return isotonic_regression(values - multiplier * weights, increasing=True).x
    upper = max(1., float(np.ptp(values)))
    multiplier = brentq(lambda value: float(weights @ fitted(value) - radius),
                        0., upper, xtol=1e-12, rtol=1e-13)
    theta = np.empty_like(centered)
    theta[order] = fitted(multiplier)
    return theta - theta.mean()


@lru_cache(maxsize=4096)
def _projection_cache(beta_bytes, radius):
    return proj_complete_graph_exact(np.frombuffer(beta_bytes, dtype=np.float64), radius)


def proj(b, r):
    return _projection_cache(np.ascontiguousarray(b, dtype=np.float64).tobytes(), float(r)).copy()


def olp(mu, d):
    pn = sigmoid(-d - mu)
    pd = sigmoid(d - mu) - pn
    pp = 1.0 - sigmoid(d - mu)
    pr = np.stack([pn, pd, pp], -1)
    pr = np.clip(pr, 1e-12, 1.0)
    return pr / pr.sum(-1, keepdims=True)


def oll(th, eta_t, X_t, h_t, y_t, d):
    mu = X_t @ th + eta_t * h_t
    p = olp(mu, d)
    return np.sum(np.log(p[np.arange(len(y_t)), (y_t + 1).astype(int)] + 1e-12))


def emis(b, eta_t, X_t, h_t, y_t, r_k, d):
    th = proj(b, r_k)
    return oll(th, eta_t, X_t, h_t, y_t, d), th


def ess_upd(bc, eta_t, X_t, h_t, y_t, r, d, sb, rng):
    nu = rng.normal(0, sb, bc.shape[0])
    lc, _ = emis(bc, eta_t, X_t, h_t, y_t, r, d)
    ly = lc + np.log(rng.uniform())
    phi = rng.uniform(0, 2 * np.pi)
    pm, pM = phi - 2 * np.pi, phi
    while True:
        bp = bc * np.cos(phi) + nu * np.sin(phi)
        lp, tp = emis(bp, eta_t, X_t, h_t, y_t, r, d)
        if lp >= ly:
            return bp, tp, lp
        if phi < 0:
            pm = phi
        else:
            pM = phi
        phi = rng.uniform(pm, pM)


def log_prior_log_r_exp(lr, alpha_r):
    return lr - alpha_r * np.exp(lr)


def sample_alpha_r(log_r, a, b, rng):
    K = len(log_r)
    sum_r = float(np.sum(np.exp(np.asarray(log_r))))
    return float(rng.gamma(a + K, 1.0 / (b + sum_r)))


def build_r_grid(alpha_r, r_grid_q):
    return np.clip(-np.log(1.0 - r_grid_q) / alpha_r, 1e-6, None)


def mh_r(k, log_r, beta, eta, states, Xl, hl, yl, d, alpha_r, psd, rng):
    idx = np.where(states == k)[0]
    if len(idx) == 0:
        return log_r, False
    lr2 = log_r.copy()
    lr2[k] += rng.normal(0.0, psd)
    cur = log_prior_log_r_exp(log_r[k], alpha_r)
    prop = log_prior_log_r_exp(lr2[k], alpha_r)
    r_cur = np.exp(log_r[k])
    r_prop = np.exp(lr2[k])
    for t in idx:
        lc, _ = emis(beta[t], eta[t], Xl[t], hl[t], yl[t], r_cur, d)
        lp, _ = emis(beta[t], eta[t], Xl[t], hl[t], yl[t], r_prop, d)
        cur += lc
        prop += lp
    if np.log(rng.uniform()) < (prop - cur):
        return lr2, True
    return log_r, False


def mh_eta_t(eta_cur, b_t, X_t, h_t, y_t, r_t, d, eta_prior_sd, psd_eta, rng):
    eta_prop = eta_cur + rng.normal(0.0, psd_eta)
    ll_cur, _ = emis(b_t, eta_cur, X_t, h_t, y_t, r_t, d)
    ll_prop, _ = emis(b_t, eta_prop, X_t, h_t, y_t, r_t, d)
    log_post_cur = ll_cur - 0.5 * (eta_cur / eta_prior_sd) ** 2
    log_post_prop = ll_prop - 0.5 * (eta_prop / eta_prior_sd) ** 2
    if np.log(rng.uniform()) < (log_post_prop - log_post_cur):
        return float(eta_prop), True
    return float(eta_cur), False


def build_cnt(states, K):
    n = np.zeros((K, K), dtype=int)
    for t in range(1, len(states)):
        n[states[t - 1], states[t]] += 1
    return n


def mfm_ratio(m, K, lam, gam):
    return (np.log(lam) - np.log(max(K, 1))
            + np.log(K + 1)
            + np.log(gam) - np.log(m + K * gam + 1e-30))


def new_state_grid(beta, eta_t, X_t, h_t, y_t, d, r_grid):
    """Log likelihoods of season t at the prior-quantile radius grid."""
    values = np.asarray([oll(proj(beta, float(r)), eta_t, X_t, h_t, y_t, d)
                         for r in np.asarray(r_grid, dtype=float)], dtype=float)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite new-state grid likelihood.")
    return values


def gibbs_sweep(states, log_r, beta, eta, Xl, hl, yl, alpha, gam, lam, d, r_grid, rng):
    Tl = len(states)
    log_r = list(log_r)

    for t in range(Tl):
        Kc = len(log_r)
        active = sorted(set(states.tolist()))
        m = len(active)
        zp = states[t - 1] if t > 0 else None
        zn = states[t + 1] if t < Tl - 1 else None

        n = build_cnt(states, Kc + 2)
        if t > 0:
            n[states[t - 1], states[t]] = max(n[states[t - 1], states[t]] - 1, 0)
        if t < Tl - 1:
            n[states[t], states[t + 1]] = max(n[states[t], states[t + 1]] - 1, 0)

        nr = n.sum(1)
        log_probs = []
        candidates = []
        bk = gam / max(m, 1)

        for k in active:
            lin = 0.0 if zp is None else np.log(max(n[zp, k] + alpha * bk, 1e-30))
            if zn is None:
                lout = 0.0
            else:
                Ip = int(zp == k)
                den = nr[k] + Ip + alpha
                num = n[k, zn] + Ip * int(zn == k) + alpha * bk
                lout = np.log(max(num, 1e-30)) - np.log(max(den, 1e-30))
            emission, _ = emis(beta[t], eta[t], Xl[t], hl[t], yl[t], np.exp(log_r[k]), d)
            log_probs.append(lin + lout + emission)
            candidates.append(k)

        lin_new = 0.0 if zp is None else np.log(alpha) + np.log(bk)
        lout_new = 0.0 if zn is None else np.log(alpha) + np.log(bk) - np.log(alpha)
        grid_ll = new_state_grid(beta[t], eta[t], Xl[t], hl[t], yl[t], d, r_grid)
        grid_max = float(np.max(grid_ll))
        log_marg = float(grid_max + np.log(np.mean(np.exp(grid_ll - grid_max))))
        log_probs.append(mfm_ratio(m, Kc, lam, gam) + lin_new + lout_new + log_marg)
        candidates.append("new")

        values = np.asarray(log_probs)
        values -= np.max(values)
        probs = np.exp(values)
        probs /= probs.sum()
        chosen = candidates[int(rng.choice(len(candidates), p=probs))]

        if chosen == "new":
            weights = np.exp(grid_ll - grid_max)
            weights /= weights.sum()
            new_index = int(rng.choice(len(r_grid), p=weights))
            log_r.append(np.log(max(float(r_grid[new_index]), 1e-8)))
            states[t] = len(log_r) - 1
        else:
            states[t] = chosen

        active2 = sorted(set(states.tolist()))
        mapping = {old: i for i, old in enumerate(active2)}
        states = np.array([mapping[z] for z in states], dtype=int)
        log_r = [log_r[old] for old in active2]

    return states, np.asarray(log_r, dtype=float)


def sample_Pi(states, K, alpha, gam, rng):
    n = build_cnt(states, K)
    bk = gam / max(len(set(states.tolist())), 1)
    P = np.zeros((K, K))
    for i in range(K):
        P[i] = rng.dirichlet(n[i] + alpha * bk * np.ones(K))
    return P


def relabel(states, log_r, Pi):
    order = np.argsort(np.exp(log_r))
    mp = {old: i for i, old in enumerate(order)}
    return np.array([mp[z] for z in states], dtype=int), log_r[order], Pi[np.ix_(order, order)]


def mcmc_step(s, data, cfg, rng, on_sweep=None):
    """One sweep over abilities, home effects, radii, alpha_r, the state path and Pi.

    `s` holds states, log_r, beta, eta and alpha_r and is updated in place. Returns the
    abilities, Pi, and the accepted and proposed radius and home-effect moves.
    """
    Xl, hl, yl = data["Xl"], data["hl"], data["yl"]
    T = len(yl)
    states, log_r, beta, eta = s["states"], s["log_r"], s["beta"], s["eta"]
    for t in range(T):
        beta[t], _, _ = ess_upd(beta[t], eta[t], Xl[t], hl[t], yl[t],
                                np.exp(log_r[states[t]]), cfg["d_f"], cfg["sb"], rng)
    acc_eta = 0
    for t in range(T):
        eta[t], a_eta = mh_eta_t(eta[t], beta[t], Xl[t], hl[t], yl[t], np.exp(log_r[states[t]]),
                                 cfg["d_f"], cfg["eta_prior_sd"], cfg["psd_eta"], rng)
        acc_eta += int(a_eta)
    acc_r, n_r = 0, len(log_r)
    for k in range(n_r):
        log_r, a_r = mh_r(k, log_r, beta, eta, states, Xl, hl, yl, cfg["d_f"], s["alpha_r"], cfg["psd_r"], rng)
        acc_r += int(a_r)

    alpha_r = sample_alpha_r(log_r, cfg["a_r"], cfg["b_r"], rng)
    r_grid = build_r_grid(alpha_r, np.linspace(0.05, 0.95, cfg["n_grid"]))
    states, log_r = gibbs_sweep(states.copy(), log_r.copy(), beta, eta, Xl, hl, yl,
                                cfg["alpha"] + cfg["kappa"], cfg["gam"], cfg["lam"], cfg["d_f"], r_grid, rng)
    if on_sweep is not None:
        on_sweep(states, log_r)
    Pi = sample_Pi(states, len(log_r), cfg["alpha"], cfg["gam"], rng)
    theta = [proj(beta[t], np.exp(log_r[states[t]])) for t in range(T)]
    s.update(states=states, log_r=log_r, alpha_r=alpha_r)
    return theta, Pi, (acc_r, n_r), (acc_eta, T)
