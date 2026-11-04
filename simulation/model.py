"""Sampler settings and MCMC run of the simulation study."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from sampler import mcmc_step, relabel  # noqa: E402

CONFIG = dict(alpha=0.5, kappa=2.0, gam=0.5, lam=0.5, d_f=0.3, sb=6.0, a_r=10.0, b_r=200.0,
              n_grid=15, psd_r=0.16, psd_eta=0.05, eta_prior_sd=1.0)


def run_mcmc(data, n_mcmc, burn_in, seed, on_sweep=None):
    T = len(data["yl"])
    rng = np.random.default_rng(seed)
    alpha_r = CONFIG["a_r"] / CONFIG["b_r"]
    s = dict(states=np.zeros(T, dtype=int), log_r=np.array([np.log(1.0 / alpha_r)], dtype=float),
             beta=[rng.normal(0.0, 0.1, data["Ns"][t]) for t in range(T)], eta=[0.0 for _ in range(T)],
             alpha_r=alpha_r)

    ss, rs, Ps, Ks, ths, etas, alphas_r = [], [], [], [], [], [], []
    acc_r = acn_r = acc_eta = acn_eta = 0

    for it in range(n_mcmc):
        theta, Pi, (a_r, n_r), (a_eta, n_eta) = mcmc_step(s, data, CONFIG, rng, on_sweep)
        acc_r, acn_r, acc_eta, acn_eta = acc_r + a_r, acn_r + n_r, acc_eta + a_eta, acn_eta + n_eta
        if it >= burn_in:
            s2, lr2, P2 = relabel(s["states"].copy(), s["log_r"].copy(), Pi)
            ss.append(s2.copy())
            rs.append(np.exp(lr2).tolist())
            Ps.append(P2.copy())
            Ks.append(len(lr2))
            ths.append([th.copy() for th in theta])
            etas.append(np.array(s["eta"], dtype=float).copy())
            alphas_r.append(s["alpha_r"])

    Ks = np.array(Ks)
    Km = int(max(int(Ks.max()), 1))
    r_arr = np.full((len(Ks), Km), np.nan)
    for i, rv in enumerate(rs):
        r_arr[i, :len(rv)] = rv
    return dict(ss=np.array(ss), r_arr=r_arr, Ks=Ks, ths=np.array(ths), etas=np.array(etas),
                alphas_r=np.array(alphas_r), Ps=Ps,
                r_accept=acc_r / max(acn_r, 1), eta_accept=acc_eta / max(acn_eta, 1))
