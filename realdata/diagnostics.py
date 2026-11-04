"""Two-chain MCMC diagnostics of the applications: R-hat, ESS, K by chain and trace plots.

    python diagnostics.py
"""
import warnings

import arviz as az
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from data import load  # noqa: E402
from fit import BURN_IN, LEAGUES, N_ITER, OUT  # noqa: E402
from summarize import load_chains  # noqa: E402

DIAG = OUT / 'diagnostics'
RHAT_LIMIT, ESS_REFERENCE = 1.01, 200
TRACE_STRIDE, BLOCK = 20, 500
CHAIN_COLORS = ('#6D8EA3', '#B77E75')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 10,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'pdf.fonttype': 42, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': '#8B8B8B', 'axes.linewidth': 0.65})


def scalar(values):
    """Rank-normalized R-hat, bulk and tail ESS of a (2, draws) array."""
    if all(np.ptp(chain) == 0 for chain in values):
        return dict(rhat=np.nan, ess_bulk=np.nan, ess_tail=np.nan, mcse_mean=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return dict(rhat=float(az.rhat(values, method='rank')), ess_bulk=float(az.ess(values, method='bulk')),
                    ess_tail=float(az.ess(values, method='tail')), mcse_mean=float(az.mcse(values, method='mean')))


def table(chains, data):
    season_r = [np.take_along_axis(c['radii'], c['states'].astype(int), axis=1) for c in chains]
    stack = lambda f: np.stack([f(i, c) for i, c in enumerate(chains)])
    rows = [dict(group=g, season='', team='', **scalar(stack(lambda i, c: c[g]))) for g in ('K', 'alpha_r')]
    for t, season in enumerate(data['seasons']):
        rows.append(dict(group='radius', season=season, team='', **scalar(stack(lambda i, c: season_r[i][:, t]))))
        rows.append(dict(group='eta', season=season, team='', **scalar(stack(lambda i, c: c['eta'][:, t]))))
        rows += [dict(group='theta', season=season, team=team, **scalar(stack(lambda i, c: c['theta'][:, t, j])))
                 for j, team in enumerate(data['teams'][t])]
    return pd.DataFrame(rows), season_r


def overview(diag):
    rows = []
    for group in ('K', 'alpha_r', 'radius', 'eta', 'theta'):
        d = diag[diag.group == group]
        rows.append(dict(quantity=group, components=len(d), max_rhat=d.rhat.max(), median_rhat=d.rhat.median(),
                         rhat_above_1p01=int((d.rhat > RHAT_LIMIT).sum()), min_bulk_ess=d.ess_bulk.min(),
                         min_tail_ess=d.ess_tail.min(),
                         ess_below_200=int(((d.ess_bulk < ESS_REFERENCE) | (d.ess_tail < ESS_REFERENCE)).sum())))
    return pd.DataFrame(rows)


def trace(ax, values, discrete=False):
    iterations = np.arange(BURN_IN + 1, N_ITER + 1)
    n = len(iterations) // BLOCK * BLOCK
    for chain, color in zip(values, CHAIN_COLORS):
        stride = 1 if discrete else TRACE_STRIDE
        ax.plot(iterations[::stride], chain[::stride], color=color, linewidth=0.35, alpha=0.16, zorder=1)
        ax.plot(iterations[:n].reshape(-1, BLOCK).mean(axis=1), chain[:n].reshape(-1, BLOCK).mean(axis=1),
                color=color, linewidth=1.2, zorder=2)
    lo, hi = values.min(), values.max()
    pad = max((hi - lo) * 0.08, 0.01)
    ax.set_xlim(BURN_IN + 1, N_ITER)
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(MaxNLocator(integer=discrete, nbins=5 if discrete else 4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:g}k'))
    ax.tick_params(length=2.5)
    ax.grid(axis='y', color='#E5E5E5', linewidth=0.6)
    ax.set_axisbelow(True)


def chain_legend(fig, y, fontsize):
    fig.legend(handles=[Line2D([0], [0], color=c, lw=2, label=f'Chain {i + 1}') for i, c in enumerate(CHAIN_COLORS)],
               loc='upper center', bbox_to_anchor=(0.5, y), ncol=2, frameon=False, fontsize=fontsize,
               columnspacing=2.8, handlelength=2.6)


def plot_K(name, chains):
    values = np.stack([c['K'] for c in chains]).astype(int)
    ks = np.arange(values.min(), values.max() + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.65), gridspec_kw={'width_ratios': [1.45, 1]})
    fig.subplots_adjust(left=.075, right=.98, bottom=.16, top=.78, wspace=.29)
    fig.suptitle(f'{name.upper()} | Number of occupied states', fontsize=17, fontweight='bold', y=.995)
    chain_legend(fig, .925, 11)
    trace(axes[0], values, discrete=True)
    axes[0].set_yticks(ks)
    axes[0].set_xticks([10000, 20000, 30000, 40000, 50000], ['10k', '20k', '30k', '40k', '50k'])
    axes[0].set(xlabel='MCMC iteration', ylabel='Occupied states K')
    axes[0].set_title('(a) K trace', loc='left', pad=10)
    for i, color in enumerate(CHAIN_COLORS):
        axes[1].bar(ks + (i - .5) * .35, [np.mean(values[i] == k) for k in ks], width=.33, color=color,
                    alpha=.94, zorder=3)
    axes[1].set_title('(b) K distribution', loc='left', pad=10)
    axes[1].set(xlabel='Occupied states K', ylabel='Empirical probability', xticks=ks,
                xlim=(ks[0] - .65, ks[-1] + .65), ylim=(0, 1))
    axes[1].grid(axis='y', color='#E5E5E5', linewidth=.6)
    axes[1].set_axisbelow(True)
    fig.savefig(DIAG / f'{name}_K_trace_distribution.pdf')
    plt.close(fig)


def plot_seasons(name, seasons, values, quantity):
    rows = int(np.ceil(len(seasons) / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(16, 2.35 * rows + 1.3), squeeze=False)
    fig.subplots_adjust(left=.060, right=.985, bottom=.065, top=.90, hspace=.51, wspace=.27)
    title = 'Season-assigned radius' if quantity == 'radius' else 'Home advantage'
    fig.suptitle(f'{name.upper()} | {title}', fontsize=18, fontweight='bold', y=.988)
    chain_legend(fig, .963, 12)
    for t, ax in enumerate(axes.flat):
        if t >= len(seasons):
            ax.set_visible(False)
            continue
        trace(ax, values[:, :, t])
        ax.set_xticks([10000, 30000, 50000], ['10k', '30k', '50k'])
        ax.tick_params(axis='both', labelsize=10, length=2.5, pad=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_xlabel(seasons[t], fontsize=12, fontweight='bold', labelpad=7)
    fig.supylabel('Radius r' if quantity == 'radius' else 'Home effect eta', x=.013, fontsize=13, fontweight='bold')
    fig.savefig(DIAG / f'{name}_{quantity}_all_seasons.pdf')
    plt.close(fig)


def main():
    DIAG.mkdir(parents=True, exist_ok=True)
    for name in LEAGUES:
        data, chains = load(name), load_chains(name)
        diag, season_r = table(chains, data)
        diag.to_csv(DIAG / f'{name}_all_diagnostics.csv', index=False)
        summary = overview(diag)
        summary.to_csv(DIAG / f'{name}_overview.csv', index=False)
        K = pd.DataFrame([dict(chain=i + 1, K=k, probability=np.mean(c['K'] == k))
                          for i, c in enumerate(chains) for k in np.unique(c['K'])])
        K.to_csv(DIAG / f'{name}_K_by_chain.csv', index=False)
        plot_K(name, chains)
        plot_seasons(name, data['seasons'], np.stack(season_r), 'radius')
        plot_seasons(name, data['seasons'], np.stack([c['eta'] for c in chains]), 'eta')
        print(name.upper())
        print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
