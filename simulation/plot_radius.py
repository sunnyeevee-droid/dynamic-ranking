"""Radius recovery figures: eight main settings and the missing-games study."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from run_simulations import N_REPLICATES, OUT
from summarize import SUMMARIES

MAIN = [('(a)', '2 states, 20 seasons', 'One pattern per state', 'K2_T20_P1'),
        ('(b)', '2 states, 40 seasons', 'One pattern per state', 'K2_T40_P1'),
        ('(c)', '3 states, 20 seasons', 'One pattern per state', 'K3_T20_P1'),
        ('(d)', '3 states, 40 seasons', 'One pattern per state', 'K3_T40_P1'),
        ('(e)', '2 states, 20 seasons', 'Two patterns at r = 200', 'K2_T20_P2'),
        ('(f)', '2 states, 40 seasons', 'Two patterns at r = 200', 'K2_T40_P2'),
        ('(g)', '3 states, 20 seasons', 'Two patterns at r = 200', 'K3_T20_P2'),
        ('(h)', '3 states, 40 seasons', 'Two patterns at r = 200', 'K3_T40_P2')]
YLIM = (0, 500)  # common radius scale for all panels
MISSING = [('(a)', 1.0, 'K2_T20_P1'), ('(b)', 0.9, 'K2_T20_miss_p90'), ('(c)', 0.8, 'K2_T20_miss_p80'),
           ('(d)', 0.7, 'K2_T20_miss_p70'), ('(e)', 0.6, 'K2_T20_miss_p60')]
STYLE = dict(
    main=dict(fill=['#A9D8E8', '#F3B6C6', '#A7BE72'], edge=['#668C9B', '#A87380', '#6F8445'],
              true='#3F4142', text='#303234', grid='#D8D5D0', ticks='#505251'),
    missing=dict(fill=['#A9D8E8', '#F3B6C6'], edge=['#668C9B', '#A87380'],
                 true='#3F444A', text='#30343A', grid='#DDE3E8', ticks='#50555A'))

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11.5, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.labelsize': 13, 'axes.labelweight': 'bold', 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
    'axes.edgecolor': '#8B8986', 'axes.linewidth': 0.8, 'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.bf': 'Arial:bold', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})


def state_estimates(key):
    """One radius estimate per replicate and true state: the median over its seasons."""
    raw = pd.read_csv(SUMMARIES / f'{key}.csv')
    assert raw.replicate.nunique() == N_REPLICATES
    return (raw.groupby(['replicate', 'true_state'], as_index=False)
            .agg(estimated_radius=('estimated_radius', 'median'), true_radius=('true_radius', 'median'))
            .sort_values(['true_state', 'replicate']))


def panel(ax, data, title, style, missing):
    states = sorted(data['true_state'].unique())
    positions = np.arange(1, len(states) + 1)
    samples = [data.loc[data.true_state == s, 'estimated_radius'].to_numpy() for s in states]
    truths = [float(data.loc[data.true_state == s, 'true_radius'].median()) for s in states]
    boxes = ax.boxplot(samples, positions=positions, widths=0.54, patch_artist=True, showfliers=True,
                       medianprops={'color': '#242627', 'linewidth': 1.7},
                       whiskerprops={'color': '#666765', 'linewidth': 1.1},
                       capprops={'color': '#666765', 'linewidth': 1.1},
                       boxprops={'linewidth': 1.0} if missing else {'edgecolor': '#666765', 'linewidth': 1.0},
                       flierprops={'marker': 'o', 'markerfacecolor': '#777875', 'markeredgecolor': 'none',
                                   'markersize': 3.2, 'alpha': 0.55})
    for i, box in enumerate(boxes['boxes']):
        box.set_facecolor(style['fill'][i % len(style['fill'])])
        box.set_edgecolor(style['edge'][i % len(style['edge'])])
        box.set_alpha(0.95)
    for position, truth in zip(positions, truths):
        ax.hlines(truth, position - 0.36, position + 0.36, colors=style['true'], linewidth=2.2,
                  linestyles=(0, (3, 2)), zorder=5)
    if missing:
        ax.set_xlim(0.48, len(states) + 0.52)
    ax.set_ylim(*YLIM)
    ax.set_yticks(np.arange(0, YLIM[1], 100))
    ax.set_xticks(positions)
    ax.set_xticklabels([f'State {int(s) + 1}' for s in states], fontweight='bold')
    if missing:
        ax.set_title(title, color=style['text'], fontweight='bold', pad=10)
        ax.set_ylabel(r'Estimated radius $\mathbf{r}$', fontsize=13, fontweight='bold', labelpad=9)
    else:
        ax.set_ylabel(r'Estimated radius $\mathbf{r}$', fontsize=13, fontweight='bold', labelpad=9)
        ax.set_title(title, color=style['text'], pad=10, fontweight='bold')
    ax.grid(axis='y', color=style['grid'], linewidth=0.7, alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=style['ticks'], labelsize=11.5)


def legend(fig, style):
    handle = Line2D([0], [0], color=style['true'], linewidth=2.2, linestyle=(0, (3, 2)), label='True radius')
    fig.legend(handles=[handle], loc='lower center', bbox_to_anchor=(0.5, 0.018), frameon=False, fontsize=12)


def save(fig, stem):
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'figures' / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'figures' / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.4), sharex=False, sharey=False)
    for ax, (label, title, subtitle, key) in zip(axes.ravel(), MAIN):
        panel(ax, state_estimates(key), f'{label}  {title}\n{subtitle}', STYLE['main'], missing=False)
    legend(fig, STYLE['main'])
    fig.subplots_adjust(left=0.065, right=0.985, top=0.95, bottom=0.09, wspace=0.34, hspace=0.43)
    save(fig, 'simulation_radius_recovery')

    fig = plt.figure(figsize=(12.5, 8.4), facecolor='white')
    grid = fig.add_gridspec(2, 6, left=0.075, right=0.985, top=0.95, bottom=0.09, wspace=0.72, hspace=0.43)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6]),
            fig.add_subplot(grid[1, 1:3]), fig.add_subplot(grid[1, 3:5])]
    for ax, (label, p, key) in zip(axes, MISSING):
        panel(ax, state_estimates(key), f'{label}  p = {p:.1f}', STYLE['missing'], missing=True)
    legend(fig, STYLE['missing'])
    save(fig, 'missing_data_radius')


if __name__ == '__main__':
    main()
