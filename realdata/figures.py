"""Figures of the two applications: radius path, home advantage, tier heatmap and two seasons.

    python figures.py
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from fit import OUT  # noqa: E402

FIGURES = OUT / 'figures'
SEASONS = {'epl': ['2010-11', '2006-07'], 'nba': ['2016-17', '2010-11']}
TITLE = {'epl': 'Premier League', 'nba': 'NBA regular season'}
# Panel boxes (left, top, width, height) on a 1600 x 1600 grid.
LAYOUT = {'epl': [(20, 20, 770, 390), (810, 20, 770, 390), (20, 535, 920, 907), (960, 430, 620, 568),
                  (960, 1012, 620, 568)],
          'nba': [(20, 15, 771, 375), (809, 15, 771, 375), (20, 620, 850, 760), (890, 410, 690, 560),
                  (890, 1000, 690, 560)]}
SIZE = 24.0
# Right edge of the radius-panel legend, placed where no interval runs.
RADIUS_LEGEND = {'epl': 0.97, 'nba': 0.56}

TIER_CMAP = LinearSegmentedColormap.from_list('tiers', ['#2E8B57', '#8FD175', '#F4E46A', '#F3A45B', '#E84A4A'])
TIER_CMAP.set_bad('white')
STATE_COLORS = ['#7DB7E8', '#E59ACB']
ETA_COLOR = '#5DADE2'
LINE = '#4A4A4A'
CI_LEVEL = 90

EPL_ABBREV = {
    'Arsenal': 'ARS', 'Aston Villa': 'AVL', 'Barnsley': 'BAR', 'Birmingham City': 'BIR', 'Blackburn Rovers': 'BLA',
    'Blackpool': 'BLP', 'Bolton Wanderers': 'BOL', 'Bournemouth': 'BOU', 'Bradford City': 'BRA',
    'Brighton & Hove Albion': 'BHA', 'Burnley': 'BUR', 'Cardiff City': 'CAR', 'Charlton Athletic': 'CHA',
    'Chelsea': 'CHE', 'Coventry City': 'COV', 'Crystal Palace': 'CRY', 'Derby County': 'DER', 'Everton': 'EVE',
    'Fulham': 'FUL', 'Huddersfield Town': 'HUD', 'Hull City': 'HUL', 'Ipswich Town': 'IPS', 'Leeds United': 'LEE',
    'Leicester City': 'LEI', 'Liverpool': 'LIV', 'Manchester City': 'MCI', 'Manchester United': 'MUN',
    'Middlesbrough': 'MID', 'Newcastle United': 'NEW', 'Norwich City': 'NOR', 'Nottingham Forest': 'NFO',
    'Portsmouth': 'POR', 'Queens Park Rangers': 'QPR', 'Reading': 'REA', 'Sheffield United': 'SHU',
    'Sheffield Wednesday': 'SHW', 'Southampton': 'SOU', 'Stoke City': 'STK', 'Sunderland': 'SUN',
    'Swansea City': 'SWA', 'Tottenham Hotspur': 'TOT', 'Watford': 'WAT', 'West Bromwich Albion': 'WBA',
    'West Ham United': 'WHU', 'Wigan Athletic': 'WIG', 'Wimbledon': 'WIM', 'Wolverhampton Wanderers': 'WOL'}
# Relocated or renamed NBA franchises keep one label.
NBA_FRANCHISE = {'NJN': 'BKN', 'SEA': 'OKC', 'NOH': 'NOP', 'NOK': 'NOP'}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def short(season):
    a, b = season.split('-')
    return f'{a[-2:]}–{b[-2:]}'


def tier_color(tier, n_tiers):
    return TIER_CMAP((tier - 1) / max(n_tiers - 1, 1))


def label(league, team):
    return EPL_ABBREV[team] if league == 'epl' else NBA_FRANCHISE.get(team, team)


def axes_in(fig, box, margins):
    """Axes inside a layout box; margins (left, top, right, bottom) in grid units."""
    left, top, width, height = box
    ml, mt, mr, mb = margins
    return fig.add_axes([(left + ml) / 1600, 1 - (top + height - mb) / 1600,
                         (width - ml - mr) / 1600, (height - mt - mb) / 1600])


def season_axis(ax, seasons, ylabel):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E6E6E6', linewidth=0.9)
    ax.grid(axis='x', color='#F1F1F1', linewidth=0.7)
    ax.set_xticks(range(len(seasons)), [short(s) for s in seasons], rotation=45, ha='right')
    ax.set_xlabel('Season', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold', labelpad=10)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#555555')
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis='both', length=3, width=0.8, color='#555555', labelsize=11.5)


def boxed_legends(ax, legends, pad=0.012):
    """Draw one white box around a group of legends."""
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    boxes = [inv.transform(leg.get_window_extent()) for leg in legends]
    x0, y0 = min(b[0, 0] for b in boxes), min(b[0, 1] for b in boxes)
    x1, y1 = max(b[1, 0] for b in boxes), max(b[1, 1] for b in boxes)
    ax.add_patch(Rectangle((x0 - 2 * pad, y0 - 2 * pad), x1 - x0 + 4 * pad, y1 - y0 + 3 * pad,
                           transform=ax.transAxes, facecolor='white', edgecolor='#C8C8C8', linewidth=0.9,
                           zorder=4))


def radius_panel(ax, seasons, legend_right):
    x = np.arange(len(seasons))
    for t, row in seasons.iterrows():
        color = STATE_COLORS[row.state - 1]
        ax.vlines(t, row.radius_low, row.radius_high, color=color, lw=2.6, alpha=0.42, zorder=1)
        ax.plot(t, row.radius_median, 'o', color=color, markeredgewidth=0, ms=6.2, zorder=3)
    ax.step(x, seasons.dahl_radius, where='mid', color=LINE, lw=2.0, zorder=2)
    season_axis(ax, list(seasons.season), 'Competitive-balance radius')
    main = ax.legend(handles=[Line2D([0], [0], color=LINE, lw=2.0, label='Dahl radius'),
                              Line2D([0], [0], color='#B5B5B5', lw=2.6, label=f'{CI_LEVEL}% CrI'),
                              Line2D([0], [0], marker='o', ls='None', markerfacecolor='#777777',
                                     markeredgecolor='none', markersize=7, label='Posterior median')],
                     frameon=False, loc='lower right', bbox_to_anchor=(legend_right, 0.19), fontsize=11.5, handlelength=1.8,
                     labelspacing=0.45, borderaxespad=0)
    ax.add_artist(main)
    states = ax.legend(handles=[Line2D([0], [0], marker='o', ls='None', markerfacecolor=c, markeredgecolor='none',
                                       markersize=8, label=str(k + 1)) for k, c in enumerate(STATE_COLORS)],
                       title='State', frameon=False, ncol=2, loc='lower left', fontsize=11.2, title_fontsize=11.5,
                       handlelength=0.5, handletextpad=0.25, columnspacing=1.0, borderaxespad=0)
    ax.figure.canvas.draw()
    x0 = ax.transAxes.inverted().transform(main.get_window_extent())[0, 0]
    states.set_bbox_to_anchor((x0, 0.06), transform=ax.transAxes)
    for leg in (main, states):
        leg.set_zorder(6)
    boxed_legends(ax, [main, states])


def eta_panel(ax, seasons):
    x = np.arange(len(seasons))
    ax.fill_between(x, seasons.eta_low, seasons.eta_high, color=ETA_COLOR, alpha=0.20, linewidth=0, zorder=1)
    ax.plot(x, seasons.eta_mean, color=ETA_COLOR, lw=2.1, marker='o', ms=6.2, markeredgewidth=0, zorder=3)
    ax.axhline(0, color='#B5B5B5', linewidth=1.0, linestyle='--', zorder=0)
    season_axis(ax, list(seasons.season), 'Home-advantage effect')
    ax.legend(handles=[Line2D([0], [0], color=ETA_COLOR, lw=2.1, marker='o', markeredgewidth=0, markersize=6.2,
                              label='Posterior mean'),
                       Patch(facecolor=ETA_COLOR, edgecolor='none', alpha=0.20, label=f'{CI_LEVEL}% CrI')],
              loc='upper right', fontsize=11.5, handlelength=2.0, labelspacing=0.55, borderpad=0.75, frameon=True,
              facecolor='white', edgecolor='#C8C8C8', framealpha=1.0, fancybox=False)


def heatmap_panel(ax, cax, league, teams, seasons):
    teams = teams.assign(team=[label(league, t) for t in teams.team],
                         position=(teams.tier - 1) / np.maximum(teams.n_tiers - 1, 1))
    grid = teams.pivot_table(index='team', columns='season', values='position').reindex(columns=seasons)
    grid = grid.loc[grid.mean(axis=1).sort_values(kind='stable').index]
    values = grid.to_numpy()
    image = ax.imshow(values, aspect='auto', interpolation='nearest', cmap=TIER_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(seasons)), [short(s) for s in seasons], rotation=45, ha='right',
                  fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(grid)), grid.index, fontsize=12, fontweight='bold')
    ax.set_xlabel('Season', fontsize=20, fontweight='bold', labelpad=8)
    ax.set_ylabel('Team', fontsize=20, fontweight='bold', labelpad=8)
    ax.tick_params(axis='both', length=0)
    for i, j in zip(*np.where(~np.isfinite(values))):
        ax.text(j, i, '×', ha='center', va='center', fontsize=8.5)
    bar = ax.figure.colorbar(image, cax=cax)
    bar.set_ticks([])
    bar.ax.invert_yaxis()
    bar.outline.set_visible(False)
    bar.ax.text(0.5, 1.02, 'Stronger', transform=bar.ax.transAxes, ha='center', va='bottom', fontsize=17,
                fontweight='bold')
    bar.ax.text(0.5, -0.02, 'Weaker', transform=bar.ax.transAxes, ha='center', va='top', fontsize=17,
                fontweight='bold')


def season_panel(ax, league, table, season):
    table = table.reset_index(drop=True)
    n_tiers = int(table.n_tiers.iloc[0])
    y = np.arange(len(table))
    lo, hi = table.theta_low.min() - 0.25, table.theta_high.max() + 0.25
    ax.set_xlim(lo, hi)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E7E7E7', linewidth=0.9)
    ax.hlines(y, lo, hi, color='#EEEEEE', linewidth=0.9, zorder=0)
    ax.axvline(0, color='#B8B8B8', linewidth=1.3, zorder=1)
    for i, row in table.iterrows():
        color = tier_color(row.tier, n_tiers)
        ax.hlines(i, row.theta_low, row.theta_high, color=color, linewidth=2.1, alpha=0.82, zorder=3)
        ax.plot(row.theta_mean, i, 'o', color=color, markeredgewidth=0, ms=6.2, zorder=4)
        ax.plot(row.tier_theta, i, '^', color='#4F4F4F', markeredgewidth=0, ms=8.3, zorder=5)
    ax.set_yticks(y, [label(league, t) for t in table.team], fontsize=11)
    ax.set_ylim(len(table) - 0.35, -0.65)
    ax.tick_params(axis='y', length=0, pad=7)
    ax.tick_params(axis='x', length=0, labelsize=11.5, pad=5)
    ax.set_xlabel('Latent ability', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title(f'{season.replace("-", "–")} {TITLE[league]}', fontsize=17, fontweight='bold', pad=9)
    for side in ax.spines.values():
        side.set_visible(False)

    stats = ax.legend(handles=[Line2D([0], [0], color='#777777', lw=2.2, label=f'{CI_LEVEL}% CrI'),
                               Line2D([0], [0], marker='o', ls='None', markerfacecolor='#777777',
                                      markeredgecolor='none', markersize=7.2, label='Posterior mean'),
                               Line2D([0], [0], marker='^', ls='None', markerfacecolor='#4F4F4F',
                                      markeredgecolor='none', markersize=8.8, label='Tier ability')],
                      frameon=False, loc='lower left', fontsize=11.3, handlelength=1.8, labelspacing=0.48,
                      borderaxespad=0)
    ax.add_artist(stats)
    columns = n_tiers if n_tiers <= 8 else int(np.ceil(n_tiers / 2))
    # Matplotlib fills legend columns first; this order makes the tiers read along the rows.
    order = np.arange(2 * columns).reshape(-1, columns).T.ravel() + 1 if columns < n_tiers else range(1, n_tiers + 1)
    tiers = ax.legend(handles=[Line2D([0], [0], marker='o', ls='None', markerfacecolor=tier_color(k, n_tiers),
                                      markeredgecolor='none', markersize=7.4, label=str(k))
                               for k in order if k <= n_tiers],
                      title='Tier (1 = strongest)', frameon=False, ncol=columns, loc='lower right',
                      bbox_to_anchor=(0.965, 0.075), fontsize=10.4, title_fontsize=11.0, handlelength=0.55,
                      handletextpad=0.12, columnspacing=0.42, borderaxespad=0)
    ax.figure.canvas.draw()
    inv = ax.transAxes.inverted()
    t_box = inv.transform(tiers.get_window_extent())
    stats.set_bbox_to_anchor((t_box[0, 0], t_box[1, 1] + 0.035), transform=ax.transAxes)
    for leg in (stats, tiers):
        leg.set_zorder(8)
    boxed_legends(ax, [stats, tiers])


def figure(league):
    seasons = pd.read_csv(OUT / league / 'season_summary.csv')
    teams = pd.read_csv(OUT / league / 'team_summary.csv')
    radius, eta, heat, first, second = LAYOUT[league]
    fig = plt.figure(figsize=(SIZE, SIZE))
    radius_panel(axes_in(fig, radius, (95, 10, 10, 110)), seasons, RADIUS_LEGEND[league])
    eta_panel(axes_in(fig, eta, (95, 10, 10, 110)), seasons)
    left, top, width, height = heat
    heatmap_panel(axes_in(fig, heat, (75, 10, 110, 85)), axes_in(fig, (left + width - 80, top, 80, height),
                                                                   (40, 130, 22, 215)),
                  league, teams, list(seasons.season))
    for box, season in zip((first, second), SEASONS[league]):
        season_panel(axes_in(fig, box, (60, 40, 10, 70)), league, teams[teams.season == season], season)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURES / f'{league}_figure.{ext}', dpi=300 if ext == 'pdf' else 100)
    plt.close(fig)


if __name__ == '__main__':
    for name in ('epl', 'nba'):
        figure(name)
