"""Premier League 1995-96 to 2019-20 and NBA 2004-05 to 2023-24 regular seasons."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'


def season_arrays(games, outcome):
    teams = sorted(set(games.home_team) | set(games.away_team))
    index = {team: i for i, team in enumerate(teams)}
    X = np.zeros((len(games), len(teams)))
    X[np.arange(len(games)), games.home_team.map(index)] = 1.0
    X[np.arange(len(games)), games.away_team.map(index)] = -1.0
    return teams, X, np.ones(len(games)), outcome.to_numpy(dtype=int)


def load(league):
    """Season labels, teams, design matrices X, home indicators h and outcomes y in {-1, 0, 1}."""
    if league == 'epl':
        games = pd.read_csv(DATA / '1978_2020_football.csv')
        for col in ('season', 'home_team', 'away_team', 'result'):
            games[col] = games[col].astype(str).str.strip()
        games = games[games.season.str[:4].astype(int).between(1995, 2019)]
        games['y'] = games.result.map({'H': 1, 'D': 0, 'A': -1})
        label = str
    else:
        games = pd.read_csv(DATA / 'nba_regular_season_2000_2024.csv')
        games = games[games.season.between(2005, 2024)].copy()
        games['home_team'], games['away_team'] = games.home_team.astype(str), games.away_team.astype(str)
        # Wins by more than five points are decisive; the rest count as close games.
        games['y'] = np.sign(games.margin) * (games.margin.abs() > 5)
        label = lambda year: f'{year - 1}-{str(year)[-2:]}'

    out = dict(seasons=[], teams=[], Xl=[], hl=[], yl=[], Ns=[])
    for season, sub in games.groupby('season', sort=True):
        teams, X, h, y = season_arrays(sub.reset_index(drop=True), sub.y)
        for key, value in zip(('seasons', 'teams', 'Xl', 'hl', 'yl', 'Ns'), (label(season), teams, X, h, y, len(teams))):
            out[key].append(value)
    return out
