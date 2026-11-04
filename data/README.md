# Data

One row per game in both files.

## Sources

- **Premier League.** The 42 season tables in `Data_Premier` of
  [francescabasini/Football_SBM](https://github.com/francescabasini/Football_SBM/tree/f520731d27d2bdf1f3dcf0b7cf6eb707f97eb502/Data_Premier),
  commit `f520731d27d2bdf1f3dcf0b7cf6eb707f97eb502` (the data of Basini et al., 2023), merged into
  one row per match. All 17,066 matches agree with the source; `Birmingham` is written
  `Birmingham City`.
- **NBA.** 22,055 of the 24,038 analysed games match `games.csv` of the Kaggle dataset
  [nathanlauga/nba-games](https://www.kaggle.com/datasets/nathanlauga/nba-games), version 10
  (collected with [Nathanlauga/nba-data-scrapping](https://github.com/Nathanlauga/nba-data-scrapping)).
  The 16,658 games from 2010/11 to 2023/24 match `regular_season_totals_2010_2024.csv` of
  [NocturneBear/NBA-Data-2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024),
  commit `a5f108b5b1f08074d78b9e8e901926a9ce4c06c5`. Together the two sources cover every analysed
  game, with no differences in date, home and away franchise or score.

## `1978_2020_football.csv`

English top-flight league matches, 1978/79 to 2019/20 (17,066 matches). The analysis uses the
25 Premier League seasons from 1995/96 to 2019/20 (9,500 matches).

| Column | Description |
|---|---|
| `season` | Season, e.g. `1995-96` |
| `home_team` | Home club |
| `away_team` | Away club |
| `home_goals` | Goals scored by the home club |
| `away_goals` | Goals scored by the away club |
| `result` | `H` home win, `D` draw, `A` away win |

## `nba_regular_season_2000_2024.csv`

NBA regular-season games, 1999/00 to 2023/24 (28,794 games). The analysis uses the 20 seasons
from 2004/05 to 2023/24 (24,038 games).

| Column | Description |
|---|---|
| `gameid` | NBA game identifier |
| `date` | Game date (YYYY-MM-DD) |
| `season` | Year in which the season ends, e.g. `2005` for 2004/05 |
| `home_team` | Home franchise abbreviation |
| `away_team` | Away franchise abbreviation |
| `home_score` | Points scored by the home team |
| `away_score` | Points scored by the away team |
| `margin` | `home_score - away_score` |
| `y_sign` | `1` if `margin > 5`, `-1` if `margin < -5`, `0` otherwise |
| `y_cat` | `y_sign + 1` (`2`, `0`, `1`) |

Abbreviations are those of the season played, so relocated or renamed franchises appear under
more than one code (`SEA`/`OKC`, `NJN`/`BKN`, `NOH`/`NOK`/`NOP`). The model uses the teams of
each season as they are; `realdata/figures.py` merges these codes for the longitudinal plots.
The code recomputes the three outcome categories from `margin`.
