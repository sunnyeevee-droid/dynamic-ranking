# When Do Sports Leagues Stop Being Competitive? Bayesian Detection of Competitive-Balance Eras

Code and data for the paper. Team abilities in each season are obtained by projecting
Gaussian auxiliary variables onto a graph-fused $\ell_1$ polytope, so that teams the data
cannot tell apart share an exact rank tier, and the radius of the polytope measures
structural competitive balance. The radius follows an adaptive finite-regime process
across seasons. The repository reproduces the simulation study (eight settings and the
missing-games study, 50 replicates each) and the analyses of the English Premier League
(1995/96 to 2019/20) and the NBA (2004/05 to 2023/24).

## Repository structure

```
data/                  EPL and NBA match results; sources and columns in data/README.md
common/
  sampler.py           MCMC updates and exact projection
  summary.py           modal K, Dahl state path and radii, SALSO tiers
  salso.py             Python wrapper of the R package salso
  salso_batch.R        SALSO estimates called by salso.py
simulation/
  model.py             sampler settings and MCMC run
  simulate.py          designs, data generation and evaluation
  run_simulations.py   12 settings x 50 replicates
  summarize.py         posterior summaries of the saved chains
  results_for_paper.py tables
  diagnostics.py       coverage, calibration and sampling diagnostics
  plot_radius.py       radius figures
realdata/
  data.py              EPL and NBA seasons
  motivating_data.R    descriptive figure and table of Section 2
  fit.py               two chains per league
  summarize.py         posterior summaries
  figures.py           figures of the applications
  diagnostics.py       R-hat, ESS and trace plots
results/               output (created when the code runs)
```

## Installation

Python 3.11 with the packages in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate dynamic
```

R 4.3 or later with the packages `salso`, `ggplot2`, `patchwork` and `ragg`, and `Rscript`
on the path:

```r
install.packages(c("salso", "ggplot2", "patchwork", "ragg"))
```

## Data

`data/1978_2020_football.csv` contains English top-flight matches from 1978/79 to
2019/20, of which the 25 Premier League seasons from 1995/96 to 2019/20 (9,500 matches)
are analysed. `data/nba_regular_season_2000_2024.csv` contains NBA regular-season games
from 1999/00 to 2023/24, of which the 20 seasons from 2004/05 to 2023/24 (24,038 games)
are analysed. Sources, columns and the coding of outcomes are given in `data/README.md`.

## Reproducing the results

Simulation study, in `simulation/`:

```bash
python run_simulations.py run      # 600 replicates; python run_simulations.py status shows progress
python summarize.py
python results_for_paper.py
python diagnostics.py
python plot_radius.py
```

With eight replicates in parallel on an Apple M5 laptop, one replicate takes about
15 minutes with 20 seasons and about 31 minutes with 40 seasons, and the 600 replicates
take about 26 hours. Finished replicates are saved in `results/simulation/parts/` and
skipped when the command is run again.

Applications, in `realdata/`:

```bash
Rscript motivating_data.R
python fit.py epl
python fit.py nba
python summarize.py
python figures.py
python diagnostics.py
```

`fit.py` runs two chains of 50,000 iterations in parallel; each chain takes about one hour
and is checkpointed every 1,000 iterations. Seeds are fixed in the scripts.

## Tables and figures in the paper

| Paper | Script | Output in `results/` |
|---|---|---|
| Premier League data table and figure (Section 2) | `realdata/motivating_data.R` | `realdata/figures/motivating_data.pdf`; table entries printed to the console |
| Premier League results figure (Section 6) | `realdata/figures.py` | `realdata/figures/epl_figure.pdf` |
| Premier League and NBA numbers in the text | `realdata/summarize.py` | `realdata/{epl,nba}/season_summary.csv`, `team_summary.csv`, `K_posterior.csv`, `transition_frequencies.csv` |
| Simulation results table | `simulation/results_for_paper.py` | `simulation/tables/table_simulation_results.tex`, `main_table.csv`, `misfits.csv` |
| Radius recovery figure | `simulation/plot_radius.py` | `simulation/figures/simulation_radius_recovery.pdf` |
| Missing-games table | `simulation/results_for_paper.py` | `simulation/tables/table_missing_data.tex`, `missing_table.csv` |
| Missing-games radius figure | `simulation/plot_radius.py` | `simulation/figures/missing_data_radius.pdf` |
| Calibration and sampling tables | `simulation/diagnostics.py` | `simulation/tables/table_calibration.tex`, `table_sampling.tex`, `coclustering_reliability.csv` |
| NBA results figure (Supplementary Material) | `realdata/figures.py` | `realdata/figures/nba_figure.pdf` |
| Real-data diagnostics and their figures | `realdata/diagnostics.py` | `realdata/diagnostics/` |

## License

MIT; see `LICENSE`.
