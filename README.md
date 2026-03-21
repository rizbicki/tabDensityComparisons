# Tabular Foundation Models for Conditional Density Estimation

Benchmark comparing tabular foundation models (TabPFN, TabICL) against
classical methods for conditional density estimation (CDE) on tabular data.

## Quick Start

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh --setup-only     # create .venv and install dependencies only
./setup_and_run.sh --sim-only       # simulated datasets only
./setup_and_run.sh                  # full run (synthetic + real datasets)
./setup_and_run.sh --real-only      # real/semi-synthetic datasets only
./setup_and_run.sh --cpu            # force CPU (slower)
```

For the SDSS-only run, first create the local virtualenv via `./setup_and_run.sh --setup-only`
or the manual setup steps below, then launch:

```bash
.venv/bin/python run_sdss_scaling_experiment.py --device cuda   # SDSS scaling study
```

Results now split by dataset type:

- simulated outputs go to `results_simulated/`
- real-data outputs go to `results_real/`

You can regenerate plots/tables from both after runs complete:

```bash
.venv/bin/python consolidate_partial_results.py
.venv/bin/python generate_plots.py      # regenerates all plots
```

If you only need the tables plus the metric-based figures on another machine,
you can transfer just `results_real/results.json` and/or
`results_simulated/results.json` and regenerate from metrics only:

```bash
.venv/bin/python generate_plots.py --metrics-only
```

This reproduces the HTML table plus the ranking, raw-metric, and performance
plots. It intentionally skips PIT histograms and density example plots, which
require the cached arrays under `results_*/cache/*.npz`.

### Plots from partial results

You can regenerate preliminary plots at any point by re-running:

```bash
.venv/bin/python consolidate_partial_results.py
.venv/bin/python generate_plots.py
```

Only datasets with at least one completed method will appear.

## Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy setuptools
pip install --no-build-isolation xbart
pip install -r requirements.txt
python run_experiments.py --sim-only --device cuda
```

### If you already have a CUDA PyTorch

Comment out the `torch` line in `requirements.txt` before installing,
so pip doesn't overwrite your CUDA build with a CPU-only one.

## Project Structure

```
run_experiments.py          Main benchmark entry point (simulated + real datasets)
run_real_experiments.py     Real/semi-synthetic experiments entry point
run_sdss_scaling_experiment.py  SDSS-only scaling benchmark over multiple n
consolidate_partial_results.py  Build results.json from partial checkpoints
generate_plots.py           Regenerate all plots from cached results
models/
  flexcode.py               FlexCodeEstimator + RF regressor wrapper
  native.py                 TabPFN / RealTabPFN native density extraction
  baselines.py              Parametric, GLM, quantile, flow, MDN, BART, CatMLP baselines
  tuning.py                 Hyperparameter search helpers for learned baselines
datasets/
  synthetic.py              Synthetic DGPs (with known true densities)
  real.py                   Semi-synthetic + real-world dataset loaders
  sdss_galaxies.csv         SDSS DR18 photometric redshift data (500k galaxies)
evaluation/
  metrics.py                CDE loss, log-lik, CRPS, PIT, coverage
visualization/
  plots.py                  HTML tables, rankings, raw metrics, density comparisons, PIT
utils/
  io.py                     Caching, formatting, summary printing
```

## Methods

### Tabular Foundation Models (native CDE)

| Method | Description |
|--------|-------------|
| TabPFN-Native | TabPFN's built-in bar distribution |
| TabPFN-2.5 | TabPFN v2.5 default regressor checkpoint |
| RealTabPFN-2.5 | TabPFN v2.5 real-data regressor checkpoint |
| TabICL-Quantiles | TabICLv2's native 999-quantile distribution |

### FlexCode (basis-expansion CDE)

| Method | Description |
|--------|-------------|
| FlexCode-RF | FlexCode with Random Forest regressor |

### Parametric Baselines

| Method | Description |
|--------|-------------|
| LinearGauss-Homo | Linear mean + constant Gaussian noise |
| LinearGauss-Hetero | Linear mean + input-dependent Gaussian noise |
| Student-t | Linear mean + constant Student-t noise (df estimated by MLE) |
| LogNormal-Homo | Log-normal with linear log-mean, constant log-variance |
| LogNormal-Hetero | Log-normal with linear log-mean, input-dependent log-variance |
| Gamma-GLM | Gamma GLM with log-link for the mean |

### Penalized (Ridge) Variants

The classical linear / GLM families above have a Ridge-regularized variant with
the penalty chosen by leave-one-out cross-validation (`RidgeCV`).

| Method | Description |
|--------|-------------|
| LinGauss-Homo-Ridge | LinearGauss-Homo with Ridge penalty (LOO-CV) |
| LinGauss-Hetero-Ridge | LinearGauss-Hetero with Ridge penalty (LOO-CV) |
| Student-t-Ridge | Student-t with Ridge penalty (LOO-CV) |
| LogNormal-Homo-Ridge | LogNormal-Homo with Ridge penalty (LOO-CV) |
| LogNormal-Hetero-Ridge | LogNormal-Hetero with Ridge penalty (LOO-CV) |
| Gamma-GLM-Ridge | Gamma-GLM with Ridge penalty (LOO-CV) |

### BART Methods

| Method | Description |
|--------|-------------|
| BART-Homo | XBART for the mean + constant residual variance (Gaussian) |
| BART-Hetero | Two-stage XBART: mean + input-dependent variance (Gaussian) |

### Nonparametric Baselines

| Method | Description |
|--------|-------------|
| Quantile-Tree | Quantile regression via XGBoost/GBM |
| Flow-Spline | Conditional neural spline flow with Gaussian base |
| CatMLP | MLP with discretized response (softmax over bins, CV-tuned) |
| MDN | Mixture Density Network (legacy method key `MDN-2mix` is still accepted); random-search CV tunes the number of Gaussian components over `{2, 3, 5}` plus hidden size and learning rate |

## Datasets

The main simulated and real-data benchmarks run `4` repetitions per dataset by
default (`--n-reps` to change this); metrics report mean ± SE across
repetitions.

### Synthetic (d ∈ {5, 10, 50}, n ∈ {50, 500, 1000, 5000, 10000, 20000})

All synthetic datasets have known true conditional densities. Tags follow
`{Base}-d{d}-{n}` (e.g. `Heteroscedastic-d10-5000`).

| Dataset | Description |
|---------|-------------|
| Heteroscedastic | Input-dependent Gaussian noise |
| Bimodal | Mixture of two Gaussians with fixed 50/50 weights |
| Skewed | Gamma-shifted response with input-dependent shape |
| Nonlinear | Nonlinear mean with heteroscedastic noise |
| LinGauss-Homo | Linear Gaussian with homoscedastic noise |
| Interaction | Interactions between covariates affecting mean and variance |
| Friedman1 | Friedman #1 DGP; extra features (d>5) are irrelevant noise |
| Friedman2 | Friedman #2 DGP; fixed d=4, tag `Friedman2-d4-{n}` |

The benchmark `Bimodal` generator in [datasets/synthetic.py](/home/rizbicki/git/tabDensityComparisons/datasets/synthetic.py)
uses fixed weights:
`0.5 * N(z; x0 + x1, 0.5^2) + 0.5 * N(z; -x0 + x1, 0.5^2)`.

For a concrete input-dependent mixture example, see
[datasets/synthetic.py](/home/rizbicki/git/tabDensityComparisons/datasets/synthetic.py)
function `make_bimodal_input_weighted_example`, which uses
`w(x) = Phi(1.5 * x0 - 0.75 * x1)` and
`f(z|x) = w(x) p1(z|x) + (1 - w(x)) p2(z|x)`.
It is example-only and is not included in the default benchmark schedule.

### Real-world (n ∈ {50, 500, 1000, 5000, 10000, 20000} where available)

From OpenML and SDSS DR18, subsampled consistently so smaller n is always
a strict subset of larger n.

| Dataset | Source | Max n |
|---------|--------|-------|
| SpaceGA | OpenML | 3107 |
| Elevators | OpenML | 16599 |
| Kin8nm | OpenML | 8192 |
| Puma8NH | OpenML | 8192 |
| Bank8FM | OpenML | 22784 |
| CpuSmall | OpenML | 8192 |
| CPUact | OpenML | 8192 |
| CalHousing | OpenML | 20640 |
| Diamonds | OpenML | 53940 |
| Abalone | OpenML | 4177 |
| Ailerons | OpenML | 13750 |
| BikeSharing | OpenML | 17379 |
| AmesHousing | OpenML | 2930 |
| Digits | OpenML (`optdigits`, numeric labels) | 5620 |
| House16H | OpenML | 22784 |
| HouseSales | OpenML | 21613 |
| NYCTaxi | OpenML (`nyc-taxi-green-dec-2016`) | 581835 |
| Sulfur | OpenML | 10081 |
| BrazilianHouses | OpenML | 10692 |
| Pol | OpenML | 15000 |
| MercedesBenz | OpenML | 4209 |
| Protein | OpenML | 45730 |
| VisualizingSoil | OpenML (`visualizing_soil`) | 8641 |
| Year | OpenML | 515345 |
| SGEMM_GPU | OpenML | 241600 |
| BlackFriday | OpenML | 166821 |
| SDSS | SDSS DR18 | 500000 |

The SDSS dataset uses ugriz photometric magnitudes as features and
spectroscopic redshift as the target.

## SDSS Scaling Study

To compare methods on SDSS across larger sample sizes:

```bash
.venv/bin/python run_sdss_scaling_experiment.py --device cuda
```

The default sample-size spec is
`500,1000,10000,50000,100000,250000,500000,full`. The parser deduplicates
repeated sizes, so with the bundled 500k-row SDSS CSV this currently becomes
the seven unique sizes `500, 1k, 10k, 50k, 100k, 250k, 500k`. Outputs are
written to
`results_real/sdss_scaling/`, metrics are averaged over `4` repetitions per
sample size by default, and the script regenerates the SDSS-only HTML table and
performance-vs-`n` plots there.

The default schedule still prunes methods that hit explicit limits or are
conservatively capped for runtime reasons. `BART-Homo`, `BART-Hetero`,
`Quantile-Tree`, and `CatMLP` remain available through `n=100,000`,
`TabICL-Quantiles` is capped below `n=50,000`, and the TabPFN variants are
capped at `n=50,000`. You can override the schedule with `--methods ...`,
disable pruning with `--all-methods-at-all-sizes`, inspect the exact chosen
methods with
`results_real/sdss_scaling/method_policy.json`, or change the repetition count
with `--n-reps`.

To refresh the current SDSS scaling HTML table and performance-vs-`n` plots
from whatever repetitions have already finished, without starting new fits:

```bash
.venv/bin/python run_sdss_scaling_experiment.py \
  --plot-partial-results-only \
  --output-dir results_real/sdss_scaling
```

That scans `results_real/sdss_scaling/cache/partial/rep*/`, aggregates the
currently completed repetitions into `results.json`, and regenerates
`results_table.html` plus the `perf_vs_n_*` figures for the finished sample
sizes and methods so far. If you want those outputs refreshed automatically
while the scaling experiment is running, add `--refresh-plots-after-each-rep`
to the main scaling command.

## Evaluation Metrics

- **CDE loss** (Izbicki & Lee, 2017): proper scoring rule for conditional densities
- **Log-likelihood**: mean log f(z_test)
- **CRPS**: Continuous Ranked Probability Score
- **PIT KS**: Kolmogorov-Smirnov statistic for calibration
- **90% coverage**: proportion of test samples inside the central 90% predictive interval (ideal is close to `0.90`; rankings and table highlighting treat closer to `0.90` as better)
- **Interval width**: mean width of that 90% predictive interval
- **Fit time**: total wall-clock time (fit + predict) in seconds

## Output

Simulated results go to `results_simulated/` by default. Real-dataset results
go to `results_real/` by default. `generate_plots.py` reads both directories
and writes each artifact back to the matching destination. The SDSS-specific
benchmark scripts write to their own subdirectories under `results_real/`.

```
results_simulated/
  results.json                      raw metric values (mean ± SE across repetitions)
  results_table.html                formatted HTML table (±SE across repetitions)
  rankings_sim_{metric}_n{n}.png    ranking heatmap — simulated, per n
  raw_sim_{metric}_n{n}.png         raw value heatmap — simulated, per n
  perf_vs_n_{metric}_sim_d{d}.png   performance vs n — simulated, per d; metric in {cde_loss, log_lik, crps, pit_ks, coverage_90, interval_width, fit_time}
  pit_calibration.png               PIT histograms for calibration assessment
  native_tab_{ds}.png               density comparison plots (synthetic)
  cache/{dataset}.npz               cached arrays (skip re-runs)

results_real/
  results.json                      raw metric values (mean ± SE across repetitions)
  results_table.html                formatted HTML table (±SE across repetitions)
  rankings_real_{metric}_n{n}.png   ranking heatmap — real, per n
  raw_real_{metric}_n{n}.png        raw value heatmap — real, per n
  perf_vs_n_{metric}_real.png       performance vs n — real datasets; metric in {cde_loss, log_lik, crps, pit_ks, coverage_90, interval_width, fit_time}
  perf_vs_n_foundational_*.png      same metrics, highlighting foundation models only
  pit_calibration.png               PIT histograms for calibration assessment
  cache/{dataset}.npz               cached arrays (skip re-runs)

results_real/sdss_scaling/
  results.json                      aggregated SDSS-by-n metrics
  method_policy.json                methods selected/skipped at each sample size
  results_table.html                SDSS scaling HTML table
  perf_vs_n_{metric}_real.png       SDSS performance vs n; metric in {cde_loss, log_lik, crps, pit_ks, coverage_90, interval_width, fit_time}
  perf_vs_n_foundational_*.png      same metrics, SDSS performance vs n with foundational focus
  cache/partial/rep*/               per-repetition partial checkpoints
```
