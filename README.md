# Tabular Foundation Models for Conditional Density Estimation

Benchmark comparing tabular foundation models (TabPFN, TabICL) against
classical methods for conditional density estimation (CDE) on tabular data.

## Quick Start

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh --quick          # sanity check (synthetic datasets only)
./setup_and_run.sh                  # full run (synthetic + real datasets)
./setup_and_run.sh --real-only      # real/semi-synthetic datasets only
./setup_and_run.sh --cpu            # force CPU (slower)
.venv/bin/python run_sdss_scaling_experiment.py --device cuda   # SDSS scaling study
```

Results now split by dataset type:

- simulated outputs go to `results_simulated/`
- real-data outputs go to `results_real/`

You can regenerate plots/tables from both after runs complete:

```bash
python consolidate_partial_results.py
python generate_plots.py                # regenerates all plots
```

If you only need the tables plus the metric-based figures on another machine,
you can transfer just `results_real/results.json` and/or
`results_simulated/results.json` and regenerate from metrics only:

```bash
python generate_plots.py --metrics-only
```

This reproduces the HTML table plus the ranking, raw-metric, and performance
plots. It intentionally skips PIT histograms and density example plots, which
require the cached arrays under `results_*/cache/*.npz`.

### Plots from partial results (while experiments are running)

You can generate preliminary plots at any point without waiting for all
experiments to finish:

```bash
# consolidate whatever has been saved so far
python consolidate_partial_results.py

# generate plots from the consolidated results
python generate_plots.py
```

Re-run these two commands whenever you want updated plots. Only datasets
with at least one completed method will appear.

## Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiments.py --quick --device cuda
```

### If you already have a CUDA PyTorch

Comment out the `torch` line in `requirements.txt` before installing,
so pip doesn't overwrite your CUDA build with a CPU-only one.

## Project Structure

```
run_experiments.py          Synthetic experiments entry point + CLI
run_real_experiments.py     Real/semi-synthetic experiments entry point
run_sdss_scaling_experiment.py  SDSS-only scaling benchmark over multiple n
consolidate_partial_results.py  Build results.json from partial checkpoints
generate_plots.py           Regenerate all plots from cached results
models/
  flexcode.py               FlexCodeEstimator + RF regressor wrapper
  native.py                 TabPFN / RealTabPFN native density extraction
  baselines.py              Parametric, GLM, quantile, flow, and MDN baselines
datasets/
  synthetic.py              Synthetic DGPs (with known true densities)
  real.py                   Semi-synthetic + real-world dataset loaders
  sdss_galaxies.csv         SDSS DR18 photometric redshift data (500k galaxies)
evaluation/
  metrics.py                CDE loss, log-lik, CRPS, PIT, coverage
visualization/
  plots.py                  Rankings, raw metrics, density comparisons, PIT
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

### Parametric / GLM Baselines

| Method | Description |
|--------|-------------|
| LinearGauss-Homo | Linear mean + constant Gaussian noise |
| LinearGauss-Hetero | Linear mean + input-dependent Gaussian noise |
| Student-t | Linear mean + constant Student-t noise (df estimated by MLE) |
| LogNormal-Homo | Log-normal with linear log-mean, constant log-variance |
| LogNormal-Hetero | Log-normal with linear log-mean, input-dependent log-variance |
| Gamma-GLM | Gamma GLM with log-link for the mean |
| MDN-2mix | Mixture Density Network (2 Gaussians, 1 hidden layer) |

### Penalized (Ridge) Variants

Each parametric method above (except MDN) also has a Ridge-regularized variant
with the penalty chosen by leave-one-out cross-validation (`RidgeCV`).

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
| Quantile-Linear | Linear quantile regression (skipped for n > 10000) |
| Flow-Spline | Conditional neural spline flow with Gaussian base |

## Datasets

Each experiment is run 20 times with different train/test splits; metrics
report mean ± SE across repetitions.

### Synthetic (d ∈ {5, 10, 50}, n ∈ {1000, 2000, 4000, 6000, 20000})

All synthetic datasets have known true conditional densities. Tags follow
`{Base}-d{d}-{n}` (e.g. `Heteroscedastic-d10-2000`).

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

### Real-world (n ∈ {1000, 2000, 4000, 6000, 20000} where available)

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

By default this runs `n = 10k, 50k, 100k, 250k, 500k, full`, writes outputs to
`results_real/sdss_scaling/`, averages metrics over `4` repetitions per sample
size by default, and generates performance-vs-`n` plots there. The script uses
a conservative default schedule that drops methods once they hit explicit
package limits or are likely to be impractical for a repeated scaling
benchmark. You can override that with `--methods ...`,
`--all-methods-at-all-sizes`, or change the repetition count with `--n-reps`.

## Evaluation Metrics

- **CDE loss** (Izbicki & Lee, 2017): proper scoring rule for conditional densities
- **Log-likelihood**: mean log f(z_test)
- **CRPS**: Continuous Ranked Probability Score
- **PIT KS**: Kolmogorov-Smirnov statistic for calibration
- **90% coverage**: proportion of test samples in 90% credible interval
- **Interval width**: mean width of 90% credible interval
- **Fit time**: total wall-clock time (fit + predict) in seconds

## Output

Simulated results go to `results_simulated/` by default. Real-dataset results
go to `results_real/` by default. `generate_plots.py` reads both directories
and writes each artifact back to the matching destination.

```
results_simulated/
  results.json                      raw metric values (mean ± SE across repetitions)
  results_table.html                formatted HTML table (±SE across repetitions)
  rankings_sim_{metric}_n{n}.png    ranking heatmap — simulated, per n
  raw_sim_{metric}_n{n}.png         raw value heatmap — simulated, per n
  perf_vs_n_{metric}_sim_d{d}.png   performance vs n — simulated, per d
  pit_calibration.png               PIT histograms for calibration assessment
  native_tab_{ds}.png               density comparison plots (synthetic)
  cache/{dataset}.npz               cached arrays (skip re-runs)

results_real/
  results.json                      raw metric values (mean ± SE across repetitions)
  results_table.html                formatted HTML table (±SE across repetitions)
  rankings_real_{metric}_n{n}.png   ranking heatmap — real, per n
  raw_real_{metric}_n{n}.png        raw value heatmap — real, per n
  perf_vs_n_{metric}_real.png       performance vs n — real datasets
  perf_vs_n_foundational_*.png      same, highlighting foundation models only
  pit_calibration.png               PIT histograms for calibration assessment
  cache/{dataset}.npz               cached arrays (skip re-runs)
```

For a lightweight cross-machine bundle, you can track or archive only
`results_*/results.json` and run:

```bash
python generate_plots.py --metrics-only
```

That reproduces `results_table.html`, `rankings_*`, `raw_*`,
`perf_vs_n_*`, and `perf_vs_n_foundational_*` without copying `cache/`.
