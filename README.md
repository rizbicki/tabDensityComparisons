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
```

Results go to `results/` by default. If you have older real-data runs in
`results_real/`, you can still generate combined plots from both after runs
complete:

```bash
python consolidate_partial_results.py   # merges results/ and results_real/
python generate_plots.py                # regenerates all plots
```

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
consolidate_partial_results.py  Build results.json from partial checkpoints
generate_plots.py           Regenerate all plots from cached results
models/
  flexcode.py               FlexCodeEstimator + RF regressor wrapper
  native.py                 TabPFN / RealTabPFN native density extraction
  baselines.py              Parametric, GLM, quantile, and MDN baselines
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

## Datasets

Each experiment is run 20 times with different train/test splits; metrics
report mean ± SE across repetitions.

### Synthetic (d ∈ {5, 10, 50}, n ∈ {1000, 2000, 4000, 6000, 20000})

All synthetic datasets have known true conditional densities. Tags follow
`{Base}-d{d}-{n}` (e.g. `Heteroscedastic-d10-2000`).

| Dataset | Description |
|---------|-------------|
| Heteroscedastic | Input-dependent Gaussian noise |
| Bimodal | Mixture of two Gaussians with input-dependent weights |
| Skewed | Gamma-shifted response with input-dependent shape |
| Nonlinear | Nonlinear mean with heteroscedastic noise |
| LinGauss-Homo | Linear Gaussian with homoscedastic noise |
| Interaction | Interactions between covariates affecting mean and variance |
| Friedman1 | Friedman #1 DGP; extra features (d>5) are irrelevant noise |
| Friedman2 | Friedman #2 DGP; fixed d=4, tag `Friedman2-d4-{n}` |

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
| CPUact | OpenML | 8192 |
| CalHousing | OpenML | 20640 |
| Diamonds | OpenML | 53940 |
| Abalone | OpenML | 4177 |
| Ailerons | OpenML | 13750 |
| Pol | OpenML | 15000 |
| MercedesBenz | OpenML | 4209 |
| Protein | OpenML | 45730 |
| Year | OpenML | 515345 |
| SGEMM_GPU | OpenML | 241600 |
| BlackFriday | OpenML | 166821 |
| SDSS | SDSS DR18 | 500000 |

The SDSS dataset uses ugriz photometric magnitudes as features and
spectroscopic redshift as the target.

## Evaluation Metrics

- **CDE loss** (Izbicki & Lee, 2017): proper scoring rule for conditional densities
- **Log-likelihood**: mean log f(z_test)
- **CRPS**: Continuous Ranked Probability Score
- **PIT KS**: Kolmogorov-Smirnov statistic for calibration
- **90% coverage**: proportion of test samples in 90% credible interval
- **Interval width**: mean width of 90% credible interval
- **Fit time**: total wall-clock time (fit + predict) in seconds

## Output

Synthetic results and real dataset results go to `results/` by default.
If you have older runs split across `results/` and `results_real/`, running
`consolidate_partial_results.py` merges both into `results/results.json`.

```
results/
  results.json                      raw metric values (mean ± SE)
  results_table.html                formatted HTML table
  rankings_sim_{metric}_n{n}.png    ranking heatmap — simulated, per n
  rankings_real_{metric}_n{n}.png   ranking heatmap — real, per n
  raw_sim_{metric}_n{n}.png         raw value heatmap — simulated, per n
  raw_real_{metric}_n{n}.png        raw value heatmap — real, per n
  perf_vs_n_{metric}_sim_d{d}.png   performance vs n — simulated, per d
  perf_vs_n_{metric}_real.png       performance vs n — real datasets
  perf_vs_n_foundational_*.png      same, highlighting foundation models only
  pit_calibration.png               PIT histograms for calibration assessment
  native_tab_{ds}.png               density comparison plots (synthetic)
  cache/{dataset}.npz               cached arrays (skip re-runs)
```
