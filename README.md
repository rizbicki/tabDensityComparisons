# Tabular Foundation Models for Conditional Density Estimation

Benchmark comparing tabular foundation models (TabPFN, TabICL) against
classical methods for conditional density estimation (CDE) on tabular data.

## Quick Start

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh --quick          # sanity check (synthetic datasets)
./setup_and_run.sh                  # full run
./setup_and_run.sh --cpu            # force CPU (slower)
```

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
runTab/
  run_experiments.py        Entry point + CLI
  models/
    flexcode.py             FlexCodeEstimator + RF regressor wrapper
    native.py               TabPFN-Native, TabICL-Quantiles extraction
    baselines.py            Parametric, GLM, quantile, and MDN baselines
  datasets/
    synthetic.py            Synthetic DGPs (with known true densities)
    real.py                 Semi-synthetic + real-world dataset loaders
  evaluation/
    metrics.py              CDE loss, log-lik, CRPS, PIT, coverage
  visualization/
    plots.py                Rankings, raw metrics, density comparisons, PIT
  utils/
    io.py                   Caching, formatting, summary printing
```

## Methods

### Tabular Foundation Models (native CDE)

| Method | Description |
|--------|-------------|
| TabPFN-Native | TabPFN's built-in bar distribution |
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

### Nonparametric Baselines

| Method | Description |
|--------|-------------|
| Quantile-Tree | Quantile regression via XGBoost/GBM |
| Quantile-Linear | Linear quantile regression |

## Datasets

### Synthetic (n = 1000, 2000, 4000)

Heteroscedastic, Bimodal, Skewed, Nonlinear, LinGauss-Homo -- each with
known true conditional density for visual comparison.

### Semi-synthetic

Friedman1 (n=1500, d=10), Friedman2 (n=1500, d=4).

### Real-world (n = 1000, 2000, 4000 where available)

SpaceGA, Kin8nm, Puma8NH, Bank8FM, CPUact -- from OpenML, at multiple
sample sizes to study scaling behavior.

## Evaluation Metrics

- **CDE loss** (Izbicki & Lee, 2017): proper scoring rule for conditional densities
- **Log-likelihood**: mean log f(z_test)
- **CRPS**: Continuous Ranked Probability Score
- **PIT KS**: Kolmogorov-Smirnov statistic for calibration
- **90% coverage**: proportion of test samples in 90% credible interval
- **Interval width**: mean width of 90% credible interval

## Output

All results go to `results/`:
- `results.json` -- raw metric values
- `results_table.html` -- formatted HTML table (green = best, yellow = 2nd best)
- `rankings_{metric}.png` -- ranking heatmap per metric
- `raw_{metric}.png` -- raw value heatmap per metric
- `native_tab_{ds}.png` -- selected method density comparison (synthetic)
- `pit_calibration.png` -- PIT histograms for calibration assessment
- `cache/{dataset}.npz` -- cached arrays (skip re-runs)
