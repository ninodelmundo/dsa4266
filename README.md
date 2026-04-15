# Multimodal Phishing Website Detection

This project detects phishing websites by combining URL, HTML/text, and webpage screenshot signals in a multimodal machine learning pipeline.

## About The Project

The goal of this project is to classify websites as either legitimate or phishing using multiple complementary data modalities:

- URL signals: character-level URL modeling plus handcrafted URL features such as length, digit count, special characters, HTTPS usage, and subdomain count.
- HTML/text signals: visible webpage text extracted from HTML and structural HTML features such as forms, inputs, password fields, scripts, iframes, and external links.
- Visual signals: webpage screenshots represented with pretrained computer vision embeddings.
- Fusion model: a multimodal classifier that combines URL, text, visual, and HTML-structure representations for binary classification.

The recommended workflow uses the fast cached-feature pipeline in `main.py`. This path extracts frozen DistilBERT and EfficientNet features once, caches them in `data/processed/`, and then trains lightweight unimodal and fusion models on top of those cached representations. The older standalone training scripts are still available, but `main.py` and the optimization scripts are the preferred workflow.

Typical outputs include model checkpoints, training curves, confusion matrices, metric summaries, optimization reports, and ablation plots under `outputs/`.

## Repository Structure

- `configs/`: YAML configuration for data paths, model settings, training parameters, optimization search spaces, and ablation variants.
- `scripts/`: command-line scripts for downloading data, preprocessing, training, evaluation, hyperparameter optimization, and ablation analysis.
- `src/data/`: dataset loading, data merging, feature extraction, dataloaders, image transforms, URL utilities, and HTML utilities.
- `src/models/`: URL, text, visual, unimodal, and multimodal fusion model definitions.
- `src/training/`: generic trainer, losses, callbacks, schedulers, early stopping, checkpointing, and mixed precision support.
- `src/evaluation/`: metric computation, threshold calibration, C-index/composite scoring, and plot generation.
- `src/experiments/`: shared experiment helpers for Optuna optimization and ablation workflows.
- `notebooks/`: exploratory data analysis notebooks.
- `outputs/`: generated logs, checkpoints, plots, optimization results, and ablation artifacts.

## Installation

From the project root:

```bash
python3 -m venv .venv
# For MacOS
source .venv/bin/activate
# For Windows
.venv/Scripts/activate
python3 -m pip install -r requirements.txt
```

The dataset downloader uses Hugging Face and Kaggle-hosted datasets. Create a `.env` file in the repo root with your Kaggle credentials:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

The default config requests CUDA:

```yaml
project:
  device: "cuda"
```

If you do not have a CUDA GPU, change `project.device` in `configs/config.yaml` to `cpu` before training.

## User Guide

`main.py` is the source-of-truth orchestrator for the current workflow. It runs the
same stage-specific scripts that can also be called directly from `scripts/`.

### Which Command Should I Run?

For a first-time baseline run:

```bash
python3 main.py
```

This runs:

```text
download -> preprocess -> auxiliary plots -> baseline unimodal training -> baseline fusion training -> evaluation
```

If the data is already downloaded and you want the full research workflow with
optimization, ablation, and explainability:

```bash
python3 main.py --skip-download --optimize --ablation --explain
```

If you also want to redownload the datasets first:

```bash
python3 main.py --optimize --ablation --explain
```

To rerun a single stage:

```bash
python3 main.py --step <step>
```

Available steps are:

```text
download, preprocess, aux, baseline, optimize, ablation, evaluate, explain, all
```

### 1. Download Data

Run this before preprocessing or training:

```bash
python3 main.py --step download
```

This downloads raw URL, HTML, and screenshot datasets into:

```text
data/raw/
```

### 2. Preprocess And Merge Data

Build the merged multimodal dataset:

```bash
python3 main.py --step preprocess
```

This creates:

```text
data/processed/merged_dataset.csv
```

After this step completes, the EDA notebook has the merged data it needs.

### 3. Generate Auxiliary Data-Study Plots

Generate class-distribution, dataset-statistics, embedding, feature-correlation,
and learning-rate-schedule plots:

```bash
python3 main.py --step aux
```

These plots are saved under:

```text
outputs/evaluation/
```

### 4. Train Baseline Unimodal And Fusion Models

Train URL-only, text-only, visual-only, HTML-structure-only, and baseline fusion
models:

```bash
python3 main.py --step baseline
```

During the first baseline run, the pipeline extracts and caches frozen text and
visual embeddings. Later runs reuse cached features when possible.

Expected generated files include:

```text
data/processed/features.pt
outputs/baseline/unimodal/
outputs/baseline/fusion/
outputs/run.log
```

### 5. Optimize Unimodal And Fusion Models

Run Optuna tuning for unimodal models and then the fusion model:

```bash
python3 main.py --step optimize
```

This saves results under:

```text
outputs/optimization/unimodal/
outputs/optimization/fusion/
```

### 6. Run Ablation Analysis

Run the tuned fusion ablation workflow:

```bash
python3 main.py --step ablation
```

This saves results under:

```text
outputs/optimization/ablation/
```

### 7. Evaluate Available Results

Run unified evaluation across available baseline and optimization outputs:

```bash
python3 main.py --step evaluate
```

Evaluation plots and summaries are saved under:

```text
outputs/evaluation/
```

### 8. Run Explainability

Run SHAP-based surrogate explanations, fusion modality Shapley diagnostics, and
local URL/text/image explanations:

```bash
python3 main.py --step explain
```

Results are saved to:

```text
outputs/explainability/
```

## Direct Optimization And Ablation Scripts

The commands above are the recommended beginner path. The underlying scripts can
also be run directly when you want finer control over one stage.

### 1. Optimize Unimodal Models

```bash
python3 scripts/optimize_unimodal.py --config configs/config.yaml --modality all
```

This tunes URL-only, text-only, visual-only, and HTML-structure-only baselines. Results are saved to:

```text
outputs/optimization/unimodal/
```

Important artifacts:

- `summary.json`
- `promoted_overrides.json`
- per-modality `trial_results.csv` and `trial_results.json`

### 2. Optimize The Fusion Model

```bash
python3 scripts/optimize_fusion.py --config configs/config.yaml
```

By default, this uses `outputs/optimization/unimodal/promoted_overrides.json` if it exists. Results are saved to:

```text
outputs/optimization/fusion/
```

Important artifacts:

- `summary.json`
- `best_fusion_overrides.json`
- `trial_results.csv`
- `trial_results.json`

### 3. Run Ablation Analysis

```bash
python3 scripts/run_ablation.py --config configs/config.yaml
```

By default, this uses `outputs/optimization/fusion/best_fusion_overrides.json` if it exists. Results are saved to:

```text
outputs/optimization/ablation/
```

Important artifacts:

- `ablation_results.csv`
- `ablation_results.json`
- `ablation_metrics.png`
- `ablation_deltas.png`
- `summary.json`

The ablation workflow evaluates the full tuned fusion model, variants with individual modalities removed, variants without handcrafted URL scalar features, and alternative fusion strategies.

## Direct Explainability Script

The recommended entrypoint is `python3 main.py --step explain`. You can also call
the explainability script directly:

```bash
python3 scripts/run_explainability.py --config configs/config.yaml
```

For a quick smoke run:

```bash
python3 scripts/run_explainability.py --config configs/config.yaml --max-samples 8 --background-samples 4 --local-samples 2
```

Results are saved to:

```text
outputs/explainability/
```

Important artifacts:

- `explainability_report.html`: static browser-openable summary report.
- `dashboard_inputs.csv` and `dashboard_predictions.csv`: interpretable dashboard table and surrogate predictions.
- `global_importance.csv` and `global_importance.png`: global readable surrogate importance.
- `fusion_modality_contributions.csv` and `fusion_modality_contributions.png`: URL/text/visual/HTML modality contribution diagnostics.
- `local/sample_XXX/`: per-sample explanations from the selected split, including `explanation.json`, URL/text HTML reports, Grad-CAM, occlusion maps, and local contribution plots.
- `shapash_explainer.pkl`: saved Shapash `SmartExplainer` object, produced only when Shapash is installed in the active Python environment.

Shapash is an interactive local web app, not a static HTML file. After generating `shapash_explainer.pkl`, launch it with:

```bash
python3 scripts/serve_shapash_dashboard.py --explainer outputs/explainability/shapash_explainer.pkl --port 8050
```

Then open:

```text
http://127.0.0.1:8050
```

## Configuration Tips

Most experiment settings live in `configs/config.yaml`.

- `data.max_samples`: set to a small integer for quick debugging runs.
- `project.device`: use `cuda` for GPU training or `cpu` if CUDA is unavailable.
- `training.num_epochs`: controls maximum training epochs.
- `training.batch_size`: controls batch size for training and evaluation.
- `optimization.n_trials`: controls how many Optuna trials to run.
- `optimization.metric_weights`: controls the composite score used for model selection.
- `fusion.strategy`: chooses the default fusion strategy, such as `attention`, `weighted`, or `concatenation`.

Feature extraction can be slow on the first run because pretrained DistilBERT and EfficientNet representations are computed. These are cached under `data/processed/` and reused on later runs when possible.

## Troubleshooting

- Missing data: run `python3 main.py --step download` first.
- Missing Kaggle credentials: create a repo-root `.env` file with `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- No CUDA device: set `project.device` to `cpu` in `configs/config.yaml`.
- Slow first training run: this is expected because text and image features are extracted and cached.
- Shapash missing: install it in the active environment with `python3 -m pip install shapash dash`, then verify with `python3 -c "import shapash; print(shapash.__version__)"`.
- Standalone script confusion: prefer `main.py` for the current workflow, and use direct scripts only when you intentionally want to run one stage by hand.
