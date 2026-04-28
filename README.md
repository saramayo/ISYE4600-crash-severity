# AV Crash Severity Prediction  
**ISYE 4600 — Spring 2026**  
Santiago Aramayo, Lauren McDonald, Luis Velez

---


Inputs are the four CSVs under **`Data/`** (bundled with the submission). All paths below are relative to the repository root.

**Software:** Python 3 with `pip`. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On **macOS**, install OpenMP before running models that use XGBoost:

```bash
brew install libomp
```

Run Python from the project root (the folder that contains `Data/` and `scripts/`). Use either activated venv (`python scripts/...`) or explicit interpreter (`.venv/bin/python scripts/...`).

Splits and models use **`random_state=42`** where applicable so runs are repeatable on the same machine and library versions.

---

## Minimal rerun (main numerical results only)

These three steps regenerate the **baseline** and **stratified model** metrics:

```bash
.venv/bin/python scripts/01_clean_incidents.py
.venv/bin/python scripts/02_run_baselines.py
.venv/bin/python scripts/05_stratified_models_ads_l2.py
```

**Check:** after this, these files should exist:

| File | What it is |
|------|------------|
| `Cleaned/sgo_cleaned_incidents.csv` | Cleaned incidents |
| `Cleaned/data_dictionary.csv` | Column summary from cleaning |
| `Modeling/baselines/baseline_results.csv` | Pooled logistic baseline metrics |
| `Modeling/logistic_regression/all_stratified_results.csv` | **Primary table:** LR / RF / XGB × ADS & L2 (precision, recall, F1, AUC, FN rate, thresholds) |

The report should cite **`Modeling/logistic_regression/all_stratified_results.csv`** as the source for headline model comparisons.

---

## Full rerun (report figures and extra analyses)

Steps build narrative comparisons, false-negative lists, clustering, and slide PNGs. They assume the minimal steps above have already run.

```bash
.venv/bin/python scripts/06_narrative_features.py
.venv/bin/python scripts/09_stratified_fn_analysis.py
.venv/bin/python scripts/10_cluster_profiling.py
.venv/bin/python scripts/make_presentation_figures.py
```

L2 clustering (same idea as script 10, other automation level):

```bash
.venv/bin/python scripts/11_cluster_profiling_by_level.py --level L2
```

**Check:** notable outputs include `Modeling/logistic_regression/narrative_*.csv`, `fn_*.csv`, `Modeling/clustering/ads_cluster_*.csv`, and PNGs under **`Presentation/figures/`**.

---

## Step summary

| Step | Script | Main outputs |
|------|--------|----------------|
| 1 | `01_clean_incidents.py` | `Cleaned/sgo_cleaned_incidents.csv`, `data_dictionary.csv` |
| 2 | `02_run_baselines.py` | `Modeling/baselines/baseline_results.csv`; LR coef / FN under `Modeling/logistic_regression/` |
| 3 | `05_stratified_models_ads_l2.py` | `all_stratified_results.csv`, coef/importance CSVs, `Presentation/figures/13_model_comparison_all.png` |
| 4 | `06_narrative_features.py` | `narrative_*` CSVs under `Modeling/logistic_regression/` |
| 5 | `09_stratified_fn_analysis.py` | `fn_*.csv`, `Presentation/figures/17_fn_analysis_stratified.png` |
| 6 | `10_cluster_profiling.py` | `Modeling/clustering/ads_cluster_*.csv`, clustering figure |
| 7 | `make_presentation_figures.py` | PNG set in `Presentation/figures/`, `SLIDE_FIGURES.txt` |

---

## Output folders

| Folder | Role |
|--------|------|
| `Data/` | Raw NHTSA extracts (inputs) |
| `Cleaned/` | Produced by step 1 |
| `Modeling/` | Baselines, regression outputs, `clustering/` |
| `Presentation/figures/` | Generated figures |

---

## Supporting code

Imported by the runnable scripts (not separate entry points): **`baseline_common.py`** (paths, splits, metrics), **`narrative_utils.py`** (narrative flags). **`logistic_regression_baseline.py`** defines the pooled LR used by **`02_run_baselines.py`**.
