# Predicting Crash Severity in NHTSA SGO Reports

**ISYE 4600 — Spring 2026**
Santiago Aramayo, Lauren McDonald, Luis Velez
April 27, 2026

---

## 1. Problem statement and goal

Companies that test self driving cars in the United States have to send a crash report to NHTSA under the Standing General Order (SGO). The reports include vehicles that use a full driving system (ADS, like Waymo or Cruise) and vehicles that use a Level 2 driver assist (L2, like Tesla Autopilot). The data set is public, but each report is hand written and uses different formats across companies and across years.

A safety analyst who reads these reports has a simple question: which incidents are likely to be **severe**? In our project a crash is severe if at least one of these things is true:

- the highest reported injury is moderate or worse,
- the airbag was deployed for the subject vehicle, or
- some vehicle had to be towed.

This is the rule used in the SGO public summary, so we kept it. We want to build a model that, given the structured fields and the short narrative the company writes, returns a probability that the incident is severe. The goal is not to replace the human review. The goal is to put the most likely severe cases on top so the analyst spends time where it matters.

The dataset is small for machine learning standards (around 5,500 incidents after cleaning) and the reports change over time, so the project is also a study of what is possible with this kind of public data.

---

## 2. Data

**Source.** Four CSV files from the NHTSA SGO portal: ADS current era, ADS archived era, L2 current era, L2 archived era. Together they cover roughly mid-2021 to early 2025. We treat the "archived" era as past reports and the "current" era as more recent reports, which gives us a natural temporal split.

**Unit of analysis.** One row per **unique incident**. The raw files contain one row per *report version*, and a single crash can have many versions and even many vehicles (subject + counterpart). Script `01_clean_incidents.py` keeps the latest version for each report id, then groups by `Same Incident ID` so the model sees one example per real-world event.

**Target.** A binary label `severe` defined by the OR rule above. To avoid imputing the outcome, we also keep a flag `severity_known` and only train and test on rows where at least one component is observed. Out of 5,576 unique incidents, 5,567 have at least one severity signal, of which 4,063 are severe and 1,504 are not.

**Features.** Two groups:

1. *Structured (tabular)* fields from the report form: roadway type, weather flags, crash partner, pre crash movement of subject and counterpart vehicles, speed, automation engagement, month, etc.
2. *Narrative flags* extracted from the free text by `narrative_utils.py`. We use 21 binary flags that describe the scenario only, like `nav_av_stopped`, `nav_other_struck_av`, `nav_at_intersection`, `nav_in_parking_lot`. We do **not** use words like "injured", "towed", or "airbag" as features, because those words define the label. Adding them would be label leakage.

**Preprocessing.** Numeric columns are filled with the median, categorical columns are filled with the string `"Unknown"`. For each column with missing values we add an `is_missing` indicator before imputation. Categorical columns are one hot encoded for the linear models.

**Train / test split.** A *temporal* split: train on the archived era, test on the current era. This is harder than a random split (the severe rate changes between eras, see Section 3), but it is the only fair way to imitate deployment. For the stratified models (Section 4) we also reserve 25% of the training data as a validation set for hyperparameter and threshold tuning.

**Limitations and biases.** SGO reporting is not a population census. Companies report what their internal systems flag, the form changed across eras, and the narrative section is sometimes redacted. ADS test fleets are also concentrated in a few US cities (San Francisco, Phoenix, Austin), so the data does not represent the full driving environment. We discuss this again in Section 7.

| Era | ADS rows | L2 rows | ADS severe rate | L2 severe rate |
|---|---|---|---|---|
| Archived | 2,295 | 4,029 | 26% | 98% |
| Current | 612 | 848 | 47% | 97% |

Table 1. Counts and severe rates by era and automation level (after cleaning).

---

## 3. What we tried first and why it did not work

A natural first attempt is to train **one logistic regression** on the union of ADS and L2 with a temporal split, balanced class weights, and `automation_level` as a feature. This is what `02_run_baselines.py` does.

The pooled metrics on the test set look fine on the surface (`baseline_results.csv`):

| Slice | Precision | Recall | F1 | AUC | FN rate |
|---|---|---|---|---|---|
| Pooled overall | 0.975 | 0.730 | 0.835 | 0.856 | 27.0% |
| L2 slice only | 0.977 | 1.000 | 0.988 | 0.927 | 0.0% |
| ADS slice only | 0.000 | 0.000 | 0.000 | 0.510 | **100.0%** |

Table 2. Pooled logistic baseline (current era test set).

The pooled F1 of 0.835 hides the real story. When we slice the same predictions by automation level, the model gets **every** severe ADS crash wrong. AUC on ADS is 0.51, basically random. Figure 1 (`Presentation/figures/06_confusion_matrices.png`) shows the same thing visually.

![Pooled logistic regression confusion matrix on the current era test set.](Presentation/figures/06_confusion_matrices.png)

**Why it fails.** Two reasons together:

1. **Distribution shift.** In the archived era, only 26% of ADS crashes are severe. In the current era, 47% are. The training set teaches the model "ADS usually means not severe." When the test era arrives with a higher severe rate, that shortcut breaks.
2. **Class proxy.** The pooled model picks up the variable `automation_level` as a strong proxy for severity (because L2 is almost always severe in the training set). It learns a quick rule like "L2 yes, ADS no", which works on average but is useless for the case we actually care about, the ADS reports.

Figure `Presentation/figures/04_reporting_bias_severe_rate_by_stratum.png` shows the strong gap in severe rate between L2 and ADS, which is exactly what the pooled model picks up on.

This is the failure mode we wanted to fix.

---

## 4. Methods

After the failure of the pooled baseline we made two changes and evaluated three model families.

**Change 1. Stratify by automation level.** We train a separate model on each level (ADS and L2). The level no longer is a feature, it is the partition. This removes the shortcut described above and forces each model to learn the actual signal in the report fields.

**Change 2. Add narrative scenario flags to ADS.** Tabular fields alone give an AUC near 0.50 on ADS (random). Adding the 21 binary scenario flags from `narrative_utils.py` gives a clear lift, and the flags are interpretable (for example `nav_av_stopped`, `nav_other_struck_av`). For L2, almost all crashes are severe and the tabular fields alone separate the few non severe cases well, so we keep L2 tabular.

**Models.** For each automation level we train three models with `05_stratified_models_ads_l2.py`:

- *Logistic regression* (LR), balanced class weights. C grid `{0.1, 0.3, 1, 3, 10}`.
- *Random forest* (RF), balanced class weights. Grid over `n_estimators in {100, 200, 300}`, `max_depth in {None, 5, 10}`, `min_samples_leaf in {1, 5, 10}`.
- *XGBoost*, with `scale_pos_weight` chosen from `{1, half class ratio, full class ratio}`, `learning_rate in {0.05, 0.1, 0.2}`, `max_depth in {3, 5, 7}`, 200 trees.

For every fit we tune the **probability threshold** on the validation slice. We pick the threshold with the highest F1 among the ones with validation precision at least 0.65 (`PREC_FLOOR`). If no threshold meets the floor the script falls back to the threshold with the best F1 alone.

**Reproducibility.** All splits and models use `random_state=42`. We fixed all hyperparameter grids in code, no manual tweaks per run.

---

## 5. Evaluation

We use four metrics, reported on the held out current era test set for each level. Each one captures something different that matters here:

- **Precision** (of the predicted severe cases, how many were actually severe). False alarms cost analyst time.
- **Recall** (of the actually severe cases, how many we found). Missing a severe case is the worst error.
- **F1**, the harmonic mean of the two.
- **ROC AUC**, threshold free comparison, useful to compare the *ranking* quality of the models.
- **False negative rate** (`FN / (FN + TP)`), the share of severe crashes we miss. We track this explicitly because it is the metric a safety team would care about most.

Validation is done **inside** the train set. The test set is only used at the end. There is no model selection that looks at the test set, which keeps the comparison fair.

---

## 6. Results

### 6.1 Stratified results (main table)

The full result file is `Modeling/logistic_regression/all_stratified_results.csv`.

| Model | Level | Precision | Recall | F1 | AUC | FN rate | Threshold |
|---|---|---|---|---|---|---|---|
| LR | ADS | 0.499 | 0.845 | 0.627 | 0.531 | 15.5% | 0.36 |
| RF | ADS | 0.804 | 0.883 | 0.842 | 0.893 | 11.7% | 0.48 |
| **XGB** | **ADS** | **0.831** | **0.922** | **0.874** | **0.921** | **7.8%** | **0.46** |
| LR | L2 | 0.985 | 0.967 | 0.976 | 0.876 | 3.3% | 0.20 |
| RF | L2 | 0.972 | 1.000 | 0.986 | 0.836 | 0.0% | 0.20 |
| XGB | L2 | 0.972 | 1.000 | 0.986 | 0.850 | 0.0% | 0.20 |

Table 3. Stratified LR / RF / XGB by automation level on the current era test.

For ADS the ranking is clear: XGBoost wins on every metric. The big gain is from going non linear (LR to RF jumps the AUC from 0.53 to 0.89). XGBoost adds another smaller step, and most importantly cuts the false negative rate from 11.7% to 7.8%. Compared to the pooled baseline FN rate of 100% on ADS, this is a meaningful change.

For L2, all three models reach an F1 above 0.97 and FN rates between 0% and 3%. RF and XGB tie at the top because L2 is almost linearly separable (98% positive class). We use logistic regression for L2 in error analysis because the FN cases LR makes are the more interesting ones, RF predicts everyone severe and trivially gets 0% FN.

The visual summary (Figure 2) is `Presentation/figures/13_model_comparison_all.png`.

![Stratified model comparison, all six combinations on the current era test set.](Presentation/figures/13_model_comparison_all.png)

### 6.2 Pooled baseline vs stratified ADS

This is the before / after view of the failure we described in Section 3. The figure is `Presentation/figures/15_ads_pooled_vs_stratified_improvement.png`.

![Pooled LR on the ADS slice (left bars in each pair) compared with the stratified ADS model (right bars).](Presentation/figures/15_ads_pooled_vs_stratified_improvement.png)

The pooled bars are zero on every metric for ADS. The stratified bars reach the numbers from Table 3. The change of split design and feature set, not just the choice of XGBoost, is what fixed the model.

### 6.3 Does the narrative help on ADS?

We compared three feature sets on the ADS logistic regression: tabular only, tabular plus narrative, narrative only. AUC results (`narrative_ads_model_comparison.csv`):

| Feature set | AUC |
|---|---|
| Tabular only | 0.500 |
| Tabular + narrative | 0.530 |
| Narrative only | 0.608 |

Table 4. ADS LR with different feature sets, current era test.

The interesting result is that **narrative alone beats tabular alone**. When we use the full XGBoost in Table 3 (which gets AUC 0.92 with tabular plus narrative on ADS), the lift becomes really big. So the structured form fields are not enough on their own for ADS, the short text in the report is what carries most of the signal.

Top narrative flags (by absolute LR coefficient on ADS, from `lr_ads_coefficients.csv`):

- `nav_av_stopped`, `nav_other_struck_av`, `nav_minor_damage_lang` are associated with **lower** severe probability.
- `nav_av_moving`, `nav_lane_change`, `nav_in_parking_lot` push the probability **up** in some clusters (see 6.5).

### 6.4 Error analysis (false negatives)

The output of `09_stratified_fn_analysis.py` gives the missed severe crashes for the best model on each level. Summary (`fn_summary.csv`):

| Model | Severe in test | FN | FN rate | Top roadway | Top crash partner |
|---|---|---|---|---|---|
| XGB on ADS | 283 | 19 | 6.7% | Street | SUV |
| LR on L2 | 766 | 19 | 2.5% | Highway / Freeway | Other, see narrative |

Table 5. False negatives per stratified model.

Figure `Presentation/figures/17_fn_analysis_stratified.png` shows the top contexts as bar charts.

![Where the stratified models miss severe cases (current era test).](Presentation/figures/17_fn_analysis_stratified.png)

What the model misses on ADS is mostly **street level crashes with a passenger vehicle**, where the AV was stopped or in slow traffic and the narrative is short. These cases look benign on paper but the airbag did fire or the vehicle was towed. On L2 the misses are concentrated on **highway / freeway** crashes that the company described under "Other, see narrative", which is exactly where the structured form gives less information.

### 6.5 ADS scenario clusters

To support the error analysis we ran k means on the ADS rows (script `10_cluster_profiling.py`). Silhouette score selected k = 7. The summary table (`ads_cluster_summary.csv`):

| Cluster | Size | Severe rate | Scenario label |
|---|---|---|---|
| 2 | 56% | 26% | Typical street crash, AV stopped |
| 5 | 21% | **48%** | AV moving at impact |
| 0 | 14% | 33% | Other vehicle struck AV |
| 1 | 4% | 42% | AV hit a fixed object (parking lot) |
| 3 | 4% | 31% | AV struck other party |
| 4 | 1% | 0% | Animal strike |
| 6 | <1% | 0% | Pedestrian / cyclist case (very few) |

Table 6. ADS scenario clusters (k = 7).

The biggest cluster (C2) is the typical "AV stopped, low speed bump" type. The most severe one, C5, is "AV moving at impact". These are the cases the operations team would want to see first if a model flags them. The animal cluster has zero severe cases, which is consistent with how those incidents are reported.

We also ran the same clustering on L2 (`11_cluster_profiling_by_level.py --level L2`) for symmetry. Almost every L2 cluster is at 95% to 100% severe, so the clustering for L2 is mostly about scenario type, not about severity contrast.

---

## 7. Interpretation, what did not work, limitations, next steps

**Interpretation for the system.** A small safety team can use the stratified ADS XGBoost as a triage tool. Rank new ADS reports by predicted probability, look at the top first. Cluster C5 ("AV moving at impact") is the most informative scenario to monitor. The L2 model is almost trivial because virtually every L2 SGO report is severe by our definition; L2 is more useful as a *consistency check* on the OR rule than as a learning problem.

**Risks and assumptions.** The label is the OR rule, not a clinical injury rating. Reports are written by the company, which means the narrative reflects each company style. The "severity_known" filter removes around 9 incidents only, so it has no real impact on the headline numbers. The model is calibrated for the current era; if the SGO form changes again the model should be retrained.

**What did not work.** A few things we tried and dropped from the final pipeline:

- *Pooled training across levels.* Discussed in Section 3. It does not just give worse numbers on ADS, it gives a 100% FN rate, which is the kind of failure we have to point out.
- *Outcome words in the narrative.* Early ablations included words like "towed" or "injured" as features, which gave very high AUC but is label leakage. We removed these.
- *Narrative for L2.* We tested adding the 21 flags to L2. Because L2 is already 98% severe in training, the flags add nothing measurable, so we kept L2 tabular only.
- *Pure logistic regression on ADS.* Even with the narrative flags, ADS LR stays at AUC 0.53 because the relationships are not linear. Tree based models are clearly the right family here.

**Limitations.**
- Reporting bias: SGO is not a random sample of all AV crashes.
- Sample size for ADS is still small (around 600 test incidents).
- The temporal split is correct but it depends on what the SGO portal looks like at submission time, the boundary between "archived" and "current" is set by NHTSA and can move.
- Narrative redactions zero out our scenario flags, which limits what the model can learn from those rows.

**Next steps (one or two realistic ones).**
1. Replace the regex narrative flags with a small LLM that produces the same JSON schema. The flags would still be binary (so the model interface does not change) but would handle paraphrases the regex misses. We would need to cache the outputs for reproducibility and avoid prompts that mention injury / airbag / tow.
2. Calibrate the threshold per cluster, not per level. Cluster C5 has a 48% severe rate, so a different operating point may catch more "AV moving at impact" cases at acceptable false alarm rate.

---

## 8. Reproducing the results

Software: Python 3, dependencies in `requirements.txt`. On macOS install OpenMP first (`brew install libomp`) so XGBoost runs.

Minimum rerun (the numbers in Tables 2 and 3):

```
python scripts/01_clean_incidents.py
python scripts/02_run_baselines.py
python scripts/05_stratified_models_ads_l2.py
```

Rest of the analyses (Tables 4 to 6 and the figures referenced above):

```
python scripts/06_narrative_features.py
python scripts/09_stratified_fn_analysis.py
python scripts/10_cluster_profiling.py
python scripts/11_cluster_profiling_by_level.py --level L2
python scripts/make_presentation_figures.py
```

All seeds are fixed at 42. The README contains the full output map.
