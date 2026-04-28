# Predicting Crash Severity in NHTSA SGO Reports

**ISYE 4600 — Spring 2026**

Santiago Aramayo, Lauren McDonald, Luis Velez
April 27, 2026


---

## 1. Problem statement and goal

Companies that test self driving cars in the United States are required to send a crash report to NHTSA under the Standing General Order (SGO). These reports include vehicles that use a full driving system (ADS, like Waymo or Cruise) and also vehicles that use a Level 2 driver assist (Tesla Autopilot). We used this public data set, however each report was hand written and used different formats that varied according to the company and the year it was published.

Some of the information and features that were included in each incident report are:

- report id, report version, same incident id, report month and year.

- Vehicle context: automation level (ADS or L2), if the system was engaged at the time, manufacturer, model and model year of the subject vehicle, and a VIN flag.

- Roadway context: roadway type (street, highway / freeway, intersection, parking lot), work zone flag, traffic incident flag.

- Weather flags: clear, cloudy, rain, snow, fog or smoke or haze, severe wind.

- Crash dynamics: crash partner ("Crash With", for example passenger car, SUV, heavy truck, animal, fixed object, pedestrian or cyclist), pre crash movement of the subject vehicle and of the counterpart vehicle, and a reported pre crash speed in miles per hour.

- Severity components used to build the label: highest reported injury level, airbag deployment, vehicle towed.

- Free text narrative: a short description that is written by the reporting company. From this we extract 21 binary scenario flags (for example, AV stopped, AV moving, AV in parking lot, other vehicle approached from behind, intersection, vulnerable road user). 

A safety analyst who reads these reports will need to define which incidents to focus on in order to maximize the safety improvement. To allow this proper focus, we propose a simple question to : which incidents are likely to be severe? Incidents that are likely to be severe will allow these analysts to spend the most time in analyzing situations that lead to severe outcomes.

In our project we consider that a crash is severe if at least one of these things is true:

- the highest reported injury is moderate or worse,
- the airbag was deployed for the subject vehicle, or
- some vehicle was  towed.

This was the rule that was used in the SGO public summary. Our project aims to build a model that, given the incident with  structured fields and the short narrative the company writes- which is unstructured, and then we return the probability that the incident is severe. Our aim is not to replace the human review, but instead to flag the most likely severe cases on top so the analyst spends time where it matters the most.


---

## 2. Data Sources, Features and Cleaning

Sources - We have four CSV files from the NHTSA SGO portal: ADS current era, ADS archived era, L2 current era, L2 archived era. These  cover a timeframe from mid-2021 to early 2025. We treat the "archived" era as past reports and the "current" era as more recent reports, which gives us a temporal split that we will use. 


- Incident Reports
These include one row per each unique incident. The raw files contain one row per report version, and a single crash can have many versions and even many vehicles. We used the script:  `01_clean_incidents.py` in order to keep the latest version for each report id, we then group by `Same Incident ID` so the model sees one example for each real-world crash.

 We use a binary label `severe` which is defined by the OR rule above. To avoid directly representing the outcome, we keep a flag `severity_known` and only train and test on rows where at least one of these components is observed. Out of the 5,576 unique incidents, 5,567 have at least one severity signal, of these 4,063 are severe and 1,504 are not considered severe.

The visual below gives us the OR rule we used.

![Severity label rule. If at least one of injury moderate or worse, airbag deployed, or vehicle towed is true, the incident is labeled severe.](Presentation/figures/02_severity_label_rule_schematic.png)

Features: 

We divided these into two groups, what we get directly from the tabular fields in the incidents reports, and what we extract from the free text narrative.

1. Structured (tabular):  These structured fields are pulled directly from the report, these include: roadway type, weather flags, crash partner, pre crash movement of subject and counterpart vehicles, speed, automation engagement, month...


2. Narrative flags: These, like mentioned above are extracted from the free text by `narrative_utils.py`. We use 21 binary flags that describe the scenario , some of these are `nav_av_stopped`, `nav_other_struck_av`, `nav_at_intersection`, `nav_in_parking_lot`. We have purposefully excluded words like  like "injured", "towed", or "airbag" as features, since those words define the label rule above, then adding these would be a label leakage and would defeat the purpose on this additional training and learning.

Preprocessing. The numeric columns are filled with the median, and we filled the categorical columns  with the string `"Unknown"`. For each column that has missing values we also added  an `is_missing` binary indicator. The categorical columns are then one hot encoded for the linear models.

Training and testing approach.

Our approach was to train on the archived era, and then test on the current era. This is harder than a random split (the severe rate changes between eras, see Section 3), but it is the only fair way to imitate deployment. For the stratified models, which are just the models separated by Automation level we also then reserved a 25% of the training data as a validation set for  threshold tuning.

The figure below shows this split, it includes row counts and severe rate on the right. 
In the case of the L2 severe rate it stays close to its ceiling,  while the ADS severe rate jumps from 26% to 47%, this is the shift the pooled baseline in Section 3 was not able to handle. 

![Temporal train and test split, row counts and severe rate by era and automation level.](Presentation/figures/03_temporal_train_test_split.png)

One of the limitations that we faced was reagrding the fact that SGO reporting is not standardized. What we mean by this is that companies report what their internal systems flag, and the way that they do this, therefore  changed across different years, and the narrative section is sometimes redacted.

 We can also see that the ADS test incidents are also concentrated in some US cities (San Francisco, Phoenix, Austin), this adds bias, since the data does not represent the full driving environment. 

| Era | ADS rows | L2 rows | ADS severe rate | L2 severe rate |
|---|---|---|---|---|
| Archived | 2,295 | 4,029 | 26% | 98% |
| Current | 612 | 848 | 47% | 97% |

Table 1. Counts and severe rates by era and automation level (after cleaning).

The figure below explains this. 

The left panel is the severe vs non severe count on all labeled rows and the right panel shows, what share fires each component of the OR rule regarding those severe incidents. We can also see that the "vehicle towed" flag is the most common one, while the "moderate or worse injury"  is the least common.

![Severity label distribution and the share of each OR component among severe rows.](Presentation/figures/01_severity_outcome_and_label_components.png)

---

## 3. Failed Pooled Baseline Approach 

Our first attempt was to train one logistic regression on the combination of the ADS and L2 with a temporal split, balanced class weights, and also `automation_level` as a feature. 

This is what our script : `02_run_baselines.py` does.

The pooled metrics on the test set look fine : (`baseline_results.csv`):

| Slice | Precision | Recall | F1 | AUC | FN rate |
|---|---|---|---|---|---|
| Pooled overall | 0.975 | 0.730 | 0.835 | 0.856 | 27.0% |
| L2 slice only | 0.977 | 1.000 | 0.988 | 0.927 | 0.0% |
| ADS slice only | 0.000 | 0.000 | 0.000 | 0.510 | **100.0%** |

Table 2. Pooled logistic baseline (current era test set).

The pooled F1 of 0.835 does not explain the error. When we then separated the same predictions by automation level we can see that the model gets  every severe ADS crash wrong. The area under curve on ADS is 0.51, this is basically random.

Figure 1 (`Presentation/figures/06_confusion_matrices.png`) shows this. 

![Pooled logistic regression confusion matrix on the current era test set.](Presentation/figures/06_confusion_matrices.png)

Failure Explanation:
In the archived years, only 26% of ADS the crashes are severe.
 In the current year, 47% are. The training set is teaching the model that "ADS more often means not severe." When then the test era comes with a higher severe rate, then this logic fails.

The pooled model also picks up the variable `automation_level` as a proxy for severity (since L2 is almost always severe in the training set). It then is able to learn a quick rule  "L2 yes, ADS no", this isually works but is useless for the ADS reports.

Figure `Presentation/figures/04_reporting_bias_severe_rate_by_stratum.png` shows the  gap in severity rate between L2 and ADS, this is what the pooled model picks up on.

![Severe rate by era and automation level. ADS goes from 26% in archived to 47% in current; L2 stays close to 98% in both eras.](Presentation/figures/04_reporting_bias_severe_rate_by_stratum.png)

This is the failure that we aimed to fix using the startisfied model. 

---

## 4. Methods

Because of the unsuccesful approach of the pooled baseline we made two changes and evaluated the models distinctly by families.

We seaparated by automation level-  We train a separate model on each level (ADS and L2). This meant that automation level was no longer  a feature and instead the separation. This helped us remove the shortcut that the model was learning  and actually forced each model to learn the real signal in the report fields.

Additionally, we also added narrative scenario flags to ADS. Since tabular fields gave us an AUC close to 0.50 on ADS (random). We added the 21 binary scenario flags from `narrative_utils.py` which gave us a solid improvement, and the flags are actually interpretable ( `nav_av_stopped`, `nav_other_struck_av`). In the case of L2, almost all crashes that are severe and the tabular fields alone separate the few non severe cases well, because of this, we decided to  keep L2 tabular.

For each automation level we later trained three models with `05_stratified_models_ads_l2.py`:

- Logistic regression (LR), balanced class weights. C grid `{0.1, 0.3, 1, 3, 10}`.
- Random forest (RF), balanced class weights. Grid over `n_estimators in {100, 200, 300}`, `max_depth in {None, 5, 10}`, `min_samples_leaf in {1, 5, 10}`.
- XGBoost, with `scale_pos_weight` chosen from `{1, half class ratio, full class ratio}`, `learning_rate in {0.05, 0.1, 0.2}`, `max_depth in {3, 5, 7}`, 200 trees.

For every one of these we tuned the probability threshold on the validation slice. We went with the threshold with the highest F1 among those  with a validation precision of at least 0.65 (`PREC_FLOOR`). If none of the threshold met the floor then the script resorted to a fallback to the threshold with the best F1.

It is important to mention that all splits and models use `random_state=42` - we fixed all hyperparameter grids. 

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

The same comparison with precision, recall and F1 is in Table 4b. We include it because AUC alone hides the operating point. The narrative only setup not only has the highest AUC, it also has the highest F1 (0.599) and the highest precision (0.548), while still keeping a recall above the tabular only setup.

| Feature set | Precision | Recall | F1 |
|---|---|---|---|
| Tabular only | 0.497 | 0.686 | 0.577 |
| Tabular + narrative | 0.494 | 0.615 | 0.548 |
| Narrative only | 0.548 | 0.661 | 0.599 |

Table 4b. ADS LR feature set comparison, point estimates at the chosen threshold.

The interesting result is that **narrative alone beats tabular alone**. When we use the full XGBoost in Table 3 (which gets AUC 0.92 with tabular plus narrative on ADS), the lift becomes really big. So the structured form fields are not enough on their own for ADS, the short text in the report is what carries most of the signal.

Top narrative flags (by absolute LR coefficient on ADS, from `lr_ads_coefficients.csv`):

- `nav_av_stopped`, `nav_other_struck_av`, `nav_minor_damage_lang` are associated with **lower** severe probability.
- `nav_av_moving`, `nav_lane_change`, `nav_in_parking_lot` push the probability **up** in some clusters (see 6.5).

The figure below shows the largest odds ratios of the ADS LR model. Bars to the right of the dashed line at 1.0 raise the predicted severe probability, bars to the left lower it. This is what makes logistic regression useful here: even when its raw accuracy is lower than the tree models, the coefficients are easy to read.

![Top odds ratios of the ADS logistic regression with tabular plus narrative features.](Presentation/figures/09_top_odds_ratios_lr.png)

For the winning ADS XGBoost model the picture is a bit different. Tree based models do not give signed coefficients, instead we read the gain importance from `xgb_ads_importances.csv`. The top 10 features are in Table 4c. The list mostly agrees with the LR view (vulnerable users, motorcycle / cyclist crash partners, AV moving at impact), but XGBoost also leans on `Report Month` and on free form fields like `SV Pre-Crash Movement_Other, see Narrative` that the linear model cannot exploit as cleanly.

| Rank | Feature | Importance |
|---|---|---|
| 1 | Report Month | 0.213 |
| 2 | nav_vulnerable_user | 0.075 |
| 3 | Crash With_Non Motorist: Cyclist | 0.069 |
| 4 | Crash With_Motorcycle | 0.045 |
| 5 | Crash With_Non Motorist: Other | 0.036 |
| 6 | nav_av_moving | 0.024 |
| 7 | SV Pre Crash Movement_Other, see Narrative | 0.023 |
| 8 | CP Pre Crash Movement_Proceeding Straight | 0.017 |
| 9 | nav_minor_damage_lang | 0.016 |
| 10 | nav_av_turning | 0.016 |

Table 4c. Top 10 ADS XGBoost feature importances (gain).

### 6.4 Error analysis (false negatives)

The output of `09_stratified_fn_analysis.py` gives the missed severe crashes for the best model on each level. Summary (`fn_summary.csv`):

| Model | Severe in test | FN | FN rate | Top roadway | Top crash partner |
|---|---|---|---|---|---|
| XGB on ADS | 283 | 19 | 6.7% | Street | SUV |
| LR on L2 | 766 | 19 | 2.5% | Highway / Freeway | Other, see narrative |

Table 5. False negatives per stratified model.

Figure `Presentation/figures/17_fn_analysis_stratified.png` shows the top contexts as bar charts.

![Where the stratified models miss severe cases (current era test).](Presentation/figures/17_fn_analysis_stratified.png)

The two confusion matrices below give the counts behind these miss patterns. ADS XGBoost has 22 false negatives out of 283 severe cases in the test set (about 7.8%), while LR on L2 misses 25 out of 766 (about 3.3%, before threshold tuning makes the L2 RF / XGB hit zero).

![Confusion matrices for the stratified models on the current era test, ADS on the left and L2 on the right.](Presentation/figures/14_stratified_confusion_matrices_ads_l2.png)

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

Before the cluster profile, the figure below shows two diagnostic views: the silhouette score across `k` from 3 to 8, and a 2D PCA projection of the ADS feature matrix colored by the cluster id. The silhouette curve is what we used to pick k = 7, and the PCA scatter gives a sense of how well separated the clusters are in feature space.

![ADS clustering diagnostics: silhouette score against k, and PCA scatter colored by cluster id.](Presentation/figures/10_clustering_silhouette_and_pca.png)

The figure below brings together the parts of the ADS clustering that are useful to read at once. The top left panel is the silhouette score against `k` from 3 to 8, with a dashed line at the value chosen by the maximum (k = 7). The next two panels show the cluster sizes and the severe rate per cluster compared with the overall ADS average. The bottom panels show the roadway type distribution per cluster and the human readable scenario label table that we used in Table 6.

![ADS clustering summary: silhouette score by k, cluster sizes, severe rate per cluster, roadway type per cluster, and the scenario label table.](Modeling/clustering/kmeans_cluster_profiles_figure.png)

For comparison, the L2 clustering looks like the figure below. Most L2 clusters sit between 95% and 100% severe, so the bars in the "severe rate per cluster" panel are almost flat at the top. This visually confirms what we said in section 4: for L2, the scenario differences are real but they do not matter much for the severity label, because almost everything that gets reported as L2 is already severe.

![L2 clustering summary, same panels as the ADS figure.](Modeling/clustering/l2_kmeans_cluster_profiles_figure.png)

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
