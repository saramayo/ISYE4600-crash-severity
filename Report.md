# Predicting Crash Severity in NHTSA SGO Reports

**ISYE 4600 — Spring 2026**

Santiago Aramayo, Lauren McDonald, Luis Velez
April 27, 2026


---

## 1. Problem statement and goal

Companies that test self driving cars in the United States are required to send a crash report to NHTSA under the Standing General Order (SGO). These reports include vehicles that use a full driving system (ADS, like Waymo or Cruise) and also vehicles that use a Level 2 driver assist (Tesla Autopilot).


We used this public dataset, however the reports are not completely standardized, as each report is written by the reporting company, and the format, wording, level of detail, and missing information varies across companies and years.


A safety analyst who reads these reports will need to define which incidents to focus on in order to maximize the safety improvement. Our project focuses on one simple but valuable question,  which incidents are likely to be severe? Incidents that are likely to be severe will allow  analysts to spend the most time in analyzing situations that lead to severe outcomes.

In our project we consider that a crash is severe if at least one of these things is true:

- the highest reported injury is moderate or worse,
- the airbag was deployed for the subject vehicle, or
- some vehicle was  towed.

This was the rule that was used in the SGO public summary. Our project aims to build a model that, given the incident with  structured fields and the short narrative the company writes- which is unstructured, and then  return the probability that the incident is severe. Our aim is not to replace the human review, but instead to flag the most likely severe cases on top so the analyst spends time where it matters the most.


---

## 2. Data Sources, Features and Cleaning

Sources - We have four CSV files from the NHTSA SGO portal: ADS current era, ADS archived era, L2 current era, L2 archived era. These  cover a timeframe from mid 2021 to early 2025. We treat the "archived" era as past reports and the "current" era as more recent reports, which gives us a good temporal split that we used for training and testing. 


As far as the incident reports, each include one row per unique incident. The raw files contain one row per report version, and a single crash can have many versions and even many vehicles. We used the script:  `01_clean_incidents.py` in order to keep the latest version for each report id, we then group by `Same Incident ID` so the model sees one example for each real-world crash.

 We use a binary label `severe` which is defined by the OR rule we had mentioned above. To avoid directly representing the outcome, we also keep a flag `severity_known` and only train and test on rows where at least one of these components is observed. Out of the 5,576 unique incidents, 5,567 have at least one severity signal, of these 4,063 are severe and 1,504 are not considered severe.

The visual below gives us the OR rule we used.

![Severity label rule. If at least one of injury moderate or worse, airbag deployed, or vehicle towed is true, the incident is labeled severe.](Presentation/figures/02_severity_label_rule_schematic.png)

Features: 

We divided these into two groups, what we get directly from the tabular fields in the incidents reports, and what we extract from the free text narrative.

1. Structured (tabular):  These structured fields are pulled directly from the report, these include: roadway type, weather flags, crash partner, pre crash movement of subject and counterpart vehicles, speed, automation engagement, month...


2. Narrative flags: These are extracted from the free text by `narrative_utils.py`. We use 21 binary flags that describe the scenario , some of these are `nav_av_stopped`, `nav_other_struck_av`, `nav_at_intersection`, `nav_in_parking_lot`. We have purposefully excluded words like  like "injured", "towed", or "airbag" as features, since those words define the label rule above, then adding these would be a label leakage and would defeat the purpose on this additional training and learning.

For additional preprocessing, the numeric columns are filled with the median, and we filled the categorical columns  with the string `"Unknown"`. For each column that has missing values we also added  an `is_missing` binary indicator. The categorical columns are then one hot encoded for the linear models.

Training and testing approach:

Our approach was to train on the archived era, and then test on the current era. This is harder than a random split (the severe rate changes between eras, see Section 3), but it is the only fair way to imitate deployment. For the stratified models, which are just the models separated by Automation level we also then reserved a 25% of the training data as a validation set for  threshold tuning.

The figure below shows this split, it includes row counts and severe rate on the right. 
In the case of the L2 severe rate it stays close to its ceiling,  while the ADS severe rate jumps from 26% to 47%, this is the shift the pooled baseline in Section 3 was not able to handle. 

![Temporal train and test split, row counts and severe rate by era and automation level.](Presentation/figures/03_temporal_train_test_split.png)

One of the limitations that we faced was regarding the fact that SGO reporting is not standardized. What we mean by this is that companies report what their internal systems flag, and the way that they do this, therefore  changed across different years, and the narrative section is sometimes redacted.

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

## 3. Initial Pooled Baseline Approach 

Our first attempt was to train one logistic regression on the combination of the ADS and L2 with a temporal split, balanced class weights, and also `automation_level` as a feature. 

This is what our script : `02_run_baselines.py` does.

The pooled metrics on the test set look fine, however through the deeper evaluation we found some key failures (`baseline_results.csv`):

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

We are testing each model on a separate test set that represents the “current era” data. The model does not train on this data, so it shows how well the model performs on crashes it has not seen before. 

The metrics used are precision, recall, F1, ROC AUC and false negative rate, allowing us to evaluate the model from a performance and safety perspective. The precision measures how many of the predicted severe crashes were actually severe. This is important because false alarms waste analysts time. Recall is a very important metric we used because it measures how many severe cases the model was actually able to find, which is the biggest concern of this project. F1 combines both the precision and recall. ROC AUC measures how well the model ranks severe crashes above non-severe crashes (for model performance) and false negative rate measures how many severe crashes missed (top priority for safety).

The training set was only used to tune and choose models. The test set was saved until the very end. This ensures that the test results are not biased by repeatedly checking and adjusting the model based on the test data.


---

## 6. Results

### 6.1 Stratified results 

Our full result file is `Modeling/logistic_regression/all_stratified_results.csv`.

| Model | Level | Precision | Recall | F1 | AUC | FN rate | Threshold |
|---|---|---|---|---|---|---|---|
| LR | ADS | 0.499 | 0.845 | 0.627 | 0.531 | 15.5% | 0.36 |
| RF | ADS | 0.804 | 0.883 | 0.842 | 0.893 | 11.7% | 0.48 |
| **XGB** | **ADS** | **0.831** | **0.922** | **0.874** | **0.921** | **7.8%** | **0.46** |
| LR | L2 | 0.985 | 0.967 | 0.976 | 0.876 | 3.3% | 0.20 |
| RF | L2 | 0.972 | 1.000 | 0.986 | 0.836 | 0.0% | 0.20 |
| XGB | L2 | 0.972 | 1.000 | 0.986 | 0.850 | 0.0% | 0.20 |

Table 3. Stratified LR / RF / XGB separated by automation level on the current era test.

For ADS , XGBoost was clearly better on every metric. This big improvement is likely from the  non linear patterns that it is able to capture. We can see a clear jump of LR to RF in the AUC results is from 0.53 to 0.89. XGBoost is then able to add another step, and is able to cut the false negative rate from 11.7% to 7.8%. Compared to the pooled baseline FN rate of 100% on ADS, this is exactly the improvement that we were aiming for.

For the L2 level, all three models were able to reach an F1 higher than 0.97 and got FN rates between 0% and 3%. Random Forest and XG boost are have similar results likely because L2 is almost linearly separable (98% positive class).

 We see that the logistic regression false negatives are the more interesting ones, since  RF predicts every L2 case as severe, and therefore  gets 0% FN by default. 

This is summarized in Figure 2 - `Presentation/figures/13_model_comparison_all.png`.

![Stratified model comparison, all six combinations on the current era test set.](Presentation/figures/13_model_comparison_all.png)

### 6.2 Pooled baseline vs stratified ADS

The figure below gives us the before and after view of the failure earlier. 

 `Presentation/figures/15_ads_pooled_vs_stratified_improvement.png`.

![Pooled LR on the ADS slice (left bars in each pair) compared with the stratified ADS model (right bars).](Presentation/figures/15_ads_pooled_vs_stratified_improvement.png)

We can see that the pooled bars are zero for every metric for ADS and the stratified bars reach the numbers from Table 3. 
The change that we created in splitting the data by automation and the addition of XGBoost is what was able to capture the improvement that we were looking for.

### 6.3 Influence of the Extracted Narrative 

We now compare the three feature sets on the ADS logistic regression: tabular only, tabular plus narrative, narrative only. 
The AUC results are in the csv below: (`narrative_ads_model_comparison.csv`):

| Feature set | AUC |
|---|---|
| Tabular only | 0.500 |
| Tabular + narrative | 0.530 |
| Narrative only | 0.608 |

Table 4. ADS LR with different feature sets, current era test.

We proceed with the same comparison looking at precision, recall and F1 in Table 4b.

 We have included it because AUC alone is not enough to show the change. We can see that the narrative only setup has the highest AUC, and it also has the highest F1 (0.599) and the highest precision (0.548), additionally it still keeps a recall above the tabular only setup.

| Feature set | Precision | Recall | F1 |
|---|---|---|---|
| Tabular only | 0.497 | 0.686 | 0.577 |
| Tabular + narrative | 0.494 | 0.615 | 0.548 |
| Narrative only | 0.548 | 0.661 | 0.599 |

Table 4b. ADS LR feature set comparison 

A result that we were not expecting was that the narrative alone beats tabular alone. When we use the full XGBoost in Table 3 (which gets AUC 0.92 with tabular plus narrative on ADS), the improvement became massive. This tells us that the structured form fields are therefore not enough on their own for ADS, and that the unstructured short narrative in the report carries real meaning and signal.

Top narrative flags (by absolute LR coefficient on ADS, from `lr_ads_coefficients.csv`):

- `nav_av_stopped`, `nav_other_struck_av`, `nav_minor_damage_lang` are associated with **lower** severe probability.
- `nav_av_moving`, `nav_lane_change`, `nav_in_parking_lot` push the probability **up** in some clusters (see 6.5).

The figure below is showing the largest odds ratios of the ADS LR model.
We can see that the bars to the right of the dashed line at 1.0 were able to raise the predicted severe probability, and the bars to the left lower it. 
This is what makes logistic regression useful, since even if raw accuracy is lower than the tree models, the coefficients are easier to read.

![Top odds ratios of the ADS logistic regression with tabular plus narrative features.](Presentation/figures/09_top_odds_ratios_lr.png)

We can see that for ADS XGBoost model the results change. The tree based did not give signed coefficients, so we read the improvement from `xgb_ads_importances.csv`. 

We can see the top 10 features are in Table 4c. This list is in accordance with the LR view - these include vulnerable users, motorcycle / cyclist crash partners, AV moving at impact. However, we can also see that XGBoost also relies on `Report Month` and on free form fields like `SV Pre-Crash Movement_Other, see Narrative` , these are some of thhe fields that the linear model cannot use cleanly.

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

The output of the `09_stratified_fn_analysis.py` gives us the missed severe crashes for the best model on each level. We can see a summary of this on the CSV: (`fn_summary.csv`):

| Model | Severe in test | FN | FN rate | Top roadway | Top crash partner |
|---|---|---|---|---|---|
| XGB on ADS | 283 | 19 | 6.7% | Street | SUV |
| LR on L2 | 766 | 19 | 2.5% | Highway / Freeway | Other, see narrative |

Table 5. False negatives per stratified model.

Figure `Presentation/figures/17_fn_analysis_stratified.png` shows the top contexts as bar charts.

![Where the stratified models miss severe cases (current era test).](Presentation/figures/17_fn_analysis_stratified.png)

The two confusion matrices below give us the counts behind the patterns that it missed. ADS XGBoost had 22 false negatives out of 283 severe cases in the test set (about 7.8%), while LR on L2 misses 25 out of 766 , which is about 3.3%, before threshold tuning makes the L2 RF / XGB hit zero .

![Confusion matrices for the stratified models on the current era test, ADS on the left and L2 on the right.](Presentation/figures/14_stratified_confusion_matrices_ads_l2.png)

In ADS the model mostly misses the  street level crashes with a passenger vehicle, these are cases where the AV was stopped or in slow traffic and the narrative is therefore very short. These cases look very soft, however the airbag did fire or the vehicle was towed. 

On L2 the most of the misses are  on highway / freeway crashes that the company just describes  "Other" this is where the structured data gives us less information and fails. 

### 6.5 ADS scenario clusters

To support our analysis we ran k means on the ADS data (script ⁠ 10_cluster_profiling.py ⁠). By evaluating the silhouette score for each cluster, we conluded that k = 7 is the choice that best fits our data. Below is the summary table for the results yielded by this analysis(⁠ ads_cluster_summary.csv ⁠):

| Cluster | Size | Severe rate | Scenario label |
|---|---|---|---|
| 2 | 56% | 26% | Typical street crash, AV stopped |
| 5 | 21% | *48%* | AV moving at impact |
| 0 | 14% | 33% | Other vehicle struck AV |
| 1 | 4% | 42% | AV hit a fixed object (parking lot) |
| 3 | 4% | 31% | AV struck other party |
| 4 | 1% | 0% | Animal strike |
| 6 | <1% | 0% | Pedestrian / cyclist case (very few) |

Table 6. ADS scenario clusters (k = 7).

The biggest cluster (C2) is the typical "AV stopped, low speed bump" type. The most severe one (C5) is "AV moving at impact". These are the cases in which the operations team would be most interested: very frequent crash scenarios and the most severe ones. The animal cluster has zero severe cases, which is consistent with how those incidents are reported. We also ran the same clustering on L2 (⁠ 11_cluster_profiling_by_level.py --level L2 ⁠) for symmetry. Almost every L2 cluster is at 95% to 100% severity rate, so using clustering models for L2 is mostly about indentifying scenario type, not about evaluating clusters that had high severity rates.

Before the cluster profile, the figure below shows two diagnostic views: the silhouette score across ⁠ k ⁠ from 3 to 8, and a 2D PCA projection of the ADS feature matrix colored by the cluster id. The silhouette curve is what we used to pick k = 7, and the PCA scatter plot gives a sense of how well separated the clusters are in the feature space.

![ADS clustering diagnostics: silhouette score against k, and PCA scatter colored by cluster id.](Presentation/figures/10_clustering_silhouette_and_pca.png)
 
The figure below displays interpretations of the results of applying clustering algorithms to ADS data. The top left panel is the silhouette score against ⁠ k ⁠ from 3 to 8, with a dashed line at the value chosen by the maximum (k = 7). The next two panels show the cluster sizes and the severity rate per cluster compared with the overall ADS average. The bottom panels show the roadway type distribution per cluster and the label of each cluster used in Table 6.

![ADS clustering summary: silhouette score by k, cluster sizes, severe rate per cluster, roadway type per cluster, and the scenario label table.](Modeling/clustering/kmeans_cluster_profiles_figure.png)

For comparison, the L2 clustering looks like the figure below. Most L2 clusters sit between 95% and 100% severe, so the bars in the "severe rate per cluster" panel are almost flat at the top. This visually confirms what we said in section 4: for L2, the scenario differences are real but they do not matter much for the severity label, because almost everything that gets reported as L2 is already severe.

![L2 clustering summary, same panels as the ADS figure.](Modeling/clustering/l2_kmeans_cluster_profiles_figure.png)

---

## 7. Interpretation, changes in our approach, limitations and next steps

Interpretation of Results: 

This means the ADS XGBoost model could help a safety team prioritize which crash reports to review first. Instead of reading every report randomly, they could rank reports by the model’s predicted probability of being severe. The reports with the highest predicted risk would be "triaged" and reviewed first by the safety analysts.

Another interpretation is that it also says Cluster C5, described as “AV moving at impact,” is the most important crash scenario. This is important to watch because it gives useful information about risk. For the L2 model, the model is not very useful as a machine learning problem because almost all L2 reports are already labeled severe. This means that the L2 model is primarily just useful as a check to make sure the severity labeling rule is consistent.


We also recognize where some of our assumptions are limited, for example the "severe" label is based on the rule we created, and is not an official score of injury or severity. So "severe" essentially means that the crash met the OR rule (injury, airbag, towing, etc). Also, the reports are written by different companies, so the language used may differ depending on who wrote it. One person might describe crashes in more detail than the other. Furthermore, the 'severity_known' filter only removed about 9 accidents, likely not changing the results drastically. 


Changes in our approach:

There were a few things we tried and dropped from the final pipeline. First, pooled training across the levels meant combining ADS and L2 data into one model. This did not work well because it missed all severe ADS cases (100% false nefative rate), which is a major safety failure. Another initial problem was the outcome words in the narrative meant using words like "towed" or "injured" from the crash description caused label leakage, making the model look really good, but unfair because those words directly reveal the label. To fix this, we removed these words. Adding the 21 text-based flags to the L2 model was also problematic since almost every L2 case was already severe, meaning the extra narrative features did not improve the model. This made us pivot back to keeping L2 tabular only. Finally, another issue we ran into is that pure logistic regression on ADS meant using a simpler linear model. This model performed poorly (ADS LR stays at AUC 0.53) because the patterns in ADS crashes are likely more complex, which is why XGBoost worked significantly better. 


Limitations:
This study has several limitations. First, the SGO dataset is not a random sample of all autonomous vehicle crashes, so the results may not fully represent all AV crash patterns. Second, the ADS test set is still relatively small, with around 600 incidents, which limits how confidently the results can be generalized. Another limitation is that the temporal split depends on how NHTSA defines the “archived” and “current” data at the time of submission, so the boundary between eras may change over time. Another limitation that is important to point out is that some crash narratives are redacted, which prevents the model from identifying certain scenario flags and limits what it can learn from those reports.


Next steps 

To further expand and improve the model some next steps include:
1.⁠ ⁠*Replace the regex narrative flags with a small LLM.* This could produce the same JSON outputs while also capturing paraphrased crash descriptions better than the current regex approach. .
2.⁠ ⁠*Maintain reporducibality and avoid label leakage.* Doing this allows for LLM outputs to be caches and prompts to avoid outcome-related terms (injury, airbag, towing, etc). 
3.⁠ ⁠*Calibrate the threshold per cluster, not per level.* Instead of using one threshold for automation level, thresholds should be adjusted by cluster so that the model can better reflect differences in crash risk.
4.⁠ ⁠*Prioritize C5.* Since Cluster C5 has a 48% severe rate, a different threshold may catch more "AV moving at impact" cases while also keeping the false alarm rates within a reasonable range.


---

## 8. Reproducing Results

For reproducibality, all analyses were run using Python 3 with the required dependencies listed in ⁠ requirements.txt ⁠. For macOS users, OpenMP should be installed first using:

⁠ bash
brew install libomp
 ⁠
This ensures that XGBoost can run correctly.

To get the results reported in Tables 2 and 3, run:


python scripts/01_clean_incidents.py
python scripts/02_run_baselines.py
python scripts/05_stratified_models_ads_l2.py


To reproduce the remaining analyses, including Tables 4–6 and the figures that were referenced, run:


python scripts/06_narrative_features.py
python scripts/09_stratified_fn_analysis.py
python scripts/10_cluster_profiling.py
python scripts/11_cluster_profiling_by_level.py -- level L2
python scripts/make_presentation_figures.py


All random seeds were fixed at 42 to support reproducibility. The README contains the full output map for the generated tables, figures, and model results.
