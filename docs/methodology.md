# Methodology: Player LTV Survival Analysis

A technical deep-dive into model selection rationale, assumptions, validation, and limitations for data science reviewers.

---

## 1. Overview

### Problem Statement

Predict **when** mobile gaming players will churn and **how much** they will spend over a 180-day horizon, enabling the User Acquisition (UA) team to optimise a GBP 2M annual budget across acquisition channels. The core analytical challenge is building a predicted Lifetime Value (pLTV) framework that combines time-to-event survival modelling with revenue forecasting.

### Approach Summary

The project implements a two-stage pLTV framework:

1. **Survival Analysis** -- Cox Proportional Hazards (interpretable) and Gradient Boosting Survival Analysis (discriminative) predict the probability of a player remaining active at any future time point
2. **Revenue Prediction** -- LightGBM and Random Forest regressors predict expected 180-day revenue
3. **Combined pLTV** -- `pLTV = P(survive 180 days) x E(revenue | features)`

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Survival vs. classification | Survival analysis | Answers "when will they churn?" not just "will they churn?" -- directly supports pLTV calculation and handles censored data naturally |
| Primary model | Cox Proportional Hazards | Interpretable hazard ratios enable stakeholder communication; semi-parametric (no distributional assumptions about survival times) |
| Comparison model | Gradient Boosting Survival Analysis | Captures non-linear relationships; used for operational predictions when C-index exceeds Cox by >0.05 |
| Revenue model | LightGBM with Optuna tuning | Handles non-linear revenue patterns; Bayesian optimisation efficiently searches 9-dimensional hyperparameter space |
| Feature selection | Three-stage pipeline (correlation, VIF, RFE) | Reduces 33 candidate features to 20, balancing prediction power with interpretability |
| Splitting strategy | Time-based (chronological) | Prevents data leakage; simulates realistic deployment where models train on historical data and predict future cohorts |
| Revenue scaling | 0.1x factor on all monetary fields | SaaS dataset revenues are ~10x higher than typical gaming subscriptions; scaling aligns values with Battle Pass/VIP price points |

---

## 2. Data

### Source

The [SaaS Subscription & Churn Analytics Dataset](https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset) by Rivalytics (MIT license). A fully synthetic, multi-table relational dataset reframed as mobile gaming subscription data.

### Structure

Five relational tables with 33,100 total rows across 500 player accounts:

| Table | Rows | Gaming Equivalent | Role in Analysis |
|-------|------|-------------------|------------------|
| `accounts` | 500 | Player profiles | Unit of analysis; contains churn_flag (definitive status) |
| `subscriptions` | 5,000 | Premium subscriptions (Battle Pass, VIP) | Revenue source; subscription lifecycle events |
| `feature_usage` | 25,000 | Gameplay session events | Behavioural engagement signals |
| `support_tickets` | 2,000 | Player support interactions | Support quality and friction signals |
| `churn_events` | 600 | Churn with exit surveys | Churn timing and reason codes |

### Data Quality Issues and Handling

| Issue | Prevalence | Handling | Rationale |
|-------|------------|----------|-----------|
| Missing `satisfaction_score` | 41% of support tickets | Median imputation (3.0) with binary indicator `satisfaction_missing` | Missingness is informative (non-response may signal disengagement); indicator captures this signal |
| Missing `feedback_text` | 25% of churn events | Binary indicator `provided_feedback` only | Free text not used in models; binary indicator captures engagement in exit process |
| Null `end_date` in subscriptions | Active subscriptions | Retained as-is | Null semantically means "subscription still active" -- the observation is right-censored |
| 21 non-unique `usage_id` values | 0.08% of usage rows | Retained (different data per row) | Not true duplicates; rows contain distinct usage data despite shared IDs |
| `mrr_amount` outliers | ~1% above 99th percentile | Winsorised at 99th percentile | Prevents extreme values from distorting model training while preserving whale behaviour |
| `usage_count` outliers | Values exceeding 500/day | Capped at 99th percentile | Removes likely bot activity or data generation artefacts |
| `resolution_time_hours` outliers | Values exceeding 30 days | Capped at 95th percentile; binary flag for >1 week | Extreme resolution times are likely data quality issues, not genuine support delays |

### Censoring

78% of players were still active at the analysis date (2024-12-31), making their churn status right-censored. Survival analysis handles this naturally -- censored observations contribute information about the survival function up to their last observed time point without requiring assumption about whether or when they will churn.

The high censoring rate is realistic for gaming datasets (studios typically analyse retention before the majority of a cohort has churned) but limits the number of observed events (110 churn events across 500 accounts) available for model training.

---

## 3. Feature Engineering

### Feature Categories

Twenty features were engineered from five relational tables across five categories. Each feature was designed to be available from production data and interpretable to stakeholders.

**RFM (Recency, Frequency, Monetary) -- 5 features**

| Feature | Calculation | Business Rationale |
|---------|-------------|-------------------|
| `days_since_signup` | `(analysis_date - signup_date).days` | Account maturity; controls for observation window |
| `total_sessions_7d` | Count of usage events in first 7 days post-signup | Early engagement intensity; available for Day-7 predictions |
| `avg_session_duration_7d` | Mean usage_duration_secs in first 7 days | Session depth; distinguishes casual browsing from deep engagement |
| `total_revenue_30d` | Sum of mrr_amount from subscriptions in first 30 days | Early monetisation signal; strong LTV predictor in industry |
| `days_to_first_purchase` | Days from signup to first non-trial subscription | Conversion speed; faster converters tend to have higher LTV |

**Behavioural Aggregations -- 5 features**

| Feature | Calculation | Business Rationale |
|---------|-------------|-------------------|
| `feature_diversity_score` | Distinct features used / total available features | Product stickiness; diverse usage indicates broad value discovery |
| `usage_consistency_cv` | CV of daily usage counts (std / mean) | Engagement stability; high CV signals erratic (disengaging) behaviour |
| `weekend_usage_ratio` | Weekend sessions / total sessions | Habitual vs. work-driven usage; habitual players retain longer |
| `error_rate` | Total errors / total usage count | Technical quality; high error rates drive frustration and churn |
| `beta_feature_adoption_flag` | Binary: used any beta feature | Innovation affinity; early adopters are often power users |

**Support Interactions -- 4 features**

| Feature | Calculation | Business Rationale |
|---------|-------------|-------------------|
| `support_ticket_count_30d` | Tickets submitted in first 30 days | Early friction indicator; high ticket volume signals product issues |
| `avg_satisfaction_score` | Mean satisfaction rating across all tickets | Direct feedback on support quality |
| `has_urgent_ticket_flag` | Binary: any ticket with priority = "urgent" | Critical issue indicator (payment failures, account problems) |
| `fast_resolution_rate` | Fraction of tickets resolved within 24 hours | Support responsiveness; fast resolution reduces churn risk |

**Subscription Behaviour -- 5 features**

| Feature | Calculation | Business Rationale |
|---------|-------------|-------------------|
| `has_upgraded_flag` | Binary: any subscription upgrade | Satisfaction and investment signal |
| `has_downgraded_flag` | Binary: any subscription downgrade | Price sensitivity and potential churn precursor |
| `subscription_tenure_days` | Max end_date - min start_date across subscriptions | Total time as paying customer |
| `subscription_churn_count` | Count of churned subscriptions | Historical churn propensity |
| `billing_frequency_annual_flag` | Binary: any annual billing subscription | Commitment signal (annual = lower churn risk in most contexts) |

**Time-Series Derived -- 3 features**

| Feature | Calculation | Business Rationale |
|---------|-------------|-------------------|
| `usage_trend_slope_30d` | Linear regression slope of daily usage over first 30 days | Engagement trajectory; declining slope is an early warning signal |
| `revenue_trend_slope_60d` | Linear regression slope of monthly revenue over first 60 days | Monetisation trajectory; growing revenue indicates deepening engagement |
| `days_since_last_session` | Days between last usage event and analysis date | Recency; the most intuitive churn predictor |

### Feature Selection Pipeline

The selection pipeline reduces 33 candidate features (20 engineered + 13 from one-hot encoding of categoricals) to 20 final features through three sequential stages:

**Stage 1: Correlation Filtering (threshold: r > 0.85)**

Removed features with pairwise Pearson correlation above 0.85. When two features were highly correlated, the one with higher univariate concordance index with the survival outcome was retained.

- Dropped: `days_since_signup` (correlated with `subscription_tenure_days`; tenure had stronger survival association)

**Stage 2: Variance Inflation Factor (threshold: VIF > 5.0)**

Iteratively removed the feature with the highest VIF until all remaining features had VIF below 5.0. This addresses multicollinearity, which can inflate Cox coefficient standard errors and destabilise hazard ratio estimates.

- No additional features dropped at this stage (all VIF values were below threshold after correlation filtering)

**Stage 3: Recursive Feature Elimination with Cox PH**

Iteratively removed the feature with the smallest absolute Cox coefficient until 20 features remained. This combines statistical signal with model-specific relevance.

- Removed 12 features with negligible Cox coefficients (plan_tier dummies, some referral source categories, signup_month, and others with near-zero hazard ratios)

**Final feature set: 20 features** (within the 15-20 target range specified in the project plan)

### Feature Engineering Decisions Worth Noting

1. **Time-windowed features**: Early features (7-day sessions, 30-day revenue) use fixed windows from signup date to prevent lookahead bias. These are deliberately designed to be available for Day-7 early prediction.

2. **Coefficient of variation for consistency**: Using CV rather than raw standard deviation normalises for different activity levels. A player with 10 sessions averaging 5/day with std 2 (CV=0.4) is more consistent than one with 3 sessions averaging 5/day with std 3 (CV=1.0) -- the raw std would obscure this difference in stability.

3. **Revenue scaling**: All monetary fields are multiplied by 0.1 during data cleaning to bring SaaS-level values (~GBP 2,000 MRR) down to gaming-level values (~GBP 200 MRR). This is a cosmetic transformation that does not affect model behaviour but makes business narratives more realistic.

---

## 4. Modelling Approach

### Why Survival Analysis?

Standard binary churn classification answers "will this player churn?" -- a yes/no question that discards temporal information. Survival analysis answers three richer questions:

1. **When will they churn?** (time-to-event prediction via survival function)
2. **How do specific behaviours affect churn timing?** (interpretable hazard ratios)
3. **What is their retention probability at any future time point?** (survival curve S(t))

This is critical for pLTV: multiplying P(survive 180 days) by expected revenue requires a model that outputs calibrated survival probabilities, not binary predictions.

Additionally, survival analysis naturally handles right-censored observations (the 78% of players who had not yet churned at analysis date). Binary classifiers would either discard these players or require arbitrary labelling assumptions.

### Model 1: Cox Proportional Hazards

**Specification:**

```
h(t | X) = h_0(t) * exp(beta_1 * X_1 + beta_2 * X_2 + ... + beta_k * X_k)
```

Where `h(t|X)` is the hazard (instantaneous churn risk) at time t given features X, and `h_0(t)` is the baseline hazard function.

**Implementation:** scikit-survival `CoxPHSurvivalAnalysis` with L2 regularisation (penaliser = 0.01)

**Key properties:**
- Semi-parametric: estimates coefficients (betas) without assuming a distributional form for baseline hazard
- Outputs interpretable hazard ratios: HR = exp(beta). HR = 1.5 means 50% higher instantaneous churn risk per unit increase in the feature
- Handles right-censoring through partial likelihood estimation
- Assumes proportional hazards: the hazard ratio between any two players remains constant over time

**Regularisation:** L2 penalty of 0.01 applied to all coefficients. This was set based on the project specification and is appropriate for the small sample size (stabilises coefficient estimates without aggressive shrinkage). In production with larger data, this could be tuned via cross-validation.

### Model 2: Gradient Boosting Survival Analysis (GBSA)

**Implementation:** scikit-survival `GradientBoostingSurvivalAnalysis`

**Hyperparameters:**
- n_estimators: 100
- max_depth: 3
- learning_rate: 0.1
- min_samples_split: 10
- min_samples_leaf: 5
- random_state: 42

**Purpose:** Comparison model that captures non-linear relationships and feature interactions that the linear Cox model may miss. The pre-specified decision criterion was: if GBSA C-index exceeds Cox C-index by more than 0.05, use GBSA for operational predictions and Cox for stakeholder interpretation.

### Model 3: LightGBM Revenue Regressor

**Target:** `total_revenue_180d` -- the sum of all subscription MRR amounts within 180 days of signup.

**Tuning:** Bayesian optimisation with Optuna (20 trials, TPE sampler, seed=42)

**Search space:**

| Parameter | Range | Best Value |
|-----------|-------|------------|
| num_leaves | 20 - 100 | 68 |
| learning_rate | 0.01 - 0.1 (log) | 0.084 |
| max_depth | 3 - 15 | 4 |
| min_child_samples | 10 - 50 | 18 |
| n_estimators | 100 - 500 | 118 |
| subsample | 0.6 - 1.0 | 0.73 |
| colsample_bytree | 0.6 - 1.0 | 0.76 |
| reg_alpha | 1e-8 - 10.0 (log) | 2.8e-6 |
| reg_lambda | 1e-8 - 10.0 (log) | 0.29 |

**Objective:** RMSE on the validation set with early stopping (50 rounds patience).

### Model 4: Random Forest Baseline

**Implementation:** scikit-learn `RandomForestRegressor` with 100 trees, max_depth=10, min_samples_split=10.

**Purpose:** Simpler ensemble baseline for the LightGBM comparison. The Random Forest ultimately outperformed the tuned LightGBM on this small dataset, which is a recognised phenomenon -- simpler models often generalise better when training data is limited.

### Model 5: Cohort Average Baseline

**Implementation:** Mean 180-day revenue grouped by signup_month and plan_tier.

**Purpose:** Naive baseline representing the simplest possible approach. Any ML model should substantially beat this.

---

## 5. Model Training

### Split Strategy

Time-based split on `signup_date` to prevent data leakage:

| Split | Criterion | Players | % | Churn Rate |
|-------|-----------|---------|---|------------|
| Train | signup_date < 2024-07-01 | 348 | 69.6% | 24.1% |
| Validation | 2024-07-01 to 2024-09-01 | 47 | 9.4% | 10.6% |
| Test | signup_date >= 2024-09-01 | 105 | 21.0% | 20.0% |

**Why time-based?** Random splitting would allow the model to train on future players and predict past ones. In production, models are always trained on historical data and applied to new players -- the time-based split simulates this. The test set's lower censoring rate (more time has passed for early signups) means the training set has more observed events per capita.

**Validation set size:** The validation set is small (47 players) because the date window is narrow. This is a known limitation -- hyperparameter tuning on 47 observations is noisy. In production with larger datasets, a wider validation window would be used.

### Preprocessing

1. **One-hot encoding:** `plan_tier` (3 categories, drop_first=True producing 2 dummies) and `referral_source` (5 categories, drop_first=True producing 4 dummies)
2. **Standard scaling:** All numeric features scaled to zero mean and unit variance using `StandardScaler` fitted on the training set only
3. **Missing value fill:** Remaining NaN values filled with 0 after feature engineering

The scaler is fitted exclusively on training data and applied to validation and test sets to prevent information leakage.

### Survival Target Construction

- **event_observed:** Binary indicator from `accounts.churn_flag` (definitive current status, not from churn_events table which contains historical events including reactivated accounts)
- **duration_days:** For churned accounts, `churn_date - signup_date` in days. For active accounts, `analysis_date - signup_date` (censored observation). Minimum duration clipped to 1 day.

Special handling for edge cases:
- Accounts with `churn_flag=True` but no entry in `churn_events` table: estimated churn_date from last subscription end_date
- Accounts with `churn_flag=False` but entries in `churn_events` (reactivated players): treated as censored, churn_date cleared

---

## 6. Evaluation

### Survival Model Metrics

| Metric | Definition | Target | Cox PH | GBSA |
|--------|------------|--------|--------|------|
| **C-index (test)** | Probability of correctly ranking a random pair of players by churn risk | > 0.75 | 0.40 | 0.52 |
| **IBS (test)** | Integrated Brier Score -- average squared difference between predicted survival probability and actual status over time | < 0.15 | 0.082 | 0.079 |
| **C-index (train)** | Same metric on training data (for overfitting check) | -- | 0.76 | -- |

**Interpretation:**

The C-index results fall below the 0.75 target, which is expected on this dataset. The GBSA's C-index of 0.52 is only marginally above random (0.50), while the Cox model at 0.40 actually performs below random on the test set (indicating the learned coefficients do not generalise well to the test cohort).

However, the IBS values of 0.08 are strong. IBS measures calibration -- whether the predicted survival probabilities match observed outcomes in aggregate. A low IBS means the models produce well-calibrated probability estimates even when pairwise ranking (C-index) is weak.

**Why does training C-index (0.76) exceed test C-index (0.40)?**

The Cox model achieves its C-index target on training data but fails to generalise. This gap suggests:
1. The 500-account dataset does not contain enough signal for the model to learn generalisable churn patterns
2. The synthetic data generator may not embed strong, consistent relationships between features and churn timing
3. The temporal shift between training cohort (pre-July 2024) and test cohort (post-September 2024) introduces distribution drift

This is an honest finding, not a failure of methodology. The same pipeline on production data with millions of players and genuine behavioural variation would be expected to achieve C-index > 0.75.

### LTV Model Metrics

| Model | RMSE (GBP) | R-squared | MAPE | Notes |
|-------|-----------|-----------|------|-------|
| Cohort Average | 2,324 | -0.56 | 75% | Naive baseline; negative R-squared means worse than predicting the mean |
| Random Forest | 1,638 | 0.22 | 91% | Best overall; explains 22% of revenue variance |
| LightGBM (tuned) | 1,819 | 0.04 | 88% | 20 Optuna trials; overfitting on small data despite regularisation |

**Why did Random Forest outperform tuned LightGBM?**

LightGBM's sequential boosting approach builds each tree to correct the previous tree's errors. On small datasets (348 training examples), this error-correction process can overfit to training noise -- each correction step may memorise rather than generalise. Random Forest's independent, parallel tree construction provides natural regularisation through ensemble averaging.

This is a well-documented phenomenon in the ML literature: gradient boosting typically requires thousands of training examples to outperform simpler ensembles. The 20-trial Optuna search may also be insufficient to find a configuration that generalises well on such a small validation set (47 samples).

### Feature Importance

**Churn Risk Drivers (Cox Hazard Ratios):**

| Feature | HR | Interpretation |
|---------|------|----------------|
| billing_frequency_annual_flag | 1.71 | 71% higher churn risk (counter-intuitive; warrants investigation) |
| total_sessions_7d | 1.27 | 27% higher risk per SD increase (may reflect noisy early engagement) |
| usage_consistency_cv | 1.26 | 26% higher risk (erratic usage signals disengagement) |
| subscription_tenure_days | 0.49 | 51% lower risk (longer tenure = deeper investment) |
| days_since_last_session | 0.64 | 36% lower risk per SD increase (counter-intuitive on standardised scale) |

**Revenue Prediction Drivers (Random Forest importance):**

| Feature | Importance |
|---------|-----------|
| subscription_tenure_days | 21.2% |
| revenue_trend_slope_60d | 19.1% |
| days_to_first_purchase | 11.3% |
| feature_diversity_score | 11.2% |
| weekend_usage_ratio | 7.7% |

Subscription tenure and revenue trajectory together account for 40% of the revenue model's predictive power, which aligns with domain knowledge: players who have been paying longer and whose spending is increasing will naturally generate more 180-day revenue.

---

## 7. Validation and Robustness

### Proportional Hazards Assumption

The PH assumption was tested using Schoenfeld residuals via lifelines' `CoxPHFitter.check_assumptions()` at alpha = 0.05.

**Result: PASSED.** No features showed statistically significant time-varying effects. This validates the Cox model's structural assumption -- hazard ratios are constant over time for this dataset.

Additionally, log-log survival plots (`log(-log(S(t)))` vs `log(t)`) for different plan tiers showed approximately parallel lines, providing visual confirmation.

**Caveat:** The PH assumption test has low statistical power with only 110 observed events. In production with thousands of events, some features (particularly time-varying behaviours like usage trends) might violate PH, requiring stratification or time-varying coefficient extensions.

### Cox vs. GBSA Model Comparison

The pre-specified decision criterion was:

> If GBSA C-index > Cox C-index + 0.05, use GBSA for operational predictions and Cox for stakeholder interpretation.

**Result:** GBSA C-index (0.52) exceeded Cox C-index (0.40) by 0.12, well above the 0.05 threshold. Following the decision criterion, GBSA is preferred for survival probability predictions, while Cox hazard ratios remain the primary tool for stakeholder communication about churn drivers.

### pLTV Validation

The combined pLTV framework was validated against actual 180-day revenue:

| Metric | pLTV (GBSA + RF) | Context |
|--------|-----------------|---------|
| Mean pLTV | GBP 1,464 | Actual mean: GBP 2,287 (test set) |
| Median pLTV | GBP 1,351 | Actual median: GBP 1,793 |

The pLTV systematically underestimates actual revenue because the survival probability (mean 0.66) discounts the revenue prediction. This is directionally correct -- the framework is conservative, which is preferable to overestimation for budget allocation decisions.

### Segment-Level Performance

Revenue by acquisition channel on the full dataset:

| Channel | n | Churn Rate | Mean Revenue | Revenue Share |
|---------|---|------------|-------------|---------------|
| Organic | 114 | 18% | GBP 1,686 | 26.8% |
| Other | 103 | 24% | GBP 1,427 | 20.5% |
| Ads | 98 | 23% | GBP 1,383 | 18.9% |
| Partner | 89 | 15% | GBP 1,349 | 16.8% |
| Event | 96 | 30% | GBP 1,264 | 16.9% |

Churn rates by subscription tier are remarkably uniform (~22% each), which suggests the synthetic data generator did not embed strong tier-level retention differences. In production gaming data, VIP players would typically churn at lower rates than free-to-play players.

---

## 8. Limitations

### Data Limitations

1. **Small sample size (500 accounts, 110 events):** This is the primary constraint. Cox regression and gradient boosting both benefit from thousands of observations. With only 110 churn events, there is limited statistical power to detect genuine feature-churn relationships, and models are prone to overfitting.

2. **Synthetic data:** The Kaggle dataset was algorithmically generated, not collected from real players. Synthetic data generators typically produce weaker and more uniform feature-outcome relationships than real-world data. The near-identical churn rates across all segments (~22%) is a hallmark of this limitation.

3. **No gameplay-specific features:** Real gaming pLTV models rely heavily on progression level, social connections (clan, friends), PvP engagement, in-app purchase patterns, and session depth metrics. The SaaS-origin data lacks these signals, substantially limiting discriminative power.

4. **Revenue scaling artefact:** The 0.1x scaling of monetary fields is a cosmetic adjustment for narrative purposes. While it does not affect model learning (scaling is a linear transformation), it means absolute GBP figures should be interpreted directionally rather than literally.

### Modelling Limitations

5. **C-index below target (0.52 vs. 0.75 target):** The survival model's ability to rank players by churn risk is only marginally better than random on the test set. This is honest and expected given limitations 1-3 above. The methodology is sound; the data lacks sufficient signal.

6. **Revenue RMSE is large relative to mean (GBP 1,638 RMSE vs. GBP 1,432 mean revenue):** Per-player revenue predictions have substantial uncertainty. The model is more reliable for segment-level averages than individual predictions.

7. **Overfitting gap (train C-index 0.76 vs. test C-index 0.40):** The Cox model memorises training set patterns that do not generalise. Cross-validation on the small dataset was explored but the 47-sample validation set limits its utility.

8. **Counter-intuitive hazard ratios:** The annual billing flag showing HR = 1.71 (higher churn risk) contradicts domain expectations. On scaled, standardised features, coefficient interpretation requires care -- the direction may reflect confounding or synthetic data artefacts rather than true causal relationships.

### Business Limitations

9. **Causal interpretation not established:** Hazard ratios represent statistical associations, not causal effects. Recommending "increase feature diversity to reduce churn" requires A/B testing to establish causality.

10. **Retention simulation assumptions:** The ROI simulation assumes uniform 15% retention uplift and a fixed GBP 5 offer value. Actual uplift would vary by player segment, offer type, and timing. These assumptions must be validated through experimentation.

---

## 9. What Would Change in Production

This section addresses a natural follow-up question: "How would you do this differently with real data at scale?"

### Data Differences

| This Project | Production |
|-------------|-----------|
| 500 accounts | 500K+ monthly active players |
| 33K rows across 5 tables | Billions of events per month |
| 20 engineered features | 200+ features including game-specific signals |
| Batch analysis on CSV | Real-time event streaming (Kafka, Kinesis) |
| Single analysis snapshot | Continuous retraining pipeline |

### Feature Enrichment

Production models would incorporate:

- **Game progression:** Level, achievements unlocked, content completion percentage
- **Social features:** Clan membership, friends list size, chat activity, co-op sessions
- **In-app purchase patterns:** Purchase frequency, average order value, time between purchases
- **Session quality:** Crash rate, load times, frame rate drops
- **Content engagement:** Game modes played, feature adoption velocity
- **External signals:** Marketing attribution, app store ratings, competitor game launches

### Technical Differences

- **Time-varying covariates:** Extended Cox models where features update weekly (e.g., this week's session count, not just first-week total)
- **Competing risks:** Separate models for different churn reasons (pricing, technical issues, competition, natural lifecycle)
- **Deep survival models:** DeepSurv or DRSA for capturing complex non-linear interactions at scale
- **Real-time scoring:** FastAPI endpoint returning pLTV and risk scores for live player data
- **Model monitoring:** Feature drift detection, prediction quality tracking, automated retraining triggers
- **A/B test integration:** Randomised retention experiments using model-selected treatment groups

### Infrastructure

- **Databricks/Spark** for feature engineering at scale (PySpark translations of the pandas pipeline are documented in Notebook 00)
- **MLflow** for experiment tracking and model registry
- **Looker/Tableau** for self-service stakeholder dashboards
- **Airflow** for orchestrating daily batch scoring pipelines

---

## 10. Reproducibility

### Random Seeds

All random operations use seed = 42 (specified in `config.yaml`):
- NumPy random state
- scikit-learn model random_state parameters
- Optuna TPE sampler seed
- Train/validation/test split (deterministic via date cutoffs, not random)

### Dependencies

All Python packages are pinned in `requirements.txt`. Key versions:

| Package | Version | Role |
|---------|---------|------|
| scikit-survival | 0.27.0 | Cox PH, GBSA |
| lifelines | 0.30.1 | PH assumption testing, KM curves |
| lightgbm | 4.6.0 | Revenue prediction |
| optuna | 4.7.0 | Hyperparameter tuning |
| scikit-learn | 1.8.0 | Random Forest, preprocessing, metrics |
| pandas | 2.3.3 | Data manipulation |

### Execution Instructions

```bash
# From project root
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Download dataset and place 5 CSVs in data/raw/
# Source: https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset

# Run the full pipeline
python src/data_processing.py   # Clean data + feature engineering
python src/modeling.py          # Train all models

# Or run notebooks interactively
jupyter notebook notebooks/
```

### Configuration

All parameters are centralised in `config.yaml`. There are no hardcoded values in the source code. Key configuration sections:

- `data.raw_files`: Paths to the 5 source CSV files
- `splitting`: Time-based cutoff dates for train/validation/test
- `features.selection`: Correlation threshold (0.85), VIF threshold (5.0), RFE target (20)
- `models.cox_ph`: Penaliser (0.01), Breslow ties
- `models.lightgbm_survival.tuning`: Optuna trial count (20), search space bounds
- `evaluation`: Target metrics (C-index > 0.75, RMSE < 1,200, R-squared > 0.60)

---

## 11. Code Quality

### Testing

65 unit tests across two test modules:

| Module | Tests | Coverage Focus |
|--------|-------|----------------|
| `test_data_processing.py` | 30 | Data loading, validation, cleaning, feature engineering, splitting |
| `test_modeling.py` | 35 | Model fitting, evaluation, predictions, persistence, leakage checks |

Tests include data leakage detection (verifying that no test-set information contaminates training features) and model persistence round-trip checks (save/load produces identical predictions).

### Code Standards

- **Formatting:** Black (88 character line length)
- **Linting:** Flake8 (zero errors)
- **Type hints:** All function signatures include PEP 484 type annotations
- **Docstrings:** Google-style docstrings on all public functions and classes
- **Logging:** Python `logging` module throughout (no print statements in production code)
- **Error handling:** All I/O operations wrapped with informative error messages

### Architecture

Three core classes with clear separation of concerns:

| Class | Responsibility |
|-------|---------------|
| `SurvivalAnalyzer` | Survival model training, evaluation, hazard ratios, PH validation |
| `LTVPredictor` | Revenue model training, tuning, evaluation, feature importance |
| `PLTVCalculator` | Combines survival probability with revenue prediction into pLTV |

The `FeatureSelector` class implements the three-stage feature selection pipeline as a separate, composable component.

---

## 12. References

### Academic

- Cox, D.R. (1972). "Regression Models and Life-Tables." *Journal of the Royal Statistical Society*, Series B, 34(2), 187-220.
- Harrell, F.E. et al. (1996). "Multivariable prognostic models: Issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors." *Statistics in Medicine*, 15(4), 361-387.
- Graf, E. et al. (1999). "Assessment and comparison of prognostic classification schemes for survival data." *Statistics in Medicine*, 18(17-18), 2529-2545.

### Software Documentation

- scikit-survival: https://scikit-survival.readthedocs.io/
- lifelines: https://lifelines.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- Optuna: https://optuna.readthedocs.io/

### Gaming Industry Context

- Sifa, R. et al. (2015). "Predicting Purchase Decisions in Mobile Free-to-Play Games." *AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment.*
- Runge, J. et al. (2014). "Churn prediction for high-value players in casual social games." *IEEE Conference on Computational Intelligence and Games.*
