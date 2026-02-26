# Data Dictionary: Player LTV Survival Analysis

Complete feature definitions, SaaS-to-gaming terminology mapping, and data quality documentation.

---

## 1. Overview

This document defines every feature used in the Player LTV Survival Analysis project. The underlying dataset is a SaaS subscription analytics dataset reframed as mobile gaming subscription data. This mapping is structurally valid: SaaS accounts map to player profiles, subscriptions to premium tiers (Battle Pass, VIP), feature usage to gameplay sessions, and support tickets to player support interactions.

**Dataset:** [SaaS Subscription & Churn Analytics Dataset](https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset) by Rivalytics (MIT license)

**Total records:** 33,100 rows across 5 relational tables for 500 player accounts

---

## 2. SaaS-to-Gaming Terminology Mapping

The following table defines the domain reframing applied throughout the project. The data structure is unchanged; only the business narrative and column labels are adapted.

| SaaS Term | Gaming Equivalent | Example | Notes |
|-----------|-------------------|---------|-------|
| account | player_account | Individual player profile | Primary key: `account_id` |
| subscription | premium_subscription | Battle Pass, VIP Gold, Season Pass | Multiple subscriptions per player |
| plan_tier: Basic | Free-to-Play | Ad-supported, limited features | Lowest tier |
| plan_tier: Pro | Premium | GBP 4.99/month Battle Pass | Mid tier |
| plan_tier: Enterprise | VIP | GBP 9.99/month VIP Gold | Highest tier |
| mrr_amount | monthly_revenue_per_player | GBP 4.99, GBP 9.99 | Scaled 0.1x from raw SaaS values |
| arr_amount | annual_revenue_per_player | GBP 59.88, GBP 119.88 | Scaled 0.1x from raw SaaS values |
| feature_usage | gameplay_session_events | Daily logins, PvP matches, feature engagement | 50 usage events per player on average |
| feature_name | game_feature | "pvp_arena", "daily_quest", "guild_chat" | 40 distinct features in the dataset |
| support_tickets | player_support_interactions | Bug reports, billing issues, account recovery | 4 tickets per player on average |
| churn_events | player_churn_with_exit_survey | Uninstalls with exit survey | 1.2 churn events per churned player |
| industry | game_genre | Casual, Strategy, RPG, Sports | Categorical; target-encoded |
| referral_source | acquisition_channel | organic, ads, partner, event, other | 5 channels in the dataset |
| seats | account_profiles | Number of profiles linked to account | Not used in final model |

---

## 3. Raw Data Tables

### 3.1 Accounts Table (`ravenstack_accounts.csv`)

500 rows. One row per player account. The definitive source for churn status.

| Column | Type | Description | Example Values | Notes |
|--------|------|-------------|----------------|-------|
| `account_id` | int | Unique player identifier (PK) | 1, 2, ..., 500 | Foreign key target for all other tables |
| `signup_date` | date | Account creation date | 2024-01-15 | Range: ~2024-01 to ~2024-11 |
| `plan_tier` | str | Current subscription tier | Basic, Pro, Enterprise | Mapped to Free-to-Play, Premium, VIP |
| `referral_source` | str | Acquisition channel | organic, ads, partner, event, other | 5 categories |
| `industry` | str | Self-reported game genre/interest | Technology, Healthcare, Finance, ... | Target-encoded in feature engineering |
| `country` | str | Player country | US, UK, CA, DE, ... | Grouped into top-5 + Other |
| `seats` | int | Account profiles | 1 - 50 | Not used in final model |
| `is_trial` | bool | Currently on trial subscription | True, False | Used to filter first-purchase timing |
| `churn_flag` | bool | **Definitive** current churn status | True (churned), False (active) | 22% overall churn rate |
| `signup_month` | int | Month extracted from signup_date | 1 - 12 | Derived during feature engineering |

### 3.2 Subscriptions Table (`ravenstack_subscriptions.csv`)

5,000 rows. Subscription lifecycle events (10 subscriptions per player on average).

| Column | Type | Description | Example Values | Notes |
|--------|------|-------------|----------------|-------|
| `subscription_id` | int | Unique subscription identifier (PK) | 1 - 5000 | |
| `account_id` | int | Player identifier (FK) | 1 - 500 | References accounts table |
| `start_date` | date | Subscription start | 2024-01-20 | Always >= signup_date |
| `end_date` | date/null | Subscription end | 2024-06-15 or null | Null = active (right-censored) |
| `plan_tier` | str | Tier for this subscription | Basic, Pro, Enterprise | May differ from account-level tier |
| `mrr_amount` | float | Monthly recurring revenue (GBP) | 22.68 (raw), 2.27 (scaled) | Scaled 0.1x during cleaning |
| `arr_amount` | float | Annual recurring revenue (GBP) | 272.16 (raw), 27.22 (scaled) | Scaled 0.1x during cleaning |
| `billing_frequency` | str | Billing cycle | monthly, annual | Annual flag is a model feature |
| `is_trial` | bool | Trial subscription | True, False | Filtered out for days_to_first_purchase |
| `churn_flag` | bool | Subscription-level churn | True, False | Historical; not definitive status |
| `upgrade_flag` | bool | This was an upgrade | True, False | Binary feature: has_upgraded_flag |
| `downgrade_flag` | bool | This was a downgrade | True, False | Binary feature: has_downgraded_flag |

### 3.3 Feature Usage Table (`ravenstack_feature_usage.csv`)

25,000 rows. Daily gameplay session events (50 events per player on average).

| Column | Type | Description | Example Values | Notes |
|--------|------|-------------|----------------|-------|
| `usage_id` | int | Usage event identifier | 1 - 25000 | 21 non-unique IDs (different data, not duplicates) |
| `subscription_id` | int | Subscription identifier (FK) | 1 - 5000 | Links to subscriptions, then to accounts |
| `usage_date` | date | Date of gameplay session | 2024-03-10 | Used for time-windowed features |
| `feature_name` | str | Game feature used | "dashboard", "reports", "api" | 40 distinct features; mapped to game features |
| `usage_count` | int | Number of uses in this session | 1 - 500 | Capped at 99th percentile during cleaning |
| `usage_duration_secs` | float | Session duration in seconds | 120.5, 3600.0 | Used for avg_session_duration_7d |
| `error_count` | int | Errors during session | 0, 1, 5 | Used for error_rate feature |
| `is_beta_feature` | bool | Beta/experimental feature | True, False | Used for beta_feature_adoption_flag |

### 3.4 Support Tickets Table (`ravenstack_support_tickets.csv`)

2,000 rows. Player support interactions (4 tickets per player on average).

| Column | Type | Description | Example Values | Notes |
|--------|------|-------------|----------------|-------|
| `ticket_id` | int | Ticket identifier (PK) | 1 - 2000 | |
| `account_id` | int | Player identifier (FK) | 1 - 500 | |
| `submitted_at` | datetime | Ticket submission time | 2024-04-15 14:30:00 | Used for time-windowed features |
| `closed_at` | datetime/null | Ticket resolution time | 2024-04-16 09:00:00 | Null if unresolved |
| `priority` | str | Ticket priority | low, medium, high, urgent | Urgent triggers has_urgent_ticket_flag |
| `satisfaction_score` | float/null | Player satisfaction (1-5) | 1.0, 3.0, 5.0, null | 41% missing; imputed with median 3.0 |
| `resolution_time_hours` | float | Hours to resolution | 2.5, 48.0, 168.0 | Capped at 95th percentile; flag for >168h |
| `first_response_time_minutes` | float | Minutes to first response | 15.0, 120.0 | Not used in final model |
| `category` | str | Issue category | billing, technical, account | Not used in final model |

### 3.5 Churn Events Table (`ravenstack_churn_events.csv`)

600 rows. Churn instances with exit survey data. Note: some accounts have multiple churn events (churned, reactivated, churned again).

| Column | Type | Description | Example Values | Notes |
|--------|------|-------------|----------------|-------|
| `churn_event_id` | int | Churn event identifier (PK) | 1 - 600 | |
| `account_id` | int | Player identifier (FK) | 1 - 500 | Not all accounts appear here |
| `churn_date` | date | Date player churned | 2024-05-20 | Used for survival duration calculation |
| `reason_code` | str | Churn reason | pricing, support, competitor, features | Not used as model feature |
| `refund_amount_usd` | float | Refund issued (GBP) | 0.0, 50.0 | Scaled 0.1x during cleaning |
| `is_reactivation` | bool | Was this a re-churn? | True, False | Reactivated accounts treated as censored |
| `feedback_text` | str/null | Exit survey text | "Too expensive", null | 25% missing; binary indicator created |
| `provided_feedback` | int | Provided exit feedback | 0, 1 | Derived: 1 if feedback_text is not null |

---

## 4. Engineered Features

These features are computed from the raw tables during the feature engineering pipeline (`src/data_processing.py`) and stored in `data/processed/master_features.csv`. All 20 features selected for modelling are listed below.

### 4.1 RFM Features

| Feature | Type | Range | Calculation | Business Rationale | Model Role |
|---------|------|-------|-------------|-------------------|------------|
| `days_since_signup` | int | 30 - 365 | `(analysis_date - signup_date).days` | Account maturity | Dropped by correlation filter (r > 0.85 with tenure) |
| `total_sessions_7d` | int | 0 - 50+ | Count of usage events in days 0-7 post-signup | Early engagement intensity | Selected (survival + LTV) |
| `avg_session_duration_7d` | float | 0 - 7200 | Mean usage_duration_secs in days 0-7 | Session depth | Dropped by RFE |
| `total_revenue_30d` | float | 0 - 500 | Sum mrr_amount for subscriptions in days 0-30 | Early monetisation | Dropped by RFE |
| `days_to_first_purchase` | int | 0 - 365 | Min subscription start_date - signup_date (non-trial) | Conversion speed | Selected (survival + LTV) |

### 4.2 Behavioural Features

| Feature | Type | Range | Calculation | Business Rationale | Model Role |
|---------|------|-------|-------------|-------------------|------------|
| `feature_diversity_score` | float | 0.0 - 1.0 | n_distinct_features / 40 | Product stickiness | Selected (survival + LTV) |
| `usage_consistency_cv` | float | 0.0 - 5.0+ | std(daily_usage) / mean(daily_usage) | Engagement stability | Selected (survival + LTV) |
| `weekend_usage_ratio` | float | 0.0 - 1.0 | weekend_sessions / total_sessions | Habitual vs. work-driven | Selected (survival + LTV) |
| `error_rate` | float | 0.0 - 1.0 | total_errors / total_usage_count | Technical quality | Selected (survival + LTV) |
| `beta_feature_adoption_flag` | int | 0 or 1 | 1 if any beta feature used | Innovation affinity | Dropped by RFE |

### 4.3 Support Features

| Feature | Type | Range | Calculation | Business Rationale | Model Role |
|---------|------|-------|-------------|-------------------|------------|
| `support_ticket_count_30d` | int | 0 - 10+ | Count tickets in days 0-30 post-signup | Early friction | Selected (survival) |
| `avg_satisfaction_score` | float | 1.0 - 5.0 | Mean satisfaction across all tickets (imputed) | Support quality feedback | Dropped by RFE |
| `has_urgent_ticket_flag` | int | 0 or 1 | 1 if any priority = "urgent" | Critical issue indicator | Selected (survival) |
| `fast_resolution_rate` | float | 0.0 - 1.0 | n_resolved_under_24h / n_total_tickets | Support responsiveness | Selected (survival) |

### 4.4 Subscription Features

| Feature | Type | Range | Calculation | Business Rationale | Model Role |
|---------|------|-------|-------------|-------------------|------------|
| `has_upgraded_flag` | int | 0 or 1 | 1 if any upgrade_flag = True | Satisfaction signal | Selected (survival) |
| `has_downgraded_flag` | int | 0 or 1 | 1 if any downgrade_flag = True | Price sensitivity | Dropped by RFE |
| `subscription_tenure_days` | int | 0 - 730 | max(end_date) - min(start_date) | Paying customer duration | Selected (survival + LTV) |
| `subscription_churn_count` | int | 0 - 5 | Count subscriptions with churn_flag = True | Historical churn propensity | Dropped by RFE |
| `billing_frequency_annual_flag` | int | 0 or 1 | 1 if any billing_frequency = "annual" | Commitment signal | Selected (survival) |

### 4.5 Time-Series Features

| Feature | Type | Range | Calculation | Business Rationale | Model Role |
|---------|------|-------|-------------|-------------------|------------|
| `usage_trend_slope_30d` | float | -10 to +10 | OLS slope of daily usage count, days 0-30 | Engagement trajectory | Selected (survival + LTV) |
| `revenue_trend_slope_60d` | float | -500 to +500 | OLS slope of monthly revenue, days 0-60 | Monetisation trajectory | Selected (survival + LTV) |
| `days_since_last_session` | int | 0 - 365 | analysis_date - max(usage_date) | Recency | Selected (survival + LTV) |

### 4.6 Encoded Categorical Features

| Feature | Type | Source | Encoding | Model Role |
|---------|------|--------|----------|------------|
| `plan_tier_Pro` | float | plan_tier | One-hot (drop_first=True) | Dropped by RFE |
| `plan_tier_Enterprise` | float | plan_tier | One-hot (drop_first=True) | Dropped by RFE |
| `referral_source_event` | float | referral_source | One-hot (drop_first=True) | Selected (survival) |
| `referral_source_organic` | float | referral_source | One-hot (drop_first=True) | Selected (survival) |
| `referral_source_other` | float | referral_source | One-hot (drop_first=True) | Selected (survival) |
| `referral_source_partner` | float | referral_source | One-hot (drop_first=True) | Dropped by RFE |

### 4.7 Indicator Features

| Feature | Type | Source | Notes | Model Role |
|---------|------|--------|-------|------------|
| `satisfaction_missing` | int | satisfaction_score | 1 if original value was null | Dropped by RFE |
| `provided_feedback` | int | feedback_text | 1 if churn exit feedback provided | Selected (survival) |

---

## 5. Target Variables

### Survival Targets

| Variable | Type | Description | Construction |
|----------|------|-------------|-------------|
| `event_observed` | int (0/1) | Churn indicator | From `accounts.churn_flag`. 1 = churned, 0 = active (censored). 22% event rate overall. |
| `duration_days` | int | Time to event or censoring | Churned: `churn_date - signup_date` in days. Active: `analysis_date (2024-12-31) - signup_date`. Minimum 1 day. |

### Revenue Target

| Variable | Type | Description | Construction |
|----------|------|-------------|-------------|
| `total_revenue_180d` | float | Actual 180-day revenue (GBP) | Sum of mrr_amount for all subscriptions with start_date within 180 days of signup. Scaled 0.1x. |

---

## 6. Feature Engineering Logic

### Key Calculations

**Coefficient of Variation (usage_consistency_cv):**

```python
daily_usage = usage.groupby(["account_id", "usage_date"])["usage_count"].sum()
cv_stats = daily_usage.groupby("account_id").agg(["mean", "std"])
cv_stats["usage_consistency_cv"] = cv_stats["std"] / cv_stats["mean"]
```

Higher CV indicates more erratic usage patterns, which is associated with disengagement and higher churn risk (HR = 1.26 in the Cox model).

**Usage Trend Slope (usage_trend_slope_30d):**

```python
from scipy.stats import linregress

def compute_slope(group, x_col, y_col):
    if len(group) < 2:
        return 0.0
    slope, _, _, _, _ = linregress(group[x_col], group[y_col])
    return slope

slopes = daily_30d.groupby("account_id").apply(
    compute_slope, x_col="days_since_signup_usage", y_col="usage_count"
)
```

Positive slope indicates increasing engagement over the first 30 days; negative slope signals declining interest.

**Feature Diversity Score:**

```python
n_total_features = usage["feature_name"].nunique()  # 40 distinct features
diversity = (
    usage.groupby("account_id")["feature_name"].nunique()
    / n_total_features
)
```

A score of 0.5 means the player has used 20 of 40 available features -- indicating broad exploration of the game's content.

**Survival Duration:**

```python
result["duration_days"] = np.where(
    (result["event_observed"] == 1) & result["churn_date"].notna(),
    (result["churn_date"] - result["signup_date"]).dt.days,  # Churned
    (analysis_date - result["signup_date"]).dt.days,          # Censored
)
result["duration_days"] = result["duration_days"].clip(lower=1)
```

---

## 7. Data Quality Notes

### Missing Values

| Column | Table | Missing % | Strategy | Indicator Created |
|--------|-------|-----------|----------|-------------------|
| `satisfaction_score` | support_tickets | 41% (825/2000) | Median imputation (3.0) | `satisfaction_missing` |
| `feedback_text` | churn_events | 25% (148/600) | Not imputed | `provided_feedback` |
| `end_date` | subscriptions | Active subs | Retained as null | None (semantically meaningful) |
| `closed_at` | support_tickets | Open tickets | Retained as null | None |

### Outlier Treatment

| Column | Detection | Treatment | Affected Rows |
|--------|-----------|-----------|---------------|
| `mrr_amount` | IQR method (> Q3 + 3xIQR) | Winsorised at 99th percentile | ~50 (1% of subscriptions) |
| `usage_count` | Threshold > 500 | Capped at 99th percentile | ~250 (1% of usage rows) |
| `resolution_time_hours` | Threshold > 720h (30 days) | Capped at 95th percentile; binary flag `extreme_resolution_time` for > 168h (1 week) | ~100 (5% of tickets) |

### Revenue Scaling

All monetary fields are scaled by 0.1x during the `clean_data()` step to bring SaaS-level revenues into a realistic gaming range:

| Field | Raw Mean | Scaled Mean | Gaming Context |
|-------|----------|-------------|----------------|
| `mrr_amount` | ~GBP 2,268 | ~GBP 227 | Monthly Battle Pass + VIP revenue |
| `arr_amount` | ~GBP 27,216 | ~GBP 2,722 | Annual subscription value |
| `refund_amount_usd` | Variable | 0.1x | Refunds on churn |

### Referential Integrity

All foreign key relationships are validated in `validate_data_quality()`:

| Check | Result |
|-------|--------|
| All subscription.account_id in accounts | PASS |
| All support_tickets.account_id in accounts | PASS |
| All churn_events.account_id in accounts | PASS |
| All feature_usage.subscription_id in subscriptions | PASS |
| subscription.start_date >= accounts.signup_date | PASS (after 0 violations) |

---

## 8. Feature Usage by Model

### Survival Models (Cox PH and GBSA)

Both survival models use the same 20 selected features. Cox PH produces interpretable hazard ratios; GBSA provides survival probability estimates for the pLTV calculation.

| Feature | Cox HR | Direction | GBSA Used |
|---------|--------|-----------|-----------|
| billing_frequency_annual_flag | 1.71 | Increases risk | Yes |
| total_sessions_7d | 1.27 | Increases risk | Yes |
| usage_consistency_cv | 1.26 | Increases risk | Yes |
| support_ticket_count_30d | 1.22 | Increases risk | Yes |
| referral_source_other | 1.20 | Increases risk | Yes |
| referral_source_event | 1.20 | Increases risk | Yes |
| fast_resolution_rate | 1.19 | Increases risk | Yes |
| signup_day_of_week | 0.91 | Decreases risk | Yes |
| referral_source_organic | 0.89 | Decreases risk | Yes |
| provided_feedback | 0.87 | Decreases risk | Yes |
| feature_diversity_score | 0.87 | Decreases risk | Yes |
| error_rate | 0.84 | Decreases risk | Yes |
| revenue_trend_slope_60d | 0.84 | Decreases risk | Yes |
| has_upgraded_flag | 0.81 | Decreases risk | Yes |
| weekend_usage_ratio | 0.80 | Decreases risk | Yes |
| has_urgent_ticket_flag | 0.78 | Decreases risk | Yes |
| days_to_first_purchase | 0.70 | Decreases risk | Yes |
| usage_trend_slope_30d | 0.68 | Decreases risk | Yes |
| days_since_last_session | 0.64 | Decreases risk | Yes |
| subscription_tenure_days | 0.49 | Decreases risk | Yes |

### LTV Models (Random Forest and LightGBM)

Both revenue models use the same 20 selected features with identical preprocessing. The Random Forest achieved the best test-set performance (RMSE GBP 1,638, R-squared 0.22).

| Feature | RF Importance | Rank |
|---------|--------------|------|
| subscription_tenure_days | 21.2% | 1 |
| revenue_trend_slope_60d | 19.1% | 2 |
| days_to_first_purchase | 11.3% | 3 |
| feature_diversity_score | 11.2% | 4 |
| weekend_usage_ratio | 7.7% | 5 |
| days_since_last_session | 7.3% | 6 |
| usage_consistency_cv | 5.9% | 7 |
| error_rate | 5.2% | 8 |
| usage_trend_slope_30d | 4.6% | 9 |
| signup_day_of_week | 1.8% | 10 |
| All remaining features | < 1.5% each | 11-20 |

---

## 9. Feature Importance Summary

Features that appear in the top 10 for both churn risk (Cox HR distance from 1.0) and revenue prediction (RF importance) are the most valuable for the combined pLTV framework:

| Feature | Churn Signal | Revenue Signal | Combined Value |
|---------|-------------|----------------|----------------|
| subscription_tenure_days | Strong (HR 0.49) | Strong (21.2%) | **Highest overall value** |
| usage_consistency_cv | Moderate (HR 1.26) | Moderate (5.9%) | High |
| feature_diversity_score | Moderate (HR 0.87) | Strong (11.2%) | High |
| days_to_first_purchase | Strong (HR 0.70) | Strong (11.3%) | High |
| usage_trend_slope_30d | Strong (HR 0.68) | Moderate (4.6%) | High |
| revenue_trend_slope_60d | Moderate (HR 0.84) | Strong (19.1%) | High |
| weekend_usage_ratio | Moderate (HR 0.80) | Moderate (7.7%) | Moderate |
| error_rate | Moderate (HR 0.84) | Moderate (5.2%) | Moderate |

These dual-signal features should be prioritised for instrumentation in production analytics pipelines, as they drive both components of the pLTV calculation.
