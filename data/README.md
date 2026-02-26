# Data Directory

## Dataset Attribution

**Dataset:** RavenStack Synthetic SaaS Dataset (Multi-Table)
**Author:** River @ Rivalytics
**Source:** [Kaggle - SaaS Subscription & Churn Analytics Dataset](https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset)
**License:** MIT (fully synthetic, no PII)
**Download Date:** 2026-02-08

> **Credit:** This dataset was created by River @ Rivalytics. It is used here
> under the MIT license for educational and portfolio purposes. Please credit
> the original author if reusing this data.

## Domain Reframing

This project reframes the SaaS subscription dataset as a **mobile gaming studio
subscription service** to align with gaming industry analytics roles. The
underlying data structure (accounts, subscriptions, feature usage, support
tickets, churn events) is structurally identical to gaming subscription services
such as Battle Pass, VIP tiers, and Season Passes.

| SaaS Term | Gaming Equivalent | Example |
|-----------|-------------------|---------|
| account | player_account | Individual player profile |
| subscription | premium_subscription | Battle Pass, VIP Gold, Season Pass |
| plan_tier (Basic/Pro/Enterprise) | subscription_tier (Free-to-Play/Premium/VIP) | Tier determines access level |
| mrr_amount | monthly_revenue_per_player | GBP 4.99 Battle Pass, GBP 9.99 VIP |
| feature_usage | gameplay_session_events | Daily logins, PvP matches, feature engagement |
| support_tickets | player_support_interactions | Bug reports, billing issues |
| churn_events | player_churn_events | Uninstalls with exit survey |

## Raw Data Files

The `raw/` directory should contain the following 5 CSV files:

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `ravenstack_accounts.csv` | 500 | 10 | Player accounts (account_id, signup_date, referral_source, plan_tier, churn_flag, industry, country) |
| `ravenstack_subscriptions.csv` | 5,000 | 14 | Subscription lifecycle (subscription_id, account_id, start_date, end_date, mrr_amount, plan_tier, churn_flag, billing_frequency) |
| `ravenstack_feature_usage.csv` | 25,000 | 8 | Daily gameplay events (usage_id, subscription_id, usage_date, feature_name, usage_count, usage_duration_secs, error_count) |
| `ravenstack_support_tickets.csv` | 2,000 | 9 | Player support tickets (ticket_id, account_id, submitted_at, priority, satisfaction_score, resolution_time_hours) |
| `ravenstack_churn_events.csv` | 600 | 9 | Churn instances (churn_event_id, account_id, churn_date, reason_code, refund_amount_usd, is_reactivation) |

**Total size:** ~2.1 MB (uncompressed)
**Total rows:** 33,100 across all tables

## Table Relationships

```
accounts (PK: account_id)
|
+-- subscriptions (FK -> accounts.account_id)
|   +-- feature_usage (FK -> subscriptions.subscription_id)
|
+-- support_tickets (FK -> accounts.account_id)
+-- churn_events (FK -> accounts.account_id)
```

All foreign key relationships are referentially complete (no orphaned records).

## Data Quality Notes

- **Completeness:** 95%+ complete across all tables
- **Strategic nulls:**
  - `satisfaction_score` in support_tickets: ~41% missing (825/2,000 non-response)
  - `feedback_text` in churn_events: ~25% missing (148/600)
  - `end_date` in subscriptions: null for active subscriptions (right-censored)
- **Duplicate keys:** 21 non-unique `usage_id` values in feature_usage (different data per row, not exact duplicates)
- **Revenue scaling:** Raw monetary values are scaled by 0.1x during cleaning to align with realistic gaming subscription price points
- **Referential integrity:** All foreign keys validated
- **Temporal validity:** signup_date <= subscription start_date <= churn_date
- **Distributions:** Exponential and Poisson distributions for realistic data
- **Fully synthetic:** No PII concerns

## Setup Instructions

Download the 5 raw CSV files and place them in `data/raw/`.

### Option 1: Kaggle CLI

```bash
# Requires a free Kaggle account and API credentials (~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download -d rivalytics/saas-subscription-and-churn-analytics-dataset -p data/raw/ --unzip
```

### Option 2: Manual Download

1. Visit: [SaaS Subscription & Churn Analytics Dataset](https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset) on Kaggle
2. Click **Download** (requires a free Kaggle account)
3. Unzip the archive and place the 5 CSV files in `data/raw/`

### Verification

After setup, verify the files are in place:

```bash
ls data/raw/
# Expected output:
# ravenstack_accounts.csv
# ravenstack_churn_events.csv
# ravenstack_feature_usage.csv
# ravenstack_subscriptions.csv
# ravenstack_support_tickets.csv
```

Quick row count check:

```bash
wc -l data/raw/*.csv
# Expected (approximate, including header rows):
#    501 ravenstack_accounts.csv
#   5001 ravenstack_subscriptions.csv
#  25001 ravenstack_feature_usage.csv
#   2001 ravenstack_support_tickets.csv
#    601 ravenstack_churn_events.csv
```

## Processed Data Files

The `processed/` directory contains the following committed files (generated by
`python src/data_processing.py`):

| File | Description |
|------|-------------|
| `master_features.csv` | Complete feature table (one row per account, all 20 engineered features) |
| `train.csv` | Training split — players signed up before 2024-07-01 (~60%) |
| `validation.csv` | Validation split — players signed up 2024-07-01 to 2024-09-01 (~20%) |
| `test.csv` | Test split — players signed up after 2024-09-01 (~20%) |

To regenerate these files from raw data:

```bash
python src/data_processing.py
```

## Gitignore Note

`data/raw/` is excluded from version control (gitignored) because the source CSVs
must be downloaded from Kaggle per the instructions above.

`data/processed/` **is committed** to the repository. The four processed files
(master_features.csv, train.csv, validation.csv, test.csv) are derived from
fully synthetic data under MIT license and are included so that notebooks and
scripts run immediately without needing to re-run the processing pipeline.
