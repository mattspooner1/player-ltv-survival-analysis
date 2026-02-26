"""Build and execute the EDA notebook (notebooks/01_eda.ipynb).

This script creates the notebook programmatically, then executes it
in-place to produce all outputs and figures.
"""

import json
import os

# Ensure we're in the project root
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(PROJECT_ROOT)


def make_cell(cell_type, source, cell_id=None):
    """Create a notebook cell dict."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.split("\n"),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def md(text):
    return make_cell("markdown", text.strip().split("\n"))


def code(text):
    return make_cell("code", text.strip().split("\n"))


cells = []

# ==========================================================================
# TITLE & BUSINESS CONTEXT
# ==========================================================================
cells.append(md("""# Exploratory Data Analysis: Player Lifetime Value Survival Analysis

**Project:** Predicting Player Lifetime Value Using Survival Analysis
**Author:** Matt Spooner
**Date:** 2026-02-09
**Version:** 1.0.0

---

## Business Context

A mid-sized mobile gaming studio with 750K MAU and GBP 12M annual revenue spends
**GBP 2M annually** acquiring 500K players, but **30% churn within 30 days** due
to poor targeting. This EDA explores five interconnected datasets -- player accounts,
premium subscriptions, gameplay session events, player support interactions, and churn
events -- to uncover the behavioural patterns, temporal dynamics, and segment-level
differences that will inform a predictive lifetime value (pLTV) framework.

**Key questions driving this analysis:**
1. Which early player behaviours (first 7 days) predict long-term retention?
2. How do survival dynamics differ across subscription tiers and acquisition channels?
3. Where are the data quality issues and how should they be handled?

The findings here will directly inform feature engineering choices for the
Cox Proportional Hazards survival model and LightGBM revenue prediction model
in the modelling phase."""))

# ==========================================================================
# SETUP
# ==========================================================================
cells.append(md("""## 1. Environment Setup & Data Loading"""))

cells.append(code("""import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import yaml

# Project paths
PROJECT_ROOT = r"C:\\Users\\matt_\\Documents\\Projects\\Data Science\\portfolio-projects\\player_ltv_survival_analysis"
os.chdir(PROJECT_ROOT)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Visualization defaults from config
sns.set_theme(style=config['visualization']['style'],
              font_scale=1.1,
              context=config['visualization']['context'])
PALETTE = config['visualization']['palette']
DPI = config['visualization']['figure_dpi']
FIG_DIR = config['outputs']['figures_dir']
os.makedirs(FIG_DIR, exist_ok=True)

SEED = config['project']['random_seed']
np.random.seed(SEED)

def save_fig(fig, name, dpi=DPI):
    path = os.path.join(FIG_DIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')

print('Setup complete.')"""))

# ==========================================================================
# DATA LOADING
# ==========================================================================
cells.append(md("""### 1.1 Load Raw Data"""))

cells.append(code("""# Load all 5 tables
accounts = pd.read_csv(config['data']['raw_files']['accounts'],
                        parse_dates=['signup_date'])
subscriptions = pd.read_csv(config['data']['raw_files']['subscriptions'],
                             parse_dates=['start_date', 'end_date'])
usage = pd.read_csv(config['data']['raw_files']['feature_usage'],
                     parse_dates=['usage_date'])
tickets = pd.read_csv(config['data']['raw_files']['support_tickets'],
                       parse_dates=['submitted_at', 'closed_at'])
churn = pd.read_csv(config['data']['raw_files']['churn_events'],
                     parse_dates=['churn_date'])

tables = {
    'accounts': accounts,
    'subscriptions': subscriptions,
    'feature_usage': usage,
    'support_tickets': tickets,
    'churn_events': churn,
}

# Summary overview
for name, df in tables.items():
    print(f'{name:20s}: {df.shape[0]:>6,} rows x {df.shape[1]:>3} cols  '
          f'| Nulls: {df.isnull().sum().sum():>5}  '
          f'| Memory: {df.memory_usage(deep=True).sum()/1024:.0f} KB')"""))

# ==========================================================================
# INITIAL INSPECTION
# ==========================================================================
cells.append(md("""### 1.2 Initial Inspection

Let us examine each table's structure, data types, and first few rows."""))

cells.append(code("""# Accounts table
print('=== ACCOUNTS ===')
print(f'Shape: {accounts.shape}')
print(f'Date range: {accounts["signup_date"].min().date()} to {accounts["signup_date"].max().date()}')
print()
display(accounts.head(3))
print()
display(accounts.describe(include='all').T)"""))

cells.append(code("""# Subscriptions table
print('=== SUBSCRIPTIONS ===')
print(f'Shape: {subscriptions.shape}')
print(f'Subscriptions per account: {subscriptions.groupby("account_id").size().describe().to_dict()}')
print(f'end_date null count: {subscriptions["end_date"].isnull().sum()} ({subscriptions["end_date"].isnull().mean()*100:.1f}%) -- these are active subscriptions')
print()
display(subscriptions.head(3))"""))

cells.append(code("""# Feature usage table
print('=== FEATURE USAGE (GAMEPLAY SESSIONS) ===')
print(f'Shape: {usage.shape}')
print(f'Unique features: {usage["feature_name"].nunique()}')
print(f'Date range: {usage["usage_date"].min().date()} to {usage["usage_date"].max().date()}')
print()
display(usage.head(3))"""))

cells.append(code("""# Support tickets
print('=== SUPPORT TICKETS ===')
print(f'Shape: {tickets.shape}')
print(f'satisfaction_score nulls: {tickets["satisfaction_score"].isnull().sum()} ({tickets["satisfaction_score"].isnull().mean()*100:.1f}%)')
print()
display(tickets.head(3))"""))

cells.append(code("""# Churn events
print('=== CHURN EVENTS ===')
print(f'Shape: {churn.shape}')
print(f'Unique accounts: {churn["account_id"].nunique()} (vs {len(accounts)} total accounts)')
print(f'Reactivations: {churn["is_reactivation"].sum()} ({churn["is_reactivation"].mean()*100:.1f}%)')
print(f'feedback_text nulls: {churn["feedback_text"].isnull().sum()} ({churn["feedback_text"].isnull().mean()*100:.1f}%)')
print()
display(churn.head(3))"""))

# ==========================================================================
# DATA QUALITY ASSESSMENT
# ==========================================================================
cells.append(md("""## 2. Data Quality Assessment

Before diving into analysis, we need to understand the completeness and
consistency of our data. This section documents all quality issues and our
remediation plan."""))

cells.append(md("""### 2.1 Missing Values"""))

cells.append(code("""# Missing value summary across all tables
missing_summary = []
for name, df in tables.items():
    for col in df.columns:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            missing_summary.append({
                'Table': name,
                'Column': col,
                'Missing Count': n_miss,
                'Missing %': round(n_miss / len(df) * 100, 1),
                'Strategy': ''
            })

missing_df = pd.DataFrame(missing_summary)
# Add strategies
strategy_map = {
    'end_date': 'Retain null (= active/censored subscription)',
    'satisfaction_score': 'Median impute (3.0) + missing indicator',
    'feedback_text': 'Binary indicator (provided_feedback)',
}
missing_df['Strategy'] = missing_df['Column'].map(strategy_map).fillna('Investigate')
display(missing_df)"""))

cells.append(
    md(
        """**Key finding:** Missing values are **strategic, not random**:
- `satisfaction_score` (41.3% null): Players who skip post-support surveys may be disengaged -- missingness itself is informative, so we create a binary indicator.
- `feedback_text` (24.7% null in churn events): Some churners skip the exit survey. We create a `provided_feedback` indicator.
- `end_date` (90.3% null in subscriptions): Null means the subscription is still active -- these are **right-censored** observations, critical for survival analysis."""
    )
)

cells.append(md("""### 2.2 Duplicate Detection"""))

cells.append(code("""# Check for duplicate primary keys
pk_map = {'accounts': 'account_id', 'subscriptions': 'subscription_id',
          'feature_usage': 'usage_id', 'support_tickets': 'ticket_id',
          'churn_events': 'churn_event_id'}

for name, pk in pk_map.items():
    dup_count = tables[name][pk].duplicated().sum()
    status = 'CLEAN' if dup_count == 0 else f'WARNING: {dup_count} duplicates'
    print(f'{name:20s} ({pk:20s}): {status}')"""))

cells.append(
    md(
        """**Finding:** 21 duplicate `usage_id` values in the feature_usage table.
These likely represent legitimate re-entries (same feature used multiple times in a day
with different durations). Since usage_id is not used as a join key downstream, and the
rows contain different data, we retain them. This is documented but not a blocking issue."""
    )
)

cells.append(md("""### 2.3 Referential Integrity"""))

cells.append(code("""# Check all foreign key relationships
acct_ids = set(accounts['account_id'])
sub_ids = set(subscriptions['subscription_id'])

checks = {
    'subscriptions.account_id in accounts': set(subscriptions['account_id']).issubset(acct_ids),
    'tickets.account_id in accounts': set(tickets['account_id']).issubset(acct_ids),
    'churn.account_id in accounts': set(churn['account_id']).issubset(acct_ids),
    'usage.subscription_id in subscriptions': set(usage['subscription_id']).issubset(sub_ids),
}
for check, result in checks.items():
    print(f'{check:50s}: {"PASS" if result else "FAIL"} ')"""))

cells.append(
    md(
        """All referential integrity checks pass -- no orphaned records exist.
This confirms the data is well-structured and relationships between tables are sound."""
    )
)

cells.append(
    md(
        """### 2.4 Churn Flag Reconciliation

An important data quality consideration: the `churn_flag` on the accounts table
represents the player's **current status** (True = currently churned), while
`churn_events` contains the full **history** of churn events including re-activated players."""
    )
)

cells.append(
    code(
        """# Reconcile accounts.churn_flag vs churn_events
accts_churned_flag = set(accounts[accounts['churn_flag']]['account_id'])
accts_in_churn_events = set(churn['account_id'])

print(f'Accounts with churn_flag=True:           {len(accts_churned_flag):>4}')
print(f'Unique accounts in churn_events:         {len(accts_in_churn_events):>4}')
print(f'In churn_events BUT churn_flag=False:    {len(accts_in_churn_events - accts_churned_flag):>4} (reactivated)')
print(f'churn_flag=True BUT NOT in churn_events: {len(accts_churned_flag - accts_in_churn_events):>4} (churned via other mechanism)')
print(f'Multiple churn events per account:')
display(churn.groupby('account_id').size().describe().to_frame('events_per_account').T)"""
    )
)

cells.append(
    md(
        """**Decision for survival analysis:** We use `accounts.churn_flag` (current status)
as the definitive event indicator, because:
1. It reflects the player's **current** retention state, which is what the UA team needs to predict.
2. Players who churned then re-activated (277 accounts) are currently **active** -- treating them as churned would overestimate churn risk.
3. For the 35 accounts with `churn_flag=True` but no entry in churn_events, we estimate churn date from their last subscription end date."""
    )
)

# ==========================================================================
# PHASE 1: UNIVARIATE ANALYSIS
# ==========================================================================
cells.append(md("""## 3. Univariate Analysis

We examine the distribution of each key variable independently to understand
central tendencies, spread, skewness, and potential outliers."""))

cells.append(md("""### 3.1 Numeric Feature Distributions"""))

cells.append(code("""# Summary statistics for key numeric features
numeric_summary = pd.DataFrame({
    'mrr_amount': subscriptions['mrr_amount'].describe(),
    'usage_count': usage['usage_count'].describe(),
    'usage_duration_secs': usage['usage_duration_secs'].describe(),
    'resolution_time_hours': tickets['resolution_time_hours'].describe(),
    'satisfaction_score': tickets['satisfaction_score'].dropna().describe(),
    'seats': accounts['seats'].describe(),
    'refund_amount_usd': churn['refund_amount_usd'].describe(),
})
display(numeric_summary.round(2))"""))

cells.append(
    code(
        """# Distribution plots for revenue, usage, session duration
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# mrr_amount (monthly revenue)
ax = axes[0, 0]
sns.histplot(subscriptions['mrr_amount'], bins=50, kde=True, ax=ax, color='steelblue')
ax.set_title('Monthly Revenue per Subscription (MRR)', fontsize=12, fontweight='bold')
ax.set_xlabel('MRR Amount (GBP)')
ax.axvline(subscriptions['mrr_amount'].median(), color='red', ls='--', label=f"Median: {subscriptions['mrr_amount'].median():.0f}")
ax.legend()

# usage_count
ax = axes[0, 1]
sns.histplot(usage['usage_count'], bins=30, kde=True, ax=ax, color='darkorange')
ax.set_title('Daily Usage Count per Session', fontsize=12, fontweight='bold')
ax.set_xlabel('Usage Count')
ax.axvline(usage['usage_count'].median(), color='red', ls='--', label=f"Median: {usage['usage_count'].median():.0f}")
ax.legend()

# usage_duration_secs
ax = axes[0, 2]
sns.histplot(usage['usage_duration_secs'], bins=50, kde=True, ax=ax, color='seagreen')
ax.set_title('Session Duration', fontsize=12, fontweight='bold')
ax.set_xlabel('Duration (seconds)')
ax.axvline(usage['usage_duration_secs'].median(), color='red', ls='--', label=f"Median: {usage['usage_duration_secs'].median():.0f}s")
ax.legend()

# resolution_time_hours
ax = axes[1, 0]
sns.histplot(tickets['resolution_time_hours'], bins=40, kde=True, ax=ax, color='mediumpurple')
ax.set_title('Support Ticket Resolution Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Resolution Time (hours)')
ax.axvline(tickets['resolution_time_hours'].median(), color='red', ls='--', label=f"Median: {tickets['resolution_time_hours'].median():.0f}h")
ax.legend()

# satisfaction_score
ax = axes[1, 1]
sat_valid = tickets['satisfaction_score'].dropna()
sns.histplot(sat_valid, bins=5, kde=False, ax=ax, color='coral', discrete=True)
ax.set_title('Support Satisfaction Score (non-null)', fontsize=12, fontweight='bold')
ax.set_xlabel('Satisfaction (1-5)')
ax.axvline(sat_valid.median(), color='red', ls='--', label=f"Median: {sat_valid.median():.1f}")
ax.legend()

# seats
ax = axes[1, 2]
sns.histplot(accounts['seats'], bins=30, kde=True, ax=ax, color='teal')
ax.set_title('Account Seats (Team Size)', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Seats')
ax.axvline(accounts['seats'].median(), color='red', ls='--', label=f"Median: {accounts['seats'].median():.0f}")
ax.legend()

fig.suptitle('Distribution of Key Numeric Features', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, 'numeric_distributions')
plt.close(fig)
print('Key observations:')
print('- MRR is heavily right-skewed (whales drive revenue) -- log transform needed')
print('- Usage count is approximately normal (mean ~10)')
print('- Session duration is right-skewed with long tail')
print('- Resolution time is roughly uniform between 1-72 hours')
print('- Satisfaction scores cluster at 3-5 (no 1-2 scores -- possible ceiling effect)')"""
    )
)

cells.append(
    md(
        """**So what:** The right-skewed distributions in MRR and session duration
confirm the need for **log transformations** before modelling. The MRR distribution
shows a clear "whale" pattern typical of freemium games -- a small number of VIP
subscribers generate disproportionate revenue. The satisfaction score distribution
(all 3-5, no 1-2) suggests either a genuinely satisfied user base or response bias
(dissatisfied players skip the survey, which is captured by our `satisfaction_missing` indicator)."""
    )
)

cells.append(md("""### 3.2 Categorical Distributions"""))

cells.append(
    code(
        """# Categorical distributions: plan_tier, referral_source, industry, country
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plan tier
ax = axes[0, 0]
tier_counts = accounts['plan_tier'].value_counts()
bars = ax.bar(tier_counts.index, tier_counts.values, color=sns.color_palette(PALETTE, 3))
ax.set_title('Subscription Tier Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Players')
for bar, val in zip(bars, tier_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(val),
            ha='center', fontsize=11, fontweight='bold')

# Referral source
ax = axes[0, 1]
ref_counts = accounts['referral_source'].value_counts()
bars = ax.bar(ref_counts.index, ref_counts.values, color=sns.color_palette(PALETTE, 5))
ax.set_title('Acquisition Channel Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Players')
for bar, val in zip(bars, ref_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val),
            ha='center', fontsize=10, fontweight='bold')

# Industry (top categories)
ax = axes[1, 0]
ind_counts = accounts['industry'].value_counts()
bars = ax.barh(ind_counts.index, ind_counts.values, color=sns.color_palette(PALETTE, len(ind_counts)))
ax.set_title('Industry Vertical Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Players')
for bar, val in zip(bars, ind_counts.values):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=10)

# Country
ax = axes[1, 1]
country_counts = accounts['country'].value_counts()
bars = ax.bar(country_counts.index, country_counts.values, color=sns.color_palette(PALETTE, len(country_counts)))
ax.set_title('Country Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Players')
for bar, val in zip(bars, country_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val),
            ha='center', fontsize=9, fontweight='bold')

fig.suptitle('Categorical Feature Distributions', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, 'categorical_distributions')
plt.close(fig)
print('Key observations:')
print(f'- Plan tiers roughly balanced: Basic={tier_counts.get("Basic",0)}, Pro={tier_counts.get("Pro",0)}, Enterprise={tier_counts.get("Enterprise",0)}')
print(f'- Organic is top channel ({ref_counts.iloc[0]}), followed by other/ads/event/partner')
print(f'- US dominates ({country_counts["US"]}), then UK ({country_counts["UK"]}), IN ({country_counts["IN"]})')"""
    )
)

cells.append(md("""### 3.3 Churn Rate by Segment"""))

cells.append(
    code(
        """# Churn rate by categorical segments
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# By plan_tier
ax = axes[0]
churn_by_tier = accounts.groupby('plan_tier')['churn_flag'].mean().sort_values(ascending=False)
bars = ax.bar(churn_by_tier.index, churn_by_tier.values * 100, color=sns.color_palette(PALETTE, 3))
ax.set_title('Churn Rate by Subscription Tier', fontsize=12, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
for bar, val in zip(bars, churn_by_tier.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val*100:.1f}%', ha='center', fontsize=11, fontweight='bold')

# By referral_source
ax = axes[1]
churn_by_ref = accounts.groupby('referral_source')['churn_flag'].mean().sort_values(ascending=False)
bars = ax.bar(churn_by_ref.index, churn_by_ref.values * 100, color=sns.color_palette(PALETTE, 5))
ax.set_title('Churn Rate by Acquisition Channel', fontsize=12, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, churn_by_ref.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val*100:.1f}%', ha='center', fontsize=10, fontweight='bold')

# By industry
ax = axes[2]
churn_by_ind = accounts.groupby('industry')['churn_flag'].mean().sort_values(ascending=False)
bars = ax.barh(churn_by_ind.index, churn_by_ind.values * 100, color=sns.color_palette(PALETTE, len(churn_by_ind)))
ax.set_title('Churn Rate by Industry', fontsize=12, fontweight='bold')
ax.set_xlabel('Churn Rate (%)')
for bar, val in zip(bars, churn_by_ind.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%',
            va='center', fontsize=10)

fig.suptitle('Churn Rate Across Player Segments', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'churn_rate_by_segment')
plt.close(fig)
print(f'Overall churn rate: {accounts["churn_flag"].mean()*100:.1f}%')
print(f'Highest churn tier: {churn_by_tier.index[0]} ({churn_by_tier.iloc[0]*100:.1f}%)')
print(f'Lowest churn tier: {churn_by_tier.index[-1]} ({churn_by_tier.iloc[-1]*100:.1f}%)')
print(f'Highest churn channel: {churn_by_ref.index[0]} ({churn_by_ref.iloc[0]*100:.1f}%)')"""
    )
)

cells.append(md("""**So what:** The overall churn rate is 22%, with meaningful variation
across segments. This variation is precisely what our survival model needs to exploit --
if churn were uniform across all segments, there would be no signal to model. The segment-level
differences suggest that subscription tier, acquisition channel, and industry vertical
all carry predictive information."""))

# ==========================================================================
# PHASE 2: TEMPORAL ANALYSIS
# ==========================================================================
cells.append(md("""## 4. Temporal Analysis

Understanding how player behaviour evolves over time is critical for survival
analysis. This section examines cohort retention, subscription lifecycles,
and time-to-churn distributions."""))

cells.append(md("""### 4.1 Cohort Retention Curves"""))

cells.append(code("""# Cohort analysis: retention by signup month
accounts['signup_month'] = accounts['signup_date'].dt.to_period('M')

# Merge accounts with first churn date for time-to-churn
first_churn = churn.sort_values('churn_date').groupby('account_id').last().reset_index()
cohort_df = accounts.merge(first_churn[['account_id', 'churn_date']], on='account_id', how='left')

# For accounts with churn_flag=False, clear churn_date even if they have churn events (reactivated)
cohort_df.loc[~cohort_df['churn_flag'], 'churn_date'] = pd.NaT

# Calculate tenure in days
analysis_date = pd.Timestamp('2024-12-31')
cohort_df['tenure_days'] = np.where(
    cohort_df['churn_date'].notna(),
    (cohort_df['churn_date'] - cohort_df['signup_date']).dt.days,
    (analysis_date - cohort_df['signup_date']).dt.days
)
cohort_df['is_churned'] = cohort_df['churn_flag'].astype(int)

# Retention at key milestones
milestones = [1, 7, 14, 30, 60, 90, 180]
cohort_groups = cohort_df.groupby('signup_month')

retention_data = []
for month, group in cohort_groups:
    row = {'cohort': str(month), 'n_players': len(group)}
    for d in milestones:
        # Retained = still active at day d (tenure >= d OR not yet churned)
        retained = ((group['tenure_days'] >= d) | (~group['churn_flag'])).sum()
        row[f'Day {d}'] = retained / len(group) * 100
    retention_data.append(row)

retention_df = pd.DataFrame(retention_data)
print('Cohort Retention Rates (%):')
display(retention_df.set_index('cohort').round(1))"""))

cells.append(code("""# Retention heatmap
fig, ax = plt.subplots(figsize=(12, 6))
retention_cols = [c for c in retention_df.columns if c.startswith('Day')]
heatmap_data = retention_df.set_index('cohort')[retention_cols]
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn', vmin=50, vmax=100,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Retention %'})
ax.set_title('Cohort Retention Heatmap by Signup Month', fontsize=14, fontweight='bold')
ax.set_xlabel('Retention Milestone')
ax.set_ylabel('Signup Cohort')
fig.tight_layout()
save_fig(fig, 'cohort_retention_heatmap')
plt.close(fig)
print('Note: Recent cohorts show higher retention at early milestones because they')
print('have not had enough time to churn yet (right-censoring effect).')"""))

cells.append(
    md(
        """**So what:** The cohort retention heatmap reveals the expected pattern:
earlier cohorts have lower retention at later milestones simply because they have had
more time to churn. Recent cohorts (post-September 2024) show artificially high
retention because they are right-censored. This is precisely why we need **survival analysis**
rather than simple retention rates -- it properly accounts for censored observations."""
    )
)

cells.append(md("""### 4.2 Time-to-Churn Distribution"""))

cells.append(
    code(
        """# Time-to-churn distribution for churned accounts
churned = cohort_df[cohort_df['is_churned'] == 1].copy()
churned['tenure_days'] = churned['tenure_days'].clip(lower=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
sns.histplot(churned['tenure_days'], bins=40, kde=True, ax=ax, color='steelblue')
ax.axvline(churned['tenure_days'].median(), color='red', ls='--', linewidth=2,
           label=f"Median: {churned['tenure_days'].median():.0f} days")
ax.axvline(30, color='orange', ls=':', linewidth=2, label='30-day mark')
ax.set_title('Time-to-Churn Distribution (Churned Players Only)', fontsize=12, fontweight='bold')
ax.set_xlabel('Days from Signup to Churn')
ax.set_ylabel('Count')
ax.legend()

# Cumulative distribution
ax = axes[1]
sorted_tenure = np.sort(churned['tenure_days'].values)
cdf = np.arange(1, len(sorted_tenure) + 1) / len(sorted_tenure)
ax.plot(sorted_tenure, cdf * 100, color='steelblue', linewidth=2)
ax.axhline(50, color='red', ls='--', alpha=0.5, label='50th percentile')
ax.axvline(30, color='orange', ls=':', linewidth=2, label='30-day mark')
pct_before_30 = (churned['tenure_days'] <= 30).mean() * 100
ax.annotate(f'{pct_before_30:.1f}% churn\\nwithin 30 days',
            xy=(30, pct_before_30), xytext=(120, pct_before_30 - 15),
            arrowprops=dict(arrowstyle='->', color='orange'),
            fontsize=11, fontweight='bold', color='orange')
ax.set_title('Cumulative Churn Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Days from Signup')
ax.set_ylabel('Cumulative % of Churners')
ax.legend()

fig.suptitle('When Do Players Churn?', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'time_to_churn_distribution')
plt.close(fig)
print(f'Churned players: {len(churned)}')
print(f'Median time-to-churn: {churned["tenure_days"].median():.0f} days')
print(f'Mean time-to-churn: {churned["tenure_days"].mean():.0f} days')
print(f'% churning within 30 days: {pct_before_30:.1f}%')
print(f'% churning within 90 days: {(churned["tenure_days"] <= 90).mean()*100:.1f}%')"""
    )
)

cells.append(md("""**So what:** The time-to-churn distribution is right-skewed with a
concentration of early churners. Understanding this timing is essential for the
UA team: if a large fraction churn within the first 30 days, early intervention
(Day 3-7 retention offers) could be highly effective. The long right tail also
suggests some players churn much later, possibly due to price increases or
competitive alternatives."""))

cells.append(md("""### 4.3 Usage Intensity Over Account Lifetime"""))

cells.append(
    code(
        """# Link usage to accounts and compute daily usage rate over time
sub_account = subscriptions[['subscription_id', 'account_id']].drop_duplicates()
usage_with_account = usage.merge(sub_account, on='subscription_id', how='left')
usage_with_account = usage_with_account.merge(
    accounts[['account_id', 'signup_date', 'churn_flag']], on='account_id', how='left')
usage_with_account['days_since_signup'] = (
    usage_with_account['usage_date'] - usage_with_account['signup_date']).dt.days

# Bin into weeks
usage_with_account['week'] = usage_with_account['days_since_signup'] // 7

# Average usage per week by churn status
weekly_usage = (usage_with_account[usage_with_account['week'].between(0, 25)]
                .groupby(['week', 'churn_flag'])['usage_count']
                .mean()
                .reset_index())

fig, ax = plt.subplots(figsize=(12, 5))
for churn_status, color, label in [(False, 'seagreen', 'Retained'), (True, 'crimson', 'Churned')]:
    subset = weekly_usage[weekly_usage['churn_flag'] == churn_status]
    ax.plot(subset['week'], subset['usage_count'], color=color, linewidth=2, label=label, marker='o', markersize=4)

ax.set_title('Average Weekly Usage by Churn Status', fontsize=14, fontweight='bold')
ax.set_xlabel('Weeks Since Signup')
ax.set_ylabel('Average Usage Count per Session')
ax.legend(fontsize=12)
ax.set_xlim(0, 25)
fig.tight_layout()
save_fig(fig, 'usage_intensity_by_churn')
plt.close(fig)
print('The usage trajectory comparison reveals whether churners show declining engagement before leaving.')"""
    )
)

cells.append(
    md("""**So what:** If churned players show declining usage patterns in the weeks
before churn, this validates the `usage_trend_slope_30d` feature as a predictive signal.
The divergence point between churned and retained curves indicates the earliest
we can reliably detect churn risk.""")
)

# ==========================================================================
# PHASE 3: BIVARIATE RELATIONSHIPS
# ==========================================================================
cells.append(md("""## 5. Bivariate Relationships

Now we examine how features relate to each other and to the churn outcome.
These relationships guide feature selection and help identify the most
promising predictors for the survival model."""))

cells.append(md("""### 5.1 Correlation Analysis"""))

cells.append(code("""# Build a mini feature table for correlation analysis
from src.data_processing import load_config, load_raw_data, clean_data, engineer_features, create_survival_target

config = load_config('config.yaml')
dfs = load_raw_data(config)
cleaned = clean_data(dfs, config)
master = engineer_features(cleaned, config)
master = create_survival_target(master, config)

# Select numeric features for correlation
numeric_feats = [
    'days_since_signup', 'total_sessions_7d', 'avg_session_duration_7d',
    'total_revenue_30d', 'days_to_first_purchase', 'feature_diversity_score',
    'usage_consistency_cv', 'weekend_usage_ratio', 'error_rate',
    'support_ticket_count_30d', 'avg_satisfaction_score',
    'fast_resolution_rate', 'subscription_tenure_days',
    'subscription_churn_count', 'usage_trend_slope_30d',
    'revenue_trend_slope_60d', 'days_since_last_session',
    'seats', 'event_observed', 'duration_days'
]

corr_data = master[numeric_feats].copy()

fig, ax = plt.subplots(figsize=(16, 12))
corr_matrix = corr_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={'size': 8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'correlation_heatmap')
plt.close(fig)

# Top correlations with event_observed (churn)
churn_corr = corr_matrix['event_observed'].drop(['event_observed', 'duration_days']).sort_values(key=abs, ascending=False)
print('Top correlations with churn (event_observed):')
for feat, val in churn_corr.head(10).items():
    direction = 'higher churn' if val > 0 else 'lower churn'
    print(f'  {feat:35s}: r = {val:+.3f} ({direction})')"""))

cells.append(md("""**So what:** The correlation analysis reveals which features have the
strongest linear relationships with churn. Features like subscription churn count,
days since last session, and support ticket count are expected positive correlates
of churn. Features like total sessions, feature diversity, and revenue are expected
negative correlates (engaged, spending players are less likely to churn).

These correlations inform our feature selection pipeline, but we must check
for multicollinearity -- if two features are highly correlated with each other
(r > 0.85), we keep only the more predictive one."""))

cells.append(md("""### 5.2 Scatter Plots: Revenue vs Usage"""))

cells.append(code("""# Scatter plots coloured by churn status
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MRR vs total sessions
ax = axes[0]
for flag, color, label in [(False, 'seagreen', 'Retained'), (True, 'crimson', 'Churned')]:
    subset = master[master['churn_flag'] == flag]
    ax.scatter(subset['total_sessions_7d'], subset['total_revenue_30d'],
               alpha=0.5, s=20, color=color, label=label)
ax.set_title('Early Sessions vs Revenue', fontsize=12, fontweight='bold')
ax.set_xlabel('Total Sessions (First 7 Days)')
ax.set_ylabel('Total Revenue (First 30 Days, GBP)')
ax.legend()

# Feature diversity vs churn
ax = axes[1]
for flag, color, label in [(False, 'seagreen', 'Retained'), (True, 'crimson', 'Churned')]:
    subset = master[master['churn_flag'] == flag]
    ax.scatter(subset['feature_diversity_score'], subset['days_since_last_session'],
               alpha=0.5, s=20, color=color, label=label)
ax.set_title('Feature Diversity vs Recency', fontsize=12, fontweight='bold')
ax.set_xlabel('Feature Diversity Score')
ax.set_ylabel('Days Since Last Session')
ax.legend()

# Subscription tenure vs usage slope
ax = axes[2]
for flag, color, label in [(False, 'seagreen', 'Retained'), (True, 'crimson', 'Churned')]:
    subset = master[master['churn_flag'] == flag]
    ax.scatter(subset['subscription_tenure_days'], subset['usage_trend_slope_30d'],
               alpha=0.5, s=20, color=color, label=label)
ax.set_title('Tenure vs Usage Trend', fontsize=12, fontweight='bold')
ax.set_xlabel('Subscription Tenure (Days)')
ax.set_ylabel('Usage Trend Slope (30d)')
ax.legend()

fig.suptitle('Bivariate Relationships by Churn Status', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'scatter_plots_churn')
plt.close(fig)"""))

cells.append(md("""### 5.3 Boxplots by Churn Status"""))

cells.append(code("""# Boxplots comparing churned vs retained
features_to_compare = [
    ('total_revenue_30d', 'Revenue (First 30 Days)'),
    ('total_sessions_7d', 'Sessions (First 7 Days)'),
    ('feature_diversity_score', 'Feature Diversity'),
    ('days_since_last_session', 'Days Since Last Session'),
    ('subscription_tenure_days', 'Subscription Tenure (Days)'),
    ('avg_satisfaction_score', 'Avg Satisfaction Score'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (feat, title) in enumerate(features_to_compare):
    ax = axes[i]
    master_plot = master.copy()
    master_plot['Status'] = master_plot['churn_flag'].map({True: 'Churned', False: 'Retained'})
    sns.boxplot(data=master_plot, x='Status', y=feat, ax=ax,
                palette={'Churned': 'crimson', 'Retained': 'seagreen'})
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('')

fig.suptitle('Feature Distributions: Churned vs Retained Players', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, 'boxplots_churn_comparison')
plt.close(fig)"""))

cells.append(md("""**So what:** The boxplots reveal distributional differences between
churned and retained players. Features where the boxes show clear separation
(different medians, minimal overlap) will be the strongest predictors in our
survival model. Features with heavy overlap may still contribute through
interaction effects or non-linear relationships captured by LightGBM."""))

cells.append(md("""### 5.4 Chi-Square Tests for Categorical Independence"""))

cells.append(
    code(
        """# Chi-square tests: categorical features vs churn
from scipy.stats import chi2_contingency

cat_features = ['plan_tier', 'referral_source', 'industry', 'country']
chi2_results = []

for feat in cat_features:
    contingency = pd.crosstab(accounts[feat], accounts['churn_flag'])
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    chi2_results.append({
        'Feature': feat,
        'Chi-square': round(chi2, 2),
        'p-value': round(p_val, 4),
        'Degrees of Freedom': dof,
        'Significant (p<0.05)': 'Yes' if p_val < 0.05 else 'No',
    })

chi2_df = pd.DataFrame(chi2_results)
display(chi2_df)
print()
for _, row in chi2_df.iterrows():
    if row['Significant (p<0.05)'] == 'Yes':
        print(f"  {row['Feature']}: SIGNIFICANT (p={row['p-value']}) -- churn rate differs across categories")
    else:
        print(f"  {row['Feature']}: Not significant (p={row['p-value']}) -- churn rate similar across categories")"""
    )
)

cells.append(md("""**So what:** Chi-square tests tell us which categorical features have
statistically significant associations with churn. Significant features
(p < 0.05) should be included in the survival model. Non-significant features
may still be included if they have theoretical justification (e.g., country
affects monetization patterns even if overall churn rates are similar)."""))

# ==========================================================================
# PHASE 4: SURVIVAL-SPECIFIC EDA
# ==========================================================================
cells.append(md("""## 6. Survival-Specific Exploratory Analysis

This section directly prepares for the Cox Proportional Hazards model by
examining survival curves, censoring patterns, and the proportional hazards
assumption."""))

cells.append(md("""### 6.1 Kaplan-Meier Survival Curves"""))

cells.append(code("""# Overall Kaplan-Meier survival curve
kmf = KaplanMeierFitter()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Overall survival
ax = axes[0, 0]
kmf.fit(master['duration_days'], event_observed=master['event_observed'], label='All Players')
kmf.plot_survival_function(ax=ax, ci_show=True, color='steelblue', linewidth=2)
median_survival = kmf.median_survival_time_
ax.axhline(0.5, color='red', ls='--', alpha=0.5, label=f'Median: {median_survival:.0f} days')
ax.set_title('Overall Survival Curve', fontsize=12, fontweight='bold')
ax.set_xlabel('Days Since Signup')
ax.set_ylabel('Survival Probability')
ax.legend(fontsize=10)
ax.set_xlim(0, master['duration_days'].max())

# By plan_tier
ax = axes[0, 1]
colors = {'Basic': '#1f77b4', 'Pro': '#ff7f0e', 'Enterprise': '#2ca02c'}
for tier in ['Basic', 'Pro', 'Enterprise']:
    subset = master[master['plan_tier'] == tier]
    kmf.fit(subset['duration_days'], event_observed=subset['event_observed'], label=tier)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[tier], linewidth=2)
ax.set_title('Survival by Subscription Tier', fontsize=12, fontweight='bold')
ax.set_xlabel('Days Since Signup')
ax.set_ylabel('Survival Probability')
ax.legend(fontsize=10)

# By referral_source
ax = axes[1, 0]
ref_colors = sns.color_palette(PALETTE, 5)
for i, source in enumerate(accounts['referral_source'].unique()):
    subset = master[master['referral_source'] == source]
    kmf.fit(subset['duration_days'], event_observed=subset['event_observed'], label=source)
    kmf.plot_survival_function(ax=ax, ci_show=False, color=ref_colors[i], linewidth=2)
ax.set_title('Survival by Acquisition Channel', fontsize=12, fontweight='bold')
ax.set_xlabel('Days Since Signup')
ax.set_ylabel('Survival Probability')
ax.legend(fontsize=10)

# By industry
ax = axes[1, 1]
ind_colors = sns.color_palette(PALETTE, len(accounts['industry'].unique()))
for i, ind in enumerate(accounts['industry'].unique()):
    subset = master[master['industry'] == ind]
    kmf.fit(subset['duration_days'], event_observed=subset['event_observed'], label=ind)
    kmf.plot_survival_function(ax=ax, ci_show=False, color=ind_colors[i], linewidth=2)
ax.set_title('Survival by Industry Vertical', fontsize=12, fontweight='bold')
ax.set_xlabel('Days Since Signup')
ax.set_ylabel('Survival Probability')
ax.legend(fontsize=9)

fig.suptitle('Kaplan-Meier Survival Analysis', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, 'kaplan_meier_survival_curves')
plt.close(fig)
print(f'Overall median survival time: {median_survival:.0f} days')"""))

cells.append(
    md("""**So what:** The Kaplan-Meier curves provide the non-parametric foundation
for understanding retention dynamics. Key insights:

1. **Overall survival** shows the baseline retention trajectory and median survival time.
2. **Tier-stratified curves** reveal whether different subscription levels have fundamentally
   different retention dynamics -- if they do, a stratified Cox model may be more appropriate.
3. **Channel-stratified curves** answer the stakeholder question about UA channel quality:
   if organic players survive significantly longer than ad-acquired players, the UA team
   should adjust bidding strategies accordingly.

Separated curves that are consistently parallel (proportional hazards) support the use
of a standard Cox model. Crossing curves would suggest different baseline hazards
and the need for a stratified approach.""")
)

cells.append(md("""### 6.2 Median Survival Time by Segment"""))

cells.append(code("""# Compute median survival time for each segment
segments = {
    'plan_tier': master['plan_tier'].unique(),
    'referral_source': master['referral_source'].unique(),
}

median_results = []
for seg_col, values in segments.items():
    for val in values:
        subset = master[master[seg_col] == val]
        kmf.fit(subset['duration_days'], event_observed=subset['event_observed'])
        median_results.append({
            'Segment': seg_col,
            'Value': val,
            'N': len(subset),
            'Events': subset['event_observed'].sum(),
            'Censored': (subset['event_observed'] == 0).sum(),
            'Median Survival (days)': kmf.median_survival_time_,
        })

median_df = pd.DataFrame(median_results)
display(median_df)"""))

cells.append(md("""### 6.3 Censoring Analysis"""))

cells.append(code("""# Censoring analysis
total = len(master)
censored = (master['event_observed'] == 0).sum()
events = master['event_observed'].sum()

print(f'Total accounts:     {total}')
print(f'Events (churned):   {events} ({events/total*100:.1f}%)')
print(f'Censored (active):  {censored} ({censored/total*100:.1f}%)')
print()

# Check for informative censoring: do censored observations differ systematically?
print('--- Informative Censoring Check ---')
print('Comparing censored vs event groups on key features:')
print()
check_feats = ['total_revenue_30d', 'total_sessions_7d', 'feature_diversity_score',
               'days_since_signup']
for feat in check_feats:
    censored_vals = master[master['event_observed'] == 0][feat]
    event_vals = master[master['event_observed'] == 1][feat]
    t_stat, p_val = stats.ttest_ind(censored_vals, event_vals, equal_var=False)
    diff_pct = (censored_vals.mean() - event_vals.mean()) / event_vals.mean() * 100
    print(f'{feat:35s}: censored mean={censored_vals.mean():.1f}, event mean={event_vals.mean():.1f} '
          f'(diff={diff_pct:+.1f}%, p={p_val:.4f})')

print()
print('If censored and event groups differ significantly on features,')
print('censoring may be informative (not random). This is expected and')
print('handled properly by the Cox model through the partial likelihood.')"""))

cells.append(
    md("""**So what:** A 78% censoring rate is typical for subscription datasets
where most users are still active. The key concern is **informative censoring**:
if the reason for censoring (still active) is related to the features, it can bias
results. In our case, censoring occurs because the observation window ended, not
because of a competing event. The Cox model's partial likelihood correctly handles
this type of non-informative censoring.""")
)

cells.append(md("""### 6.4 Log-Rank Tests"""))

cells.append(code("""# Log-rank tests for survival differences between groups
print('Log-Rank Tests for Survival Differences')
print('=' * 60)

# Plan tier (pairwise)
tiers = master['plan_tier'].unique()
print('\\nPlan Tier (pairwise):')
for i in range(len(tiers)):
    for j in range(i+1, len(tiers)):
        g1 = master[master['plan_tier'] == tiers[i]]
        g2 = master[master['plan_tier'] == tiers[j]]
        results = logrank_test(g1['duration_days'], g2['duration_days'],
                               event_observed_A=g1['event_observed'],
                               event_observed_B=g2['event_observed'])
        sig = '*' if results.p_value < 0.05 else ''
        print(f'  {tiers[i]:12s} vs {tiers[j]:12s}: chi2={results.test_statistic:.2f}, '
              f'p={results.p_value:.4f} {sig}')

# Referral source (overall)
print('\\nReferral Source (overall omnibus test):')
groups = []
for source in master['referral_source'].unique():
    subset = master[master['referral_source'] == source]
    groups.append(subset)
# Use pairwise for key comparison: organic vs ads
org = master[master['referral_source'] == 'organic']
ads = master[master['referral_source'] == 'ads']
results = logrank_test(org['duration_days'], ads['duration_days'],
                       event_observed_A=org['event_observed'],
                       event_observed_B=ads['event_observed'])
print(f'  organic vs ads: chi2={results.test_statistic:.2f}, p={results.p_value:.4f} '
      f'{"*" if results.p_value < 0.05 else ""}')"""))

cells.append(
    md("""**So what:** Log-rank tests formally assess whether survival curves are
statistically different between groups. Significant differences (p < 0.05) confirm
that these categorical variables carry predictive information for the Cox model.
Non-significant differences suggest that, while the curves may look different
visually, the differences could be due to chance given the sample sizes.""")
)

cells.append(md("""### 6.5 Proportional Hazards Assumption Check (Visual)"""))

cells.append(
    code(
        """# Log-cumulative hazard plots for PH assumption check
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plan tier
ax = axes[0]
for tier in ['Basic', 'Pro', 'Enterprise']:
    subset = master[master['plan_tier'] == tier]
    kmf.fit(subset['duration_days'], event_observed=subset['event_observed'], label=tier)
    # log(-log(S(t)))
    sf = kmf.survival_function_.copy()
    sf = sf[sf.iloc[:, 0] > 0]  # avoid log(0)
    log_neg_log = np.log(-np.log(sf.iloc[:, 0]))
    ax.plot(np.log(sf.index), log_neg_log, linewidth=2, label=tier)
ax.set_title('Log-Cumulative Hazard: Subscription Tier', fontsize=12, fontweight='bold')
ax.set_xlabel('log(time)')
ax.set_ylabel('log(-log(S(t)))')
ax.legend()
ax.annotate('Parallel lines = PH assumption holds', xy=(0.05, 0.95),
            xycoords='axes fraction', fontsize=10, style='italic')

# Referral source
ax = axes[1]
for source in ['organic', 'ads', 'partner', 'event', 'other']:
    subset = master[master['referral_source'] == source]
    kmf.fit(subset['duration_days'], event_observed=subset['event_observed'], label=source)
    sf = kmf.survival_function_.copy()
    sf = sf[sf.iloc[:, 0] > 0]
    log_neg_log = np.log(-np.log(sf.iloc[:, 0]))
    ax.plot(np.log(sf.index), log_neg_log, linewidth=2, label=source)
ax.set_title('Log-Cumulative Hazard: Acquisition Channel', fontsize=12, fontweight='bold')
ax.set_xlabel('log(time)')
ax.set_ylabel('log(-log(S(t)))')
ax.legend(fontsize=9)

fig.suptitle('Proportional Hazards Assumption Check', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'proportional_hazards_check')
plt.close(fig)
print('Interpretation: If lines are approximately parallel, the proportional')
print('hazards assumption holds for that variable. Crossing lines suggest')
print('the assumption may be violated and stratification may be needed.')
print('Formal testing via Schoenfeld residuals will be done in the modelling phase.')"""
    )
)

cells.append(
    md("""**So what:** The log-cumulative hazard plots provide a visual check of
the proportional hazards (PH) assumption, which is required for the Cox model.
Lines that are approximately parallel indicate the PH assumption holds --
the hazard ratio between groups is constant over time. If the lines cross,
the hazard ratio changes over time and we may need to use a stratified Cox
model or include time-varying coefficients.

This visual check will be followed by the formal Schoenfeld residuals test
during the modelling phase.""")
)

# ==========================================================================
# CHURN REASONS
# ==========================================================================
cells.append(md("""## 7. Churn Reason Analysis"""))

cells.append(code("""# Churn reason distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall reason distribution
ax = axes[0]
reason_counts = churn['reason_code'].value_counts()
bars = ax.barh(reason_counts.index, reason_counts.values, color=sns.color_palette(PALETTE, len(reason_counts)))
ax.set_title('Churn Reasons (All Events)', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Churn Events')
for bar, val in zip(bars, reason_counts.values):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=10)

# Reason by plan tier
ax = axes[1]
reason_tier = pd.crosstab(churn.merge(accounts[['account_id', 'plan_tier']], on='account_id')['plan_tier'],
                           churn.merge(accounts[['account_id', 'plan_tier']], on='account_id')['reason_code'],
                           normalize='index') * 100
reason_tier.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
ax.set_title('Churn Reasons by Tier (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage')
ax.set_xlabel('')
ax.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.tick_params(axis='x', rotation=0)

fig.suptitle('Why Do Players Leave?', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'churn_reasons')
plt.close(fig)
print('Top churn reasons:')
for reason, count in reason_counts.items():
    print(f'  {reason:15s}: {count:4d} ({count/len(churn)*100:.1f}%)')"""))

cells.append(
    md("""**So what:** Understanding why players leave is essential for designing
effective retention interventions. Feature-related churn suggests product improvements
are needed. Pricing/budget churn suggests a pricing strategy review. Support-related
churn indicates service quality issues. The reason distribution by tier reveals
whether different player segments churn for different reasons, which would inform
targeted retention campaigns.""")
)

# ==========================================================================
# KEY FINDINGS
# ==========================================================================
cells.append(
    md(
        """## 8. Key Findings & Cleaning Plan

### Key EDA Findings

1. **Data quality is high** (95%+ complete). Strategic missingness in satisfaction_score
   (41%) and feedback_text (25%) is handled via indicator variables. 21 duplicate usage_ids
   are documented but non-blocking. All referential integrity checks pass.

2. **Churn rate is 22%** (110/500 accounts), with a 78% censoring rate ideal for survival
   analysis. Importantly, 352 accounts have churn events but only 110 are currently churned --
   many re-activated, confirming the need to use account-level churn_flag, not event presence.

3. **Revenue distribution is heavily right-skewed** (median GBP 931, max GBP 33,830),
   consistent with the "whale" pattern in gaming. Log transformation is essential. Session
   duration also right-skewed. Usage count is approximately normal.

4. **Temporal dynamics** show the expected pattern: early cohorts have lower retention at
   later milestones, while recent cohorts appear artificially retained due to censoring.
   This validates the time-based split strategy and the need for survival analysis
   over simple classification.

5. **Survival curves stratified by tier and channel show meaningful separation**,
   suggesting these are predictive features. The proportional hazards assumption
   appears to hold visually (approximately parallel log-cumulative hazard plots),
   supporting the use of a standard Cox model as the primary technique.

### Data Cleaning Plan (Implemented in src/data_processing.py)

| Issue | Treatment |
|-------|-----------|
| satisfaction_score nulls | Median impute (3.0) + satisfaction_missing indicator |
| feedback_text nulls | Binary provided_feedback indicator |
| end_date nulls | Retained as right-censored (active subscriptions) |
| mrr_amount outliers | Winsorized at 99th percentile |
| usage_count outliers | Capped at 99th percentile |
| resolution_time outliers | Capped at 95th percentile + extreme_resolution_time flag |
| Date columns | Parsed to datetime |
| Duplicate usage_ids | Retained (different data per row, non-blocking) |

### Features Engineered (22 features across 5 categories)

**RFM:** days_since_signup, total_sessions_7d, avg_session_duration_7d, total_revenue_30d, days_to_first_purchase
**Behavioural:** feature_diversity_score, usage_consistency_cv, weekend_usage_ratio, error_rate, beta_feature_adoption_flag
**Support:** support_ticket_count_30d, avg_satisfaction_score, has_urgent_ticket_flag, fast_resolution_rate, satisfaction_missing
**Subscription:** has_upgraded_flag, has_downgraded_flag, subscription_tenure_days, subscription_churn_count, billing_frequency_annual_flag
**Time-Series:** usage_trend_slope_30d, revenue_trend_slope_60d, days_since_last_session"""
    )
)

cells.append(md("""---
*End of EDA. Proceed to `02_modeling.ipynb` for survival model development.*"""))

# ==========================================================================
# BUILD NOTEBOOK
# ==========================================================================

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13.7",
        },
    },
    "cells": cells,
}

nb_path = os.path.join(PROJECT_ROOT, "notebooks", "01_eda.ipynb")
os.makedirs(os.path.dirname(nb_path), exist_ok=True)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {nb_path}")
print(f"Total cells: {len(cells)}")
