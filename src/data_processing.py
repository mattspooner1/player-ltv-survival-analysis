"""Data processing pipeline for Player LTV Survival Analysis.

This module handles loading, validating, cleaning, feature engineering,
and splitting of the raw SaaS/gaming subscription data. All parameters
are loaded from config.yaml -- no hardcoded values.

Functions:
    load_config: Load YAML configuration file.
    load_raw_data: Load all 5 raw CSV tables.
    validate_data_quality: Schema checks, referential integrity, outliers.
    clean_data: Missing values, outliers, type conversions.
    engineer_features: Build master feature table (one row per account).
    create_survival_target: Survival time-to-event and censoring indicator.
    create_train_val_test_split: Time-based split per config cutoff dates.
    save_processed_data: Write processed CSVs to disk.
    main: Execute full pipeline end-to-end.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Absolute or relative path to config.yaml.

    Returns:
        Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_raw_data(config: dict) -> dict[str, pd.DataFrame]:
    """Load all raw CSV tables specified in config.

    Args:
        config: Project configuration dictionary containing data.raw_files.

    Returns:
        Dictionary mapping table names to DataFrames:
        {'accounts': df, 'subscriptions': df, 'feature_usage': df,
         'support_tickets': df, 'churn_events': df}.

    Raises:
        FileNotFoundError: If any expected CSV is missing.
        ValueError: If row counts do not match expected values.
    """
    raw_files = config["data"]["raw_files"]
    expected_counts = config["data"]["expected_row_counts"]
    dfs: dict[str, pd.DataFrame] = {}

    for table_name, file_path in raw_files.items():
        full_path = Path(file_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")
        df = pd.read_csv(full_path)
        expected = expected_counts.get(table_name)
        if expected is not None and len(df) != expected:
            raise ValueError(
                f"Row count mismatch for {table_name}: "
                f"got {len(df)}, expected {expected}"
            )
        dfs[table_name] = df
        logger.info(
            "Loaded %s: %d rows x %d cols", table_name, len(df), len(df.columns)
        )

    return dfs


# ---------------------------------------------------------------------------
# Data Validation
# ---------------------------------------------------------------------------


def validate_data_quality(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Run comprehensive data quality checks across all tables.

    Checks include: missing value counts, duplicate detection, referential
    integrity between tables, value range validation, and data type
    verification.

    Args:
        dfs: Dictionary of table name to DataFrame (from load_raw_data).

    Returns:
        Dictionary summarising quality findings:
        {
            'missing_values': {table: {col: count}},
            'duplicates': {table: count},
            'referential_integrity': {check_name: bool},
            'value_ranges': {table: {col: {min, max, outlier_count}}},
            'warnings': [str],
        }
    """
    report: dict[str, Any] = {
        "missing_values": {},
        "duplicates": {},
        "referential_integrity": {},
        "value_ranges": {},
        "warnings": [],
    }

    # --- Missing values ---
    for name, df in dfs.items():
        missing = df.isnull().sum()
        missing = missing[missing > 0].to_dict()
        report["missing_values"][name] = missing
        if missing:
            logger.info("Missing values in %s: %s", name, missing)

    # --- Duplicates on primary keys ---
    pk_map = {
        "accounts": "account_id",
        "subscriptions": "subscription_id",
        "feature_usage": "usage_id",
        "support_tickets": "ticket_id",
        "churn_events": "churn_event_id",
    }
    for name, pk in pk_map.items():
        if name in dfs and pk in dfs[name].columns:
            dup_count = dfs[name][pk].duplicated().sum()
            report["duplicates"][name] = int(dup_count)
            if dup_count > 0:
                report["warnings"].append(f"Duplicate {pk} in {name}: {dup_count} rows")

    # --- Referential integrity ---
    acct_ids = set(dfs["accounts"]["account_id"])

    sub_accts = set(dfs["subscriptions"]["account_id"])
    report["referential_integrity"]["subs_in_accounts"] = sub_accts.issubset(acct_ids)
    if not sub_accts.issubset(acct_ids):
        orphans = sub_accts - acct_ids
        report["warnings"].append(f"Orphan subscription accounts: {len(orphans)}")

    ticket_accts = set(dfs["support_tickets"]["account_id"])
    report["referential_integrity"]["tickets_in_accounts"] = ticket_accts.issubset(
        acct_ids
    )

    churn_accts = set(dfs["churn_events"]["account_id"])
    report["referential_integrity"]["churn_in_accounts"] = churn_accts.issubset(
        acct_ids
    )

    sub_ids = set(dfs["subscriptions"]["subscription_id"])
    usage_subs = set(dfs["feature_usage"]["subscription_id"])
    report["referential_integrity"]["usage_in_subs"] = usage_subs.issubset(sub_ids)
    if not usage_subs.issubset(sub_ids):
        report["warnings"].append(
            f"Orphan usage subscription_ids: {len(usage_subs - sub_ids)}"
        )

    # --- Value ranges for numeric columns ---
    numeric_checks = {
        "subscriptions": ["mrr_amount", "arr_amount", "seats"],
        "feature_usage": ["usage_count", "usage_duration_secs", "error_count"],
        "support_tickets": [
            "resolution_time_hours",
            "first_response_time_minutes",
            "satisfaction_score",
        ],
        "churn_events": ["refund_amount_usd"],
    }
    for name, cols in numeric_checks.items():
        report["value_ranges"][name] = {}
        for col in cols:
            if col in dfs[name].columns:
                series = dfs[name][col].dropna()
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                outlier_count = int(
                    ((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).sum()
                )
                report["value_ranges"][name][col] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "outlier_count_3iqr": outlier_count,
                }

    # --- Temporal validation ---
    accts = dfs["accounts"].copy()
    accts["signup_date"] = pd.to_datetime(accts["signup_date"])
    subs = dfs["subscriptions"].copy()
    subs["start_date"] = pd.to_datetime(subs["start_date"])
    merged = subs.merge(accts[["account_id", "signup_date"]], on="account_id")
    bad_temporal = (merged["start_date"] < merged["signup_date"]).sum()
    report["referential_integrity"]["subscription_after_signup"] = (
        int(bad_temporal) == 0
    )
    if bad_temporal > 0:
        report["warnings"].append(
            f"Subscriptions starting before signup: {bad_temporal}"
        )

    logger.info("Validation complete. Warnings: %d", len(report["warnings"]))
    return report


# ---------------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------------


def clean_data(dfs: dict[str, pd.DataFrame], config: dict) -> dict[str, pd.DataFrame]:
    """Clean all tables according to config rules.

    Handles missing values, outlier treatment, and date parsing as specified
    in config.yaml's missing_values and outliers sections.

    Args:
        dfs: Dictionary of raw DataFrames (from load_raw_data).
        config: Project configuration dictionary.

    Returns:
        Dictionary of cleaned DataFrames (same keys, modified in-place copy).
    """
    cleaned = {name: df.copy() for name, df in dfs.items()}

    # --- Parse dates ---
    date_cols = {
        "accounts": ["signup_date"],
        "subscriptions": ["start_date", "end_date"],
        "feature_usage": ["usage_date"],
        "support_tickets": ["submitted_at", "closed_at"],
        "churn_events": ["churn_date"],
    }
    for table, cols in date_cols.items():
        for col in cols:
            if col in cleaned[table].columns:
                cleaned[table][col] = pd.to_datetime(
                    cleaned[table][col], errors="coerce"
                )
                logger.info("Parsed %s.%s as datetime", table, col)

    # --- Support tickets: satisfaction_score ---
    mv_config = config.get("missing_values", {})
    sat_cfg = mv_config.get("satisfaction_score", {})
    if sat_cfg:
        tickets = cleaned["support_tickets"]
        indicator_col = sat_cfg.get("indicator_column", "satisfaction_missing")
        impute_val = sat_cfg.get("impute_value", 3.0)
        tickets[indicator_col] = tickets["satisfaction_score"].isnull().astype(int)
        tickets["satisfaction_score"] = tickets["satisfaction_score"].fillna(impute_val)
        logger.info(
            "Imputed satisfaction_score with %s, created %s indicator",
            impute_val,
            indicator_col,
        )

    # --- Churn events: feedback_text ---
    fb_cfg = mv_config.get("feedback_text", {})
    if fb_cfg:
        churn_df = cleaned["churn_events"]
        indicator_col = fb_cfg.get("indicator_column", "provided_feedback")
        churn_df[indicator_col] = churn_df["feedback_text"].notna().astype(int)
        logger.info("Created %s indicator for feedback_text", indicator_col)

    # --- Revenue scaling ---
    # The raw dataset uses SaaS-scale revenue figures that are unrealistically
    # high for a gaming context (mean MRR ~ GBP 2,268). We scale all monetary
    # fields by 0.1 to bring values into a plausible range for a gaming
    # subscription service (mean MRR ~ GBP 227).
    revenue_scale = config.get("revenue_scale_factor", 0.1)
    if revenue_scale != 1.0:
        subs = cleaned["subscriptions"]
        for col in ["mrr_amount", "arr_amount"]:
            if col in subs.columns:
                subs[col] = subs[col] * revenue_scale
                logger.info(
                    "Scaled %s by %.2f (gaming realism adjustment)", col, revenue_scale
                )
        churn_ev = cleaned["churn_events"]
        if "refund_amount_usd" in churn_ev.columns:
            churn_ev["refund_amount_usd"] = (
                churn_ev["refund_amount_usd"] * revenue_scale
            )
            logger.info(
                "Scaled refund_amount_usd by %.2f (gaming realism adjustment)",
                revenue_scale,
            )

    # --- Outlier treatment ---
    outlier_cfg = config.get("outliers", {})

    # mrr_amount: winsorize at 99th percentile
    if "mrr_amount" in outlier_cfg:
        subs = cleaned["subscriptions"]
        p99 = subs["mrr_amount"].quantile(0.99)
        before_count = (subs["mrr_amount"] > p99).sum()
        subs["mrr_amount"] = subs["mrr_amount"].clip(upper=p99)
        logger.info(
            "Winsorized mrr_amount at 99th (%s): %d values clipped",
            p99,
            before_count,
        )

    # usage_count: cap at 99th percentile
    if "usage_count" in outlier_cfg:
        usage = cleaned["feature_usage"]
        p99 = usage["usage_count"].quantile(0.99)
        before_count = (usage["usage_count"] > p99).sum()
        usage["usage_count"] = usage["usage_count"].clip(upper=p99)
        logger.info(
            "Capped usage_count at 99th (%s): %d values clipped",
            p99,
            before_count,
        )

    # resolution_time_hours: cap at 95th, create extreme flag
    if "resolution_time_hours" in outlier_cfg:
        rt_cfg = outlier_cfg["resolution_time_hours"]
        tickets = cleaned["support_tickets"]
        p95 = tickets["resolution_time_hours"].quantile(0.95)
        flag_threshold = rt_cfg.get("flag_threshold", 168)
        flag_col = rt_cfg.get("create_flag", "extreme_resolution_time")
        tickets[flag_col] = (tickets["resolution_time_hours"] > flag_threshold).astype(
            int
        )
        before_count = (tickets["resolution_time_hours"] > p95).sum()
        tickets["resolution_time_hours"] = tickets["resolution_time_hours"].clip(
            upper=p95
        )
        logger.info(
            "Capped resolution_time_hours at 95th (%s): %d values clipped. "
            "Created %s flag (threshold=%s)",
            p95,
            before_count,
            flag_col,
            flag_threshold,
        )

    return cleaned


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


def engineer_features(dfs: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    """Build master feature table with one row per account.

    Creates RFM, behavioural, support, subscription, and time-series derived
    features as specified in config.yaml. Also creates the survival target
    variables (duration and event indicator).

    Args:
        dfs: Dictionary of *cleaned* DataFrames.
        config: Project configuration dictionary.

    Returns:
        DataFrame with one row per account_id and all engineered features.
    """
    accounts = dfs["accounts"].copy()
    subs = dfs["subscriptions"].copy()
    usage = dfs["feature_usage"].copy()
    tickets = dfs["support_tickets"].copy()
    churn_df = dfs["churn_events"].copy()

    # Use the analysis reference date from config (avoids leakage)
    analysis_date = pd.Timestamp(config["project"]["analysis_date"])
    logger.info("Analysis reference date: %s", analysis_date)

    # Start master table from accounts
    master = accounts[
        [
            "account_id",
            "signup_date",
            "plan_tier",
            "referral_source",
            "industry",
            "country",
            "seats",
            "is_trial",
            "churn_flag",
        ]
    ].copy()

    # -----------------------------------------------------------------------
    # RFM Features
    # -----------------------------------------------------------------------
    master["days_since_signup"] = (analysis_date - master["signup_date"]).dt.days

    # Link usage to accounts through subscriptions
    sub_usage = usage.merge(
        subs[["subscription_id", "account_id"]],
        on="subscription_id",
        how="left",
    )

    # Merge signup_date for time-windowed features
    sub_usage = sub_usage.merge(
        master[["account_id", "signup_date"]], on="account_id", how="left"
    )
    sub_usage["days_since_signup_usage"] = (
        sub_usage["usage_date"] - sub_usage["signup_date"]
    ).dt.days

    # total_sessions_7d
    first_7d = sub_usage[sub_usage["days_since_signup_usage"].between(0, 7)]
    sessions_7d = (
        first_7d.groupby("account_id")["usage_id"].count().rename("total_sessions_7d")
    )
    master = master.merge(sessions_7d, on="account_id", how="left")
    master["total_sessions_7d"] = master["total_sessions_7d"].fillna(0)

    # avg_session_duration_7d
    duration_7d = (
        first_7d.groupby("account_id")["usage_duration_secs"]
        .mean()
        .rename("avg_session_duration_7d")
    )
    master = master.merge(duration_7d, on="account_id", how="left")
    master["avg_session_duration_7d"] = master["avg_session_duration_7d"].fillna(0)

    # total_revenue_30d
    subs_with_signup = subs.merge(
        master[["account_id", "signup_date"]], on="account_id", how="left"
    )
    subs_with_signup["days_from_signup"] = (
        subs_with_signup["start_date"] - subs_with_signup["signup_date"]
    ).dt.days
    first_30d_subs = subs_with_signup[
        subs_with_signup["days_from_signup"].between(0, 30)
    ]
    rev_30d = (
        first_30d_subs.groupby("account_id")["mrr_amount"]
        .sum()
        .rename("total_revenue_30d")
    )
    master = master.merge(rev_30d, on="account_id", how="left")
    master["total_revenue_30d"] = master["total_revenue_30d"].fillna(0)

    # days_to_first_purchase (first non-trial subscription)
    non_trial_subs = subs_with_signup[~subs_with_signup["is_trial"]]
    first_purchase = (
        non_trial_subs.groupby("account_id")["days_from_signup"]
        .min()
        .rename("days_to_first_purchase")
    )
    master = master.merge(first_purchase, on="account_id", how="left")
    # Fill accounts with no purchase with large sentinel (capped later)
    max_days = int(master["days_since_signup"].max())
    master["days_to_first_purchase"] = master["days_to_first_purchase"].fillna(max_days)

    # -----------------------------------------------------------------------
    # Behavioural Aggregations
    # -----------------------------------------------------------------------
    # We know from data inspection there are 40 distinct features
    n_total_features = usage["feature_name"].nunique()

    # feature_diversity_score
    diversity = (
        sub_usage.groupby("account_id")["feature_name"]
        .nunique()
        .rename("feature_diversity_score")
        / n_total_features
    )
    master = master.merge(diversity, on="account_id", how="left")
    master["feature_diversity_score"] = master["feature_diversity_score"].fillna(0)

    # usage_consistency_cv (coefficient of variation of daily usage count)
    daily_usage = (
        sub_usage.groupby(["account_id", "usage_date"])["usage_count"]
        .sum()
        .reset_index()
    )
    cv_stats = daily_usage.groupby("account_id")["usage_count"].agg(["mean", "std"])
    cv_stats["usage_consistency_cv"] = cv_stats["std"] / cv_stats["mean"]
    cv_stats["usage_consistency_cv"] = (
        cv_stats["usage_consistency_cv"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    master = master.merge(
        cv_stats[["usage_consistency_cv"]], on="account_id", how="left"
    )
    master["usage_consistency_cv"] = master["usage_consistency_cv"].fillna(0)

    # weekend_usage_ratio
    sub_usage["is_weekend"] = sub_usage["usage_date"].dt.dayofweek.isin([5, 6])
    weekend_stats = sub_usage.groupby("account_id")["is_weekend"].agg(["sum", "count"])
    weekend_stats["weekend_usage_ratio"] = weekend_stats["sum"] / weekend_stats["count"]
    master = master.merge(
        weekend_stats[["weekend_usage_ratio"]], on="account_id", how="left"
    )
    master["weekend_usage_ratio"] = master["weekend_usage_ratio"].fillna(0)

    # error_rate
    error_stats = sub_usage.groupby("account_id").agg(
        total_errors=("error_count", "sum"),
        total_usage=("usage_count", "sum"),
    )
    error_stats["error_rate"] = error_stats["total_errors"] / error_stats[
        "total_usage"
    ].replace(0, np.nan)
    error_stats["error_rate"] = error_stats["error_rate"].fillna(0)
    master = master.merge(error_stats[["error_rate"]], on="account_id", how="left")
    master["error_rate"] = master["error_rate"].fillna(0)

    # beta_feature_adoption_flag
    beta_usage = sub_usage[sub_usage["is_beta_feature"]].groupby("account_id").size()
    master["beta_feature_adoption_flag"] = (
        master["account_id"].isin(beta_usage.index).astype(int)
    )

    # -----------------------------------------------------------------------
    # Support Interaction Features
    # -----------------------------------------------------------------------
    tickets_with_signup = tickets.merge(
        master[["account_id", "signup_date"]], on="account_id", how="left"
    )
    tickets_with_signup["days_from_signup"] = (
        tickets_with_signup["submitted_at"] - tickets_with_signup["signup_date"]
    ).dt.days

    # support_ticket_count_30d
    first_30d_tickets = tickets_with_signup[
        tickets_with_signup["days_from_signup"].between(0, 30)
    ]
    ticket_count_30d = (
        first_30d_tickets.groupby("account_id")["ticket_id"]
        .count()
        .rename("support_ticket_count_30d")
    )
    master = master.merge(ticket_count_30d, on="account_id", how="left")
    master["support_ticket_count_30d"] = master["support_ticket_count_30d"].fillna(0)

    # avg_satisfaction_score (after imputation in clean step)
    sat_scores = (
        tickets.groupby("account_id")["satisfaction_score"]
        .mean()
        .rename("avg_satisfaction_score")
    )
    master = master.merge(sat_scores, on="account_id", how="left")
    # Impute missing avg satisfaction with config value
    sat_impute = (
        config.get("missing_values", {})
        .get("satisfaction_score", {})
        .get("impute_value", 3.0)
    )
    sat_indicator_col = (
        config.get("missing_values", {})
        .get("satisfaction_score", {})
        .get("indicator_column", "satisfaction_missing")
    )
    master[sat_indicator_col] = master["avg_satisfaction_score"].isnull().astype(int)
    master["avg_satisfaction_score"] = master["avg_satisfaction_score"].fillna(
        sat_impute
    )

    # has_urgent_ticket_flag
    urgent_accounts = set(tickets[tickets["priority"] == "urgent"]["account_id"])
    master["has_urgent_ticket_flag"] = (
        master["account_id"].isin(urgent_accounts).astype(int)
    )

    # fast_resolution_rate (resolved < 24 hours)
    tickets["fast_resolved"] = (tickets["resolution_time_hours"] < 24).astype(int)
    fast_rate = tickets.groupby("account_id").agg(
        fast_count=("fast_resolved", "sum"),
        total_tickets=("ticket_id", "count"),
    )
    fast_rate["fast_resolution_rate"] = (
        fast_rate["fast_count"] / fast_rate["total_tickets"]
    )
    master = master.merge(
        fast_rate[["fast_resolution_rate"]], on="account_id", how="left"
    )
    master["fast_resolution_rate"] = master["fast_resolution_rate"].fillna(0)

    # -----------------------------------------------------------------------
    # Subscription Behaviour Features
    # -----------------------------------------------------------------------
    # has_upgraded_flag
    upgrade_accts = set(subs[subs["upgrade_flag"]]["account_id"])
    master["has_upgraded_flag"] = master["account_id"].isin(upgrade_accts).astype(int)

    # has_downgraded_flag
    downgrade_accts = set(subs[subs["downgrade_flag"]]["account_id"])
    master["has_downgraded_flag"] = (
        master["account_id"].isin(downgrade_accts).astype(int)
    )

    # subscription_tenure_days
    tenure_df = subs.copy()
    tenure_df["end_date_filled"] = tenure_df["end_date"].fillna(analysis_date)
    tenure_stats = tenure_df.groupby("account_id").agg(
        min_start=("start_date", "min"),
        max_end=("end_date_filled", "max"),
    )
    tenure_stats["subscription_tenure_days"] = (
        tenure_stats["max_end"] - tenure_stats["min_start"]
    ).dt.days
    master = master.merge(
        tenure_stats[["subscription_tenure_days"]],
        on="account_id",
        how="left",
    )
    master["subscription_tenure_days"] = master["subscription_tenure_days"].fillna(0)

    # subscription_churn_count
    sub_churn_count = (
        subs[subs["churn_flag"]]
        .groupby("account_id")
        .size()
        .rename("subscription_churn_count")
    )
    master = master.merge(sub_churn_count, on="account_id", how="left")
    master["subscription_churn_count"] = master["subscription_churn_count"].fillna(0)

    # billing_frequency_annual_flag
    annual_accts = set(subs[subs["billing_frequency"] == "annual"]["account_id"])
    master["billing_frequency_annual_flag"] = (
        master["account_id"].isin(annual_accts).astype(int)
    )

    # -----------------------------------------------------------------------
    # Time-Series Derived Features
    # -----------------------------------------------------------------------
    # usage_trend_slope_30d (linear regression slope of daily usage)
    first_30d_usage = sub_usage[sub_usage["days_since_signup_usage"].between(0, 30)]
    daily_30d = (
        first_30d_usage.groupby(["account_id", "days_since_signup_usage"])[
            "usage_count"
        ]
        .sum()
        .reset_index()
    )

    def _compute_slope(group: pd.DataFrame, x_col: str, y_col: str) -> float:
        """Compute linear regression slope for a group."""
        if len(group) < 2:
            return 0.0
        x = group[x_col].values.astype(float)
        y = group[y_col].values.astype(float)
        if np.std(x) == 0:
            return 0.0
        slope, _, _, _, _ = scipy_stats.linregress(x, y)
        return float(slope)

    slopes_30d = (
        daily_30d.groupby("account_id")
        .apply(
            _compute_slope,
            x_col="days_since_signup_usage",
            y_col="usage_count",
            include_groups=False,
        )
        .rename("usage_trend_slope_30d")
    )
    master = master.merge(slopes_30d, on="account_id", how="left")
    master["usage_trend_slope_30d"] = master["usage_trend_slope_30d"].fillna(0)

    # revenue_trend_slope_60d (monthly revenue slope)
    first_60d_subs = subs_with_signup[
        subs_with_signup["days_from_signup"].between(0, 60)
    ].copy()
    first_60d_subs["month_offset"] = first_60d_subs["days_from_signup"] // 30
    monthly_rev = (
        first_60d_subs.groupby(["account_id", "month_offset"])["mrr_amount"]
        .sum()
        .reset_index()
    )
    slopes_60d = (
        monthly_rev.groupby("account_id")
        .apply(
            _compute_slope,
            x_col="month_offset",
            y_col="mrr_amount",
            include_groups=False,
        )
        .rename("revenue_trend_slope_60d")
    )
    master = master.merge(slopes_60d, on="account_id", how="left")
    master["revenue_trend_slope_60d"] = master["revenue_trend_slope_60d"].fillna(0)

    # days_since_last_session
    last_session = (
        sub_usage.groupby("account_id")["usage_date"].max().rename("last_session_date")
    )
    master = master.merge(last_session, on="account_id", how="left")
    master["days_since_last_session"] = (
        analysis_date - master["last_session_date"]
    ).dt.days
    master["days_since_last_session"] = master["days_since_last_session"].fillna(
        master["days_since_signup"]
    )
    master = master.drop(columns=["last_session_date"], errors="ignore")

    # -----------------------------------------------------------------------
    # Temporal features from signup_date
    # -----------------------------------------------------------------------
    temporal_cfg = config.get("transformations", {}).get("temporal", {})
    if temporal_cfg:
        if "signup_month" in temporal_cfg.get("features", []):
            master["signup_month"] = master["signup_date"].dt.month
        if "signup_day_of_week" in temporal_cfg.get("features", []):
            master["signup_day_of_week"] = master["signup_date"].dt.dayofweek

    # -----------------------------------------------------------------------
    # Churn event enrichment (for survival target)
    # -----------------------------------------------------------------------
    # Note: accounts.churn_flag is the definitive current churn status.
    # churn_events table contains historical churn events (some accounts
    # churned then reactivated, so churn_flag=False despite having events).
    # For survival analysis we use: event = accounts.churn_flag (current)
    # and churn_date from the *last* churn event (most recent churn).
    last_churn = (
        churn_df.sort_values("churn_date").groupby("account_id").last().reset_index()
    )
    master = master.merge(
        last_churn[["account_id", "churn_date", "reason_code"]],
        on="account_id",
        how="left",
    )

    # For accounts with churn_flag=True but no churn_event entry,
    # estimate churn_date from last subscription end_date.
    missing_churn_date = master["churn_flag"] & master["churn_date"].isna()
    if missing_churn_date.any():
        last_end = (
            subs[subs["end_date"].notna()].groupby("account_id")["end_date"].max()
        )
        for idx in master[missing_churn_date].index:
            acct = master.loc[idx, "account_id"]
            if acct in last_end.index:
                master.loc[idx, "churn_date"] = last_end[acct]
            else:
                # Fallback: use signup_date + median tenure
                master.loc[idx, "churn_date"] = master.loc[
                    idx, "signup_date"
                ] + pd.Timedelta(days=90)
        logger.info(
            "Estimated churn_date for %d accounts missing from churn_events",
            missing_churn_date.sum(),
        )

    # For accounts with churn_flag=False but having churn events (reactivated),
    # clear the churn_date so they are treated as censored.
    reactivated = ~master["churn_flag"] & master["churn_date"].notna()
    if reactivated.any():
        master.loc[reactivated, "churn_date"] = pd.NaT
        master.loc[reactivated, "reason_code"] = np.nan
        logger.info("Cleared churn_date for %d reactivated accounts", reactivated.sum())

    # provided_feedback (from churn events, for accounts that churned)
    if "provided_feedback" in churn_df.columns:
        feedback_flag = last_churn[["account_id", "provided_feedback"]].set_index(
            "account_id"
        )
        master = master.merge(feedback_flag, on="account_id", how="left")
        master["provided_feedback"] = master["provided_feedback"].fillna(0).astype(int)

    logger.info("Master feature table shape: %s", master.shape)
    return master


# ---------------------------------------------------------------------------
# Survival Target Creation
# ---------------------------------------------------------------------------


def create_survival_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create survival analysis target variables.

    Uses the account-level churn_flag as the definitive event indicator
    (current status). Adds 'duration_days' (time to event or censoring)
    and 'event_observed' (1 = currently churned, 0 = active/censored).

    Args:
        df: Master feature DataFrame with signup_date, churn_flag,
            and churn_date columns.
        config: Project configuration dictionary.

    Returns:
        DataFrame with added survival target columns.
    """
    result = df.copy()
    analysis_date = pd.Timestamp(config["project"]["analysis_date"])

    # event_observed: use account-level churn_flag (current status)
    result["event_observed"] = result["churn_flag"].astype(int)

    # duration_days: churn_date - signup_date (for churned accounts)
    # or analysis_date - signup_date (for active/censored accounts)
    result["duration_days"] = np.where(
        (result["event_observed"] == 1) & result["churn_date"].notna(),
        (result["churn_date"] - result["signup_date"]).dt.days,
        (analysis_date - result["signup_date"]).dt.days,
    )

    # Ensure duration > 0 (minimum 1 day)
    result["duration_days"] = result["duration_days"].clip(lower=1)

    logger.info(
        "Survival target: %d events, %d censored (%.1f%% censoring rate)",
        result["event_observed"].sum(),
        (result["event_observed"] == 0).sum(),
        (result["event_observed"] == 0).mean() * 100,
    )
    return result


def _compute_total_revenue_180d(
    subs: pd.DataFrame, accounts: pd.DataFrame
) -> pd.Series:
    """Compute actual 180-day revenue per account from subscription data.

    Args:
        subs: Cleaned subscriptions DataFrame.
        accounts: Cleaned accounts DataFrame with signup_date.

    Returns:
        Series indexed by account_id with total_revenue_180d values.
    """
    merged = subs.merge(
        accounts[["account_id", "signup_date"]], on="account_id", how="left"
    )
    merged["days_from_signup"] = (merged["start_date"] - merged["signup_date"]).dt.days
    within_180 = merged[merged["days_from_signup"].between(0, 180)]
    rev_180 = within_180.groupby("account_id")["mrr_amount"].sum()
    rev_180.name = "total_revenue_180d"
    return rev_180


# ---------------------------------------------------------------------------
# Train / Validation / Test Split
# ---------------------------------------------------------------------------


def create_train_val_test_split(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data using time-based strategy from config.

    Uses signup_date cutoffs to prevent data leakage:
    - Train: signup_date < train_cutoff
    - Validation: train_cutoff <= signup_date < validation_cutoff
    - Test: signup_date >= validation_cutoff

    Args:
        df: Master feature DataFrame with signup_date column.
        config: Project configuration dictionary with splitting params.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If any split is empty.
    """
    split_cfg = config["splitting"]
    train_cutoff = pd.Timestamp(split_cfg["train_cutoff"])
    val_cutoff = pd.Timestamp(split_cfg["validation_cutoff"])

    train = df[df["signup_date"] < train_cutoff].copy()
    val = df[
        (df["signup_date"] >= train_cutoff) & (df["signup_date"] < val_cutoff)
    ].copy()
    test = df[df["signup_date"] >= val_cutoff].copy()

    for name, split in [("train", train), ("val", val), ("test", test)]:
        if len(split) == 0:
            raise ValueError(f"Empty {name} split. Check cutoff dates.")

    logger.info(
        "Split sizes - Train: %d (%.1f%%), Val: %d (%.1f%%), Test: %d (%.1f%%)",
        len(train),
        len(train) / len(df) * 100,
        len(val),
        len(val) / len(df) * 100,
        len(test),
        len(test) / len(df) * 100,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Save Processed Data
# ---------------------------------------------------------------------------


def save_processed_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str,
    master: pd.DataFrame | None = None,
) -> None:
    """Save processed DataFrames to CSV files.

    Args:
        train: Training set DataFrame.
        val: Validation set DataFrame.
        test: Test set DataFrame.
        output_dir: Directory path for output CSVs.
        master: Optional full master features table to save.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train.to_csv(out_path / "train.csv", index=False)
    val.to_csv(out_path / "validation.csv", index=False)
    test.to_csv(out_path / "test.csv", index=False)

    if master is not None:
        master.to_csv(out_path / "master_features.csv", index=False)
        logger.info("Saved master_features.csv (%d rows)", len(master))

    logger.info(
        "Saved splits to %s: train=%d, val=%d, test=%d",
        output_dir,
        len(train),
        len(val),
        len(test),
    )


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main(config_path: str = "config.yaml") -> pd.DataFrame:
    """Execute the full data processing pipeline.

    Steps:
        1. Load configuration
        2. Load raw data
        3. Validate data quality
        4. Clean data
        5. Engineer features
        6. Create survival targets
        7. Compute actual 180-day revenue
        8. Split into train/val/test
        9. Save processed data

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Master feature table DataFrame.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting data processing pipeline")

    # 1. Load config
    config = load_config(config_path)

    # 2. Load raw data
    dfs = load_raw_data(config)

    # 3. Validate
    report = validate_data_quality(dfs)
    if report["warnings"]:
        for w in report["warnings"]:
            logger.warning("Data quality warning: %s", w)

    # 4. Clean
    cleaned = clean_data(dfs, config)

    # 5. Engineer features
    master = engineer_features(cleaned, config)

    # 6. Survival targets
    master = create_survival_target(master, config)

    # 7. Compute actual 180-day revenue (overwrite proxy)
    rev_180 = _compute_total_revenue_180d(cleaned["subscriptions"], cleaned["accounts"])
    master = master.drop(columns=["total_revenue_180d"], errors="ignore")
    master = master.merge(rev_180.reset_index(), on="account_id", how="left")
    master["total_revenue_180d"] = master["total_revenue_180d"].fillna(0)

    # 8. Split
    train, val, test = create_train_val_test_split(master, config)

    # 9. Save
    output_dir = config["data"]["processed_dir"]
    save_processed_data(train, val, test, output_dir, master=master)

    logger.info("Pipeline complete. Master table: %s", master.shape)
    return master


if __name__ == "__main__":
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path=cfg)
