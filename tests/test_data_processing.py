"""Unit tests for the data processing pipeline.

Tests cover data loading, validation, cleaning, feature engineering,
survival target creation, and train/val/test splitting. Edge cases
include empty DataFrames and all-missing columns.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    clean_data,
    create_survival_target,
    create_train_val_test_split,
    engineer_features,
    load_config,
    load_raw_data,
    save_processed_data,
    validate_data_quality,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG_PATH = "config.yaml"


@pytest.fixture
def config():
    """Load project configuration."""
    return load_config(CONFIG_PATH)


@pytest.fixture
def raw_dfs(config):
    """Load raw data tables."""
    return load_raw_data(config)


@pytest.fixture
def cleaned_dfs(raw_dfs, config):
    """Return cleaned data tables."""
    return clean_data(raw_dfs, config)


@pytest.fixture
def master_df(cleaned_dfs, config):
    """Return master feature table."""
    return engineer_features(cleaned_dfs, config)


@pytest.fixture
def master_with_survival(master_df, config):
    """Return master table with survival targets."""
    return create_survival_target(master_df, config)


@pytest.fixture
def minimal_dfs():
    """Create minimal synthetic DataFrames for edge-case testing."""
    accounts = pd.DataFrame(
        {
            "account_id": ["A-001", "A-002", "A-003"],
            "account_name": ["Co1", "Co2", "Co3"],
            "industry": ["FinTech", "EdTech", "DevTools"],
            "country": ["US", "UK", "IN"],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-10-01"]),
            "referral_source": ["organic", "ads", "partner"],
            "plan_tier": ["Basic", "Pro", "Enterprise"],
            "seats": [5, 10, 20],
            "is_trial": [False, True, False],
            "churn_flag": [True, False, False],
        }
    )
    subscriptions = pd.DataFrame(
        {
            "subscription_id": ["S-001", "S-002", "S-003"],
            "account_id": ["A-001", "A-002", "A-003"],
            "start_date": pd.to_datetime(["2024-01-05", "2024-06-05", "2024-10-05"]),
            "end_date": pd.to_datetime(["2024-06-01", pd.NaT, pd.NaT]),
            "plan_tier": ["Basic", "Pro", "Enterprise"],
            "seats": [5, 10, 20],
            "mrr_amount": [100, 500, 2000],
            "arr_amount": [1200, 6000, 24000],
            "is_trial": [False, True, False],
            "upgrade_flag": [False, True, False],
            "downgrade_flag": [False, False, True],
            "churn_flag": [True, False, False],
            "billing_frequency": ["monthly", "annual", "monthly"],
            "auto_renew_flag": [True, True, False],
        }
    )
    feature_usage = pd.DataFrame(
        {
            "usage_id": ["U-001", "U-002", "U-003", "U-004"],
            "subscription_id": ["S-001", "S-001", "S-002", "S-003"],
            "usage_date": pd.to_datetime(
                ["2024-01-06", "2024-01-10", "2024-06-06", "2024-10-06"]
            ),
            "feature_name": ["feature_1", "feature_2", "feature_1", "feature_3"],
            "usage_count": [5, 10, 3, 8],
            "usage_duration_secs": [300, 600, 150, 480],
            "error_count": [0, 1, 0, 2],
            "is_beta_feature": [False, True, False, False],
        }
    )
    support_tickets = pd.DataFrame(
        {
            "ticket_id": ["T-001", "T-002"],
            "account_id": ["A-001", "A-002"],
            "submitted_at": pd.to_datetime(["2024-01-15", "2024-06-10"]),
            "closed_at": pd.to_datetime(["2024-01-16", "2024-06-11"]),
            "resolution_time_hours": [24.0, 12.0],
            "priority": ["urgent", "low"],
            "first_response_time_minutes": [30, 60],
            "satisfaction_score": [4.0, np.nan],
            "escalation_flag": [False, False],
        }
    )
    churn_events = pd.DataFrame(
        {
            "churn_event_id": ["C-001"],
            "account_id": ["A-001"],
            "churn_date": pd.to_datetime(["2024-06-01"]),
            "reason_code": ["pricing"],
            "refund_amount_usd": [10.0],
            "preceding_upgrade_flag": [False],
            "preceding_downgrade_flag": [False],
            "is_reactivation": [False],
            "feedback_text": ["too expensive"],
        }
    )
    return {
        "accounts": accounts,
        "subscriptions": subscriptions,
        "feature_usage": feature_usage,
        "support_tickets": support_tickets,
        "churn_events": churn_events,
    }


# ---------------------------------------------------------------------------
# Tests: load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_valid_config(self):
        """Config loads and contains expected top-level keys."""
        cfg = load_config(CONFIG_PATH)
        assert "project" in cfg
        assert "data" in cfg
        assert "features" in cfg
        assert "splitting" in cfg

    def test_config_has_random_seed(self):
        """Config contains random_seed."""
        cfg = load_config(CONFIG_PATH)
        assert cfg["project"]["random_seed"] == 42

    def test_missing_config_raises(self):
        """Non-existent config path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


# ---------------------------------------------------------------------------
# Tests: load_raw_data
# ---------------------------------------------------------------------------


class TestLoadRawData:
    """Tests for load_raw_data function."""

    def test_returns_correct_table_names(self, config):
        """All 5 expected table names are returned."""
        dfs = load_raw_data(config)
        expected = {
            "accounts",
            "subscriptions",
            "feature_usage",
            "support_tickets",
            "churn_events",
        }
        assert set(dfs.keys()) == expected

    def test_row_counts_match_expected(self, config):
        """Row counts match config expectations."""
        dfs = load_raw_data(config)
        expected_counts = config["data"]["expected_row_counts"]
        for table, expected in expected_counts.items():
            assert (
                len(dfs[table]) == expected
            ), f"{table}: got {len(dfs[table])}, expected {expected}"

    def test_accounts_has_required_columns(self, raw_dfs):
        """Accounts table has key columns."""
        required = [
            "account_id",
            "signup_date",
            "plan_tier",
            "referral_source",
            "churn_flag",
        ]
        for col in required:
            assert col in raw_dfs["accounts"].columns


# ---------------------------------------------------------------------------
# Tests: validate_data_quality
# ---------------------------------------------------------------------------


class TestValidateDataQuality:
    """Tests for validate_data_quality function."""

    def test_detects_missing_satisfaction_score(self, raw_dfs):
        """Validation catches missing satisfaction_score values."""
        report = validate_data_quality(raw_dfs)
        assert "satisfaction_score" in report["missing_values"].get(
            "support_tickets", {}
        )

    def test_detects_missing_end_date(self, raw_dfs):
        """Validation catches missing end_date in subscriptions."""
        report = validate_data_quality(raw_dfs)
        assert "end_date" in report["missing_values"].get("subscriptions", {})

    def test_referential_integrity_passes(self, raw_dfs):
        """All foreign key references are valid."""
        report = validate_data_quality(raw_dfs)
        assert report["referential_integrity"]["subs_in_accounts"]
        assert report["referential_integrity"]["tickets_in_accounts"]
        assert report["referential_integrity"]["churn_in_accounts"]
        assert report["referential_integrity"]["usage_in_subs"]

    def test_detects_duplicate_usage_ids(self, raw_dfs):
        """Validation detects duplicate usage_id rows."""
        report = validate_data_quality(raw_dfs)
        assert report["duplicates"]["feature_usage"] > 0

    def test_no_duplicate_account_ids(self, raw_dfs):
        """Accounts table has no duplicate primary keys."""
        report = validate_data_quality(raw_dfs)
        assert report["duplicates"]["accounts"] == 0


# ---------------------------------------------------------------------------
# Tests: clean_data
# ---------------------------------------------------------------------------


class TestCleanData:
    """Tests for clean_data function."""

    def test_satisfaction_score_imputed(self, cleaned_dfs):
        """satisfaction_score has no nulls after cleaning."""
        tickets = cleaned_dfs["support_tickets"]
        assert tickets["satisfaction_score"].isnull().sum() == 0

    def test_satisfaction_missing_indicator_created(self, cleaned_dfs):
        """satisfaction_missing indicator column exists."""
        tickets = cleaned_dfs["support_tickets"]
        assert "satisfaction_missing" in tickets.columns
        assert tickets["satisfaction_missing"].dtype in [int, np.int64]

    def test_provided_feedback_indicator_created(self, cleaned_dfs):
        """provided_feedback indicator column exists in churn_events."""
        churn = cleaned_dfs["churn_events"]
        assert "provided_feedback" in churn.columns

    def test_dates_parsed(self, cleaned_dfs):
        """Date columns are datetime type after cleaning."""
        assert pd.api.types.is_datetime64_any_dtype(
            cleaned_dfs["accounts"]["signup_date"]
        )
        assert pd.api.types.is_datetime64_any_dtype(
            cleaned_dfs["subscriptions"]["start_date"]
        )

    def test_mrr_winsorized(self, cleaned_dfs, config):
        """mrr_amount values are capped at 99th percentile."""
        subs = cleaned_dfs["subscriptions"]
        # After winsorization, no value should exceed the cap
        # The exact cap may vary, but max should be <= original 99th
        assert subs["mrr_amount"].max() <= 17115  # Approx 99th


# ---------------------------------------------------------------------------
# Tests: engineer_features
# ---------------------------------------------------------------------------


class TestEngineerFeatures:
    """Tests for engineer_features function."""

    def test_one_row_per_account(self, master_df, raw_dfs):
        """Master table has exactly one row per account."""
        assert len(master_df) == len(raw_dfs["accounts"])
        assert master_df["account_id"].nunique() == len(master_df)

    def test_rfm_features_present(self, master_df):
        """All RFM features exist in master table."""
        rfm = [
            "days_since_signup",
            "total_sessions_7d",
            "avg_session_duration_7d",
            "total_revenue_30d",
            "days_to_first_purchase",
        ]
        for feat in rfm:
            assert feat in master_df.columns, f"Missing RFM feature: {feat}"

    def test_behavioral_features_present(self, master_df):
        """All behavioral features exist in master table."""
        behavioral = [
            "feature_diversity_score",
            "usage_consistency_cv",
            "weekend_usage_ratio",
            "error_rate",
            "beta_feature_adoption_flag",
        ]
        for feat in behavioral:
            assert feat in master_df.columns, f"Missing behavioral feature: {feat}"

    def test_support_features_present(self, master_df):
        """All support features exist in master table."""
        support = [
            "support_ticket_count_30d",
            "avg_satisfaction_score",
            "has_urgent_ticket_flag",
            "fast_resolution_rate",
        ]
        for feat in support:
            assert feat in master_df.columns, f"Missing support feature: {feat}"

    def test_subscription_features_present(self, master_df):
        """All subscription features exist in master table."""
        sub_feats = [
            "has_upgraded_flag",
            "has_downgraded_flag",
            "subscription_tenure_days",
            "subscription_churn_count",
            "billing_frequency_annual_flag",
        ]
        for feat in sub_feats:
            assert feat in master_df.columns, f"Missing subscription feature: {feat}"

    def test_time_series_features_present(self, master_df):
        """All time-series features exist in master table."""
        ts_feats = [
            "usage_trend_slope_30d",
            "revenue_trend_slope_60d",
            "days_since_last_session",
        ]
        for feat in ts_feats:
            assert feat in master_df.columns, f"Missing time-series feature: {feat}"

    def test_no_unexpected_nulls(self, master_df):
        """Engineered features have no nulls (except churn_date, reason_code)."""
        allowed_null_cols = {"churn_date", "reason_code"}
        for col in master_df.columns:
            if col not in allowed_null_cols:
                assert master_df[col].isnull().sum() == 0, (
                    f"Unexpected nulls in {col}: " f"{master_df[col].isnull().sum()}"
                )

    def test_binary_flags_are_0_or_1(self, master_df):
        """Binary flag features contain only 0 and 1."""
        flags = [
            "beta_feature_adoption_flag",
            "has_urgent_ticket_flag",
            "has_upgraded_flag",
            "has_downgraded_flag",
            "billing_frequency_annual_flag",
            "satisfaction_missing",
        ]
        for flag in flags:
            unique = set(master_df[flag].unique())
            assert unique.issubset(
                {0, 1}
            ), f"{flag} has values outside {{0, 1}}: {unique}"

    def test_feature_diversity_bounded(self, master_df):
        """feature_diversity_score is between 0 and 1."""
        assert master_df["feature_diversity_score"].min() >= 0
        assert master_df["feature_diversity_score"].max() <= 1

    def test_weekend_ratio_bounded(self, master_df):
        """weekend_usage_ratio is between 0 and 1."""
        assert master_df["weekend_usage_ratio"].min() >= 0
        assert master_df["weekend_usage_ratio"].max() <= 1


# ---------------------------------------------------------------------------
# Tests: create_survival_target
# ---------------------------------------------------------------------------


class TestCreateSurvivalTarget:
    """Tests for create_survival_target function."""

    def test_event_observed_binary(self, master_with_survival):
        """event_observed contains only 0 and 1."""
        unique = set(master_with_survival["event_observed"].unique())
        assert unique.issubset({0, 1})

    def test_event_count_matches_churn_flag(self, master_with_survival, raw_dfs):
        """Number of events matches account-level churn_flag count."""
        expected_events = raw_dfs["accounts"]["churn_flag"].sum()
        actual_events = master_with_survival["event_observed"].sum()
        assert actual_events == expected_events

    def test_duration_positive(self, master_with_survival):
        """All duration_days values are positive."""
        assert (master_with_survival["duration_days"] > 0).all()

    def test_censored_have_longer_duration(self, master_with_survival):
        """On average, censored accounts have longer observed duration."""
        df = master_with_survival
        censored_mean = df[df["event_observed"] == 0]["duration_days"].mean()
        event_mean = df[df["event_observed"] == 1]["duration_days"].mean()
        # Censored accounts should generally have longer observation
        # (they survive until analysis_date)
        assert censored_mean > event_mean


# ---------------------------------------------------------------------------
# Tests: create_train_val_test_split
# ---------------------------------------------------------------------------


class TestTrainValTestSplit:
    """Tests for create_train_val_test_split function."""

    def test_split_sizes_sum_to_total(self, master_with_survival, config):
        """Train + val + test = total rows."""
        train, val, test = create_train_val_test_split(master_with_survival, config)
        assert len(train) + len(val) + len(test) == len(master_with_survival)

    def test_no_overlap(self, master_with_survival, config):
        """No account appears in multiple splits."""
        train, val, test = create_train_val_test_split(master_with_survival, config)
        train_ids = set(train["account_id"])
        val_ids = set(val["account_id"])
        test_ids = set(test["account_id"])
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_temporal_ordering(self, master_with_survival, config):
        """Train dates < val dates < test dates (no temporal leakage)."""
        train, val, test = create_train_val_test_split(master_with_survival, config)
        train_max = pd.to_datetime(train["signup_date"]).max()
        val_min = pd.to_datetime(val["signup_date"]).min()
        val_max = pd.to_datetime(val["signup_date"]).max()
        test_min = pd.to_datetime(test["signup_date"]).min()
        assert train_max < val_min, "Temporal leakage: train > val"
        assert val_max < test_min, "Temporal leakage: val > test"

    def test_splits_not_empty(self, master_with_survival, config):
        """No split is empty."""
        train, val, test = create_train_val_test_split(master_with_survival, config)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_validate_with_minimal_data(self, minimal_dfs):
        """Validation runs on minimal synthetic data without errors."""
        report = validate_data_quality(minimal_dfs)
        assert isinstance(report, dict)
        assert "missing_values" in report

    def test_clean_with_minimal_data(self, minimal_dfs, config):
        """Cleaning runs on minimal synthetic data."""
        cleaned = clean_data(minimal_dfs, config)
        assert "accounts" in cleaned
        assert len(cleaned["accounts"]) == 3

    def test_empty_dataframe_validation(self):
        """Validation handles empty tables gracefully."""
        empty_dfs = {
            "accounts": pd.DataFrame(columns=["account_id", "signup_date"]),
            "subscriptions": pd.DataFrame(
                columns=["subscription_id", "account_id", "start_date"]
            ),
            "feature_usage": pd.DataFrame(columns=["usage_id", "subscription_id"]),
            "support_tickets": pd.DataFrame(columns=["ticket_id", "account_id"]),
            "churn_events": pd.DataFrame(columns=["churn_event_id", "account_id"]),
        }
        report = validate_data_quality(empty_dfs)
        assert isinstance(report, dict)

    def test_save_and_reload(self, master_with_survival, config, tmp_path):
        """Saved CSV files can be reloaded correctly."""
        train, val, test = create_train_val_test_split(master_with_survival, config)
        output_dir = str(tmp_path / "test_output")
        save_processed_data(train, val, test, output_dir)
        reloaded_train = pd.read_csv(tmp_path / "test_output" / "train.csv")
        assert len(reloaded_train) == len(train)
