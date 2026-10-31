"""Synthetic raw tables for tests.

The real dataset is not committed to the repository (see data/README.md), so
the data-processing tests run against small synthetic tables that match the
schema and row counts in config.yaml. The tables are deterministic for a given
seed, keep referential integrity, and include the data quality quirks the
tests check for: missing satisfaction scores, open-ended subscriptions, and
duplicate usage IDs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANALYSIS_DATE = pd.Timestamp("2024-12-31")
SIGNUP_START = pd.Timestamp("2023-01-01")
SIGNUP_END = pd.Timestamp("2024-11-30")


def _random_dates(
    rng: np.random.Generator,
    start: pd.Series | pd.Timestamp,
    max_offset_days: int,
    n: int,
) -> pd.Series:
    """Return n dates at a random offset after start, capped at ANALYSIS_DATE."""
    offsets = pd.to_timedelta(rng.integers(0, max_offset_days + 1, size=n), unit="D")
    base = pd.Series(start) if isinstance(start, pd.Timestamp) else start
    dates = pd.Series(pd.to_datetime(base.values) + offsets.values)
    return dates.clip(upper=ANALYSIS_DATE)


def make_raw_tables(
    expected_counts: dict[str, int], seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Build the five raw tables with the row counts the config expects.

    Args:
        expected_counts: Row counts per table (config data.expected_row_counts).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary of raw DataFrames keyed by table name.
    """
    rng = np.random.default_rng(seed)
    n_acc = expected_counts["accounts"]
    n_sub = expected_counts["subscriptions"]
    n_use = expected_counts["feature_usage"]
    n_tix = expected_counts["support_tickets"]
    n_churn = expected_counts["churn_events"]

    # --- accounts -----------------------------------------------------------
    span_days = (SIGNUP_END - SIGNUP_START).days
    signup = SIGNUP_START + pd.to_timedelta(
        rng.integers(0, span_days + 1, size=n_acc), unit="D"
    )
    churned = np.zeros(n_acc, dtype=bool)
    churned[rng.choice(n_acc, size=int(n_acc * 0.22), replace=False)] = True
    accounts = pd.DataFrame(
        {
            "account_id": [f"A-{i:04d}" for i in range(n_acc)],
            "account_name": [f"Company{i}" for i in range(n_acc)],
            "industry": rng.choice(["FinTech", "EdTech", "DevTools", "Retail"], n_acc),
            "country": rng.choice(["US", "UK", "IN", "DE"], n_acc),
            "signup_date": signup,
            "referral_source": rng.choice(
                ["organic", "ads", "partner", "event", "other"], n_acc
            ),
            "plan_tier": rng.choice(["Basic", "Pro", "Enterprise"], n_acc),
            "seats": rng.integers(1, 50, n_acc),
            "is_trial": rng.random(n_acc) < 0.2,
            "churn_flag": churned,
        }
    )

    # --- subscriptions ------------------------------------------------------
    sub_acc_idx = np.concatenate(
        [np.arange(n_acc), rng.integers(0, n_acc, size=n_sub - n_acc)]
    )
    sub_signup = accounts["signup_date"].iloc[sub_acc_idx].reset_index(drop=True)
    sub_start = _random_dates(rng, sub_signup, 20, n_sub)
    open_ended = rng.random(n_sub) < 0.7
    sub_end = _random_dates(rng, sub_start, 200, n_sub)
    sub_end = sub_end.where(~open_ended, other=pd.NaT)
    mrr = np.round(rng.gamma(2.0, 400.0, n_sub), 2)
    subscriptions = pd.DataFrame(
        {
            "subscription_id": [f"S-{i:05d}" for i in range(n_sub)],
            "account_id": accounts["account_id"].iloc[sub_acc_idx].values,
            "start_date": sub_start,
            "end_date": sub_end,
            "plan_tier": accounts["plan_tier"].iloc[sub_acc_idx].values,
            "seats": accounts["seats"].iloc[sub_acc_idx].values,
            "mrr_amount": mrr,
            "arr_amount": np.round(mrr * 12, 2),
            "is_trial": rng.random(n_sub) < 0.1,
            "upgrade_flag": rng.random(n_sub) < 0.15,
            "downgrade_flag": rng.random(n_sub) < 0.1,
            "churn_flag": ~open_ended,
            "billing_frequency": rng.choice(["monthly", "annual"], n_sub),
            "auto_renew_flag": rng.random(n_sub) < 0.8,
        }
    )

    # --- feature usage ------------------------------------------------------
    use_sub_idx = np.concatenate(
        [np.arange(n_sub), rng.integers(0, n_sub, size=n_use - n_sub)]
    )[:n_use]
    use_start = subscriptions["start_date"].iloc[use_sub_idx].reset_index(drop=True)
    usage_ids = [f"U-{i:06d}" for i in range(n_use)]
    for dup in range(1, 41):  # duplicate IDs the validator should detect
        usage_ids[dup] = usage_ids[0]
    feature_usage = pd.DataFrame(
        {
            "usage_id": usage_ids,
            "subscription_id": subscriptions["subscription_id"]
            .iloc[use_sub_idx]
            .values,
            "usage_date": _random_dates(rng, use_start, 150, n_use),
            "feature_name": rng.choice([f"feature_{i}" for i in range(1, 9)], n_use),
            "usage_count": rng.integers(1, 40, n_use),
            "usage_duration_secs": rng.integers(30, 3600, n_use),
            "error_count": rng.poisson(0.3, n_use),
            "is_beta_feature": rng.random(n_use) < 0.15,
        }
    )

    # --- support tickets ----------------------------------------------------
    tix_acc_idx = rng.integers(0, n_acc, size=n_tix)
    submitted = _random_dates(
        rng,
        accounts["signup_date"].iloc[tix_acc_idx].reset_index(drop=True),
        120,
        n_tix,
    )
    resolution = np.round(rng.gamma(2.0, 12.0, n_tix), 1)
    satisfaction = rng.integers(1, 6, n_tix).astype(float)
    satisfaction[rng.random(n_tix) < 0.4] = np.nan
    support_tickets = pd.DataFrame(
        {
            "ticket_id": [f"T-{i:05d}" for i in range(n_tix)],
            "account_id": accounts["account_id"].iloc[tix_acc_idx].values,
            "submitted_at": submitted,
            "closed_at": submitted + pd.to_timedelta(resolution, unit="h"),
            "resolution_time_hours": resolution,
            "priority": rng.choice(["low", "medium", "high", "urgent"], n_tix),
            "first_response_time_minutes": rng.integers(5, 240, n_tix),
            "satisfaction_score": satisfaction,
            "escalation_flag": rng.random(n_tix) < 0.1,
        }
    )

    # --- churn events -------------------------------------------------------
    churned_idx = np.flatnonzero(churned)
    extra_idx = rng.integers(0, n_acc, size=n_churn - len(churned_idx))
    ev_acc_idx = np.concatenate([churned_idx, extra_idx])
    ev_signup = accounts["signup_date"].iloc[ev_acc_idx].reset_index(drop=True)
    ev_date = ev_signup + pd.to_timedelta(rng.integers(30, 240, size=n_churn), unit="D")
    reasons = rng.choice(["pricing", "support", "features", "competitor"], n_churn)
    feedback = np.where(rng.random(n_churn) < 0.5, "feedback text", None)
    churn_events = pd.DataFrame(
        {
            "churn_event_id": [f"C-{i:04d}" for i in range(n_churn)],
            "account_id": accounts["account_id"].iloc[ev_acc_idx].values,
            "churn_date": ev_date.clip(upper=ANALYSIS_DATE),
            "reason_code": reasons,
            "refund_amount_usd": np.round(rng.gamma(1.5, 40.0, n_churn), 2),
            "preceding_upgrade_flag": rng.random(n_churn) < 0.1,
            "preceding_downgrade_flag": rng.random(n_churn) < 0.1,
            "is_reactivation": rng.random(n_churn) < 0.1,
            "feedback_text": feedback,
        }
    )

    return {
        "accounts": accounts,
        "subscriptions": subscriptions,
        "feature_usage": feature_usage,
        "support_tickets": support_tickets,
        "churn_events": churn_events,
    }
