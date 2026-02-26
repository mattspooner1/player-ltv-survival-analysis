"""Quick data inspection script."""

import pandas as pd
import os

os.chdir(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

a = pd.read_csv("data/raw/ravenstack_accounts.csv")
s = pd.read_csv("data/raw/ravenstack_subscriptions.csv")
u = pd.read_csv("data/raw/ravenstack_feature_usage.csv")
t = pd.read_csv("data/raw/ravenstack_support_tickets.csv")
c = pd.read_csv("data/raw/ravenstack_churn_events.csv")

print("=== ACCOUNTS ===")
print("plan_tier:", a["plan_tier"].value_counts().to_dict())
print("referral_source:", a["referral_source"].value_counts().to_dict())
print("industry:", a["industry"].value_counts().to_dict())
print("country:", a["country"].value_counts().to_dict())
print("churn_flag:", a["churn_flag"].value_counts().to_dict())
print("is_trial:", a["is_trial"].value_counts().to_dict())
print("seats:", a["seats"].describe().to_dict())
print("signup range:", a["signup_date"].min(), "-", a["signup_date"].max())
print()

print("=== SUBSCRIPTIONS ===")
print("plan_tier:", s["plan_tier"].value_counts().to_dict())
print("billing:", s["billing_frequency"].value_counts().to_dict())
print("churn:", s["churn_flag"].value_counts().to_dict())
print("mrr:", s["mrr_amount"].describe().to_dict())
print("end_date nulls:", s["end_date"].isnull().sum(), "of", len(s))
print("upgrades:", s["upgrade_flag"].sum(), "downgrades:", s["downgrade_flag"].sum())
print("start range:", s["start_date"].min(), "-", s["start_date"].max())
print()

print("=== USAGE ===")
print("features unique:", u["feature_name"].nunique())
print("usage_count:", u["usage_count"].describe().to_dict())
print("duration:", u["usage_duration_secs"].describe().to_dict())
print("error_count:", u["error_count"].describe().to_dict())
print("beta:", u["is_beta_feature"].sum())
print("date range:", u["usage_date"].min(), "-", u["usage_date"].max())
print()

print("=== TICKETS ===")
print("priority:", t["priority"].value_counts().to_dict())
print("satisfaction:", t["satisfaction_score"].describe().to_dict())
print("satisfaction nulls:", t["satisfaction_score"].isnull().sum())
print("resolution:", t["resolution_time_hours"].describe().to_dict())
print("escalation:", t["escalation_flag"].sum())
print()

print("=== CHURN ===")
print("reason:", c["reason_code"].value_counts().to_dict())
print("reactivation:", c["is_reactivation"].value_counts().to_dict())
print("feedback nulls:", c["feedback_text"].isnull().sum(), "of", len(c))
print("refund:", c["refund_amount_usd"].describe().to_dict())
print("date range:", c["churn_date"].min(), "-", c["churn_date"].max())
print()

print("Unique accounts in churn:", c["account_id"].nunique())
print("Total accounts:", len(a))
print("Accounts with churn=True:", a["churn_flag"].sum())

# Check referential integrity
acct_ids = set(a["account_id"])
sub_acct_ids = set(s["account_id"])
ticket_acct_ids = set(t["account_id"])
churn_acct_ids = set(c["account_id"])
print()
print("Sub accounts not in accounts:", len(sub_acct_ids - acct_ids))
print("Ticket accounts not in accounts:", len(ticket_acct_ids - acct_ids))
print("Churn accounts not in accounts:", len(churn_acct_ids - acct_ids))

# Check subscription to usage link
sub_ids = set(s["subscription_id"])
usage_sub_ids = set(u["subscription_id"])
print("Usage subs not in subs:", len(usage_sub_ids - sub_ids))

# Subs per account
print("Subs per account:", s.groupby("account_id").size().describe().to_dict())
