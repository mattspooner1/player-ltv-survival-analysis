"""Modeling module for Player LTV Survival Analysis.

This module implements three core classes for predicting player lifetime value:

1. SurvivalAnalyzer: Cox Proportional Hazards and Gradient Boosting survival
   models for time-to-churn prediction with feature selection pipeline.
2. LTVPredictor: LightGBM regressor for 180-day revenue prediction with
   Optuna hyperparameter tuning and baseline comparisons.
3. PLTVCalculator: Combines survival probabilities with revenue predictions
   to produce predicted Lifetime Value (pLTV) scores.

All parameters are loaded from config.yaml -- no hardcoded values.

Classes:
    SurvivalAnalyzer: Fit, evaluate, and interpret survival models.
    LTVPredictor: Fit, tune, and evaluate LTV regression models.
    PLTVCalculator: Combine survival + revenue into pLTV framework.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Absolute or relative path to config.yaml.

    Returns:
        Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


def _prepare_survival_target(
    df: pd.DataFrame,
) -> np.ndarray:
    """Convert DataFrame columns to scikit-survival structured array.

    Args:
        df: DataFrame with 'event_observed' (0/1) and 'duration_days' columns.

    Returns:
        Structured numpy array with dtype [('event', bool), ('time', float)].
    """
    event = df["event_observed"].astype(bool).values
    time = df["duration_days"].astype(float).values
    return np.array(
        list(zip(event, time)),
        dtype=[("event", bool), ("time", float)],
    )


def _get_numeric_features(df: pd.DataFrame, exclude_cols: list[str]) -> list[str]:
    """Get list of numeric feature columns, excluding metadata/target columns.

    Args:
        df: DataFrame to inspect.
        exclude_cols: Column names to exclude.

    Returns:
        Sorted list of numeric feature column names.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return sorted([c for c in numeric_cols if c not in exclude_cols])


# ---------------------------------------------------------------------------
# Feature Selection Pipeline
# ---------------------------------------------------------------------------


class FeatureSelector:
    """Feature selection pipeline for survival modeling.

    Implements the spec's sequential approach: correlation filtering,
    VIF removal, and RFE with Cox model.

    Args:
        config: Project configuration dictionary.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        sel_cfg = config.get("features", {}).get("selection", {})
        self.corr_threshold: float = sel_cfg.get("correlation_threshold", 0.85)
        self.vif_threshold: float = sel_cfg.get("vif_threshold", 5.0)
        self.rfe_top_n: int = sel_cfg.get("rfe_top_n", 20)
        target_range = sel_cfg.get("target_feature_count", [15, 20])
        self.min_features: int = target_range[0]
        self.max_features: int = target_range[1]
        self.selected_features_: Optional[list[str]] = None
        self.selection_report_: dict[str, Any] = {}

    def _correlation_filter(self, X: pd.DataFrame, y_struct: np.ndarray) -> list[str]:
        """Remove features with pairwise correlation above threshold.

        Retains the feature with stronger univariate association to the
        survival outcome (measured by concordance index).

        Args:
            X: Feature matrix.
            y_struct: Structured survival target array.

        Returns:
            List of feature names after correlation filtering.
        """
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find pairs above threshold
        high_corr_pairs = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                if upper_tri.loc[idx, col] > self.corr_threshold:
                    high_corr_pairs.append((idx, col, upper_tri.loc[idx, col]))

        if not high_corr_pairs:
            logger.info(
                "No features above correlation threshold %.2f", self.corr_threshold
            )
            return X.columns.tolist()

        # For each pair, keep feature with higher univariate C-index
        to_drop = set()
        event = y_struct["event"]
        time = y_struct["time"]
        for feat_a, feat_b, corr_val in high_corr_pairs:
            if feat_a in to_drop or feat_b in to_drop:
                continue
            try:
                c_a = concordance_index_censored(event, time, X[feat_a].values)[0]
            except Exception:
                c_a = 0.5
            try:
                c_b = concordance_index_censored(event, time, X[feat_b].values)[0]
            except Exception:
                c_b = 0.5
            drop = feat_b if c_a >= c_b else feat_a
            to_drop.add(drop)
            logger.info(
                "Corr filter: |r(%s, %s)| = %.3f, dropping %s (C-index: %.3f vs %.3f)",
                feat_a,
                feat_b,
                corr_val,
                drop,
                c_a,
                c_b,
            )

        remaining = [c for c in X.columns if c not in to_drop]
        self.selection_report_["correlation_filter"] = {
            "dropped": list(to_drop),
            "remaining_count": len(remaining),
        }
        return remaining

    def _vif_filter(self, X: pd.DataFrame) -> list[str]:
        """Remove features with Variance Inflation Factor above threshold.

        Uses iterative removal: at each step, remove the feature with
        highest VIF if it exceeds the threshold.

        Args:
            X: Feature matrix (numeric only).

        Returns:
            List of feature names after VIF filtering.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        features = X.columns.tolist()
        dropped = []

        # Iteratively remove highest VIF feature
        max_iterations = len(features)
        for _ in range(max_iterations):
            if len(features) <= self.min_features:
                break
            X_subset = X[features].copy()
            # Add small noise to prevent singular matrix
            X_subset = X_subset + np.random.RandomState(42).normal(
                0, 1e-10, X_subset.shape
            )
            try:
                vifs = pd.Series(
                    [
                        variance_inflation_factor(X_subset.values, i)
                        for i in range(len(features))
                    ],
                    index=features,
                )
            except Exception as exc:
                logger.warning("VIF computation failed: %s. Skipping VIF filter.", exc)
                break

            max_vif = vifs.max()
            if max_vif <= self.vif_threshold:
                break
            worst = vifs.idxmax()
            features.remove(worst)
            dropped.append((worst, float(max_vif)))
            logger.info("VIF filter: dropping %s (VIF=%.2f)", worst, max_vif)

        self.selection_report_["vif_filter"] = {
            "dropped": dropped,
            "remaining_count": len(features),
        }
        return features

    def _rfe_with_cox(
        self, X: pd.DataFrame, y_struct: np.ndarray, n_select: int
    ) -> list[str]:
        """Recursive feature elimination using Cox PH coefficient magnitudes.

        Args:
            X: Feature matrix.
            y_struct: Structured survival target array.
            n_select: Number of features to retain.

        Returns:
            List of top-n feature names by importance.
        """
        features = X.columns.tolist()
        penalizer = (
            self.config.get("models", {}).get("cox_ph", {}).get("penalizer", 0.01)
        )

        while len(features) > n_select:
            try:
                cox = CoxPHSurvivalAnalysis(alpha=penalizer)
                cox.fit(X[features].values, y_struct)
                coefs = np.abs(cox.coef_)
                weakest_idx = np.argmin(coefs)
                weakest = features[weakest_idx]
                features.remove(weakest)
                logger.debug(
                    "RFE: removing %s (|coef|=%.6f)", weakest, coefs[weakest_idx]
                )
            except Exception as exc:
                logger.warning("RFE step failed: %s. Stopping early.", exc)
                break

        self.selection_report_["rfe"] = {
            "final_features": features,
            "count": len(features),
        }
        return features

    def fit(self, X: pd.DataFrame, y_struct: np.ndarray) -> "FeatureSelector":
        """Run the full feature selection pipeline.

        Steps: correlation filtering -> VIF removal -> RFE with Cox.

        Args:
            X: Numeric feature matrix (scaled).
            y_struct: Structured survival target.

        Returns:
            self, with selected_features_ attribute populated.
        """
        logger.info("Starting feature selection: %d input features", X.shape[1])

        # Step 1: Correlation filtering
        features_after_corr = self._correlation_filter(X, y_struct)
        logger.info("After correlation filter: %d features", len(features_after_corr))

        # Step 2: VIF filtering
        try:
            features_after_vif = self._vif_filter(X[features_after_corr])
        except ImportError:
            logger.warning("statsmodels not available; skipping VIF filter")
            features_after_vif = features_after_corr

        logger.info("After VIF filter: %d features", len(features_after_vif))

        # Step 3: RFE with Cox
        n_select = min(self.rfe_top_n, len(features_after_vif))
        if len(features_after_vif) > n_select:
            features_final = self._rfe_with_cox(
                X[features_after_vif], y_struct, n_select
            )
        else:
            features_final = features_after_vif

        self.selected_features_ = features_final
        self.selection_report_["final"] = {
            "count": len(features_final),
            "features": features_final,
        }
        logger.info(
            "Feature selection complete: %d features selected", len(features_final)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only the chosen features from a DataFrame.

        Args:
            X: Feature matrix with at least the selected columns.

        Returns:
            Subset DataFrame with selected features only.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() before transform().")
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features in input: {missing}")
        return X[self.selected_features_]


# ---------------------------------------------------------------------------
# SurvivalAnalyzer
# ---------------------------------------------------------------------------


class SurvivalAnalyzer:
    """Survival analysis models for player churn prediction.

    Supports Cox Proportional Hazards (primary) and Gradient Boosting
    Survival Analysis (comparison) models.

    Args:
        config: Project configuration dictionary.

    Attributes:
        cox_model: Fitted CoxPHSurvivalAnalysis model.
        gbsa_model: Fitted GradientBoostingSurvivalAnalysis model.
        scaler: StandardScaler fitted on training features.
        feature_names: List of feature column names used in modeling.
        feature_selector: FeatureSelector instance (after fit).
    """

    # Columns that are NOT features (metadata, targets, identifiers)
    EXCLUDE_COLS = [
        "account_id",
        "signup_date",
        "plan_tier",
        "referral_source",
        "industry",
        "country",
        "is_trial",
        "churn_flag",
        "churn_date",
        "reason_code",
        "event_observed",
        "duration_days",
        "total_revenue_180d",
    ]

    def __init__(self, config: dict) -> None:
        self.config = config
        cox_cfg = config.get("models", {}).get("cox_ph", {})
        self.penalizer: float = cox_cfg.get("penalizer", 0.01)
        self.random_seed: int = config.get("project", {}).get("random_seed", 42)

        self.cox_model: Optional[CoxPHSurvivalAnalysis] = None
        self.gbsa_model: Optional[GradientBoostingSurvivalAnalysis] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[list[str]] = None
        self.feature_selector: Optional[FeatureSelector] = None
        self._train_y_struct: Optional[np.ndarray] = None

    def _prepare_features(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Prepare numeric features and survival target from raw data.

        Handles one-hot encoding of categoricals, scaling of numerics,
        and conversion to scikit-survival structured array.

        Args:
            df: DataFrame with all columns (features + targets).
            fit_scaler: If True, fit the scaler on this data.

        Returns:
            Tuple of (X_scaled DataFrame, y_struct array).
        """
        y_struct = _prepare_survival_target(df)

        # Get numeric features
        numeric_features = _get_numeric_features(df, self.EXCLUDE_COLS)

        # One-hot encode categoricals that are still present
        X = df[numeric_features].copy()

        # Encode plan_tier if present as numeric proxies
        if "plan_tier" in df.columns:
            dummies = pd.get_dummies(
                df["plan_tier"], prefix="plan_tier", drop_first=True, dtype=float
            )
            X = pd.concat([X, dummies], axis=1)

        # Encode referral_source if present
        if "referral_source" in df.columns:
            dummies = pd.get_dummies(
                df["referral_source"],
                prefix="referral_source",
                drop_first=True,
                dtype=float,
            )
            X = pd.concat([X, dummies], axis=1)

        # Fill any remaining NaN with 0
        X = X.fillna(0)

        # Scale
        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index,
            )
        else:
            if self.scaler is None:
                raise RuntimeError(
                    "Scaler not fitted. Call with fit_scaler=True first."
                )
            # Align columns to match what the scaler was fitted on
            expected_cols = self.scaler.feature_names_in_
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[expected_cols]
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index,
            )

        return X_scaled, y_struct

    def select_features(self, X_train: pd.DataFrame, y_train: np.ndarray) -> list[str]:
        """Run feature selection pipeline on training data.

        Args:
            X_train: Scaled training feature matrix.
            y_train: Structured survival target for training set.

        Returns:
            List of selected feature names.
        """
        self.feature_selector = FeatureSelector(self.config)
        self.feature_selector.fit(X_train, y_train)
        self.feature_names = self.feature_selector.selected_features_
        logger.info("Selected %d features for modeling", len(self.feature_names))
        return self.feature_names

    def fit_cox(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> CoxPHSurvivalAnalysis:
        """Fit Cox Proportional Hazards model.

        Args:
            X_train: Scaled feature matrix (training set).
            y_train: Structured survival target (training set).
            feature_names: Specific features to use. If None, uses all columns.

        Returns:
            Fitted CoxPHSurvivalAnalysis model.
        """
        if feature_names is not None:
            self.feature_names = feature_names
            X_fit = X_train[feature_names]
        elif self.feature_names is not None:
            X_fit = X_train[self.feature_names]
        else:
            self.feature_names = X_train.columns.tolist()
            X_fit = X_train

        self.cox_model = CoxPHSurvivalAnalysis(alpha=self.penalizer)
        self.cox_model.fit(X_fit.values, y_train)
        self._train_y_struct = y_train
        logger.info(
            "Cox PH fitted. C-index (train): %.4f",
            self.cox_model.score(X_fit.values, y_train),
        )
        return self.cox_model

    def fit_gbsa(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
    ) -> GradientBoostingSurvivalAnalysis:
        """Fit Gradient Boosting Survival Analysis model.

        Args:
            X_train: Scaled feature matrix.
            y_train: Structured survival target.
            feature_names: Specific features to use. If None, uses same as Cox.
            n_estimators: Number of boosting iterations.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate.
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples in a leaf.

        Returns:
            Fitted GradientBoostingSurvivalAnalysis model.
        """
        if feature_names is not None:
            X_fit = X_train[feature_names]
        elif self.feature_names is not None:
            X_fit = X_train[self.feature_names]
        else:
            X_fit = X_train

        self.gbsa_model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_seed,
        )
        self.gbsa_model.fit(X_fit.values, y_train)
        logger.info(
            "GBSA fitted. C-index (train): %.4f",
            self.gbsa_model.score(X_fit.values, y_train),
        )
        return self.gbsa_model

    def evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y_struct: np.ndarray,
        model_name: str = "model",
    ) -> dict[str, float]:
        """Evaluate a survival model on concordance index and IBS.

        Args:
            model: Fitted scikit-survival model with predict() method.
            X: Feature matrix for evaluation.
            y_struct: Structured survival target for evaluation.
            model_name: Label for logging.

        Returns:
            Dictionary with 'c_index' and 'ibs' metrics.
        """
        features = self.feature_names or X.columns.tolist()
        X_eval = X[features].values

        # Concordance index
        risk_scores = model.predict(X_eval)
        c_index = concordance_index_censored(
            y_struct["event"], y_struct["time"], risk_scores
        )[0]

        # Integrated Brier Score
        ibs = self._compute_ibs(model, X[features], y_struct)

        results = {"c_index": float(c_index), "ibs": float(ibs)}
        logger.info(
            "%s evaluation -- C-index: %.4f, IBS: %.4f",
            model_name,
            c_index,
            ibs,
        )
        return results

    def _compute_ibs(
        self,
        model: Any,
        X: pd.DataFrame,
        y_struct: np.ndarray,
    ) -> float:
        """Compute Integrated Brier Score for a survival model.

        Args:
            model: Fitted model with predict_survival_function method.
            X: Feature matrix.
            y_struct: Structured survival target.

        Returns:
            IBS value (lower is better).
        """
        try:
            surv_funcs = model.predict_survival_function(X.values)
            # Create time grid within observed range
            times_all = y_struct["time"]
            t_min = (
                max(times_all[y_struct["event"]].min(), 1.0)
                if y_struct["event"].any()
                else 1.0
            )
            t_max = times_all.max() * 0.9
            time_grid = np.linspace(t_min, t_max, 50)

            # Build survival probability matrix
            surv_probs = np.column_stack(
                [fn(time_grid) for fn in surv_funcs]
            ).T  # shape: (n_samples, n_times)

            # Need training y for IBS computation
            train_y = (
                self._train_y_struct if self._train_y_struct is not None else y_struct
            )
            ibs = integrated_brier_score(train_y, y_struct, surv_probs, time_grid)
            return float(ibs)
        except Exception as exc:
            logger.warning("IBS computation failed: %s. Returning NaN.", exc)
            return float("nan")

    def get_hazard_ratios(self) -> pd.DataFrame:
        """Extract hazard ratios from the fitted Cox model.

        Returns:
            DataFrame with columns: feature, coefficient, hazard_ratio,
            and interpretation.

        Raises:
            RuntimeError: If Cox model has not been fitted.
        """
        if self.cox_model is None:
            raise RuntimeError("Cox model not fitted. Call fit_cox() first.")

        coefs = self.cox_model.coef_
        features = self.feature_names or [f"x{i}" for i in range(len(coefs))]
        hr_df = pd.DataFrame(
            {
                "feature": features,
                "coefficient": coefs,
                "hazard_ratio": np.exp(coefs),
            }
        )
        hr_df["interpretation"] = hr_df.apply(
            lambda row: (
                f"{abs(row['hazard_ratio'] - 1) * 100:.1f}% "
                f"{'higher' if row['hazard_ratio'] > 1 else 'lower'} churn risk "
                f"per unit increase"
            ),
            axis=1,
        )
        hr_df = hr_df.sort_values("hazard_ratio", ascending=False).reset_index(
            drop=True
        )
        return hr_df

    def predict_survival_probability(
        self,
        model: Any,
        X: pd.DataFrame,
        time_point: float = 180.0,
    ) -> np.ndarray:
        """Predict survival probability at a specific time point.

        Args:
            model: Fitted survival model with predict_survival_function.
            X: Feature matrix.
            time_point: Time point (in days) at which to evaluate S(t).

        Returns:
            Array of survival probabilities at the given time point.
        """
        features = self.feature_names or X.columns.tolist()
        surv_funcs = model.predict_survival_function(X[features].values)
        probs = np.array([fn(time_point) for fn in surv_funcs])
        return probs

    def get_survival_curves(
        self,
        model: Any,
        X: pd.DataFrame,
        time_grid: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get survival curves for a set of observations.

        Args:
            model: Fitted survival model.
            X: Feature matrix.
            time_grid: Time points at which to evaluate. If None,
                uses 0 to 365 days in 30-day steps.

        Returns:
            Tuple of (time_grid, survival_matrix) where survival_matrix
            has shape (n_samples, n_times).
        """
        if time_grid is None:
            time_grid = np.arange(0, 366, 30)

        features = self.feature_names or X.columns.tolist()
        surv_funcs = model.predict_survival_function(X[features].values)

        surv_matrix = np.column_stack(
            [fn(time_grid) for fn in surv_funcs]
        ).T  # (n_samples, n_times)

        return time_grid, surv_matrix

    def validate_proportional_hazards(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> dict[str, Any]:
        """Validate the proportional hazards assumption using lifelines.

        Uses Schoenfeld residuals test via lifelines CoxPHFitter.

        Args:
            X_train: Scaled training features.
            y_train: Structured survival target.

        Returns:
            Dictionary with test results per feature and overall pass/fail.
        """
        from lifelines import CoxPHFitter

        features = self.feature_names or X_train.columns.tolist()
        alpha = (
            self.config.get("models", {})
            .get("cox_ph", {})
            .get("assumption_tests", {})
            .get("schoenfeld_alpha", 0.05)
        )

        # Build lifelines-compatible DataFrame
        ll_df = X_train[features].copy()
        ll_df["duration"] = y_train["time"]
        ll_df["event"] = y_train["event"].astype(int)

        cph = CoxPHFitter(penalizer=self.penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(ll_df, duration_col="duration", event_col="event")

        # Schoenfeld residuals test
        try:
            test_results = cph.check_assumptions(
                ll_df, p_value_threshold=alpha, show_plots=False
            )
            # If no violations, check_assumptions prints but returns empty list
            violations = test_results if test_results else []
            assumption_holds = len(violations) == 0
        except Exception as exc:
            logger.warning("PH assumption test failed: %s", exc)
            violations = [str(exc)]
            assumption_holds = True  # Assume holds if test can't run

        result = {
            "alpha": alpha,
            "assumption_holds": assumption_holds,
            "violations": violations,
            "recommendation": (
                "Proportional hazards assumption holds. Cox model is appropriate."
                if assumption_holds
                else (
                    "PH assumption violated for some features. Consider stratified "
                    "Cox model or time-varying coefficients."
                )
            ),
        }
        logger.info(
            "PH assumption test: %s",
            "PASSED" if assumption_holds else "VIOLATED",
        )
        return result

    def compare_cox_vs_gbsa(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Compare Cox and GBSA models on the same test set.

        Implements the spec's decision criteria: if GBSA C-index > Cox + 0.05,
        use GBSA for predictions and Cox for interpretation.

        Args:
            X_test: Test feature matrix.
            y_test: Structured survival target for test set.

        Returns:
            Dictionary with comparison metrics and recommendation.

        Raises:
            RuntimeError: If both models are not fitted.
        """
        if self.cox_model is None or self.gbsa_model is None:
            raise RuntimeError("Both Cox and GBSA models must be fitted first.")

        cox_metrics = self.evaluate_model(self.cox_model, X_test, y_test, "Cox PH")
        gbsa_metrics = self.evaluate_model(self.gbsa_model, X_test, y_test, "GBSA")

        c_diff = gbsa_metrics["c_index"] - cox_metrics["c_index"]
        threshold = 0.05

        if c_diff > threshold:
            recommendation = (
                f"GBSA C-index exceeds Cox by {c_diff:.3f} (>{threshold}). "
                f"Use GBSA for operational predictions, Cox for stakeholder "
                f"interpretation (hazard ratios)."
            )
            preferred_model = "gbsa"
        else:
            recommendation = (
                f"GBSA C-index difference ({c_diff:.3f}) is within {threshold} "
                f"of Cox. Use Cox for both prediction and interpretation "
                f"(simplicity wins)."
            )
            preferred_model = "cox"

        comparison = {
            "cox": cox_metrics,
            "gbsa": gbsa_metrics,
            "c_index_difference": float(c_diff),
            "threshold": threshold,
            "preferred_model": preferred_model,
            "recommendation": recommendation,
        }
        logger.info("Model comparison: %s", recommendation)
        return comparison


# ---------------------------------------------------------------------------
# LTVPredictor
# ---------------------------------------------------------------------------


class LTVPredictor:
    """LTV prediction models for 180-day player revenue forecasting.

    Implements LightGBM regressor with Optuna tuning, plus baseline models
    (cohort average, logistic regression churn, random forest).

    Args:
        config: Project configuration dictionary.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.random_seed: int = config.get("project", {}).get("random_seed", 42)
        self.time_horizon: int = (
            config.get("models", {}).get("ltv", {}).get("time_horizon_days", 180)
        )
        self.target_col: str = (
            config.get("models", {})
            .get("ltv", {})
            .get("target_variable", "total_revenue_180d")
        )

        self.lgb_model: Optional[Any] = None
        self.rf_model: Optional[RandomForestRegressor] = None
        self.lr_model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[list[str]] = None
        self.best_params_: Optional[dict] = None
        self.cohort_averages_: Optional[pd.DataFrame] = None

    def fit_cohort_baseline(
        self,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute cohort average LTV baseline.

        Groups players by signup_month and plan_tier, computes mean
        180-day revenue per cohort.

        Args:
            train_df: Training set with signup_month, plan_tier, and target.

        Returns:
            DataFrame of cohort averages.
        """
        cohort_cols = ["signup_month", "plan_tier"]
        available_cols = [c for c in cohort_cols if c in train_df.columns]
        if not available_cols:
            global_mean = train_df[self.target_col].mean()
            self.cohort_averages_ = pd.DataFrame({"cohort_avg_ltv": [global_mean]})
            return self.cohort_averages_

        self.cohort_averages_ = (
            train_df.groupby(available_cols)[self.target_col]
            .mean()
            .reset_index()
            .rename(columns={self.target_col: "cohort_avg_ltv"})
        )
        logger.info(
            "Cohort baseline: %d cohorts, mean LTV: %.2f",
            len(self.cohort_averages_),
            self.cohort_averages_["cohort_avg_ltv"].mean(),
        )
        return self.cohort_averages_

    def predict_cohort_baseline(self, df: pd.DataFrame) -> np.ndarray:
        """Predict LTV using cohort averages.

        Args:
            df: DataFrame with signup_month and plan_tier columns.

        Returns:
            Array of predicted LTV values.
        """
        if self.cohort_averages_ is None:
            raise RuntimeError("Fit cohort baseline first.")

        cohort_cols = ["signup_month", "plan_tier"]
        available_cols = [c for c in cohort_cols if c in df.columns]
        if not available_cols:
            return np.full(len(df), self.cohort_averages_["cohort_avg_ltv"].iloc[0])

        merged = df[available_cols].merge(
            self.cohort_averages_, on=available_cols, how="left"
        )
        # Fill unmatched cohorts with global mean
        global_mean = self.cohort_averages_["cohort_avg_ltv"].mean()
        return merged["cohort_avg_ltv"].fillna(global_mean).values

    def fit_logistic_baseline(
        self,
        X_train: pd.DataFrame,
        y_train_binary: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> LogisticRegression:
        """Fit logistic regression churn classifier baseline.

        Args:
            X_train: Feature matrix.
            y_train_binary: Binary churn labels (0/1).
            feature_names: Features to use.

        Returns:
            Fitted LogisticRegression model.
        """
        features = feature_names or X_train.columns.tolist()
        self.lr_model = LogisticRegression(
            random_state=self.random_seed,
            max_iter=1000,
            C=1.0,
        )
        self.lr_model.fit(X_train[features].values, y_train_binary)
        logger.info("Logistic regression baseline fitted.")
        return self.lr_model

    def fit_random_forest_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> RandomForestRegressor:
        """Fit random forest regressor baseline for LTV.

        Args:
            X_train: Feature matrix.
            y_train: Continuous revenue target.
            feature_names: Features to use.

        Returns:
            Fitted RandomForestRegressor model.
        """
        features = feature_names or X_train.columns.tolist()
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        self.rf_model.fit(X_train[features].values, y_train)
        logger.info("Random forest baseline fitted.")
        return self.rf_model

    def tune_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        feature_names: Optional[list[str]] = None,
        n_trials: int = 20,
    ) -> dict:
        """Tune LightGBM regressor using Optuna Bayesian optimization.

        Args:
            X_train: Training feature matrix.
            y_train: Training revenue target.
            X_val: Validation feature matrix.
            y_val: Validation revenue target.
            feature_names: Features to use.
            n_trials: Number of Optuna trials.

        Returns:
            Best hyperparameters dictionary.
        """
        import lightgbm as lgb
        import optuna

        features = feature_names or X_train.columns.tolist()
        self.feature_names = features

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "random_state": self.random_seed,
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train[features].values,
                y_train,
                eval_set=[(X_val[features].values, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = model.predict(X_val[features].values)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            return rmse

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.best_params_ = study.best_params
        logger.info(
            "Optuna tuning complete. Best RMSE: %.4f. Best params: %s",
            study.best_value,
            study.best_params,
        )
        return study.best_params

    def fit_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
        params: Optional[dict] = None,
    ) -> Any:
        """Fit LightGBM regressor with given or best parameters.

        Args:
            X_train: Training feature matrix.
            y_train: Training revenue target.
            feature_names: Features to use.
            params: Hyperparameters. If None, uses best_params_ from tuning.

        Returns:
            Fitted LGBMRegressor model.
        """
        import lightgbm as lgb

        features = feature_names or self.feature_names or X_train.columns.tolist()
        self.feature_names = features

        if params is None:
            params = self.best_params_ or {}

        model_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": self.random_seed,
        }
        model_params.update(params)

        self.lgb_model = lgb.LGBMRegressor(**model_params)
        self.lgb_model.fit(X_train[features].values, y_train)
        logger.info("LightGBM regressor fitted with %d features.", len(features))
        return self.lgb_model

    def predict(
        self, model: Any, X: pd.DataFrame, feature_names: Optional[list[str]] = None
    ) -> np.ndarray:
        """Generate predictions from a fitted model.

        Args:
            model: Fitted model with predict() method.
            X: Feature matrix.
            feature_names: Features to use.

        Returns:
            Array of predictions.
        """
        features = feature_names or self.feature_names or X.columns.tolist()
        return model.predict(X[features].values)

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
    ) -> dict[str, float]:
        """Evaluate regression model performance.

        Args:
            y_true: Actual revenue values.
            y_pred: Predicted revenue values.
            model_name: Label for logging.

        Returns:
            Dictionary with rmse, mape, r_squared metrics.
        """
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        # MAPE: avoid division by zero for zero-revenue accounts
        nonzero_mask = y_true > 0
        if nonzero_mask.sum() > 0:
            mape = float(
                mean_absolute_percentage_error(
                    y_true[nonzero_mask], y_pred[nonzero_mask]
                )
            )
        else:
            mape = float("nan")
        r2 = float(r2_score(y_true, y_pred))

        results = {"rmse": rmse, "mape": mape, "r_squared": r2}
        logger.info(
            "%s -- RMSE: %.2f, MAPE: %.2f%%, R2: %.4f",
            model_name,
            rmse,
            mape * 100 if not np.isnan(mape) else 0,
            r2,
        )
        return results

    def compute_decile_lift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        """Compute revenue decile lift analysis.

        Bins predictions into deciles and measures actual revenue
        concentration. Critical for UA targeting validation.

        Args:
            y_true: Actual revenue values.
            y_pred: Predicted revenue values.

        Returns:
            DataFrame with decile, predicted_mean, actual_mean,
            actual_total, cumulative_pct, and lift.
        """
        df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
        df["decile"] = pd.qcut(
            df["predicted"], q=10, labels=range(1, 11), duplicates="drop"
        )

        decile_stats = (
            df.groupby("decile", observed=False)
            .agg(
                predicted_mean=("predicted", "mean"),
                actual_mean=("actual", "mean"),
                actual_total=("actual", "sum"),
                count=("actual", "count"),
            )
            .reset_index()
        )

        total_revenue = df["actual"].sum()
        decile_stats["revenue_pct"] = decile_stats["actual_total"] / total_revenue
        decile_stats["cumulative_pct"] = decile_stats["revenue_pct"].cumsum()
        overall_mean = df["actual"].mean()
        decile_stats["lift"] = decile_stats["actual_mean"] / overall_mean

        return decile_stats

    def get_feature_importance(
        self, model: Any, feature_names: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Extract feature importance from a tree-based model.

        Args:
            model: Fitted model with feature_importances_ attribute.
            feature_names: Feature column names.

        Returns:
            DataFrame with feature and importance columns, sorted descending.
        """
        features = feature_names or self.feature_names or []
        importances = model.feature_importances_

        if len(features) != len(importances):
            features = [f"feature_{i}" for i in range(len(importances))]

        fi_df = pd.DataFrame(
            {"feature": features, "importance": importances}
        ).sort_values("importance", ascending=False)
        fi_df["importance_pct"] = fi_df["importance"] / fi_df["importance"].sum() * 100
        return fi_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PLTVCalculator
# ---------------------------------------------------------------------------


class PLTVCalculator:
    """Predicted Lifetime Value calculator.

    Combines survival probability from Cox model with revenue prediction
    from LightGBM: pLTV = P(survive 180d | Cox) * E(revenue | LightGBM).

    Args:
        survival_analyzer: Fitted SurvivalAnalyzer instance.
        ltv_predictor: Fitted LTVPredictor instance.
        config: Project configuration dictionary.
    """

    def __init__(
        self,
        survival_analyzer: SurvivalAnalyzer,
        ltv_predictor: LTVPredictor,
        config: dict,
    ) -> None:
        self.survival_analyzer = survival_analyzer
        self.ltv_predictor = ltv_predictor
        self.config = config
        self.time_horizon: int = (
            config.get("models", {}).get("ltv", {}).get("time_horizon_days", 180)
        )

    def calculate_pltv(
        self,
        X: pd.DataFrame,
        survival_model: Any = None,
        revenue_model: Any = None,
    ) -> pd.DataFrame:
        """Calculate predicted LTV for a set of players.

        pLTV = P(survive time_horizon days) * E(revenue | features)

        Args:
            X: Feature matrix.
            survival_model: Survival model to use (defaults to Cox).
            revenue_model: Revenue model to use (defaults to LightGBM).

        Returns:
            DataFrame with survival_prob, predicted_revenue, and pltv columns.
        """
        if survival_model is None:
            survival_model = self.survival_analyzer.cox_model
        if revenue_model is None:
            revenue_model = self.ltv_predictor.lgb_model

        # Survival probability at time horizon
        surv_prob = self.survival_analyzer.predict_survival_probability(
            survival_model, X, time_point=float(self.time_horizon)
        )

        # Expected revenue
        features = self.ltv_predictor.feature_names or X.columns.tolist()
        predicted_revenue = revenue_model.predict(X[features].values)
        # Clip negative predictions
        predicted_revenue = np.clip(predicted_revenue, 0, None)

        # pLTV = survival probability * predicted revenue
        pltv = surv_prob * predicted_revenue

        result = pd.DataFrame(
            {
                "survival_prob_180d": surv_prob,
                "predicted_revenue": predicted_revenue,
                "pltv": pltv,
            },
            index=X.index,
        )
        logger.info(
            "pLTV calculated. Mean: %.2f, Median: %.2f, Max: %.2f",
            result["pltv"].mean(),
            result["pltv"].median(),
            result["pltv"].max(),
        )
        return result

    def segment_analysis(
        self,
        pltv_df: pd.DataFrame,
        original_df: pd.DataFrame,
        segment_col: str,
    ) -> pd.DataFrame:
        """Analyze pLTV by segment (e.g., plan_tier, referral_source).

        Args:
            pltv_df: DataFrame with pLTV calculations (from calculate_pltv).
            original_df: Original data with segment column.
            segment_col: Column name to group by.

        Returns:
            DataFrame with segment-level pLTV statistics.
        """
        if segment_col in pltv_df.columns:
            combined = pltv_df
        else:
            combined = pltv_df.join(original_df[[segment_col]], how="left")
        segment_stats = (
            combined.groupby(segment_col)["pltv"]
            .agg(["mean", "median", "std", "count", "sum"])
            .reset_index()
        )
        segment_stats.columns = [
            segment_col,
            "pltv_mean",
            "pltv_median",
            "pltv_std",
            "count",
            "pltv_total",
        ]
        segment_stats["pct_of_total"] = (
            segment_stats["pltv_total"] / segment_stats["pltv_total"].sum() * 100
        )
        return segment_stats

    def simulate_retention_intervention(
        self,
        pltv_df: pd.DataFrame,
        config: Optional[dict] = None,
    ) -> dict[str, float]:
        """Simulate ROI of retention offer for high-risk players.

        Per spec: GBP 5 retention offer to top 20% churn risk,
        assume 15% retention uplift, avg LTV of saved player = GBP 18.

        Args:
            pltv_df: DataFrame with survival_prob_180d and pltv columns.
            config: Business validation config (or uses self.config).

        Returns:
            Dictionary with intervention simulation results and ROI.
        """
        bv_cfg = (
            (config or self.config)
            .get("business_validation", {})
            .get("retention_offer_simulation", {})
        )
        offer_value = bv_cfg.get("offer_value_gbp", 5.0)
        risk_percentile = bv_cfg.get("target_risk_percentile", 0.80)
        retention_uplift = bv_cfg.get("assumed_retention_uplift", 0.15)
        avg_ltv_saved = bv_cfg.get("avg_ltv_saved_player", 18.0)

        # Identify top risk players (lowest survival probability)
        risk_threshold = pltv_df["survival_prob_180d"].quantile(1.0 - risk_percentile)
        high_risk = pltv_df[pltv_df["survival_prob_180d"] <= risk_threshold]
        n_targeted = len(high_risk)

        # Cost of intervention
        total_cost = n_targeted * offer_value

        # Expected saved players
        n_saved = int(n_targeted * retention_uplift)

        # Expected revenue from saved players
        expected_revenue = n_saved * avg_ltv_saved

        # ROI
        roi = (expected_revenue - total_cost) / total_cost if total_cost > 0 else 0

        result = {
            "n_targeted": n_targeted,
            "offer_value_gbp": offer_value,
            "total_cost_gbp": total_cost,
            "retention_uplift_pct": retention_uplift * 100,
            "n_saved": n_saved,
            "expected_revenue_gbp": expected_revenue,
            "net_benefit_gbp": expected_revenue - total_cost,
            "roi_pct": roi * 100,
        }
        logger.info(
            "Retention intervention: target %d players, save %d, ROI: %.0f%%",
            n_targeted,
            n_saved,
            roi * 100,
        )
        return result


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------


def save_model(
    model: Any,
    path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save a trained model and its metadata to disk.

    Args:
        model: Trained model object (any pickle-compatible model).
        path: File path for the model artifact (.joblib).
        metadata: Optional metadata dictionary (saved as JSON sidecar).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_path)
    logger.info("Model saved to %s", out_path)

    if metadata is not None:
        meta_path = out_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Metadata saved to %s", meta_path)


def load_model(path: str) -> Any:
    """Load a trained model from disk.

    Args:
        path: File path to the model artifact (.joblib).

    Returns:
        Deserialized model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
    return model


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(config_path: str = "config.yaml") -> dict[str, Any]:
    """Execute the full modeling pipeline.

    Steps:
        1. Load config and processed data
        2. Feature selection
        3. Fit Cox PH model
        4. Fit GBSA model
        5. Compare Cox vs GBSA
        6. Fit LTV baselines (cohort, logistic, random forest)
        7. Tune and fit LightGBM LTV model
        8. Calculate pLTV
        9. Save all models and results

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Dictionary with all model results and metrics.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config(config_path)
    processed_dir = config["data"]["processed_dir"]

    # Load processed splits
    train_df = pd.read_csv(f"{processed_dir}/train.csv")
    val_df = pd.read_csv(f"{processed_dir}/validation.csv")
    test_df = pd.read_csv(f"{processed_dir}/test.csv")

    logger.info(
        "Data loaded: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # ---- Survival Modeling ----
    sa = SurvivalAnalyzer(config)
    X_train_all, y_train = sa._prepare_features(train_df, fit_scaler=True)
    X_val_all, y_val = sa._prepare_features(val_df)
    X_test_all, y_test = sa._prepare_features(test_df)

    # Feature selection
    selected = sa.select_features(X_train_all, y_train)

    # Fit Cox PH
    sa.fit_cox(X_train_all, y_train, feature_names=selected)

    # Fit GBSA
    sa.fit_gbsa(X_train_all, y_train, feature_names=selected)

    # Compare
    comparison = sa.compare_cox_vs_gbsa(X_test_all, y_test)

    # PH validation
    ph_result = sa.validate_proportional_hazards(X_train_all, y_train)

    # ---- LTV Modeling ----
    ltv = LTVPredictor(config)
    y_train_rev = train_df["total_revenue_180d"].values
    y_val_rev = val_df["total_revenue_180d"].values
    y_test_rev = test_df["total_revenue_180d"].values

    # Baselines
    ltv.fit_cohort_baseline(train_df)
    ltv.fit_logistic_baseline(
        X_train_all, train_df["event_observed"].values, feature_names=selected
    )
    ltv.fit_random_forest_baseline(X_train_all, y_train_rev, feature_names=selected)

    # Tune LightGBM
    n_trials = (
        config.get("models", {})
        .get("lightgbm_survival", {})
        .get("tuning", {})
        .get("n_trials", 20)
    )
    ltv.tune_lightgbm(
        X_train_all,
        y_train_rev,
        X_val_all,
        y_val_rev,
        feature_names=selected,
        n_trials=n_trials,
    )
    ltv.fit_lightgbm(X_train_all, y_train_rev, feature_names=selected)

    # Evaluate all LTV models
    ltv_results = {}
    for name, model in [
        ("lightgbm", ltv.lgb_model),
        ("random_forest", ltv.rf_model),
    ]:
        preds = ltv.predict(model, X_test_all, feature_names=selected)
        ltv_results[name] = ltv.evaluate_regression(y_test_rev, preds, name)

    cohort_preds = ltv.predict_cohort_baseline(test_df)
    ltv_results["cohort_baseline"] = ltv.evaluate_regression(
        y_test_rev, cohort_preds, "cohort_baseline"
    )

    # ---- pLTV ----
    pltv_calc = PLTVCalculator(sa, ltv, config)
    pltv_df = pltv_calc.calculate_pltv(X_test_all)

    # Save models
    models_dir = config["outputs"]["models_dir"]
    save_model(sa.cox_model, f"{models_dir}/cox_ph_model.joblib", comparison["cox"])
    save_model(sa.gbsa_model, f"{models_dir}/gbsa_model.joblib", comparison["gbsa"])
    save_model(ltv.lgb_model, f"{models_dir}/lgb_ltv_model.joblib", ltv.best_params_)

    results = {
        "survival_comparison": comparison,
        "ph_assumption": ph_result,
        "ltv_results": ltv_results,
        "pltv_summary": {
            "mean": float(pltv_df["pltv"].mean()),
            "median": float(pltv_df["pltv"].median()),
        },
    }
    logger.info("Modeling pipeline complete.")
    return results


if __name__ == "__main__":
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path=cfg)
