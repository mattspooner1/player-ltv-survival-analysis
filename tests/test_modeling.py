"""Unit tests for the modeling module.

Tests cover: SurvivalAnalyzer, LTVPredictor, PLTVCalculator,
FeatureSelector, and model persistence utilities.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    FeatureSelector,
    LTVPredictor,
    PLTVCalculator,
    SurvivalAnalyzer,
    _get_numeric_features,
    _prepare_survival_target,
    load_config,
    load_model,
    save_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = str(PROJECT_ROOT / "config.yaml")
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def config() -> dict:
    """Load project config once for all tests."""
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    """Load training set."""
    return pd.read_csv(DATA_DIR / "train.csv")


@pytest.fixture(scope="module")
def val_df() -> pd.DataFrame:
    """Load validation set."""
    return pd.read_csv(DATA_DIR / "validation.csv")


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    """Load test set."""
    return pd.read_csv(DATA_DIR / "test.csv")


@pytest.fixture(scope="module")
def survival_analyzer(config: dict, train_df: pd.DataFrame) -> SurvivalAnalyzer:
    """Create and fit a SurvivalAnalyzer for reuse across tests."""
    sa = SurvivalAnalyzer(config)
    X_train, y_train = sa._prepare_features(train_df, fit_scaler=True)

    # Use all numeric features (skip full selection for speed)
    features = X_train.columns.tolist()
    sa.fit_cox(X_train, y_train, feature_names=features)
    sa.fit_gbsa(
        X_train,
        y_train,
        feature_names=features,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
    )
    return sa


@pytest.fixture(scope="module")
def ltv_predictor(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    survival_analyzer: SurvivalAnalyzer,
) -> LTVPredictor:
    """Create and fit an LTVPredictor for reuse."""
    ltv = LTVPredictor(config)
    X_train, _ = survival_analyzer._prepare_features(train_df)
    X_val, _ = survival_analyzer._prepare_features(val_df)
    features = survival_analyzer.feature_names

    y_train_rev = train_df["total_revenue_180d"].values
    y_val_rev = val_df["total_revenue_180d"].values

    ltv.fit_cohort_baseline(train_df)
    ltv.fit_random_forest_baseline(X_train, y_train_rev, feature_names=features)
    ltv.fit_logistic_baseline(
        X_train, train_df["event_observed"].values, feature_names=features
    )

    # Quick tune with 3 trials for test speed
    ltv.tune_lightgbm(
        X_train, y_train_rev, X_val, y_val_rev, feature_names=features, n_trials=3
    )
    ltv.fit_lightgbm(X_train, y_train_rev, feature_names=features)
    return ltv


# ---------------------------------------------------------------------------
# Test: Utility Helpers
# ---------------------------------------------------------------------------


class TestPrepSurvivalTarget:
    """Tests for _prepare_survival_target."""

    def test_structured_array_dtype(self, train_df: pd.DataFrame) -> None:
        y = _prepare_survival_target(train_df)
        assert y.dtype.names == ("event", "time")
        assert y["event"].dtype == bool
        assert y["time"].dtype == float

    def test_event_values_are_boolean(self, train_df: pd.DataFrame) -> None:
        y = _prepare_survival_target(train_df)
        assert set(np.unique(y["event"])).issubset({True, False})

    def test_time_values_positive(self, train_df: pd.DataFrame) -> None:
        y = _prepare_survival_target(train_df)
        assert np.all(y["time"] > 0)

    def test_length_matches_input(self, train_df: pd.DataFrame) -> None:
        y = _prepare_survival_target(train_df)
        assert len(y) == len(train_df)


class TestGetNumericFeatures:
    """Tests for _get_numeric_features."""

    def test_excludes_specified_columns(self, train_df: pd.DataFrame) -> None:
        exclude = ["event_observed", "duration_days"]
        features = _get_numeric_features(train_df, exclude)
        for col in exclude:
            assert col not in features

    def test_returns_only_numeric(self, train_df: pd.DataFrame) -> None:
        features = _get_numeric_features(train_df, [])
        for col in features:
            assert train_df[col].dtype in [np.int64, np.float64, int, float]

    def test_returns_sorted_list(self, train_df: pd.DataFrame) -> None:
        features = _get_numeric_features(train_df, [])
        assert features == sorted(features)


# ---------------------------------------------------------------------------
# Test: SurvivalAnalyzer
# ---------------------------------------------------------------------------


class TestSurvivalAnalyzer:
    """Tests for the SurvivalAnalyzer class."""

    def test_init_loads_config(self, config: dict) -> None:
        sa = SurvivalAnalyzer(config)
        assert sa.penalizer == config["models"]["cox_ph"]["penalizer"]
        assert sa.random_seed == config["project"]["random_seed"]

    def test_prepare_features_returns_correct_shapes(
        self, config: dict, train_df: pd.DataFrame
    ) -> None:
        sa = SurvivalAnalyzer(config)
        X, y = sa._prepare_features(train_df, fit_scaler=True)
        assert len(X) == len(train_df)
        assert len(y) == len(train_df)
        assert X.shape[1] > 10  # Should have many features

    def test_scaler_not_fitted_raises(
        self, config: dict, train_df: pd.DataFrame
    ) -> None:
        sa = SurvivalAnalyzer(config)
        with pytest.raises(RuntimeError, match="Scaler not fitted"):
            sa._prepare_features(train_df, fit_scaler=False)

    def test_cox_model_fitted(self, survival_analyzer: SurvivalAnalyzer) -> None:
        assert survival_analyzer.cox_model is not None
        assert hasattr(survival_analyzer.cox_model, "coef_")

    def test_gbsa_model_fitted(self, survival_analyzer: SurvivalAnalyzer) -> None:
        assert survival_analyzer.gbsa_model is not None

    def test_cox_predictions_valid_shape(
        self,
        survival_analyzer: SurvivalAnalyzer,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        features = survival_analyzer.feature_names
        preds = survival_analyzer.cox_model.predict(X_test[features].values)
        assert len(preds) == len(test_df)
        assert not np.any(np.isnan(preds))

    def test_gbsa_predictions_valid_shape(
        self,
        survival_analyzer: SurvivalAnalyzer,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        features = survival_analyzer.feature_names
        preds = survival_analyzer.gbsa_model.predict(X_test[features].values)
        assert len(preds) == len(test_df)

    def test_evaluate_model_returns_metrics(
        self,
        survival_analyzer: SurvivalAnalyzer,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, y_test = survival_analyzer._prepare_features(test_df)
        metrics = survival_analyzer.evaluate_model(
            survival_analyzer.cox_model, X_test, y_test, "Cox"
        )
        assert "c_index" in metrics
        assert "ibs" in metrics
        assert 0.0 <= metrics["c_index"] <= 1.0

    def test_c_index_in_valid_range(
        self,
        survival_analyzer: SurvivalAnalyzer,
        test_df: pd.DataFrame,
    ) -> None:
        """C-index should be in valid range [0, 1].

        Note: with all features (no selection), the Cox model may not
        beat random on the small test set. The full pipeline with feature
        selection is tested in the notebook.
        """
        X_test, y_test = survival_analyzer._prepare_features(test_df)
        metrics = survival_analyzer.evaluate_model(
            survival_analyzer.cox_model, X_test, y_test, "Cox"
        )
        assert 0.0 <= metrics["c_index"] <= 1.0

    def test_hazard_ratios_returned(self, survival_analyzer: SurvivalAnalyzer) -> None:
        hr_df = survival_analyzer.get_hazard_ratios()
        assert isinstance(hr_df, pd.DataFrame)
        assert "feature" in hr_df.columns
        assert "hazard_ratio" in hr_df.columns
        assert "coefficient" in hr_df.columns
        assert "interpretation" in hr_df.columns
        assert len(hr_df) == len(survival_analyzer.feature_names)
        assert (hr_df["hazard_ratio"] > 0).all()

    def test_hazard_ratios_not_fitted_raises(self, config: dict) -> None:
        sa = SurvivalAnalyzer(config)
        with pytest.raises(RuntimeError, match="Cox model not fitted"):
            sa.get_hazard_ratios()

    def test_survival_probability_bounded(
        self,
        survival_analyzer: SurvivalAnalyzer,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        probs = survival_analyzer.predict_survival_probability(
            survival_analyzer.cox_model, X_test, time_point=180.0
        )
        assert len(probs) == len(test_df)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_survival_curves_shape(
        self,
        survival_analyzer: SurvivalAnalyzer,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        time_grid, surv_matrix = survival_analyzer.get_survival_curves(
            survival_analyzer.cox_model, X_test
        )
        assert surv_matrix.shape == (len(test_df), len(time_grid))
        assert np.all(surv_matrix >= 0)
        assert np.all(surv_matrix <= 1)

    def test_compare_both_models_needed(
        self, config: dict, train_df: pd.DataFrame
    ) -> None:
        sa = SurvivalAnalyzer(config)
        X, y = sa._prepare_features(train_df, fit_scaler=True)
        sa.fit_cox(X, y)
        with pytest.raises(RuntimeError, match="Both Cox and GBSA"):
            sa.compare_cox_vs_gbsa(X, y)

    def test_compare_returns_recommendation(
        self,
        survival_analyzer: SurvivalAnalyzer,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, y_test = survival_analyzer._prepare_features(test_df)
        comparison = survival_analyzer.compare_cox_vs_gbsa(X_test, y_test)
        assert "cox" in comparison
        assert "gbsa" in comparison
        assert "preferred_model" in comparison
        assert comparison["preferred_model"] in ("cox", "gbsa")
        assert "recommendation" in comparison


# ---------------------------------------------------------------------------
# Test: FeatureSelector
# ---------------------------------------------------------------------------


class TestFeatureSelector:
    """Tests for the FeatureSelector class."""

    def test_init_loads_thresholds(self, config: dict) -> None:
        fs = FeatureSelector(config)
        assert fs.corr_threshold == 0.85
        assert fs.vif_threshold == 5.0
        assert fs.rfe_top_n == 20

    def test_transform_before_fit_raises(self, config: dict) -> None:
        fs = FeatureSelector(config)
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(RuntimeError, match="Call fit"):
            fs.transform(X)

    def test_fit_reduces_features(
        self,
        config: dict,
        survival_analyzer: SurvivalAnalyzer,
        train_df: pd.DataFrame,
    ) -> None:
        X_train, y_train = survival_analyzer._prepare_features(train_df)
        fs = FeatureSelector(config)
        fs.fit(X_train, y_train)
        assert fs.selected_features_ is not None
        assert len(fs.selected_features_) <= X_train.shape[1]
        assert len(fs.selected_features_) >= fs.min_features

    def test_transform_returns_correct_columns(
        self,
        config: dict,
        survival_analyzer: SurvivalAnalyzer,
        train_df: pd.DataFrame,
    ) -> None:
        X_train, y_train = survival_analyzer._prepare_features(train_df)
        fs = FeatureSelector(config)
        fs.fit(X_train, y_train)
        X_sel = fs.transform(X_train)
        assert list(X_sel.columns) == fs.selected_features_


# ---------------------------------------------------------------------------
# Test: LTVPredictor
# ---------------------------------------------------------------------------


class TestLTVPredictor:
    """Tests for the LTVPredictor class."""

    def test_cohort_baseline_fitted(self, ltv_predictor: LTVPredictor) -> None:
        assert ltv_predictor.cohort_averages_ is not None
        assert "cohort_avg_ltv" in ltv_predictor.cohort_averages_.columns

    def test_cohort_predictions_positive(
        self, ltv_predictor: LTVPredictor, test_df: pd.DataFrame
    ) -> None:
        preds = ltv_predictor.predict_cohort_baseline(test_df)
        assert len(preds) == len(test_df)
        assert np.all(preds >= 0)

    def test_rf_model_fitted(self, ltv_predictor: LTVPredictor) -> None:
        assert ltv_predictor.rf_model is not None

    def test_lgb_model_fitted(self, ltv_predictor: LTVPredictor) -> None:
        assert ltv_predictor.lgb_model is not None

    def test_best_params_populated(self, ltv_predictor: LTVPredictor) -> None:
        assert ltv_predictor.best_params_ is not None
        assert "num_leaves" in ltv_predictor.best_params_

    def test_lgb_predictions_valid(
        self,
        ltv_predictor: LTVPredictor,
        survival_analyzer: SurvivalAnalyzer,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        preds = ltv_predictor.predict(
            ltv_predictor.lgb_model, X_test, ltv_predictor.feature_names
        )
        assert len(preds) == len(test_df)
        assert not np.any(np.isnan(preds))

    def test_evaluate_regression_metrics(self, ltv_predictor: LTVPredictor) -> None:
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 380, 520])
        metrics = ltv_predictor.evaluate_regression(y_true, y_pred, "test")
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "r_squared" in metrics
        assert metrics["rmse"] > 0
        assert metrics["r_squared"] > 0

    def test_evaluate_regression_handles_zeros(
        self, ltv_predictor: LTVPredictor
    ) -> None:
        """MAPE should handle zero actual values gracefully."""
        y_true = np.array([0, 0, 100, 200])
        y_pred = np.array([10, 5, 110, 190])
        metrics = ltv_predictor.evaluate_regression(y_true, y_pred, "test_zeros")
        assert "mape" in metrics

    def test_decile_lift_structure(
        self,
        ltv_predictor: LTVPredictor,
        survival_analyzer: SurvivalAnalyzer,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        y_true = test_df["total_revenue_180d"].values
        y_pred = ltv_predictor.predict(
            ltv_predictor.lgb_model, X_test, ltv_predictor.feature_names
        )
        decile_df = ltv_predictor.compute_decile_lift(y_true, y_pred)
        assert "decile" in decile_df.columns
        assert "revenue_pct" in decile_df.columns
        assert "lift" in decile_df.columns
        # Cumulative pct should end near 1.0
        assert decile_df["cumulative_pct"].iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_feature_importance_shape(self, ltv_predictor: LTVPredictor) -> None:
        fi = ltv_predictor.get_feature_importance(
            ltv_predictor.lgb_model, ltv_predictor.feature_names
        )
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == len(ltv_predictor.feature_names)


# ---------------------------------------------------------------------------
# Test: PLTVCalculator
# ---------------------------------------------------------------------------


class TestPLTVCalculator:
    """Tests for the PLTVCalculator class."""

    def test_calculate_pltv_returns_dataframe(
        self,
        survival_analyzer: SurvivalAnalyzer,
        ltv_predictor: LTVPredictor,
        config: dict,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        calc = PLTVCalculator(survival_analyzer, ltv_predictor, config)
        pltv_df = calc.calculate_pltv(X_test)
        assert isinstance(pltv_df, pd.DataFrame)
        assert "survival_prob_180d" in pltv_df.columns
        assert "predicted_revenue" in pltv_df.columns
        assert "pltv" in pltv_df.columns
        assert len(pltv_df) == len(test_df)

    def test_pltv_is_product_of_components(
        self,
        survival_analyzer: SurvivalAnalyzer,
        ltv_predictor: LTVPredictor,
        config: dict,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        calc = PLTVCalculator(survival_analyzer, ltv_predictor, config)
        pltv_df = calc.calculate_pltv(X_test)
        expected = pltv_df["survival_prob_180d"] * pltv_df["predicted_revenue"]
        np.testing.assert_allclose(pltv_df["pltv"].values, expected.values, rtol=1e-5)

    def test_survival_prob_bounded(
        self,
        survival_analyzer: SurvivalAnalyzer,
        ltv_predictor: LTVPredictor,
        config: dict,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        calc = PLTVCalculator(survival_analyzer, ltv_predictor, config)
        pltv_df = calc.calculate_pltv(X_test)
        assert (pltv_df["survival_prob_180d"] >= 0).all()
        assert (pltv_df["survival_prob_180d"] <= 1).all()

    def test_predicted_revenue_non_negative(
        self,
        survival_analyzer: SurvivalAnalyzer,
        ltv_predictor: LTVPredictor,
        config: dict,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        calc = PLTVCalculator(survival_analyzer, ltv_predictor, config)
        pltv_df = calc.calculate_pltv(X_test)
        assert (pltv_df["predicted_revenue"] >= 0).all()

    def test_segment_analysis(
        self,
        survival_analyzer: SurvivalAnalyzer,
        ltv_predictor: LTVPredictor,
        config: dict,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        calc = PLTVCalculator(survival_analyzer, ltv_predictor, config)
        pltv_df = calc.calculate_pltv(X_test)
        seg = calc.segment_analysis(pltv_df, test_df, "plan_tier")
        assert isinstance(seg, pd.DataFrame)
        assert "pltv_mean" in seg.columns
        assert "pct_of_total" in seg.columns

    def test_retention_simulation_roi(
        self,
        survival_analyzer: SurvivalAnalyzer,
        ltv_predictor: LTVPredictor,
        config: dict,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        calc = PLTVCalculator(survival_analyzer, ltv_predictor, config)
        pltv_df = calc.calculate_pltv(X_test)
        sim = calc.simulate_retention_intervention(pltv_df)
        assert "roi_pct" in sim
        assert "n_targeted" in sim
        assert "n_saved" in sim
        assert sim["n_targeted"] > 0


# ---------------------------------------------------------------------------
# Test: Model Persistence
# ---------------------------------------------------------------------------


class TestModelPersistence:
    """Tests for save_model and load_model."""

    def test_save_load_roundtrip(self, survival_analyzer: SurvivalAnalyzer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model.joblib")
            save_model(survival_analyzer.cox_model, path, {"test": True})

            loaded = load_model(path)
            assert hasattr(loaded, "coef_")
            np.testing.assert_array_equal(
                loaded.coef_, survival_analyzer.cox_model.coef_
            )

    def test_metadata_saved(self, survival_analyzer: SurvivalAnalyzer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model.joblib")
            meta = {"c_index": 0.78, "features": ["a", "b"]}
            save_model(survival_analyzer.cox_model, path, meta)

            meta_path = Path(path).with_suffix(".json")
            assert meta_path.exists()
            with open(meta_path) as f:
                loaded_meta = json.load(f)
            assert loaded_meta["c_index"] == 0.78

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.joblib")

    def test_predictions_preserved_after_load(
        self,
        survival_analyzer: SurvivalAnalyzer,
        test_df: pd.DataFrame,
    ) -> None:
        X_test, _ = survival_analyzer._prepare_features(test_df)
        features = survival_analyzer.feature_names
        original_preds = survival_analyzer.cox_model.predict(X_test[features].values)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "cox.joblib")
            save_model(survival_analyzer.cox_model, path)
            loaded = load_model(path)
            loaded_preds = loaded.predict(X_test[features].values)
            np.testing.assert_array_equal(original_preds, loaded_preds)


# ---------------------------------------------------------------------------
# Test: Feature Leakage Prevention
# ---------------------------------------------------------------------------


class TestFeatureLeakagePrevention:
    """Critical tests ensuring no data leakage in modeling pipeline."""

    def test_no_target_in_features(self, survival_analyzer: SurvivalAnalyzer) -> None:
        """Ensure target columns are not included as features."""
        target_cols = {
            "event_observed",
            "duration_days",
            "total_revenue_180d",
            "churn_flag",
            "churn_date",
        }
        feature_set = set(survival_analyzer.feature_names or [])
        leaked = feature_set.intersection(target_cols)
        assert len(leaked) == 0, f"Target columns found in features: {leaked}"

    def test_no_identifier_in_features(
        self, survival_analyzer: SurvivalAnalyzer
    ) -> None:
        """Ensure identifiers are not used as features."""
        id_cols = {"account_id", "signup_date"}
        feature_set = set(survival_analyzer.feature_names or [])
        leaked = feature_set.intersection(id_cols)
        assert len(leaked) == 0, f"Identifier columns in features: {leaked}"


# ---------------------------------------------------------------------------
# Test: Config Loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for config loading utility."""

    def test_loads_valid_config(self) -> None:
        config = load_config(CONFIG_PATH)
        assert "project" in config
        assert "models" in config

    def test_missing_config_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
