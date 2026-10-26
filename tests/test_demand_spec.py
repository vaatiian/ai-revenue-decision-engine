import numpy as np
import pytest

from src.models.demand_spec import (
    DEMAND_APPROVED_USE,
    DEMAND_FINAL_DECISION,
    DEMAND_FINAL_DECISION_DESCRIPTION,
    DEMAND_FINAL_HOLDOUT_STATUS,
    DEMAND_FINAL_MODEL_NAME,
    DEMAND_FINAL_PRISTINE_HOLDOUT_AVAILABLE,
    DEMAND_FORECAST_FORMULA,
    DEMAND_FORECAST_FREQUENCY,
    DEMAND_FORECAST_HORIZONS,
    DEMAND_FUTURE_INFORMATION_ALLOWED,
    DEMAND_GRADIENT_BOOSTING_IS_FINAL_MODEL,
    DEMAND_INTERVAL_INDEPENDENTLY_CALIBRATED,
    DEMAND_MODEL_FROZEN,
    DEMAND_MODEL_METADATA,
    DEMAND_MULTIHORIZON_RULE,
    DEMAND_PLANNING_INTERVAL_LEVELS,
    DEMAND_PLANNING_INTERVAL_METHOD,
    DEMAND_PRIMARY_EVALUATION_METRIC,
    DEMAND_PRIMARY_PLANNING_HORIZON_WEEKS,
    DEMAND_PROHIBITED_INTERPRETATION,
    DEMAND_RANDOM_SPLIT_PERMITTED,
    DEMAND_RECENT_COMPONENT,
    DEMAND_RECENT_WEIGHT,
    DEMAND_RECENT_WINDOW_WEEKS,
    DEMAND_REPORTED_APPROX_FOUR_WEEK_AGGREGATE_FORECAST_ACTUAL,
    DEMAND_REPORTED_APPROX_FOUR_WEEK_AGGREGATE_WAPE,
    DEMAND_REPORTED_APPROX_HORIZON_FORECAST_ACTUAL,
    DEMAND_REPORTED_APPROX_HORIZON_WAPE,
    DEMAND_REPORTED_APPROX_INTERVAL_COVERAGE,
    DEMAND_REPORTED_APPROX_Q4_FORECAST_ACTUAL,
    DEMAND_RESEARCH_CHALLENGER_FEATURES,
    DEMAND_RESEARCH_CHALLENGERS,
    DEMAND_RIDGE_IS_FINAL_MODEL,
    DEMAND_SEASONAL_COMPONENT,
    DEMAND_SEASONAL_LAG_WEEKS,
    DEMAND_SEASONAL_WEIGHT,
    DEMAND_SECONDARY_EVALUATION_METRICS,
    DEMAND_SERIALIZED_MODEL_REQUIRED,
    DEMAND_TARGET_DESCRIPTION,
    DEMAND_VALIDATION_METHOD,
    DEMAND_WEIGHT_TUNING_CLOSED,
    calculate_demand_forecast,
    validate_demand_horizon,
    validate_demand_specification,
)


# ================================================================
# METADATA
# ================================================================

def test_demand_model_metadata():
    assert (
        DEMAND_MODEL_METADATA.model_name
        ==
        "Seasonal / Recent Weekly Revenue Blend"
    )

    assert (
        DEMAND_MODEL_METADATA.model_version
        ==
        "1.0.0"
    )

    assert (
        DEMAND_MODEL_METADATA.model_type
        ==
        "deterministic_forecast"
    )

    assert (
        DEMAND_MODEL_METADATA.target_name
        ==
        "weekly_positive_merchandise_revenue"
    )


def test_demand_target_is_weekly_positive_merchandise_revenue():
    assert (
        DEMAND_TARGET_DESCRIPTION
        ==
        "Weekly qualifying positive merchandise revenue"
    )

    assert (
        DEMAND_FORECAST_FREQUENCY
        ==
        "Weekly"
    )


# ================================================================
# FROZEN ARCHITECTURE
# ================================================================

def test_final_model_name():
    assert (
        DEMAND_FINAL_MODEL_NAME
        ==
        "Seasonal Blend 0.9"
    )


def test_seasonal_component():
    assert (
        DEMAND_SEASONAL_COMPONENT
        ==
        "Same target week 52 weeks earlier"
    )

    assert (
        DEMAND_SEASONAL_LAG_WEEKS
        ==
        52
    )


def test_recent_component():
    assert (
        DEMAND_RECENT_COMPONENT
        ==
        "Trailing 4-week average at forecast origin"
    )

    assert (
        DEMAND_RECENT_WINDOW_WEEKS
        ==
        4
    )


def test_final_weights_are_90_10():
    assert (
        DEMAND_SEASONAL_WEIGHT
        ==
        pytest.approx(
            0.90
        )
    )

    assert (
        DEMAND_RECENT_WEIGHT
        ==
        pytest.approx(
            0.10
        )
    )

    assert (
        DEMAND_SEASONAL_WEIGHT
        +
        DEMAND_RECENT_WEIGHT
        ==
        pytest.approx(
            1.0
        )
    )


def test_model_is_frozen():
    assert (
        DEMAND_MODEL_FROZEN
        is True
    )


def test_weight_tuning_is_closed():
    assert (
        DEMAND_WEIGHT_TUNING_CLOSED
        is True
    )


def test_no_serialized_artifact_required():
    assert (
        DEMAND_SERIALIZED_MODEL_REQUIRED
        is False
    )


# ================================================================
# HORIZONS
# ================================================================

def test_frozen_forecast_horizons():
    assert (
        DEMAND_FORECAST_HORIZONS
        ==
        (
            1,
            2,
            4,
        )
    )


def test_primary_planning_horizon_is_four_weeks():
    assert (
        DEMAND_PRIMARY_PLANNING_HORIZON_WEEKS
        ==
        4
    )


@pytest.mark.parametrize(
    "horizon",
    [
        1,
        2,
        4,
    ],
)
def test_validate_demand_horizon_accepts_frozen_horizons(
    horizon,
):
    assert (
        validate_demand_horizon(
            horizon
        )
        ==
        horizon
    )


@pytest.mark.parametrize(
    "horizon",
    [
        0,
        3,
        5,
        52,
        -1,
    ],
)
def test_validate_demand_horizon_rejects_unsupported_integer(
    horizon,
):
    with pytest.raises(
        ValueError,
        match="Unsupported demand forecast horizon",
    ):
        validate_demand_horizon(
            horizon
        )


@pytest.mark.parametrize(
    "horizon",
    [
        1.0,
        "1",
        None,
        True,
    ],
)
def test_validate_demand_horizon_rejects_non_integer(
    horizon,
):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        validate_demand_horizon(
            horizon
        )


# ================================================================
# FORECAST FORMULA
# ================================================================

def test_forecast_formula_name():
    assert (
        DEMAND_FORECAST_FORMULA
        ==
        "0.90 * seasonal_52_week_analogue "
        "+ 0.10 * trailing_4_week_average_at_origin"
    )


def test_forecast_formula_scalar():
    result = (
        calculate_demand_forecast(
            seasonal_analogue=100.0,
            recent_four_week_average=200.0,
        )
    )

    assert (
        result
        ==
        pytest.approx(
            110.0
        )
    )


def test_forecast_formula_equal_inputs():
    result = (
        calculate_demand_forecast(
            seasonal_analogue=500.0,
            recent_four_week_average=500.0,
        )
    )

    assert (
        result
        ==
        pytest.approx(
            500.0
        )
    )


def test_forecast_formula_array():
    seasonal = np.array(
        [
            100.0,
            200.0,
        ]
    )

    recent = np.array(
        [
            200.0,
            400.0,
        ]
    )

    result = (
        calculate_demand_forecast(
            seasonal,
            recent,
        )
    )

    expected = np.array(
        [
            110.0,
            220.0,
        ]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


@pytest.mark.parametrize(
    "seasonal,recent",
    [
        (
            np.nan,
            100.0,
        ),
        (
            np.inf,
            100.0,
        ),
        (
            100.0,
            np.nan,
        ),
        (
            100.0,
            np.inf,
        ),
    ],
)
def test_forecast_formula_rejects_nonfinite_inputs(
    seasonal,
    recent,
):
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        calculate_demand_forecast(
            seasonal,
            recent,
        )


# ================================================================
# TEMPORAL GOVERNANCE
# ================================================================

def test_validation_method_is_temporal():
    assert (
        DEMAND_VALIDATION_METHOD
        ==
        "Temporal backtesting / multi-horizon historical evaluation"
    )


def test_random_split_is_prohibited():
    assert (
        DEMAND_RANDOM_SPLIT_PERMITTED
        is False
    )


def test_future_information_is_prohibited():
    assert (
        DEMAND_FUTURE_INFORMATION_ALLOWED
        is False
    )


def test_multihorizon_rule_blocks_intermediate_realised_demand():
    assert (
        "between the forecast origin"
        in
        DEMAND_MULTIHORIZON_RULE
    )

    assert (
        "may enter prediction"
        in
        DEMAND_MULTIHORIZON_RULE
    )


def test_no_pristine_final_holdout_remains():
    assert (
        DEMAND_FINAL_PRISTINE_HOLDOUT_AVAILABLE
        is False
    )

    assert (
        DEMAND_FINAL_HOLDOUT_STATUS
        ==
        "No pristine final holdout remains"
    )


# ================================================================
# METRICS
# ================================================================

def test_primary_metric_is_wape():
    assert (
        DEMAND_PRIMARY_EVALUATION_METRIC
        ==
        "WAPE"
    )


def test_secondary_metrics():
    assert (
        DEMAND_SECONDARY_EVALUATION_METRICS
        ==
        (
            "MAE",
            "RMSE",
            "Bias",
            "Forecast / Actual",
        )
    )


# ================================================================
# REPORTED HISTORICAL PERFORMANCE
# ================================================================

def test_reported_horizon_wape():
    assert (
        DEMAND_REPORTED_APPROX_HORIZON_WAPE
        ==
        {
            1: 0.1652,
            2: 0.1668,
            4: 0.1684,
        }
    )


def test_reported_horizon_forecast_actual():
    assert (
        DEMAND_REPORTED_APPROX_HORIZON_FORECAST_ACTUAL
        ==
        {
            1: 0.974,
            2: 0.974,
            4: 0.975,
        }
    )


def test_reported_four_week_aggregate_performance():
    assert (
        DEMAND_REPORTED_APPROX_FOUR_WEEK_AGGREGATE_WAPE
        ==
        pytest.approx(
            0.0916
        )
    )

    assert (
        DEMAND_REPORTED_APPROX_FOUR_WEEK_AGGREGATE_FORECAST_ACTUAL
        ==
        pytest.approx(
            0.977
        )
    )


def test_reported_q4_forecast_actual():
    assert (
        DEMAND_REPORTED_APPROX_Q4_FORECAST_ACTUAL
        ==
        {
            1: 0.943,
            4: 0.933,
        }
    )


def test_reported_interval_coverage():
    assert (
        DEMAND_REPORTED_APPROX_INTERVAL_COVERAGE[
            0.80
        ]
        ==
        pytest.approx(
            0.804
        )
    )

    assert (
        DEMAND_REPORTED_APPROX_INTERVAL_COVERAGE[
            0.90
        ]
        ==
        pytest.approx(
            0.882
        )
    )


# ================================================================
# PLANNING RANGES
# ================================================================

def test_planning_interval_method():
    assert (
        DEMAND_PLANNING_INTERVAL_METHOD
        ==
        "Empirical historical residual quantiles"
    )


def test_planning_interval_levels():
    assert (
        DEMAND_PLANNING_INTERVAL_LEVELS
        ==
        (
            0.80,
            0.90,
        )
    )


def test_planning_intervals_are_not_independently_calibrated():
    assert (
        DEMAND_INTERVAL_INDEPENDENTLY_CALIBRATED
        is False
    )


# ================================================================
# GOVERNANCE DECISION
# ================================================================

def test_final_decision_is_go():
    assert (
        DEMAND_FINAL_DECISION
        ==
        "GO"
    )

    assert (
        DEMAND_FINAL_DECISION_DESCRIPTION
        ==
        "GO for operational demand planning"
    )


def test_approved_use_is_operational_planning():
    assert (
        "operational demand planning"
        in
        DEMAND_APPROVED_USE
    )

    assert (
        "working-capital"
        in
        DEMAND_APPROVED_USE
    )


def test_accounting_guarantee_is_not_approved():
    assert (
        "accounting forecast"
        in
        DEMAND_PROHIBITED_INTERPRETATION
    )

    assert (
        "guarantee"
        in
        DEMAND_PROHIBITED_INTERPRETATION
    )


# ================================================================
# CHALLENGER LINEAGE
# ================================================================

def test_research_challengers_are_retained():
    assert (
        "Ridge Lag Forecast"
        in
        DEMAND_RESEARCH_CHALLENGERS
    )

    assert (
        "Conservative Gradient Boosting"
        in
        DEMAND_RESEARCH_CHALLENGERS
    )


def test_ridge_is_not_final_model():
    assert (
        DEMAND_RIDGE_IS_FINAL_MODEL
        is False
    )


def test_gradient_boosting_is_not_final_model():
    assert (
        DEMAND_GRADIENT_BOOSTING_IS_FINAL_MODEL
        is False
    )


def test_challenger_feature_count():
    assert (
        len(
            DEMAND_RESEARCH_CHALLENGER_FEATURES
        )
        ==
        25
    )


def test_challenger_feature_register_has_no_duplicates():
    assert (
        len(
            DEMAND_RESEARCH_CHALLENGER_FEATURES
        )
        ==
        len(
            set(
                DEMAND_RESEARCH_CHALLENGER_FEATURES
            )
        )
    )


def test_challenger_register_contains_annual_revenue_lag():
    assert (
        "revenue_lag_52"
        in
        DEMAND_RESEARCH_CHALLENGER_FEATURES
    )


def test_challenger_register_contains_year_end_indicator():
    assert (
        "year_end_shutdown_week"
        in
        DEMAND_RESEARCH_CHALLENGER_FEATURES
    )


# ================================================================
# INTERNAL SPEC VALIDATION
# ================================================================

def test_frozen_specification_validates():
    validate_demand_specification()