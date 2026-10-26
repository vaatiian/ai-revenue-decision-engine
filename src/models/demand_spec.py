"""
Frozen production specification for the business-level weekly
demand forecasting model from Notebook 07.

The final research architecture is deterministic.

It is NOT the Ridge or Gradient Boosting challenger developed during
research.

Frozen production formula
-------------------------
For a target week at horizon h:

    forecast =
        0.90 * seasonal_analogue
        +
        0.10 * recent_four_week_average

where:

    seasonal_analogue
        = revenue observed for the same target week 52 weeks earlier

    recent_four_week_average
        = trailing four-week average revenue known at the forecast origin

Supported horizons
------------------
1, 2 and 4 weeks ahead.

Target
------
Weekly qualifying positive merchandise revenue.

Governance
----------
The available weekly history does not contain a pristine final holdout
because earlier diagnostics consumed the eligible historical period.

The final architecture was therefore selected using temporal
backtesting and multi-horizon historical evaluation.

No further blend-weight tuning is permitted on the existing history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ================================================================
# MODEL METADATA
# ================================================================

@dataclass(
    frozen=True
)
class DemandModelMetadata:
    """
    Governance metadata for the frozen demand forecasting architecture.
    """

    model_name: str
    model_version: str
    model_type: str
    target_name: str


DEMAND_MODEL_METADATA = DemandModelMetadata(
    model_name=(
        "Seasonal / Recent Weekly Revenue Blend"
    ),
    model_version="1.0.0",
    model_type="deterministic_forecast",
    target_name="weekly_positive_merchandise_revenue",
)


# ================================================================
# BUSINESS OBJECTIVE / TARGET
# ================================================================

DEMAND_BUSINESS_OBJECTIVE = (
    "Forecast qualifying positive merchandise revenue for "
    "operational demand planning."
)

DEMAND_TARGET_DESCRIPTION = (
    "Weekly qualifying positive merchandise revenue"
)

DEMAND_FORECAST_FREQUENCY = (
    "Weekly"
)

DEMAND_APPROVED_USE = (
    "Weekly and four-week operational demand planning, budgeting, "
    "capacity guidance and working-capital planning"
)

DEMAND_PROHIBITED_INTERPRETATION = (
    "Precise accounting forecast or guarantee of realised weekly revenue"
)


# ================================================================
# FORECAST HORIZONS
# ================================================================

DEMAND_FORECAST_HORIZONS = (
    1,
    2,
    4,
)

DEMAND_PRIMARY_PLANNING_HORIZON_WEEKS = 4


def validate_demand_horizon(
    horizon: int,
) -> int:
    """
    Validate that a requested forecast horizon belongs to the frozen
    production horizon register.
    """

    if (
        isinstance(
            horizon,
            bool,
        )
        or
        not isinstance(
            horizon,
            int,
        )
    ):
        raise TypeError(
            "Demand forecast horizon must be an integer."
        )

    if horizon not in DEMAND_FORECAST_HORIZONS:
        raise ValueError(
            "Unsupported demand forecast horizon. "
            f"Supported horizons: {DEMAND_FORECAST_HORIZONS}."
        )

    return horizon


# ================================================================
# FINAL FROZEN ARCHITECTURE
# ================================================================

DEMAND_FINAL_MODEL_NAME = (
    "Seasonal Blend 0.9"
)

DEMAND_SEASONAL_COMPONENT = (
    "Same target week 52 weeks earlier"
)

DEMAND_RECENT_COMPONENT = (
    "Trailing 4-week average at forecast origin"
)

DEMAND_SEASONAL_WEIGHT = 0.90
DEMAND_RECENT_WEIGHT = 0.10

DEMAND_SEASONAL_LAG_WEEKS = 52
DEMAND_RECENT_WINDOW_WEEKS = 4

DEMAND_MODEL_FROZEN = True
DEMAND_WEIGHT_TUNING_CLOSED = True

DEMAND_SERIALIZED_MODEL_REQUIRED = False


# ================================================================
# FINAL FORECAST FORMULA
# ================================================================

DEMAND_FORECAST_FORMULA = (
    "0.90 * seasonal_52_week_analogue "
    "+ 0.10 * trailing_4_week_average_at_origin"
)


def calculate_demand_forecast(
    seasonal_analogue,
    recent_four_week_average,
):
    """
    Apply the exact frozen Notebook 07 point-forecast formula.

    Supports scalars and numpy-compatible arrays.

    No clipping is applied here because the research point-forecast
    formula itself is the weighted blend. Planning-interval lower bounds
    are handled separately in the research architecture.
    """

    seasonal = np.asarray(
        seasonal_analogue,
        dtype=float,
    )

    recent = np.asarray(
        recent_four_week_average,
        dtype=float,
    )

    if (
        not np.isfinite(
            seasonal
        ).all()
    ):
        raise ValueError(
            "Seasonal analogue values must be finite."
        )

    if (
        not np.isfinite(
            recent
        ).all()
    ):
        raise ValueError(
            "Recent four-week average values must be finite."
        )

    forecast = (
        DEMAND_SEASONAL_WEIGHT
        *
        seasonal
        +
        DEMAND_RECENT_WEIGHT
        *
        recent
    )

    if (
        np.ndim(
            forecast
        )
        ==
        0
    ):
        return float(
            forecast
        )

    return forecast


# ================================================================
# TEMPORAL GOVERNANCE
# ================================================================

DEMAND_VALIDATION_METHOD = (
    "Temporal backtesting / multi-horizon historical evaluation"
)

DEMAND_RANDOM_SPLIT_PERMITTED = False

DEMAND_FUTURE_INFORMATION_ALLOWED = False

DEMAND_ORIGIN_INFORMATION_RULE = (
    "All forecast inputs must be known at or before the forecast origin"
)

DEMAND_MULTIHORIZON_RULE = (
    "No realised demand occurring between the forecast origin and "
    "the future target week may enter prediction"
)

DEMAND_FINAL_PRISTINE_HOLDOUT_AVAILABLE = False

DEMAND_FINAL_HOLDOUT_STATUS = (
    "No pristine final holdout remains"
)


# ================================================================
# EVALUATION METRICS
# ================================================================

DEMAND_PRIMARY_EVALUATION_METRIC = (
    "WAPE"
)

DEMAND_SECONDARY_EVALUATION_METRICS = (
    "MAE",
    "RMSE",
    "Bias",
    "Forecast / Actual",
)


# ================================================================
# PLANNING INTERVALS
# ================================================================

DEMAND_PLANNING_INTERVAL_METHOD = (
    "Empirical historical residual quantiles"
)

DEMAND_PLANNING_INTERVAL_LEVELS = (
    0.80,
    0.90,
)

DEMAND_INTERVAL_INTERPRETATION = (
    "Historical planning ranges rather than statistically guaranteed "
    "future prediction intervals"
)

DEMAND_INTERVAL_INDEPENDENTLY_CALIBRATED = False


# ================================================================
# REPORTED HISTORICAL PERFORMANCE
# ================================================================
#
# These values are explicitly described by Notebook 07 as approximate
# summary results. They are not represented as an untouched holdout.
# ================================================================

DEMAND_REPORTED_APPROX_HORIZON_WAPE = {
    1: 0.1652,
    2: 0.1668,
    4: 0.1684,
}

DEMAND_REPORTED_APPROX_HORIZON_FORECAST_ACTUAL = {
    1: 0.974,
    2: 0.974,
    4: 0.975,
}

DEMAND_REPORTED_APPROX_FOUR_WEEK_AGGREGATE_WAPE = (
    0.0916
)

DEMAND_REPORTED_APPROX_FOUR_WEEK_AGGREGATE_FORECAST_ACTUAL = (
    0.977
)

DEMAND_REPORTED_APPROX_Q4_FORECAST_ACTUAL = {
    1: 0.943,
    4: 0.933,
}

DEMAND_REPORTED_APPROX_INTERVAL_COVERAGE = {
    0.80: 0.804,
    0.90: 0.882,
}


# ================================================================
# GOVERNANCE DECISION
# ================================================================

DEMAND_FINAL_DECISION = (
    "GO"
)

DEMAND_FINAL_DECISION_DESCRIPTION = (
    "GO for operational demand planning"
)

DEMAND_PRIMARY_LIMITATION = (
    "Weekly demand remains volatile and peak-season Q4 forecasts "
    "show greater aggregate underprediction"
)


# ================================================================
# RESEARCH CHALLENGERS — NOT FINAL PRODUCTION MODEL
# ================================================================

DEMAND_RESEARCH_CHALLENGERS = (
    "Seasonal Naive — 52 Weeks",
    "Seasonal / Recent Blend",
    "Ridge Lag Forecast",
    "Conservative Gradient Boosting",
)

DEMAND_RIDGE_IS_FINAL_MODEL = False
DEMAND_GRADIENT_BOOSTING_IS_FINAL_MODEL = False


# ================================================================
# RESEARCH CHALLENGER FEATURE REGISTER
# ================================================================
#
# This register is retained for research lineage only.
#
# The final frozen production architecture does not require the Ridge /
# Gradient Boosting feature matrix for point forecasting.
# ================================================================

DEMAND_RESEARCH_CHALLENGER_FEATURES = (
    "revenue_lag_1",
    "revenue_lag_2",
    "revenue_lag_3",
    "revenue_lag_4",
    "revenue_lag_8",
    "revenue_lag_13",
    "revenue_lag_26",
    "revenue_lag_52",
    "revenue_ma_4",
    "revenue_ma_8",
    "revenue_ma_13",
    "revenue_std_4",
    "revenue_std_8",
    "orders_lag_1",
    "orders_ma_4",
    "units_lag_1",
    "units_ma_4",
    "active_customers_lag_1",
    "active_customers_ma_4",
    "active_products_lag_1",
    "active_products_ma_4",
    "week_sin",
    "week_cos",
    "time_index",
    "year_end_shutdown_week",
)


# ================================================================
# SPECIFICATION VALIDATION
# ================================================================

def validate_demand_specification() -> None:
    """
    Fail closed if frozen demand forecasting constants are internally
    inconsistent.
    """

    if DEMAND_MODEL_FROZEN is not True:
        raise ValueError(
            "Demand model must remain frozen."
        )

    if DEMAND_WEIGHT_TUNING_CLOSED is not True:
        raise ValueError(
            "Demand blend-weight tuning must remain closed."
        )

    if DEMAND_SERIALIZED_MODEL_REQUIRED is not False:
        raise ValueError(
            "Final deterministic demand model must not require "
            "a serialized sklearn artifact."
        )

    if not np.isclose(
        DEMAND_SEASONAL_WEIGHT
        +
        DEMAND_RECENT_WEIGHT,
        1.0,
    ):
        raise ValueError(
            "Demand forecast weights must sum to one."
        )

    if not np.isclose(
        DEMAND_SEASONAL_WEIGHT,
        0.90,
    ):
        raise ValueError(
            "Frozen seasonal weight must equal 0.90."
        )

    if not np.isclose(
        DEMAND_RECENT_WEIGHT,
        0.10,
    ):
        raise ValueError(
            "Frozen recent-demand weight must equal 0.10."
        )

    if (
        DEMAND_SEASONAL_LAG_WEEKS
        !=
        52
    ):
        raise ValueError(
            "Frozen seasonal lag must equal 52 weeks."
        )

    if (
        DEMAND_RECENT_WINDOW_WEEKS
        !=
        4
    ):
        raise ValueError(
            "Frozen recent-demand window must equal four weeks."
        )

    if (
        DEMAND_FORECAST_HORIZONS
        !=
        (
            1,
            2,
            4,
        )
    ):
        raise ValueError(
            "Frozen forecast horizons must be 1, 2 and 4 weeks."
        )

    if DEMAND_RANDOM_SPLIT_PERMITTED is not False:
        raise ValueError(
            "Random forecasting splits are prohibited."
        )

    if DEMAND_FUTURE_INFORMATION_ALLOWED is not False:
        raise ValueError(
            "Future information is prohibited."
        )

    if (
        DEMAND_FINAL_PRISTINE_HOLDOUT_AVAILABLE
        is not False
    ):
        raise ValueError(
            "Research history does not contain a pristine final "
            "weekly holdout."
        )

    if (
        DEMAND_RIDGE_IS_FINAL_MODEL
        is not False
    ):
        raise ValueError(
            "Ridge is a research challenger, not the final model."
        )

    if (
        DEMAND_GRADIENT_BOOSTING_IS_FINAL_MODEL
        is not False
    ):
        raise ValueError(
            "Gradient Boosting is a research challenger, not the "
            "final model."
        )


validate_demand_specification()