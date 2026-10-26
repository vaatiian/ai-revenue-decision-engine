"""
Production scorer for the frozen Notebook 07 weekly demand forecast.

Final architecture
------------------
For each supported horizon h in {1, 2, 4}:

    target_week = origin_week + h

    seasonal_analogue =
        revenue observed for target_week - 52 weeks

    recent_component =
        mean revenue over the four observed weeks ending at the
        forecast origin

    forecast =
        0.90 * seasonal_analogue
        +
        0.10 * recent_component

No sklearn artifact is required.

No realised demand occurring after the forecast origin is used.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.features.demand_model_features import (
    build_weekly_demand_model,
)

from src.models.demand_spec import (
    DEMAND_FINAL_DECISION,
    DEMAND_FINAL_MODEL_NAME,
    DEMAND_FORECAST_HORIZONS,
    DEMAND_MODEL_METADATA,
    DEMAND_RECENT_WINDOW_WEEKS,
    DEMAND_SEASONAL_LAG_WEEKS,
    DEMAND_SEASONAL_WEIGHT,
    DEMAND_RECENT_WEIGHT,
    calculate_demand_forecast,
    validate_demand_horizon,
)


# ================================================================
# OUTPUT CONTRACT
# ================================================================

DEMAND_SCORE_OUTPUT_COLUMNS = (
    "forecast_origin_week",
    "target_week",
    "horizon_weeks",
    "seasonal_analogue_week",
    "seasonal_analogue_revenue",
    "recent_four_week_average",
    "forecast_revenue",
    "seasonal_weight",
    "recent_weight",
    "model_name",
    "model_version",
    "model_type",
    "target_name",
    "governance_decision",
    "scored_at_utc",
)


# ================================================================
# ORIGIN NORMALISATION
# ================================================================

def _normalise_origin_week(
    forecast_origin_week,
) -> pd.Timestamp:
    """
    Normalise a forecast origin to its W-SUN week start.

    The production contract requires Monday week-start timestamps.
    """

    origin = pd.Timestamp(
        forecast_origin_week
    )

    if pd.isna(
        origin
    ):
        raise ValueError(
            "Forecast origin week must not be missing."
        )

    origin = origin.normalize()

    period = origin.to_period(
        "W-SUN"
    )

    expected_start = (
        period.start_time.normalize()
    )

    if origin != expected_start:
        raise ValueError(
            "Forecast origin must be the Monday week_start of a "
            "W-SUN forecasting period."
        )

    return origin


# ================================================================
# SINGLE-HORIZON FORECAST
# ================================================================

def forecast_weekly_revenue(
    weekly_demand_model: pd.DataFrame,
    forecast_origin_week,
    horizon_weeks: int,
) -> dict:
    """
    Forecast one target week using the exact frozen 90/10 architecture.
    """

    horizon = validate_demand_horizon(
        horizon_weeks
    )

    origin = _normalise_origin_week(
        forecast_origin_week
    )

    required = (
        "week_start",
        "revenue",
    )

    missing = [
        column
        for column in required
        if column not in weekly_demand_model.columns
    ]

    if missing:
        raise ValueError(
            "Weekly demand model is missing scorer columns: "
            + ", ".join(
                missing
            )
        )

    history = (
        weekly_demand_model.loc[
            weekly_demand_model[
                "week_start"
            ]
            <=
            origin
        ]
        .copy()
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    if history.empty:
        raise ValueError(
            "No weekly history exists at or before forecast origin."
        )

    if (
        history[
            "week_start"
        ].iloc[
            -1
        ]
        !=
        origin
    ):
        raise ValueError(
            "Forecast origin is not a complete observed week in the "
            "locked weekly demand series."
        )

    if (
        len(
            history
        )
        <
        DEMAND_RECENT_WINDOW_WEEKS
    ):
        raise ValueError(
            "Insufficient history for trailing four-week average."
        )

    target_week = (
        origin
        +
        pd.Timedelta(
            weeks=horizon
        )
    )

    seasonal_analogue_week = (
        target_week
        -
        pd.Timedelta(
            weeks=DEMAND_SEASONAL_LAG_WEEKS
        )
    )

    seasonal_match = (
        history.loc[
            history[
                "week_start"
            ]
            ==
            seasonal_analogue_week
        ]
    )

    if seasonal_match.empty:
        raise ValueError(
            "Insufficient history for 52-week seasonal analogue."
        )

    if len(
        seasonal_match
    ) != 1:
        raise RuntimeError(
            "Seasonal analogue week is not unique."
        )

    seasonal_revenue = float(
        seasonal_match[
            "revenue"
        ].iloc[
            0
        ]
    )

    recent_average = float(
        history[
            "revenue"
        ]
        .tail(
            DEMAND_RECENT_WINDOW_WEEKS
        )
        .mean()
    )

    forecast = (
        calculate_demand_forecast(
            seasonal_analogue=(
                seasonal_revenue
            ),
            recent_four_week_average=(
                recent_average
            ),
        )
    )

    return {
        "forecast_origin_week":
            origin,

        "target_week":
            target_week,

        "horizon_weeks":
            horizon,

        "seasonal_analogue_week":
            seasonal_analogue_week,

        "seasonal_analogue_revenue":
            seasonal_revenue,

        "recent_four_week_average":
            recent_average,

        "forecast_revenue":
            forecast,
    }


# ================================================================
# PRODUCTION SCORER
# ================================================================

class DemandScorer:
    """
    Production scorer for the frozen deterministic weekly demand model.
    """

    def __init__(
        self,
        horizons=DEMAND_FORECAST_HORIZONS,
    ):
        horizons = tuple(
            horizons
        )

        if not horizons:
            raise ValueError(
                "At least one forecast horizon is required."
            )

        for horizon in horizons:
            validate_demand_horizon(
                horizon
            )

        if len(
            horizons
        ) != len(
            set(
                horizons
            )
        ):
            raise ValueError(
                "Forecast horizons must not contain duplicates."
            )

        self.horizons = horizons

    def score(
        self,
        transactions: pd.DataFrame,
        forecast_origin_week,
        scored_at_utc=None,
    ) -> pd.DataFrame:
        """
        Produce one forecast row per requested horizon.
        """

        weekly = build_weekly_demand_model(
            transactions
        )

        origin = _normalise_origin_week(
            forecast_origin_week
        )

        rows = [
            forecast_weekly_revenue(
                weekly_demand_model=weekly,
                forecast_origin_week=origin,
                horizon_weeks=horizon,
            )
            for horizon in self.horizons
        ]

        output = pd.DataFrame(
            rows
        )

        if scored_at_utc is None:
            scored_at_utc = (
                datetime.now(
                    timezone.utc
                )
            )

        scored_at = pd.Timestamp(
            scored_at_utc
        )

        if scored_at.tzinfo is None:
            scored_at = scored_at.tz_localize(
                "UTC"
            )
        else:
            scored_at = scored_at.tz_convert(
                "UTC"
            )

        output[
            "seasonal_weight"
        ] = DEMAND_SEASONAL_WEIGHT

        output[
            "recent_weight"
        ] = DEMAND_RECENT_WEIGHT

        output[
            "model_name"
        ] = (
            DEMAND_MODEL_METADATA.model_name
        )

        output[
            "model_version"
        ] = (
            DEMAND_MODEL_METADATA.model_version
        )

        output[
            "model_type"
        ] = (
            DEMAND_MODEL_METADATA.model_type
        )

        output[
            "target_name"
        ] = (
            DEMAND_MODEL_METADATA.target_name
        )

        output[
            "governance_decision"
        ] = (
            DEMAND_FINAL_DECISION
        )

        output[
            "scored_at_utc"
        ] = scored_at

        output = (
            output.loc[
                :,
                DEMAND_SCORE_OUTPUT_COLUMNS,
            ]
            .sort_values(
                "horizon_weeks"
            )
            .reset_index(
                drop=True
            )
        )

        if output[
            "horizon_weeks"
        ].duplicated().any():
            raise RuntimeError(
                "Demand scorer returned duplicate horizon rows."
            )

        if output[
            "forecast_revenue"
        ].isna().any():
            raise RuntimeError(
                "Demand scorer returned missing forecasts."
            )

        return output