"""
Scheduled weekly Demand Forecast batch.

This module provides the deployment boundary for the frozen
weekly deterministic demand forecasting architecture.

Responsibilities
----------------
- require an explicit weekly forecast origin;
- require an explicit supported horizon register;
- delegate all governed forecasting calculations and history checks
  to DemandScorer;
- validate the returned production score contract;
- verify that the scorer returned exactly the requested horizons;
- return one immutable orchestration result.

Non-responsibilities
--------------------
This module does not:
- retrain a forecasting model;
- redefine the 90/10 forecast formula;
- change forecast horizons;
- impute missing historical weeks outside the governed feature pipeline;
- feed Demand Forecast into the Customer Decision Engine;
- write files;
- schedule execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.models.demand_scorer import (
    DEMAND_SCORE_OUTPUT_COLUMNS,
    DemandScorer,
)
from src.models.demand_spec import (
    DEMAND_FORECAST_HORIZONS,
)


# ================================================================
# RESULT CONTRACT
# ================================================================


@dataclass(
    frozen=True,
)
class DemandBatchResult:
    """
    Immutable result from one governed weekly demand forecast batch.
    """

    forecast_origin_week: pd.Timestamp
    horizons: tuple[int, ...]
    forecasts: pd.DataFrame


# ================================================================
# VALIDATION
# ================================================================


def _validate_transactions(
    transactions: pd.DataFrame,
) -> None:
    """
    Validate the orchestration-level transaction input type.
    """

    if not isinstance(
        transactions,
        pd.DataFrame,
    ):
        raise TypeError(
            "transactions must be a pandas DataFrame."
        )


def _validate_forecast_origin_week(
    forecast_origin_week,
) -> None:
    """
    Require an explicit production forecast origin.

    Exact Monday week-start validation remains owned by DemandScorer.
    """

    if forecast_origin_week is None:
        raise ValueError(
            "Demand batch requires an explicit forecast_origin_week."
        )


def _normalise_horizons(
    horizons,
) -> tuple[int, ...]:
    """
    Normalize and validate the requested production horizon register.
    """

    try:

        horizon_tuple = tuple(
            horizons
        )

    except TypeError as exc:

        raise TypeError(
            "horizons must be an iterable of supported integer horizons."
        ) from exc

    if not horizon_tuple:
        raise ValueError(
            "Demand batch requires at least one forecast horizon."
        )

    if len(
        horizon_tuple
    ) != len(
        set(
            horizon_tuple
        )
    ):
        raise ValueError(
            "Demand batch horizons must not contain duplicates."
        )

    unsupported = [
        horizon
        for horizon
        in horizon_tuple
        if horizon
        not in DEMAND_FORECAST_HORIZONS
    ]

    if unsupported:
        raise ValueError(
            "Demand batch contains unsupported forecast horizons: "
            f"{unsupported}. "
            f"Supported horizons: {DEMAND_FORECAST_HORIZONS}."
        )

    return horizon_tuple


def _validate_score_contract(
    forecasts: pd.DataFrame,
    horizons: tuple[int, ...],
) -> None:
    """
    Fail closed if the governed scorer output drifts.
    """

    if not isinstance(
        forecasts,
        pd.DataFrame,
    ):
        raise TypeError(
            "Demand scorer must return a pandas DataFrame."
        )

    if tuple(
        forecasts.columns
    ) != DEMAND_SCORE_OUTPUT_COLUMNS:
        raise RuntimeError(
            "Demand scorer output does not match "
            "DEMAND_SCORE_OUTPUT_COLUMNS."
        )

    if forecasts.empty:
        raise RuntimeError(
            "Demand scorer returned an empty forecast table."
        )

    observed_horizons = tuple(
        forecasts[
            "horizon_weeks"
        ]
        .tolist()
    )

    if observed_horizons != horizons:
        raise RuntimeError(
            "Demand scorer output horizons do not match "
            "the requested horizon register."
        )

    if forecasts[
        "forecast_origin_week"
    ].nunique(
        dropna=False
    ) != 1:
        raise RuntimeError(
            "Demand scorer returned inconsistent forecast origins."
        )


# ================================================================
# SCHEDULED BATCH
# ================================================================


def run_demand_batch(
    transactions: pd.DataFrame,
    forecast_origin_week,
    *,
    horizons=DEMAND_FORECAST_HORIZONS,
    scored_at_utc: datetime | None = None,
) -> DemandBatchResult:
    """
    Execute one governed weekly Demand Forecast batch.

    Parameters
    ----------
    transactions:
        Canonical retail transactions.

    forecast_origin_week:
        Explicit Monday week-start timestamp representing the final
        complete observed week used as the forecast origin.

        Exact week-start and available-history validation are delegated
        to the governed DemandScorer.

    horizons:
        Supported forecast horizons. Defaults to the frozen production
        register.

    scored_at_utc:
        Optional scoring timestamp passed through for deterministic
        provenance when required by scheduled execution or testing.

    Returns
    -------
    DemandBatchResult
        Immutable orchestration result containing the resolved forecast
        origin, frozen requested horizon register, and canonical forecasts.
    """

    _validate_transactions(
        transactions
    )

    _validate_forecast_origin_week(
        forecast_origin_week
    )

    resolved_horizons = (
        _normalise_horizons(
            horizons
        )
    )

    scorer = DemandScorer(
        horizons=(
            resolved_horizons
        )
    )

    forecasts = scorer.score(
        transactions=transactions,
        forecast_origin_week=(
            forecast_origin_week
        ),
        scored_at_utc=(
            scored_at_utc
        ),
    )

    _validate_score_contract(
        forecasts=forecasts,
        horizons=resolved_horizons,
    )

    resolved_origin = pd.Timestamp(
        forecasts[
            "forecast_origin_week"
        ].iloc[
            0
        ]
    )

    return DemandBatchResult(
        forecast_origin_week=(
            resolved_origin
        ),
        horizons=(
            resolved_horizons
        ),
        forecasts=(
            forecasts
        ),
    )