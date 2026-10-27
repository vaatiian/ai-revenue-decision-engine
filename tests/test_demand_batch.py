from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pandas as pd
import pytest

import src.deployment.demand_batch as demand_batch_module

from src.deployment.demand_batch import (
    DemandBatchResult,
    run_demand_batch,
)
from src.models.demand_scorer import (
    DEMAND_SCORE_OUTPUT_COLUMNS,
)
from src.models.demand_spec import (
    DEMAND_FORECAST_HORIZONS,
)


FORECAST_ORIGIN = pd.Timestamp(
    "2011-01-03"
)

FIXED_SCORED_AT = datetime(
    2011,
    1,
    3,
    12,
    0,
    tzinfo=timezone.utc,
)


# ================================================================
# FIXTURES
# ================================================================


def make_transactions():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-01-03 10:00:00",
                ]
            ),
            "customer": [
                "A",
            ],
            "invoice": [
                "A1",
            ],
            "product": [
                "P1",
            ],
            "region": [
                "United Kingdom",
            ],
            "quantity": [
                1,
            ],
            "unit_price": [
                10.0,
            ],
            "net_merchandise_revenue": [
                10.0,
            ],
        }
    )


def make_forecasts(
    horizons=DEMAND_FORECAST_HORIZONS,
):

    rows = []

    for horizon in horizons:

        rows.append(
            {
                "forecast_origin_week":
                    FORECAST_ORIGIN,

                "target_week":
                    FORECAST_ORIGIN
                    +
                    pd.Timedelta(
                        weeks=horizon
                    ),

                "horizon_weeks":
                    horizon,

                "seasonal_analogue_week":
                    FORECAST_ORIGIN
                    +
                    pd.Timedelta(
                        weeks=(
                            horizon
                            -
                            52
                        )
                    ),

                "seasonal_analogue_revenue":
                    100.0,

                "recent_four_week_average":
                    110.0,

                "forecast_revenue":
                    101.0,

                "seasonal_weight":
                    0.90,

                "recent_weight":
                    0.10,

                "model_name":
                    "demand-model",

                "model_version":
                    "1.0.0",

                "model_type":
                    "deterministic",

                "target_name":
                    "weekly_positive_merchandise_revenue",

                "governance_decision":
                    "GO",

                "scored_at_utc":
                    FIXED_SCORED_AT,
            }
        )

    return pd.DataFrame(
        rows,
        columns=(
            DEMAND_SCORE_OUTPUT_COLUMNS
        ),
    )


# ================================================================
# INPUT VALIDATION
# ================================================================


def test_demand_batch_requires_dataframe_transactions():

    with pytest.raises(
        TypeError,
        match=(
            "transactions must be a pandas DataFrame"
        ),
    ):
        run_demand_batch(
            transactions=[],
            forecast_origin_week=(
                FORECAST_ORIGIN
            ),
        )


def test_demand_batch_requires_explicit_forecast_origin():

    with pytest.raises(
        ValueError,
        match=(
            "requires an explicit forecast_origin_week"
        ),
    ):
        run_demand_batch(
            transactions=make_transactions(),
            forecast_origin_week=None,
        )


def test_demand_batch_requires_nonempty_horizons():

    with pytest.raises(
        ValueError,
        match=(
            "requires at least one forecast horizon"
        ),
    ):
        run_demand_batch(
            transactions=make_transactions(),
            forecast_origin_week=(
                FORECAST_ORIGIN
            ),
            horizons=(),
        )


def test_demand_batch_rejects_duplicate_horizons():

    with pytest.raises(
        ValueError,
        match=(
            "must not contain duplicates"
        ),
    ):
        run_demand_batch(
            transactions=make_transactions(),
            forecast_origin_week=(
                FORECAST_ORIGIN
            ),
            horizons=(
                1,
                1,
            ),
        )


def test_demand_batch_rejects_unsupported_horizons():

    with pytest.raises(
        ValueError,
        match=(
            "unsupported forecast horizons"
        ),
    ):
        run_demand_batch(
            transactions=make_transactions(),
            forecast_origin_week=(
                FORECAST_ORIGIN
            ),
            horizons=(
                1,
                3,
            ),
        )


# ================================================================
# ORCHESTRATION
# ================================================================


def test_demand_batch_calls_governed_scorer(
    monkeypatch,
):

    transactions = make_transactions()

    requested_horizons = (
        1,
        4,
    )

    forecasts = make_forecasts(
        requested_horizons
    )

    captured = {}

    class FakeDemandScorer:

        def __init__(
            self,
            horizons,
        ):

            captured[
                "constructor_horizons"
            ] = horizons

        def score(
            self,
            transactions,
            forecast_origin_week,
            scored_at_utc=None,
        ):

            captured[
                "transactions"
            ] = transactions.copy()

            captured[
                "forecast_origin_week"
            ] = forecast_origin_week

            captured[
                "scored_at_utc"
            ] = scored_at_utc

            return forecasts.copy()

    monkeypatch.setattr(
        demand_batch_module,
        "DemandScorer",
        FakeDemandScorer,
    )

    result = run_demand_batch(
        transactions=transactions,
        forecast_origin_week=(
            FORECAST_ORIGIN
        ),
        horizons=(
            requested_horizons
        ),
        scored_at_utc=(
            FIXED_SCORED_AT
        ),
    )

    assert isinstance(
        result,
        DemandBatchResult,
    )

    assert (
        captured[
            "constructor_horizons"
        ]
        ==
        requested_horizons
    )

    pd.testing.assert_frame_equal(
        captured[
            "transactions"
        ],
        transactions,
    )

    assert (
        captured[
            "forecast_origin_week"
        ]
        ==
        FORECAST_ORIGIN
    )

    assert (
        captured[
            "scored_at_utc"
        ]
        ==
        FIXED_SCORED_AT
    )

    assert (
        result.forecast_origin_week
        ==
        FORECAST_ORIGIN
    )

    assert (
        result.horizons
        ==
        requested_horizons
    )

    pd.testing.assert_frame_equal(
        result.forecasts,
        forecasts,
    )


def test_demand_batch_uses_frozen_default_horizons(
    monkeypatch,
):

    captured = {}

    class FakeDemandScorer:

        def __init__(
            self,
            horizons,
        ):

            captured[
                "horizons"
            ] = horizons

        def score(
            self,
            transactions,
            forecast_origin_week,
            scored_at_utc=None,
        ):

            return make_forecasts(
                DEMAND_FORECAST_HORIZONS
            )

    monkeypatch.setattr(
        demand_batch_module,
        "DemandScorer",
        FakeDemandScorer,
    )

    result = run_demand_batch(
        transactions=make_transactions(),
        forecast_origin_week=(
            FORECAST_ORIGIN
        ),
    )

    assert (
        captured[
            "horizons"
        ]
        ==
        DEMAND_FORECAST_HORIZONS
    )

    assert (
        result.horizons
        ==
        DEMAND_FORECAST_HORIZONS
    )


# ================================================================
# FAIL-CLOSED OUTPUT CONTRACT
# ================================================================


def test_demand_batch_rejects_score_schema_drift(
    monkeypatch,
):

    class FakeDemandScorer:

        def __init__(
            self,
            horizons,
        ):
            pass

        def score(
            self,
            transactions,
            forecast_origin_week,
            scored_at_utc=None,
        ):

            return pd.DataFrame(
                {
                    "forecast_origin_week": [
                        FORECAST_ORIGIN,
                    ],
                }
            )

    monkeypatch.setattr(
        demand_batch_module,
        "DemandScorer",
        FakeDemandScorer,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "does not match "
            "DEMAND_SCORE_OUTPUT_COLUMNS"
        ),
    ):
        run_demand_batch(
            transactions=make_transactions(),
            forecast_origin_week=(
                FORECAST_ORIGIN
            ),
        )


def test_demand_batch_rejects_empty_scorer_output(
    monkeypatch,
):

    class FakeDemandScorer:

        def __init__(
            self,
            horizons,
        ):
            pass

        def score(
            self,
            transactions,
            forecast_origin_week,
            scored_at_utc=None,
        ):

            return pd.DataFrame(
                columns=(
                    DEMAND_SCORE_OUTPUT_COLUMNS
                )
            )

    monkeypatch.setattr(
        demand_batch_module,
        "DemandScorer",
        FakeDemandScorer,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "returned an empty forecast table"
        ),
    ):
        run_demand_batch(
            transactions=make_transactions(),
            forecast_origin_week=(
                FORECAST_ORIGIN
            ),
        )


def test_demand_batch_rejects_horizon_drift(
    monkeypatch,
):

    class FakeDemandScorer:

        def __init__(
            self,
            horizons,
        ):
            pass

        def score(
            self,
            transactions,
            forecast_origin_week,
            scored_at_utc=None,
        ):

            return make_forecasts(
                (
                    1,
                    2,
                )
            )

    monkeypatch.setattr(
        demand_batch_module,
        "DemandScorer",
        FakeDemandScorer,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "output horizons do not match"
        ),
    ):
        run_demand_batch(
            transactions=make_transactions(),
            forecast_origin_week=(
                FORECAST_ORIGIN
            ),
            horizons=(
                1,
                4,
            ),
        )


def test_demand_batch_rejects_inconsistent_forecast_origins(
    monkeypatch,
):

    forecasts = make_forecasts(
        (
            1,
            2,
        )
    )

    forecasts.loc[
        1,
        "forecast_origin_week",
    ] = pd.Timestamp(
        "2011-01-10"
    )

    class FakeDemandScorer:

        def __init__(
            self,
            horizons,
        ):
            pass

        def score(
            self,
            transactions,
            forecast_origin_week,
            scored_at_utc=None,
        ):

            return forecasts.copy()

    monkeypatch.setattr(
        demand_batch_module,
        "DemandScorer",
        FakeDemandScorer,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "inconsistent forecast origins"
        ),
    ):
        run_demand_batch(
            transactions=make_transactions(),
            forecast_origin_week=(
                FORECAST_ORIGIN
            ),
            horizons=(
                1,
                2,
            ),
        )


# ================================================================
# GOVERNED SCORER VALIDATION PASSTHROUGH
# ================================================================


def test_demand_batch_preserves_scorer_origin_validation(
    monkeypatch,
):

    class FakeDemandScorer:

        def __init__(
            self,
            horizons,
        ):
            pass

        def score(
            self,
            transactions,
            forecast_origin_week,
            scored_at_utc=None,
        ):

            raise ValueError(
                "Forecast origin must be the Monday week_start of a "
                "W-SUN forecasting period."
            )

    monkeypatch.setattr(
        demand_batch_module,
        "DemandScorer",
        FakeDemandScorer,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Forecast origin must be the Monday week_start"
        ),
    ):
        run_demand_batch(
            transactions=make_transactions(),
            forecast_origin_week=(
                pd.Timestamp(
                    "2011-01-04"
                )
            ),
        )


# ================================================================
# RESULT IMMUTABILITY
# ================================================================


def test_demand_batch_result_is_frozen():

    result = DemandBatchResult(
        forecast_origin_week=(
            FORECAST_ORIGIN
        ),
        horizons=(
            DEMAND_FORECAST_HORIZONS
        ),
        forecasts=(
            make_forecasts()
        ),
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.forecast_origin_week = (
            pd.Timestamp(
                "2011-01-10"
            )
        )