"""
End-to-end validation for the frozen Notebook 07 weekly demand forecaster.

This script validates the production implementation against the canonical
cleaned transaction data and the frozen research specification.

Validation layers
-----------------
1. Load canonical cleaned_online_retail.parquet.
2. Build the production weekly demand series.
3. Verify exact W-SUN / Monday-start regular weekly calendar.
4. Verify the two audited year-end zero-trading weeks.
5. Reproduce the frozen 90% seasonal / 10% recent forecast architecture.
6. Recreate Notebook 07 multi-horizon historical backtesting.
7. Compare production historical WAPE and Forecast/Actual to the
   approximate frozen Notebook 07 results.
8. Validate the governance manifest and all governed source hashes.
9. Run the production scorer at the latest complete historical origin.

No model training occurs here.
No future information is used in forecast construction.
"""

from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# PROJECT IMPORT PATH
# ================================================================

PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
)

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from src.features.demand_model_features import (  # noqa: E402
    DEMAND_WEEK_FREQUENCY,
    DEMAND_ZERO_TRADING_WEEK_PERIODS,
    build_weekly_demand_model,
)

from src.models.demand_manifest import (  # noqa: E402
    DEMAND_MANIFEST_PATH,
    validate_demand_manifest_file,
)

from src.models.demand_scorer import (  # noqa: E402
    DemandScorer,
)

from src.models.demand_spec import (  # noqa: E402
    DEMAND_FINAL_DECISION,
    DEMAND_FORECAST_HORIZONS,
    DEMAND_MODEL_METADATA,
    DEMAND_RECENT_WEIGHT,
    DEMAND_RECENT_WINDOW_WEEKS,
    DEMAND_REPORTED_APPROX_HORIZON_FORECAST_ACTUAL,
    DEMAND_REPORTED_APPROX_HORIZON_WAPE,
    DEMAND_SEASONAL_LAG_WEEKS,
    DEMAND_SEASONAL_WEIGHT,
    calculate_demand_forecast,
)


# ================================================================
# PATHS / TOLERANCES
# ================================================================

DATA_PATH = (
    PROJECT_ROOT
    /
    "cleaned_online_retail.parquet"
)

# Notebook 07 reports these final metrics approximately rather than
# preserving them as untouched six-decimal holdout metrics.
#
# We therefore validate research parity to sensible summary tolerances.
WAPE_ABSOLUTE_TOLERANCE = 0.0025

FORECAST_ACTUAL_ABSOLUTE_TOLERANCE = 0.005


# ================================================================
# HELPERS
# ================================================================

def calculate_wape(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """
    Weighted Absolute Percentage Error.
    """

    actual_values = np.asarray(
        actual,
        dtype=float,
    )

    predicted_values = np.asarray(
        predicted,
        dtype=float,
    )

    denominator = np.abs(
        actual_values
    ).sum()

    if denominator == 0:
        raise ValueError(
            "Cannot calculate WAPE when total actual revenue is zero."
        )

    return float(
        np.abs(
            actual_values
            -
            predicted_values
        ).sum()
        /
        denominator
    )


def calculate_forecast_actual_ratio(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """
    Aggregate predicted revenue divided by aggregate actual revenue.
    """

    actual_total = float(
        np.asarray(
            actual,
            dtype=float,
        ).sum()
    )

    if actual_total == 0:
        raise ValueError(
            "Cannot calculate Forecast/Actual when actual total is zero."
        )

    predicted_total = float(
        np.asarray(
            predicted,
            dtype=float,
        ).sum()
    )

    return (
        predicted_total
        /
        actual_total
    )


# ================================================================
# NOTEBOOK 07 MULTI-HORIZON REPRODUCTION
# ================================================================

def build_historical_multihorizon_forecasts(
    weekly_demand_model: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the frozen Notebook 07 historical multi-horizon logic.

    Research logic:
    - recent component = trailing four-week average known at origin;
    - target = h weeks after origin;
    - seasonal analogue = target week minus 52 weeks;
    - no realised intermediate future demand enters prediction;
    - horizons = 1, 2, 4;
    - final point forecast = 90% seasonal + 10% recent.
    """

    data = (
        weekly_demand_model[
            [
                "week_start",
                "week_end",
                "revenue",
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    data[
        "recent_ma4_at_origin"
    ] = (
        data[
            "revenue"
        ]
        .rolling(
            window=DEMAND_RECENT_WINDOW_WEEKS,
            min_periods=DEMAND_RECENT_WINDOW_WEEKS,
        )
        .mean()
    )

    records = []

    for origin_index in range(
        len(
            data
        )
    ):

        origin_row = (
            data.iloc[
                origin_index
            ]
        )

        recent_ma4 = (
            origin_row[
                "recent_ma4_at_origin"
            ]
        )

        if pd.isna(
            recent_ma4
        ):
            continue

        for horizon in (
            DEMAND_FORECAST_HORIZONS
        ):

            target_index = (
                origin_index
                +
                horizon
            )

            if (
                target_index
                >=
                len(
                    data
                )
            ):
                continue

            seasonal_index = (
                target_index
                -
                DEMAND_SEASONAL_LAG_WEEKS
            )

            if seasonal_index < 0:
                continue

            target_row = (
                data.iloc[
                    target_index
                ]
            )

            seasonal_row = (
                data.iloc[
                    seasonal_index
                ]
            )

            actual = float(
                target_row[
                    "revenue"
                ]
            )

            seasonal_prediction = float(
                seasonal_row[
                    "revenue"
                ]
            )

            prediction = (
                calculate_demand_forecast(
                    seasonal_analogue=(
                        seasonal_prediction
                    ),
                    recent_four_week_average=(
                        float(
                            recent_ma4
                        )
                    ),
                )
            )

            records.append(
                {
                    "origin_week":
                        origin_row[
                            "week_start"
                        ],

                    "target_week":
                        target_row[
                            "week_start"
                        ],

                    "horizon":
                        horizon,

                    "actual":
                        actual,

                    "seasonal_prediction":
                        seasonal_prediction,

                    "recent_ma4_at_origin":
                        float(
                            recent_ma4
                        ),

                    "prediction":
                        prediction,
                }
            )

    forecasts = pd.DataFrame(
        records
    )

    if forecasts.empty:
        raise RuntimeError(
            "No historical multi-horizon demand forecasts were generated."
        )

    return forecasts


def summarise_historical_forecasts(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the principal frozen Notebook 07 horizon metrics.
    """

    rows = []

    for horizon in (
        DEMAND_FORECAST_HORIZONS
    ):

        group = (
            forecasts.loc[
                forecasts[
                    "horizon"
                ]
                ==
                horizon
            ]
            .copy()
        )

        if group.empty:
            raise RuntimeError(
                f"No historical forecasts found for horizon {horizon}."
            )

        actual = group[
            "actual"
        ]

        predicted = group[
            "prediction"
        ]

        errors = (
            predicted
            -
            actual
        )

        mae = float(
            np.abs(
                errors
            ).mean()
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        errors
                    )
                )
            )
        )

        bias = float(
            errors.mean()
        )

        wape = calculate_wape(
            actual,
            predicted,
        )

        forecast_actual = (
            calculate_forecast_actual_ratio(
                actual,
                predicted,
            )
        )

        rows.append(
            {
                "horizon":
                    horizon,

                "observations":
                    len(
                        group
                    ),

                "mae":
                    mae,

                "rmse":
                    rmse,

                "wape":
                    wape,

                "bias":
                    bias,

                "forecast_actual":
                    forecast_actual,
            }
        )

    return pd.DataFrame(
        rows
    )


# ================================================================
# VALIDATIONS
# ================================================================

def validate_weekly_calendar(
    weekly: pd.DataFrame,
) -> None:
    """
    Validate regular production weekly series.
    """

    if weekly.empty:
        raise AssertionError(
            "Weekly production series is empty."
        )

    if weekly[
        "revenue"
    ].isna().any():
        raise AssertionError(
            "Weekly production series contains missing revenue."
        )

    spacing = (
        weekly[
            "week_start"
        ]
        .diff()
        .dropna()
    )

    if not spacing.eq(
        pd.Timedelta(
            days=7
        )
    ).all():
        raise AssertionError(
            "Weekly production series is not exactly seven days apart."
        )

    for week_start in (
        weekly[
            "week_start"
        ]
    ):

        period = pd.Timestamp(
            week_start
        ).to_period(
            DEMAND_WEEK_FREQUENCY
        )

        if (
            period.start_time.normalize()
            !=
            pd.Timestamp(
                week_start
            ).normalize()
        ):
            raise AssertionError(
                "Weekly series contains a non-Monday W-SUN week start."
            )


def validate_zero_trading_weeks(
    weekly: pd.DataFrame,
) -> None:
    """
    Validate Notebook 07 audited year-end zero-trading periods.
    """

    expected_periods = pd.PeriodIndex(
        DEMAND_ZERO_TRADING_WEEK_PERIODS,
        freq=DEMAND_WEEK_FREQUENCY,
    )

    for period in expected_periods:

        match = (
            weekly.loc[
                weekly[
                    "week_period"
                ]
                ==
                period
            ]
        )

        if len(
            match
        ) != 1:
            raise AssertionError(
                "Expected exactly one governed zero-trading week for "
                f"{period}; found {len(match)}."
            )

        row = match.iloc[
            0
        ]

        if float(
            row[
                "revenue"
            ]
        ) != 0.0:
            raise AssertionError(
                f"Governed zero-trading week {period} does not have "
                "zero revenue."
            )

        if float(
            row[
                "orders"
            ]
        ) != 0.0:
            raise AssertionError(
                f"Governed zero-trading week {period} does not have "
                "zero orders."
            )

        if float(
            row[
                "units"
            ]
        ) != 0.0:
            raise AssertionError(
                f"Governed zero-trading week {period} does not have "
                "zero units."
            )

        if (
            bool(
                row[
                    "is_year_end_zero_trading_week"
                ]
            )
            is not True
        ):
            raise AssertionError(
                f"Governed zero-trading week {period} is not flagged."
            )


def validate_historical_research_parity(
    scorecard: pd.DataFrame,
) -> None:
    """
    Compare production historical results to the approximate frozen
    Notebook 07 scorecard.
    """

    for row in scorecard.itertuples(
        index=False
    ):

        horizon = int(
            row.horizon
        )

        expected_wape = (
            DEMAND_REPORTED_APPROX_HORIZON_WAPE[
                horizon
            ]
        )

        if not np.isclose(
            row.wape,
            expected_wape,
            atol=WAPE_ABSOLUTE_TOLERANCE,
            rtol=0.0,
        ):
            raise AssertionError(
                "Demand WAPE research-parity failure for "
                f"horizon {horizon}: "
                f"production={row.wape:.6f}, "
                f"notebook≈{expected_wape:.6f}."
            )

        expected_ratio = (
            DEMAND_REPORTED_APPROX_HORIZON_FORECAST_ACTUAL[
                horizon
            ]
        )

        if not np.isclose(
            row.forecast_actual,
            expected_ratio,
            atol=FORECAST_ACTUAL_ABSOLUTE_TOLERANCE,
            rtol=0.0,
        ):
            raise AssertionError(
                "Demand Forecast/Actual research-parity failure for "
                f"horizon {horizon}: "
                f"production={row.forecast_actual:.6f}, "
                f"notebook≈{expected_ratio:.6f}."
            )


def validate_latest_production_score(
    transactions: pd.DataFrame,
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the real production scorer using the latest complete origin that
    still has the necessary 52-week seasonal history.
    """

    origin = (
        weekly[
            "week_start"
        ].max()
    )

    scorer = DemandScorer()

    scores = scorer.score(
        transactions=transactions,
        forecast_origin_week=origin,
        scored_at_utc=(
            "2026-10-26T18:00:00Z"
        ),
    )

    if len(
        scores
    ) != len(
        DEMAND_FORECAST_HORIZONS
    ):
        raise AssertionError(
            "Production demand scorer returned an unexpected number "
            "of forecast rows."
        )

    if (
        tuple(
            scores[
                "horizon_weeks"
            ].tolist()
        )
        !=
        DEMAND_FORECAST_HORIZONS
    ):
        raise AssertionError(
            "Production demand scorer returned the wrong horizon order."
        )

    if scores[
        "forecast_revenue"
    ].isna().any():
        raise AssertionError(
            "Production demand scorer returned missing revenue forecasts."
        )

    if not np.isfinite(
        scores[
            "forecast_revenue"
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise AssertionError(
            "Production demand scorer returned non-finite forecasts."
        )

    if not scores[
        "seasonal_weight"
    ].eq(
        DEMAND_SEASONAL_WEIGHT
    ).all():
        raise AssertionError(
            "Production demand scorer seasonal weight drift detected."
        )

    if not scores[
        "recent_weight"
    ].eq(
        DEMAND_RECENT_WEIGHT
    ).all():
        raise AssertionError(
            "Production demand scorer recent weight drift detected."
        )

    if not scores[
        "model_name"
    ].eq(
        DEMAND_MODEL_METADATA.model_name
    ).all():
        raise AssertionError(
            "Production demand scorer model-name drift detected."
        )

    if not scores[
        "governance_decision"
    ].eq(
        DEMAND_FINAL_DECISION
    ).all():
        raise AssertionError(
            "Production demand scorer governance-decision drift detected."
        )

    return scores


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    """
    Execute the complete demand forecasting production validation.
    """

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Canonical demand dataset not found: {DATA_PATH}"
        )

    print(
        "Loading canonical demand data..."
    )

    transactions = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"Canonical transaction rows: {len(transactions):,}"
    )

    print(
        "Building research-parity weekly demand series..."
    )

    weekly = build_weekly_demand_model(
        transactions
    )

    print(
        f"Locked weekly observations: {len(weekly):,}"
    )

    print(
        "First complete week:",
        weekly[
            "week_start"
        ].min().date(),
    )

    print(
        "Last complete week:",
        weekly[
            "week_start"
        ].max().date(),
    )

    validate_weekly_calendar(
        weekly
    )

    print(
        "Weekly calendar validation: PASS"
    )

    validate_zero_trading_weeks(
        weekly
    )

    print(
        "Year-end zero-trading week validation: PASS"
    )

    print(
        "Reproducing Notebook 07 multi-horizon backtest..."
    )

    historical_forecasts = (
        build_historical_multihorizon_forecasts(
            weekly
        )
    )

    print(
        "Historical forecast records:",
        f"{len(historical_forecasts):,}",
    )

    scorecard = (
        summarise_historical_forecasts(
            historical_forecasts
        )
    )

    validate_historical_research_parity(
        scorecard
    )

    print(
        "Historical research-parity validation: PASS"
    )

    print()
    print(
        "Frozen historical horizon scorecard:"
    )

    for row in scorecard.itertuples(
        index=False
    ):

        print(
            f"  {int(row.horizon)} week(s): "
            f"n={int(row.observations):,}, "
            f"WAPE={row.wape:.4%}, "
            f"Forecast/Actual={row.forecast_actual:.4f}, "
            f"MAE={row.mae:,.2f}, "
            f"RMSE={row.rmse:,.2f}, "
            f"Bias={row.bias:,.2f}"
        )

    print()
    print(
        "Validating governed manifest and source hashes..."
    )

    manifest = validate_demand_manifest_file(
        DEMAND_MANIFEST_PATH,
        verify_source_hashes=True,
    )

    print(
        "Demand manifest/source hash validation: PASS"
    )

    print(
        "Research notebook SHA-256:",
        manifest[
            "source_sha256"
        ][
            "research_notebook"
        ],
    )

    print(
        "Production scorer SHA-256:",
        manifest[
            "source_sha256"
        ][
            "production_scorer"
        ],
    )

    print()
    print(
        "Running latest-origin production forecast..."
    )

    production_scores = (
        validate_latest_production_score(
            transactions=transactions,
            weekly=weekly,
        )
    )

    print(
        "Latest-origin production scoring: PASS"
    )

    print()

    display_columns = [
        "forecast_origin_week",
        "target_week",
        "horizon_weeks",
        "seasonal_analogue_revenue",
        "recent_four_week_average",
        "forecast_revenue",
    ]

    print(
        production_scores[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "DEMAND END-TO-END VALIDATION: PASS"
    )

    print(
        "Model:",
        DEMAND_MODEL_METADATA.model_name,
    )

    print(
        "Version:",
        DEMAND_MODEL_METADATA.model_version,
    )

    print(
        "Architecture:",
        f"{DEMAND_SEASONAL_WEIGHT:.0%} seasonal / "
        f"{DEMAND_RECENT_WEIGHT:.0%} recent",
    )

    print(
        "Horizons:",
        list(
            DEMAND_FORECAST_HORIZONS
        ),
    )

    print(
        "Governance decision:",
        DEMAND_FINAL_DECISION,
    )


if __name__ == "__main__":
    main()