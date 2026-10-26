"""
Production demand forecasting feature pipeline.

This module builds leakage-safe aggregate demand features from canonical
retail transactions.

The demand layer is time-series based rather than customer based.

Core principles:
- only qualifying positive merchandise activity is used;
- transactions after an explicit cutoff date are excluded;
- demand is aggregated to daily grain;
- calendar features are deterministic;
- lag and rolling features use only prior observations;
- rolling statistics are shifted before calculation so the current
  day's target is never included in its own predictors.

No future demand target is created in this module.
"""

from __future__ import annotations

import pandas as pd

from src.features.customer_snapshot import (
    build_qualifying_merchandise_history,
    normalise_decision_date,
)


DEFAULT_LAGS = (
    1,
    7,
    14,
    28,
)

DEFAULT_ROLLING_WINDOWS = (
    7,
    28,
)


# ================================================================
# DAILY DEMAND
# ================================================================

def build_daily_demand(
    transactions: pd.DataFrame,
    cutoff_date,
) -> pd.DataFrame:
    """
    Aggregate qualifying merchandise transactions to daily demand.

    Output fields:
    - date
    - revenue
    - units
    - orders
    - customers

    Missing calendar dates between the first and last observed dates
    are inserted with zero demand.
    """

    score_date = normalise_decision_date(
        cutoff_date
    )

    history = (
        build_qualifying_merchandise_history(
            transactions,
            score_date,
        )
    )

    output_columns = [
        "date",
        "revenue",
        "units",
        "orders",
        "customers",
    ]

    if history.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    history = history.copy()

    history[
        "demand_date"
    ] = (
        history[
            "date"
        ]
        .dt.normalize()
    )

    daily = (
        history
        .groupby(
            "demand_date",
            as_index=False,
        )
        .agg(
            revenue=(
                "net_merchandise_revenue",
                "sum",
            ),

            units=(
                "quantity",
                "sum",
            ),

            orders=(
                "invoice",
                "nunique",
            ),

            customers=(
                "customer",
                "nunique",
            ),
        )
        .rename(
            columns={
                "demand_date":
                    "date",
            }
        )
    )

    full_calendar = pd.DataFrame(
        {
            "date": pd.date_range(
                start=daily[
                    "date"
                ].min(),
                end=daily[
                    "date"
                ].max(),
                freq="D",
            )
        }
    )

    daily = (
        full_calendar
        .merge(
            daily,
            on="date",
            how="left",
        )
    )

    fill_columns = [
        "revenue",
        "units",
        "orders",
        "customers",
    ]

    daily[
        fill_columns
    ] = (
        daily[
            fill_columns
        ]
        .fillna(0)
    )

    daily[
        "orders"
    ] = (
        daily[
            "orders"
        ]
        .astype(int)
    )

    daily[
        "customers"
    ] = (
        daily[
            "customers"
        ]
        .astype(int)
    )

    if (
        daily[
            "date"
        ]
        >
        score_date.normalize()
    ).any():
        raise RuntimeError(
            "Future demand leakage detected."
        )

    return (
        daily
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )


# ================================================================
# CALENDAR FEATURES
# ================================================================

def add_calendar_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add deterministic calendar and seasonality features.
    """

    result = daily.copy()

    if result.empty:
        return result

    result[
        "year"
    ] = result[
        "date"
    ].dt.year

    result[
        "month"
    ] = result[
        "date"
    ].dt.month

    result[
        "quarter"
    ] = result[
        "date"
    ].dt.quarter

    result[
        "day_of_month"
    ] = result[
        "date"
    ].dt.day

    result[
        "day_of_week"
    ] = result[
        "date"
    ].dt.dayofweek

    result[
        "week_of_year"
    ] = (
        result[
            "date"
        ]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    result[
        "is_weekend"
    ] = (
        result[
            "day_of_week"
        ]
        >=
        5
    ).astype(int)

    result[
        "is_month_start"
    ] = (
        result[
            "date"
        ]
        .dt.is_month_start
        .astype(int)
    )

    result[
        "is_month_end"
    ] = (
        result[
            "date"
        ]
        .dt.is_month_end
        .astype(int)
    )

    return result


# ================================================================
# LAG FEATURES
# ================================================================

def add_lag_features(
    daily: pd.DataFrame,
    lags=DEFAULT_LAGS,
) -> pd.DataFrame:
    """
    Add historical lag features.

    Example:

        revenue_lag_7

    for day t contains revenue observed on day t-7.

    The current day's demand is never used in its own lag feature.
    """

    result = daily.copy()

    for lag in lags:

        if lag <= 0:
            raise ValueError(
                "Lag values must be greater than zero."
            )

        for metric in [
            "revenue",
            "units",
            "orders",
            "customers",
        ]:

            result[
                f"{metric}_lag_{lag}"
            ] = (
                result[
                    metric
                ]
                .shift(
                    lag
                )
            )

    return result


# ================================================================
# ROLLING FEATURES
# ================================================================

def add_rolling_features(
    daily: pd.DataFrame,
    rolling_windows=DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Add leakage-safe rolling statistics.

    The series is shifted by one day before rolling statistics are
    calculated.

    Therefore, for date t:

        rolling feature =
            function(demand before t)

    rather than including demand observed on date t.
    """

    result = daily.copy()

    for window in rolling_windows:

        if window <= 0:
            raise ValueError(
                "Rolling window values must be greater than zero."
            )

        for metric in [
            "revenue",
            "units",
            "orders",
            "customers",
        ]:

            prior_values = (
                result[
                    metric
                ]
                .shift(1)
            )

            result[
                f"{metric}_rolling_mean_{window}"
            ] = (
                prior_values
                .rolling(
                    window=window,
                    min_periods=1,
                )
                .mean()
            )

            result[
                f"{metric}_rolling_std_{window}"
            ] = (
                prior_values
                .rolling(
                    window=window,
                    min_periods=2,
                )
                .std()
            )

    return result


# ================================================================
# TREND FEATURES
# ================================================================

def add_demand_trend_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add simple historical demand trend features.

    Requires 7-day and 28-day rolling revenue means to already exist.
    """

    result = daily.copy()

    required = [
        "revenue_rolling_mean_7",
        "revenue_rolling_mean_28",
    ]

    missing = [
        column
        for column in required
        if column not in result.columns
    ]

    if missing:
        raise ValueError(
            "Demand trend features require rolling columns: "
            + ", ".join(missing)
        )

    denominator = (
        result[
            "revenue_rolling_mean_28"
        ]
        .replace(
            0,
            pd.NA,
        )
    )

    result[
        "revenue_short_vs_long_ratio"
    ] = (
        result[
            "revenue_rolling_mean_7"
        ]
        /
        denominator
    )

    result[
        "revenue_short_vs_long_change"
    ] = (
        result[
            "revenue_rolling_mean_7"
        ]
        -
        result[
            "revenue_rolling_mean_28"
        ]
    )

    return result


# ================================================================
# FINAL DEMAND FEATURE TABLE
# ================================================================

def build_demand_features(
    transactions: pd.DataFrame,
    cutoff_date,
    lags=DEFAULT_LAGS,
    rolling_windows=DEFAULT_ROLLING_WINDOWS,
    drop_incomplete_history: bool = False,
) -> pd.DataFrame:
    """
    Build the production demand forecasting feature table.

    Parameters
    ----------
    transactions:
        Canonical validated transaction data.

    cutoff_date:
        Latest transaction timestamp allowed into the feature table.

    lags:
        Historical lag lengths.

    rolling_windows:
        Leakage-safe rolling window lengths.

    drop_incomplete_history:
        If True, remove rows that do not yet have the maximum requested
        lag history.

    Returns
    -------
    pd.DataFrame
        Daily demand observations with historical predictors.
    """

    score_date = normalise_decision_date(
        cutoff_date
    )

    daily = build_daily_demand(
        transactions,
        score_date,
    )

    if daily.empty:
        return pd.DataFrame()

    features = add_calendar_features(
        daily
    )

    features = add_lag_features(
        features,
        lags=lags,
    )

    features = add_rolling_features(
        features,
        rolling_windows=rolling_windows,
    )

    if (
        7 in rolling_windows
        and
        28 in rolling_windows
    ):
        features = add_demand_trend_features(
            features
        )

    features[
        "feature_cutoff_date"
    ] = score_date

    features[
        "feature_period"
    ] = (
        features[
            "date"
        ]
        .dt.to_period(
            "M"
        )
        .astype(str)
    )

    if drop_incomplete_history:

        maximum_lag = max(
            lags
        )

        required_lag_columns = [
            f"revenue_lag_{maximum_lag}",
            f"units_lag_{maximum_lag}",
            f"orders_lag_{maximum_lag}",
            f"customers_lag_{maximum_lag}",
        ]

        features = (
            features
            .dropna(
                subset=required_lag_columns
            )
            .copy()
        )

    # ------------------------------------------------------------
    # SAFETY CHECKS
    # ------------------------------------------------------------

    if (
        features[
            "date"
        ]
        >
        score_date.normalize()
    ).any():
        raise RuntimeError(
            "Future demand observations entered the feature table."
        )

    if not features[
        "date"
    ].is_unique:
        raise RuntimeError(
            "Demand feature table contains duplicate dates."
        )

    if not features[
        "date"
    ].is_monotonic_increasing:
        raise RuntimeError(
            "Demand feature table is not ordered chronologically."
        )

    preferred_column_order = [
        "date",
        "feature_cutoff_date",
        "feature_period",

        "revenue",
        "units",
        "orders",
        "customers",

        "year",
        "month",
        "quarter",
        "day_of_month",
        "day_of_week",
        "week_of_year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
    ]

    remaining_columns = [
        column
        for column in features.columns
        if column not in preferred_column_order
    ]

    features = features[
        preferred_column_order
        +
        remaining_columns
    ]

    return (
        features
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )