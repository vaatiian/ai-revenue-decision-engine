"""
Research-parity weekly feature engine for the frozen Notebook 07
business-level demand forecasting architecture.

This module is intentionally separate from demand_features.py.

demand_features.py
    = general daily analytical feature pipeline.

demand_model_features.py
    = exact weekly production representation required by the final
      Notebook 07 forecasting architecture.

Frozen Notebook 07 rules
------------------------
Observation boundary:
    2011-11-30 23:59:59

Qualifying positive merchandise:
    is_merchandise
    AND quantity > 0
    AND unit_price > 0
    AND NOT is_cancellation
    AND date <= observation boundary

Customer identification:
    Both identified and unidentified customer purchases are retained.

Calendar:
    W-SUN periods.
    Week start = Monday.
    Week end = Sunday.

Weekly measures:
    revenue
    orders
    units
    active_customers
    active_products
    merchandise_lines

Missing calendar weeks:
    A complete regular weekly calendar is materialised.

Year-end zero-trading weeks:
    2009-12-28 / 2010-01-03
    2010-12-27 / 2011-01-02

These were explicitly audited in Notebook 07 and are retained as
observed zero-trading weeks.

Partial weeks:
    Partial start and partial end weeks are excluded from the locked
    modelling series.

The final forecasting formula is implemented in:
    src.models.demand_scorer
"""

from __future__ import annotations

import pandas as pd


# ================================================================
# FROZEN RESEARCH CONSTANTS
# ================================================================

DEMAND_WEEK_FREQUENCY = "W-SUN"

DEMAND_END_DATE = pd.Timestamp(
    "2011-11-30 23:59:59"
)

DEMAND_ZERO_TRADING_WEEK_PERIODS = (
    "2009-12-28/2010-01-03",
    "2010-12-27/2011-01-02",
)


# ================================================================
# REQUIRED CANONICAL INPUT
# ================================================================

DEMAND_REQUIRED_TRANSACTION_COLUMNS = (
    "date",
    "invoice",
    "customer",
    "product",
    "quantity",
    "unit_price",
    "net_merchandise_revenue",
    "is_cancellation",
)


DEMAND_WEEKLY_METRIC_COLUMNS = (
    "revenue",
    "orders",
    "units",
    "active_customers",
    "active_products",
    "merchandise_lines",
)


# ================================================================
# COLUMN VALIDATION
# ================================================================

def _validate_transaction_columns(
    transactions: pd.DataFrame,
) -> None:
    """
    Validate the canonical production columns needed to reconstruct
    Notebook 07 demand eligibility.

    Notebook 07 itself also contained an is_merchandise column.

    Production supports two valid representations:

    1. is_merchandise is already present; or
    2. canonical merchandise/non-merchandise revenue fields are present,
       allowing the merchandise flag to be reconstructed safely.
    """

    missing = [
        column
        for column in DEMAND_REQUIRED_TRANSACTION_COLUMNS
        if column not in transactions.columns
    ]

    if missing:
        raise ValueError(
            "Demand forecasting transactions are missing required "
            "columns: "
            + ", ".join(
                missing
            )
        )

    if (
        "is_merchandise"
        not in transactions.columns
        and
        "non_merchandise_revenue"
        not in transactions.columns
    ):
        raise ValueError(
            "Demand forecasting requires either is_merchandise or "
            "non_merchandise_revenue so the Notebook 07 merchandise "
            "eligibility rule can be reproduced."
        )


# ================================================================
# MERCHANDISE FLAG
# ================================================================

def _build_is_merchandise(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Return the Notebook-07-equivalent merchandise flag.

    Preferred path
    --------------
    If the canonical dataset already exposes is_merchandise, use it
    directly. This is exact research parity.

    Fallback path
    -------------
    The canonical production contract separates merchandise revenue
    from non-merchandise revenue. For positive-price, positive-quantity
    activity, a row with no non-merchandise revenue is the production
    equivalent of a merchandise row.

    The eligibility mask still separately applies:
        quantity > 0
        unit_price > 0
        NOT is_cancellation

    Therefore we do not add extra product/non-null or net-revenue
    restrictions that were absent from Notebook 07.
    """

    if (
        "is_merchandise"
        in data.columns
    ):
        return (
            data[
                "is_merchandise"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

    non_merchandise = pd.to_numeric(
        data[
            "non_merchandise_revenue"
        ],
        errors="coerce",
    )

    return (
        non_merchandise
        .fillna(
            0.0
        )
        .eq(
            0.0
        )
    )


# ================================================================
# QUALIFYING DEMAND LINES
# ================================================================

def build_demand_model_lines(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce Notebook 07 Step 63.6 exactly at production level.

    Notebook rule:

        df["is_merchandise"]
        &
        (df["quantity"] > 0)
        &
        (df["unit_price"] > 0)
        &
        (~df["is_cancellation"])
        &
        (df["date"] <= DEMAND_END_DATE)

    Important:
    - unidentified customers remain eligible;
    - no explicit product.notna() rule is added;
    - no explicit net_merchandise_revenue > 0 rule is added;
    - December 2011 after 30 November is excluded before weekly
      calendar construction.
    """

    if transactions.empty:
        raise ValueError(
            "Demand forecasting transactions must not be empty."
        )

    _validate_transaction_columns(
        transactions
    )

    data = transactions.copy()

    data[
        "date"
    ] = pd.to_datetime(
        data[
            "date"
        ],
        format="mixed",
        errors="raise",
    )

    data[
        "_is_merchandise"
    ] = _build_is_merchandise(
        data
    )

    quantity = pd.to_numeric(
        data[
            "quantity"
        ],
        errors="coerce",
    )

    unit_price = pd.to_numeric(
        data[
            "unit_price"
        ],
        errors="coerce",
    )

    cancellation = (
        data[
            "is_cancellation"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    qualifying_mask = (
        data[
            "_is_merchandise"
        ]
        &
        quantity.gt(
            0
        )
        &
        unit_price.gt(
            0
        )
        &
        ~cancellation
        &
        data[
            "date"
        ].le(
            DEMAND_END_DATE
        )
    )

    retained_columns = list(
        DEMAND_REQUIRED_TRANSACTION_COLUMNS
    )

    if (
        "is_merchandise"
        in data.columns
    ):
        retained_columns.append(
            "is_merchandise"
        )

    if (
        "non_merchandise_revenue"
        in data.columns
    ):
        retained_columns.append(
            "non_merchandise_revenue"
        )

    lines = (
        data.loc[
            qualifying_mask,
            retained_columns,
        ]
        .copy()
        .sort_values(
            [
                "date",
                "invoice",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    if lines.empty:
        raise ValueError(
            "No qualifying positive merchandise transactions remain "
            "for weekly demand forecasting."
        )

    if (
        lines[
            "date"
        ].max()
        >
        DEMAND_END_DATE
    ):
        raise RuntimeError(
            "Demand observation boundary leakage detected."
        )

    lines[
        "week_period"
    ] = (
        lines[
            "date"
        ]
        .dt.to_period(
            DEMAND_WEEK_FREQUENCY
        )
    )

    lines[
        "week_start"
    ] = (
        lines[
            "week_period"
        ]
        .dt.start_time
    )

    lines[
        "week_end"
    ] = (
        lines[
            "week_period"
        ]
        .dt.end_time
    )

    return lines


# ================================================================
# OBSERVED WEEKLY AGGREGATION
# ================================================================

def aggregate_weekly_demand(
    demand_lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce Notebook 07 Step 63.9 weekly aggregation.

    revenue
        sum(net_merchandise_revenue)

    orders
        unique invoices

    units
        sum(quantity)

    active_customers
        unique non-null customers

    active_products
        unique products

    merchandise_lines
        row count using product column size
    """

    required = (
        "week_period",
        "invoice",
        "customer",
        "product",
        "quantity",
        "net_merchandise_revenue",
    )

    missing = [
        column
        for column in required
        if column not in demand_lines.columns
    ]

    if missing:
        raise ValueError(
            "Demand lines are missing weekly aggregation columns: "
            + ", ".join(
                missing
            )
        )

    if demand_lines.empty:
        raise ValueError(
            "Demand lines must not be empty."
        )

    weekly = (
        demand_lines
        .groupby(
            "week_period",
            as_index=False,
        )
        .agg(
            revenue=(
                "net_merchandise_revenue",
                "sum",
            ),

            orders=(
                "invoice",
                "nunique",
            ),

            units=(
                "quantity",
                "sum",
            ),

            active_customers=(
                "customer",
                lambda values:
                    values
                    .dropna()
                    .nunique()
            ),

            active_products=(
                "product",
                "nunique",
            ),

            merchandise_lines=(
                "product",
                "size",
            ),
        )
    )

    weekly[
        "week_start"
    ] = (
        weekly[
            "week_period"
        ]
        .dt.start_time
    )

    weekly[
        "week_end"
    ] = (
        weekly[
            "week_period"
        ]
        .dt.end_time
    )

    weekly = (
        weekly
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    return weekly


# ================================================================
# COMPLETE WEEKLY CALENDAR
# ================================================================

def build_complete_weekly_calendar(
    demand_lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce Notebook 07 Step 63.10.

    Materialise every W-SUN period between the first and last
    qualifying demand week.
    """

    if demand_lines.empty:
        raise ValueError(
            "Demand lines must not be empty."
        )

    if (
        "week_period"
        not in demand_lines.columns
    ):
        raise ValueError(
            "Demand lines must contain week_period."
        )

    first_week = (
        demand_lines[
            "week_period"
        ]
        .min()
    )

    last_week = (
        demand_lines[
            "week_period"
        ]
        .max()
    )

    complete_week_periods = (
        pd.period_range(
            start=first_week,
            end=last_week,
            freq=DEMAND_WEEK_FREQUENCY,
        )
    )

    calendar = pd.DataFrame(
        {
            "week_period":
                complete_week_periods,
        }
    )

    calendar[
        "week_start"
    ] = (
        calendar[
            "week_period"
        ]
        .dt.start_time
    )

    calendar[
        "week_end"
    ] = (
        calendar[
            "week_period"
        ]
        .dt.end_time
    )

    return calendar


# ================================================================
# COMPLETE WEEKLY SERIES
# ================================================================

def build_complete_weekly_demand(
    demand_lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce Notebook 07 Steps 63.11–63.13.

    Observed demand is joined to a complete W-SUN calendar.

    Missing weeks remain NaN at this stage for explicit audit.
    """

    observed = aggregate_weekly_demand(
        demand_lines
    )

    calendar = build_complete_weekly_calendar(
        demand_lines
    )

    weekly = (
        calendar
        .merge(
            observed.drop(
                columns=[
                    "week_start",
                    "week_end",
                ]
            ),
            on="week_period",
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    weekly[
        "is_missing_week"
    ] = (
        weekly[
            "revenue"
        ]
        .isna()
    )

    data_start_date = (
        demand_lines[
            "date"
        ]
        .min()
    )

    data_end_date = (
        demand_lines[
            "date"
        ]
        .max()
    )

    weekly[
        "is_partial_start_week"
    ] = (
        weekly[
            "week_start"
        ]
        <
        data_start_date.normalize()
    )

    weekly[
        "is_partial_end_week"
    ] = (
        weekly[
            "week_end"
        ]
        >
        data_end_date
    )

    return weekly


# ================================================================
# YEAR-END ZERO-TRADING RESOLUTION
# ================================================================

def resolve_zero_trading_weeks(
    weekly_demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce Notebook 07 Step 63.36.

    Only the two audited year-end weeks are allowed to be converted
    from missing calendar observations to zero demand.

    Any other unexplained interior gap fails closed.
    """

    weekly = weekly_demand.copy()

    required = (
        "week_period",
        "is_missing_week",
        "is_partial_start_week",
        "is_partial_end_week",
        *DEMAND_WEEKLY_METRIC_COLUMNS,
    )

    missing = [
        column
        for column in required
        if column not in weekly.columns
    ]

    if missing:
        raise ValueError(
            "Weekly demand is missing zero-trading resolution columns: "
            + ", ".join(
                missing
            )
        )

    frozen_periods = pd.PeriodIndex(
        DEMAND_ZERO_TRADING_WEEK_PERIODS,
        freq=DEMAND_WEEK_FREQUENCY,
    )

    zero_trading_week_mask = (
        weekly[
            "week_period"
        ]
        .isin(
            frozen_periods
        )
        &
        weekly[
            "is_missing_week"
        ]
    )

    weekly.loc[
        zero_trading_week_mask,
        list(
            DEMAND_WEEKLY_METRIC_COLUMNS
        ),
    ] = 0

    weekly[
        "is_year_end_zero_trading_week"
    ] = (
        zero_trading_week_mask
    )

    unresolved_interior = (
        weekly[
            "is_missing_week"
        ]
        &
        ~weekly[
            "is_year_end_zero_trading_week"
        ]
        &
        ~weekly[
            "is_partial_start_week"
        ]
        &
        ~weekly[
            "is_partial_end_week"
        ]
    )

    if unresolved_interior.any():

        periods = (
            weekly.loc[
                unresolved_interior,
                "week_period",
            ]
            .astype(
                str
            )
            .tolist()
        )

        raise ValueError(
            "Unexplained interior missing weekly observations remain: "
            + ", ".join(
                periods
            )
        )

    return weekly


# ================================================================
# LOCK REGULAR WEEKLY MODELLING SERIES
# ================================================================

def lock_weekly_demand_model(
    weekly_demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the final Notebook 07 weekly-series lock.

    After the two governed zero-trading weeks have been filled, complete
    modelling weeks are defined solely as:

        NOT partial_start_week
        AND
        NOT partial_end_week

    The year-end zero-trading periods therefore remain in the model.
    """

    weekly = weekly_demand.copy()

    required = (
        "week_start",
        "week_end",
        "revenue",
        "is_partial_start_week",
        "is_partial_end_week",
        "is_year_end_zero_trading_week",
    )

    missing = [
        column
        for column in required
        if column not in weekly.columns
    ]

    if missing:
        raise ValueError(
            "Weekly demand is missing model-lock columns: "
            + ", ".join(
                missing
            )
        )

    weekly[
        "is_complete_week"
    ] = (
        ~weekly[
            "is_partial_start_week"
        ]
        &
        ~weekly[
            "is_partial_end_week"
        ]
    )

    model = (
        weekly.loc[
            weekly[
                "is_complete_week"
            ]
        ]
        .copy()
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    if model.empty:
        raise ValueError(
            "No complete weekly forecasting observations remain."
        )

    if (
        model[
            "revenue"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Locked weekly demand model contains missing revenue."
        )

    spacing = (
        model[
            "week_start"
        ]
        .diff()
        .dropna()
    )

    if not (
        spacing.eq(
            pd.Timedelta(
                days=7
            )
        )
        .all()
    ):
        raise RuntimeError(
            "Locked weekly demand model is not spaced exactly "
            "seven days apart."
        )

    return model


# ================================================================
# END-TO-END WEEKLY MODEL TABLE
# ================================================================

def build_weekly_demand_model(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete research-parity Notebook 07 weekly modelling
    series from canonical transaction data.
    """

    demand_lines = build_demand_model_lines(
        transactions
    )

    weekly = build_complete_weekly_demand(
        demand_lines
    )

    weekly = resolve_zero_trading_weeks(
        weekly
    )

    model = lock_weekly_demand_model(
        weekly
    )

    return model