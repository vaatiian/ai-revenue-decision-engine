import pandas as pd
import pytest

from src.features.demand_model_features import (
    DEMAND_END_DATE,
    DEMAND_WEEK_FREQUENCY,
    DEMAND_ZERO_TRADING_WEEK_PERIODS,
    aggregate_weekly_demand,
    build_complete_weekly_calendar,
    build_complete_weekly_demand,
    build_demand_model_lines,
    build_weekly_demand_model,
    lock_weekly_demand_model,
    resolve_zero_trading_weeks,
)


# ================================================================
# TEST DATA
# ================================================================

def make_transactions(
    rows,
):
    """
    Build synthetic transaction rows using the exact Notebook 07
    eligibility fields.

    Default row is:
    - merchandise;
    - positive quantity;
    - positive unit price;
    - non-cancellation;
    - before DEMAND_END_DATE.
    """

    base = {
        "date": "2010-01-04 10:00",
        "invoice": "I1",
        "customer": "C1",
        "product": "P1",
        "quantity": 1,
        "unit_price": 10.0,
        "is_merchandise": True,
        "net_merchandise_revenue": 10.0,
        "is_cancellation": False,
    }

    records = []

    for row in rows:
        record = base.copy()

        record.update(
            row
        )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ================================================================
# FROZEN CONSTANTS
# ================================================================

def test_week_frequency_is_w_sun():
    assert (
        DEMAND_WEEK_FREQUENCY
        ==
        "W-SUN"
    )


def test_observation_boundary_is_frozen():
    assert (
        DEMAND_END_DATE
        ==
        pd.Timestamp(
            "2011-11-30 23:59:59"
        )
    )


def test_frozen_zero_trading_periods():
    assert (
        DEMAND_ZERO_TRADING_WEEK_PERIODS
        ==
        (
            "2009-12-28/2010-01-03",
            "2010-12-27/2011-01-02",
        )
    )


# ================================================================
# NOTEBOOK 07 ELIGIBILITY
# ================================================================

def test_demand_lines_allow_missing_customer():
    transactions = make_transactions(
        [
            {
                "customer": None,
            }
        ]
    )

    result = build_demand_model_lines(
        transactions
    )

    assert len(
        result
    ) == 1

    assert pd.isna(
        result.loc[
            0,
            "customer",
        ]
    )


@pytest.mark.parametrize(
    "override",
    [
        {
            "quantity": -1,
        },
        {
            "quantity": 0,
        },
        {
            "unit_price": 0,
        },
        {
            "unit_price": -1,
        },
        {
            "is_cancellation": True,
        },
        {
            "is_merchandise": False,
        },
        {
            "date":
                "2011-12-01 00:00:00",
        },
    ],
)
def test_notebook_07_ineligible_demand_line_is_excluded(
    override,
):
    transactions = make_transactions(
        [
            {},
            override,
        ]
    )

    result = build_demand_model_lines(
        transactions
    )

    assert len(
        result
    ) == 1


@pytest.mark.parametrize(
    "override",
    [
        {
            "net_merchandise_revenue": 0.0,
        },
        {
            "net_merchandise_revenue": -10.0,
        },
        {
            "product": None,
        },
        {
            "customer": None,
        },
    ],
)
def test_conditions_not_present_in_notebook_filter_are_not_silently_excluded(
    override,
):
    transactions = make_transactions(
        [
            {},
            {
                "invoice": "I2",
                **override,
            },
        ]
    )

    result = build_demand_model_lines(
        transactions
    )

    assert len(
        result
    ) == 2


def test_observation_boundary_is_inclusive():
    transactions = make_transactions(
        [
            {
                "date":
                    "2011-11-30 23:59:59",
            }
        ]
    )

    result = build_demand_model_lines(
        transactions
    )

    assert len(
        result
    ) == 1


def test_transaction_after_observation_boundary_is_excluded():
    transactions = make_transactions(
        [
            {
                "date":
                    "2011-11-30 23:59:59",
                "invoice":
                    "I1",
            },
            {
                "date":
                    "2011-12-01 00:00:00",
                "invoice":
                    "I2",
            },
        ]
    )

    result = build_demand_model_lines(
        transactions
    )

    assert len(
        result
    ) == 1

    assert (
        result[
            "invoice"
        ].tolist()
        ==
        [
            "I1",
        ]
    )


def test_is_merchandise_false_is_excluded():
    transactions = make_transactions(
        [
            {},
            {
                "invoice":
                    "I2",
                "is_merchandise":
                    False,
            },
        ]
    )

    result = build_demand_model_lines(
        transactions
    )

    assert (
        result[
            "invoice"
        ].tolist()
        ==
        [
            "I1",
        ]
    )


def test_missing_is_merchandise_and_missing_fallback_fail_closed():
    transactions = make_transactions(
        [
            {},
        ]
    ).drop(
        columns=[
            "is_merchandise",
        ]
    )

    with pytest.raises(
        ValueError,
        match="either is_merchandise or non_merchandise_revenue",
    ):
        build_demand_model_lines(
            transactions
        )


# ================================================================
# WEEKLY PERIOD
# ================================================================

def test_week_period_is_w_sun():
    transactions = make_transactions(
        [
            {
                "date":
                    "2010-01-06 12:00",
            }
        ]
    )

    result = build_demand_model_lines(
        transactions
    )

    assert str(
        result.loc[
            0,
            "week_period",
        ]
    ) == (
        "2010-01-04/2010-01-10"
    )

    assert (
        result.loc[
            0,
            "week_start",
        ]
        ==
        pd.Timestamp(
            "2010-01-04"
        )
    )


# ================================================================
# WEEKLY AGGREGATION
# ================================================================

def test_weekly_aggregation():
    transactions = make_transactions(
        [
            {
                "invoice":
                    "I1",
                "customer":
                    "C1",
                "product":
                    "P1",
                "quantity":
                    2,
                "net_merchandise_revenue":
                    20.0,
            },
            {
                "invoice":
                    "I1",
                "customer":
                    "C1",
                "product":
                    "P2",
                "quantity":
                    1,
                "net_merchandise_revenue":
                    15.0,
            },
            {
                "invoice":
                    "I2",
                "customer":
                    None,
                "product":
                    "P1",
                "quantity":
                    3,
                "net_merchandise_revenue":
                    30.0,
            },
        ]
    )

    lines = build_demand_model_lines(
        transactions
    )

    weekly = aggregate_weekly_demand(
        lines
    )

    row = weekly.iloc[
        0
    ]

    assert row[
        "revenue"
    ] == pytest.approx(
        65.0
    )

    assert row[
        "orders"
    ] == 2

    assert row[
        "units"
    ] == 6

    assert row[
        "active_customers"
    ] == 1

    assert row[
        "active_products"
    ] == 2

    assert row[
        "merchandise_lines"
    ] == 3


def test_weekly_revenue_can_include_zero_revenue_eligible_row():
    transactions = make_transactions(
        [
            {
                "invoice":
                    "I1",
                "net_merchandise_revenue":
                    10.0,
            },
            {
                "invoice":
                    "I2",
                "net_merchandise_revenue":
                    0.0,
            },
        ]
    )

    lines = build_demand_model_lines(
        transactions
    )

    assert len(
        lines
    ) == 2

    weekly = aggregate_weekly_demand(
        lines
    )

    assert (
        weekly.loc[
            0,
            "revenue",
        ]
        ==
        pytest.approx(
            10.0
        )
    )


# ================================================================
# COMPLETE CALENDAR
# ================================================================

def test_complete_calendar_fills_week_periods():
    transactions = make_transactions(
        [
            {
                "date":
                    "2010-01-04 10:00",
            },
            {
                "date":
                    "2010-01-25 10:00",
                "invoice":
                    "I2",
            },
        ]
    )

    lines = build_demand_model_lines(
        transactions
    )

    calendar = build_complete_weekly_calendar(
        lines
    )

    assert len(
        calendar
    ) == 4

    spacing = (
        calendar[
            "week_start"
        ]
        .diff()
        .dropna()
    )

    assert (
        spacing.eq(
            pd.Timedelta(
                days=7
            )
        )
        .all()
    )


def test_missing_week_is_audited_before_resolution():
    transactions = make_transactions(
        [
            {
                "date":
                    "2010-01-04 10:00",
            },
            {
                "date":
                    "2010-01-18 10:00",
                "invoice":
                    "I2",
            },
        ]
    )

    lines = build_demand_model_lines(
        transactions
    )

    weekly = build_complete_weekly_demand(
        lines
    )

    assert (
        weekly[
            "is_missing_week"
        ].sum()
        ==
        1
    )


# ================================================================
# MISSING-WEEK GOVERNANCE
# ================================================================

def test_unknown_interior_missing_week_fails_closed():
    transactions = make_transactions(
        [
            {
                "date":
                    "2010-01-04 10:00",
            },
            {
                "date":
                    "2010-01-18 10:00",
                "invoice":
                    "I2",
            },
        ]
    )

    lines = build_demand_model_lines(
        transactions
    )

    weekly = build_complete_weekly_demand(
        lines
    )

    with pytest.raises(
        ValueError,
        match="Unexplained interior missing",
    ):
        resolve_zero_trading_weeks(
            weekly
        )


def test_known_year_end_missing_week_is_zero_filled():
    transactions = make_transactions(
        [
            {
                "date":
                    "2009-12-21 10:00",
            },
            {
                "date":
                    "2010-01-04 10:00",
                "invoice":
                    "I2",
            },
        ]
    )

    lines = build_demand_model_lines(
        transactions
    )

    weekly = build_complete_weekly_demand(
        lines
    )

    resolved = resolve_zero_trading_weeks(
        weekly
    )

    zero_week = (
        resolved.loc[
            resolved[
                "is_year_end_zero_trading_week"
            ]
        ]
        .iloc[
            0
        ]
    )

    assert zero_week[
        "revenue"
    ] == 0

    assert zero_week[
        "orders"
    ] == 0

    assert zero_week[
        "units"
    ] == 0

    assert zero_week[
        "active_customers"
    ] == 0

    assert zero_week[
        "active_products"
    ] == 0

    assert zero_week[
        "merchandise_lines"
    ] == 0


# ================================================================
# PARTIAL-WEEK GOVERNANCE
# ================================================================

def test_partial_start_and_end_weeks_are_excluded():
    transactions = make_transactions(
        [
            {
                "date":
                    "2010-01-06 10:00",
            },
            {
                "date":
                    "2010-01-13 10:00",
                "invoice":
                    "I2",
            },
            {
                "date":
                    "2010-01-20 10:00",
                "invoice":
                    "I3",
            },
        ]
    )

    lines = build_demand_model_lines(
        transactions
    )

    weekly = build_complete_weekly_demand(
        lines
    )

    weekly = resolve_zero_trading_weeks(
        weekly
    )

    model = lock_weekly_demand_model(
        weekly
    )

    assert (
        pd.Timestamp(
            "2010-01-04"
        )
        not in
        set(
            model[
                "week_start"
            ]
        )
    )

    assert (
        pd.Timestamp(
            "2010-01-18"
        )
        not in
        set(
            model[
                "week_start"
            ]
        )
    )


def test_notebook_boundary_makes_final_week_partial():
    transactions = make_transactions(
        [
            {
                "date":
                    "2011-11-21 10:00",
                "invoice":
                    "I1",
            },
            {
                "date":
                    "2011-11-28 10:00",
                "invoice":
                    "I2",
            },
            {
                "date":
                    "2011-11-30 20:00",
                "invoice":
                    "I3",
            },
            {
                "date":
                    "2011-12-01 10:00",
                "invoice":
                    "I4",
            },
        ]
    )

    lines = build_demand_model_lines(
        transactions
    )

    weekly = build_complete_weekly_demand(
        lines
    )

    final_week = (
        weekly.loc[
            weekly[
                "week_start"
            ]
            ==
            pd.Timestamp(
                "2011-11-28"
            )
        ]
        .iloc[
            0
        ]
    )

    assert bool(
        final_week[
            "is_partial_end_week"
        ]
    ) is True


# ================================================================
# FINAL LOCKED SERIES
# ================================================================

def test_locked_model_is_exactly_weekly():
    dates = pd.date_range(
        "2010-01-04",
        periods=8,
        freq="W-MON",
    )

    transactions = make_transactions(
        [
            {
                "date":
                    date
                    +
                    pd.Timedelta(
                        hours=10
                    ),
                "invoice":
                    f"I{i}",
            }
            for i, date
            in enumerate(
                dates
            )
        ]
    )

    transactions = pd.concat(
        [
            transactions,
            make_transactions(
                [
                    {
                        "date":
                            "2010-01-10 12:00",
                        "invoice":
                            "FIRST_END",
                    },
                    {
                        "date":
                            "2010-02-28 12:00",
                        "invoice":
                            "LAST_END",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    model = build_weekly_demand_model(
        transactions
    )

    spacing = (
        model[
            "week_start"
        ]
        .diff()
        .dropna()
    )

    assert (
        spacing.eq(
            pd.Timedelta(
                days=7
            )
        )
        .all()
    )

    assert (
        model[
            "revenue"
        ]
        .notna()
        .all()
    )