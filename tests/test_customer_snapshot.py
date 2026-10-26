import pandas as pd
import pytest

from src.features.customer_snapshot import (
    aggregate_customer_window,
    build_customer_snapshot,
    build_qualifying_merchandise_history,
    classify_customer_state,
)


# ================================================================
# TEST DATA FACTORY
# ================================================================

def make_transaction_data():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-01-01 10:00:00",
                    "2011-08-15 10:00:00",
                    "2011-07-01 10:00:00",
                    "2011-04-01 10:00:00",
                    "2010-12-01 10:00:00",
                    "2011-09-15 10:00:00",
                    "2011-08-20 10:00:00",
                    "2011-08-25 10:00:00",
                ]
            ),

            "customer": [
                "A",
                "A",
                "B",
                "C",
                "D",
                "A",
                None,
                "E",
            ],

            "invoice": [
                "I1",
                "I2",
                "I3",
                "I4",
                "I5",
                "I6",
                "I7",
                "I8",
            ],

            "quantity": [
                1,
                2,
                1,
                1,
                1,
                1,
                1,
                -1,
            ],

            "unit_price": [
                10.0,
                15.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                15.0,
            ],

            "net_merchandise_revenue": [
                10.0,
                30.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                -15.0,
            ],

            "is_cancellation": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],

            "is_return": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        }
    )


DECISION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


# ================================================================
# QUALIFYING HISTORY
# ================================================================

def test_future_transactions_are_excluded():

    df = make_transaction_data()

    history = (
        build_qualifying_merchandise_history(
            df,
            DECISION_DATE,
        )
    )

    assert (
        history[
            "date"
        ]
        <=
        DECISION_DATE
    ).all()

    assert "I6" not in set(
        history[
            "invoice"
        ]
    )


def test_missing_customer_is_excluded_from_customer_snapshot_history():

    df = make_transaction_data()

    history = (
        build_qualifying_merchandise_history(
            df,
            DECISION_DATE,
        )
    )

    assert history[
        "customer"
    ].notna().all()


def test_returns_are_excluded_from_positive_merchandise_history():

    df = make_transaction_data()

    history = (
        build_qualifying_merchandise_history(
            df,
            DECISION_DATE,
        )
    )

    assert "I8" not in set(
        history[
            "invoice"
        ]
    )


# ================================================================
# LIFECYCLE STATES
# ================================================================

@pytest.mark.parametrize(
    "days, expected",
    [
        (0, "Active"),
        (30, "Active"),
        (31, "At Risk"),
        (90, "At Risk"),
        (91, "Dormant"),
        (180, "Dormant"),
        (181, "Long Dormant"),
        (365, "Long Dormant"),
    ],
)
def test_customer_state_boundaries(
    days,
    expected,
):

    assert (
        classify_customer_state(
            days
        )
        ==
        expected
    )


# ================================================================
# CUSTOMER SNAPSHOT
# ================================================================

def test_snapshot_has_one_row_per_customer():

    df = make_transaction_data()

    snapshot = (
        build_customer_snapshot(
            df,
            DECISION_DATE,
        )
    )

    assert snapshot[
        "customer"
    ].is_unique


def test_snapshot_excludes_future_revenue():

    df = make_transaction_data()

    snapshot = (
        build_customer_snapshot(
            df,
            DECISION_DATE,
        )
    )

    customer_a = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "lifetime_revenue"
        ]
        ==
        40.0
    )


def test_snapshot_lifetime_orders():

    df = make_transaction_data()

    snapshot = (
        build_customer_snapshot(
            df,
            DECISION_DATE,
        )
    )

    customer_a = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "lifetime_orders"
        ]
        ==
        2
    )


def test_snapshot_recent_windows():

    df = make_transaction_data()

    snapshot = (
        build_customer_snapshot(
            df,
            DECISION_DATE,
        )
    )

    customer_a = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "orders_prior_30d"
        ]
        ==
        1
    )

    assert (
        customer_a[
            "revenue_prior_30d"
        ]
        ==
        30.0
    )


def test_snapshot_customer_states():

    df = make_transaction_data()

    snapshot = (
        build_customer_snapshot(
            df,
            DECISION_DATE,
        )
    )

    states = (
        snapshot.set_index(
            "customer"
        )[
            "customer_state"
        ]
        .to_dict()
    )

    assert states["A"] == "Active"
    assert states["B"] == "At Risk"
    assert states["C"] == "Dormant"
    assert states["D"] == "Long Dormant"


def test_snapshot_decision_date_is_constant():

    df = make_transaction_data()

    snapshot = (
        build_customer_snapshot(
            df,
            DECISION_DATE,
        )
    )

    assert (
        snapshot[
            "decision_date"
        ]
        ==
        DECISION_DATE
    ).all()


def test_invalid_window_fails():

    df = make_transaction_data()

    history = (
        build_qualifying_merchandise_history(
            df,
            DECISION_DATE,
        )
    )

    with pytest.raises(
        ValueError
    ):
        aggregate_customer_window(
            history,
            DECISION_DATE,
            0,
        )