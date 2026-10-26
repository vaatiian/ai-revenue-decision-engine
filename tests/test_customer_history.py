import pandas as pd
import pytest

from src.features.customer_history import (
    add_inactivity_episode_flags,
    add_inter_order_gaps,
    build_customer_history_features,
    build_customer_order_history,
)


DECISION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


# ================================================================
# TEST DATA
# ================================================================

def make_transaction_data():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-01-01 10:00:00",
                    "2011-01-01 10:02:00",
                    "2011-02-01 10:00:00",
                    "2011-06-15 10:00:00",
                    "2011-08-15 10:00:00",
                    "2011-09-15 10:00:00",
                    "2011-08-20 10:00:00",
                ]
            ),

            "customer": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "A",
                None,
            ],

            "invoice": [
                "I1",
                "I1",
                "I2",
                "I3",
                "I4",
                "I5",
                "I6",
            ],

            "product": [
                "P1",
                "P2",
                "P1",
                "P3",
                "P4",
                "P5",
                "P6",
            ],

            "quantity": [
                1,
                2,
                1,
                1,
                3,
                1,
                1,
            ],

            "unit_price": [
                10.0,
                5.0,
                20.0,
                30.0,
                10.0,
                50.0,
                60.0,
            ],

            "net_merchandise_revenue": [
                10.0,
                10.0,
                20.0,
                30.0,
                30.0,
                50.0,
                60.0,
            ],

            "is_cancellation": [
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
            ],
        }
    )


# ================================================================
# ORDER HISTORY
# ================================================================

def test_order_history_collapses_invoice_lines():

    df = make_transaction_data()

    orders = build_customer_order_history(
        df,
        DECISION_DATE,
    )

    customer_a = orders.loc[
        orders["customer"] == "A"
    ]

    assert len(customer_a) == 3

    first_order = (
        customer_a.loc[
            customer_a["invoice"] == "I1"
        ]
        .iloc[0]
    )

    assert first_order["order_revenue"] == 20.0
    assert first_order["order_units"] == 3
    assert first_order["distinct_products"] == 2


def test_future_order_is_excluded():

    df = make_transaction_data()

    orders = build_customer_order_history(
        df,
        DECISION_DATE,
    )

    assert "I5" not in set(
        orders["invoice"]
    )


# ================================================================
# INTER-ORDER GAPS
# ================================================================

def test_inter_order_gap_calculation():

    df = make_transaction_data()

    orders = build_customer_order_history(
        df,
        DECISION_DATE,
    )

    orders = add_inter_order_gaps(
        orders
    )

    customer_a = (
        orders.loc[
            orders["customer"] == "A"
        ]
        .sort_values("order_date")
        .reset_index(drop=True)
    )

    assert pd.isna(
        customer_a.loc[
            0,
            "gap_days"
        ]
    )

    assert (
        round(
            customer_a.loc[
                1,
                "gap_days"
            ]
        )
        ==
        31
    )


# ================================================================
# REACTIVATION HISTORY
# ================================================================

def test_reactivation_flag_after_long_gap():

    df = make_transaction_data()

    orders = build_customer_order_history(
        df,
        DECISION_DATE,
    )

    flagged = add_inactivity_episode_flags(
        orders,
        inactivity_threshold_days=90,
    )

    customer_a = (
        flagged.loc[
            flagged["customer"] == "A"
        ]
        .sort_values("order_date")
        .reset_index(drop=True)
    )

    assert bool(
        customer_a.loc[
            2,
            "is_reactivation_order"
        ]
    ) is True


def test_invalid_inactivity_threshold_fails():

    df = make_transaction_data()

    orders = build_customer_order_history(
        df,
        DECISION_DATE,
    )

    with pytest.raises(
        ValueError
    ):
        add_inactivity_episode_flags(
            orders,
            inactivity_threshold_days=0,
        )


# ================================================================
# CUSTOMER FEATURES
# ================================================================

def test_history_features_one_row_per_customer():

    df = make_transaction_data()

    features = (
        build_customer_history_features(
            df,
            DECISION_DATE,
        )
    )

    assert features[
        "customer"
    ].is_unique


def test_history_features_order_count():

    df = make_transaction_data()

    features = (
        build_customer_history_features(
            df,
            DECISION_DATE,
        )
    )

    customer_a = (
        features.loc[
            features["customer"] == "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "historical_order_count"
        ]
        ==
        3
    )


def test_history_features_revenue():

    df = make_transaction_data()

    features = (
        build_customer_history_features(
            df,
            DECISION_DATE,
        )
    )

    customer_a = (
        features.loc[
            features["customer"] == "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "historical_revenue"
        ]
        ==
        70.0
    )


def test_history_features_units():

    df = make_transaction_data()

    features = (
        build_customer_history_features(
            df,
            DECISION_DATE,
        )
    )

    customer_a = (
        features.loc[
            features["customer"] == "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "historical_units"
        ]
        ==
        5
    )


def test_history_features_unique_products():

    df = make_transaction_data()

    features = (
        build_customer_history_features(
            df,
            DECISION_DATE,
        )
    )

    customer_a = (
        features.loc[
            features["customer"] == "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "historical_unique_products"
        ]
        ==
        3
    )


def test_history_features_average_order_value():

    df = make_transaction_data()

    features = (
        build_customer_history_features(
            df,
            DECISION_DATE,
        )
    )

    customer_a = (
        features.loc[
            features["customer"] == "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "historical_average_order_value"
        ]
        ==
        pytest.approx(
            70.0 / 3.0
        )
    )


def test_last_order_value():

    df = make_transaction_data()

    features = (
        build_customer_history_features(
            df,
            DECISION_DATE,
        )
    )

    customer_a = (
        features.loc[
            features["customer"] == "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "last_order_value"
        ]
        ==
        30.0
    )


def test_prediction_date_constant():

    df = make_transaction_data()

    features = (
        build_customer_history_features(
            df,
            DECISION_DATE,
        )
    )

    assert (
        features[
            "prediction_date"
        ]
        ==
        DECISION_DATE
    ).all()


def test_no_future_transaction_leakage():

    df = make_transaction_data()

    features = (
        build_customer_history_features(
            df,
            DECISION_DATE,
        )
    )

    assert (
        features[
            "last_order_date"
        ]
        <=
        DECISION_DATE
    ).all()