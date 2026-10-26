import pandas as pd
import pytest

from src.features.fcv_features import (
    build_fcv_customer_history,
    build_fcv_features,
    build_fcv_prior_90d_summary,
    build_fcv_window_summary,
)


PREDICTION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


# ================================================================
# TEST DATA
# ================================================================

def make_fcv_data():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    # Customer A older history
                    "2011-01-01",

                    # Customer A previous 90-day window
                    "2011-04-15",
                    "2011-05-15",

                    # Customer A current 90-day window
                    "2011-07-01",
                    "2011-08-15",

                    # Customer B active in last 30 days
                    "2011-08-20",

                    # Customer C too old for 180-day active population
                    "2010-12-01",

                    # Future A transaction — must never enter
                    "2011-09-15",

                    # Return — must not enter positive merchandise
                    "2011-08-25",
                ]
            ),

            "customer": [
                "A",
                "A",
                "A",
                "A",
                "A",
                "B",
                "C",
                "A",
                "D",
            ],

            "invoice": [
                "A0",
                "A1",
                "A2",
                "A3",
                "A4",
                "B1",
                "C1",
                "A5",
                "D1",
            ],

            "product": [
                "P0",
                "P1",
                "P2",
                "P1",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
            ],

            "region": [
                "UK",
                "UK",
                "UK",
                "UK",
                "UK",
                "France",
                "Germany",
                "UK",
                "UK",
            ],

            "quantity": [
                1,
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                -1,
            ],

            "unit_price": [
                5.0,
                10.0,
                20.0,
                15.0,
                40.0,
                50.0,
                60.0,
                70.0,
                10.0,
            ],

            "net_merchandise_revenue": [
                5.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                -10.0,
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
                False,
                True,
            ],
        }
    )


# ================================================================
# WINDOW FEATURES
# ================================================================

def test_fcv_30d_window():

    df = make_fcv_data()

    summary = (
        build_fcv_window_summary(
            df,
            PREDICTION_DATE,
            window_days=30,
        )
    )

    customer_a = (
        summary.loc[
            summary[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "revenue_prior_30d"
        ]
        ==
        40.0
    )

    assert (
        customer_a[
            "orders_prior_30d"
        ]
        ==
        1
    )


def test_fcv_90d_window():

    df = make_fcv_data()

    summary = (
        build_fcv_window_summary(
            df,
            PREDICTION_DATE,
            window_days=90,
        )
    )

    customer_a = (
        summary.loc[
            summary[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "revenue_prior_90d"
        ]
        ==
        70.0
    )

    assert (
        customer_a[
            "orders_prior_90d"
        ]
        ==
        2
    )


def test_fcv_180d_window():

    df = make_fcv_data()

    summary = (
        build_fcv_window_summary(
            df,
            PREDICTION_DATE,
            window_days=180,
        )
    )

    customer_a = (
        summary.loc[
            summary[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "revenue_prior_180d"
        ]
        ==
        100.0
    )

    assert (
        customer_a[
            "orders_prior_180d"
        ]
        ==
        4
    )


def test_fcv_window_excludes_future_transaction():

    df = make_fcv_data()

    summary = (
        build_fcv_window_summary(
            df,
            PREDICTION_DATE,
            window_days=180,
        )
    )

    customer_a = (
        summary.loc[
            summary[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "revenue_prior_180d"
        ]
        ==
        100.0
    )


def test_fcv_window_excludes_return():

    df = make_fcv_data()

    summary = (
        build_fcv_window_summary(
            df,
            PREDICTION_DATE,
            window_days=30,
        )
    )

    assert "D" not in set(
        summary[
            "customer"
        ]
    )


# ================================================================
# PREVIOUS 90 DAYS
# ================================================================

def test_previous_90d_summary():

    df = make_fcv_data()

    summary = (
        build_fcv_prior_90d_summary(
            df,
            PREDICTION_DATE,
        )
    )

    customer_a = (
        summary.loc[
            summary[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "revenue_previous_90d"
        ]
        ==
        30.0
    )

    assert (
        customer_a[
            "orders_previous_90d"
        ]
        ==
        2
    )


# ================================================================
# LIFETIME HISTORY
# ================================================================

def test_fcv_lifetime_history():

    df = make_fcv_data()

    history = (
        build_fcv_customer_history(
            df,
            PREDICTION_DATE,
        )
    )

    customer_a = (
        history.loc[
            history[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    # Future transaction excluded.
    assert (
        customer_a[
            "lifetime_revenue"
        ]
        ==
        105.0
    )

    assert (
        customer_a[
            "lifetime_orders"
        ]
        ==
        5
    )


def test_lifetime_average_order_value():

    df = make_fcv_data()

    history = (
        build_fcv_customer_history(
            df,
            PREDICTION_DATE,
        )
    )

    customer_a = (
        history.loc[
            history[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "lifetime_average_order_value"
        ]
        ==
        pytest.approx(
            105.0 / 5.0
        )
    )


# ================================================================
# FINAL FEATURE TABLE
# ================================================================

def test_fcv_features_one_row_per_customer():

    df = make_fcv_data()

    features = (
        build_fcv_features(
            df,
            PREDICTION_DATE,
        )
    )

    assert features[
        "customer"
    ].is_unique


def test_fcv_active_population():

    df = make_fcv_data()

    features = (
        build_fcv_features(
            df,
            PREDICTION_DATE,
        )
    )

    customers = set(
        features[
            "customer"
        ]
    )

    assert "A" in customers
    assert "B" in customers

    # Too old for the default 180-day active population.
    assert "C" not in customers

    # Return-only customer.
    assert "D" not in customers


def test_fcv_revenue_change():

    df = make_fcv_data()

    features = (
        build_fcv_features(
            df,
            PREDICTION_DATE,
        )
    )

    customer_a = (
        features.loc[
            features[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "revenue_change_90d"
        ]
        ==
        40.0
    )


def test_fcv_revenue_trend_ratio():

    df = make_fcv_data()

    features = (
        build_fcv_features(
            df,
            PREDICTION_DATE,
        )
    )

    customer_a = (
        features.loc[
            features[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "revenue_trend_ratio_90d"
        ]
        ==
        pytest.approx(
            70.0 / 30.0
        )
    )


def test_fcv_recent_average_order_value():

    df = make_fcv_data()

    features = (
        build_fcv_features(
            df,
            PREDICTION_DATE,
        )
    )

    customer_a = (
        features.loc[
            features[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "average_order_value_prior_90d"
        ]
        ==
        pytest.approx(
            35.0
        )
    )


def test_fcv_prediction_metadata():

    df = make_fcv_data()

    features = (
        build_fcv_features(
            df,
            PREDICTION_DATE,
        )
    )

    assert (
        features[
            "prediction_date"
        ]
        ==
        PREDICTION_DATE
    ).all()

    assert (
        features[
            "prediction_period"
        ]
        ==
        "2011-08"
    ).all()


def test_fcv_no_future_leakage():

    df = make_fcv_data()

    features = (
        build_fcv_features(
            df,
            PREDICTION_DATE,
        )
    )

    assert (
        features[
            "last_order_date"
        ]
        <=
        PREDICTION_DATE
    ).all()


def test_fcv_required_columns():

    df = make_fcv_data()

    features = (
        build_fcv_features(
            df,
            PREDICTION_DATE,
        )
    )

    required = {
        "customer",
        "prediction_date",
        "prediction_period",
        "revenue_prior_30d",
        "orders_prior_30d",
        "revenue_prior_90d",
        "orders_prior_90d",
        "revenue_previous_90d",
        "orders_previous_90d",
        "revenue_prior_180d",
        "orders_prior_180d",
        "units_prior_180d",
        "revenue_change_90d",
        "revenue_trend_ratio_90d",
        "lifetime_orders",
        "lifetime_revenue",
        "lifetime_average_order_value",
        "days_since_last_order",
        "customer_tenure_days",
        "orders_per_30d_recent_180d",
        "revenue_per_30d_recent_180d",
    }

    assert required.issubset(
        set(
            features.columns
        )
    )


def test_invalid_fcv_window_fails():

    df = make_fcv_data()

    with pytest.raises(
        ValueError
    ):
        build_fcv_window_summary(
            df,
            PREDICTION_DATE,
            window_days=0,
        )


def test_invalid_active_window_fails():

    df = make_fcv_data()

    with pytest.raises(
        ValueError
    ):
        build_fcv_features(
            df,
            PREDICTION_DATE,
            active_window_days=0,
        )