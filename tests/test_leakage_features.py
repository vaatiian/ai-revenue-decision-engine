import pandas as pd
import pytest

from src.features.leakage_features import (
    build_leakage_customer_history,
    build_leakage_features,
    build_leakage_window_summary,
    build_prior_leakage_window_summary,
)


PREDICTION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


# ================================================================
# TEST DATA
# ================================================================

def make_leakage_data():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    # Customer A prior 90-day window
                    "2011-04-15",
                    "2011-05-15",

                    # Customer A current baseline window
                    "2011-07-01",
                    "2011-08-15",

                    # Customer B current baseline only
                    "2011-08-01",

                    # Customer C too old for current baseline
                    "2011-01-01",

                    # Future transaction — must never enter
                    "2011-09-15",

                    # Return — excluded
                    "2011-08-10",
                ]
            ),

            "customer": [
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
                "France",
                "Germany",
                "UK",
                "UK",
            ],

            "quantity": [
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


# ================================================================
# BASELINE WINDOW
# ================================================================

def test_baseline_window_summary():

    df = make_leakage_data()

    baseline = (
        build_leakage_window_summary(
            df,
            PREDICTION_DATE,
            window_days=90,
        )
    )

    customer_a = (
        baseline.loc[
            baseline[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "baseline_revenue"
        ]
        ==
        70.0
    )

    assert (
        customer_a[
            "baseline_orders"
        ]
        ==
        2
    )


def test_baseline_excludes_future_transaction():

    df = make_leakage_data()

    baseline = (
        build_leakage_window_summary(
            df,
            PREDICTION_DATE,
            window_days=90,
        )
    )

    customer_a = (
        baseline.loc[
            baseline[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "baseline_revenue"
        ]
        ==
        70.0
    )


def test_return_excluded_from_baseline():

    df = make_leakage_data()

    baseline = (
        build_leakage_window_summary(
            df,
            PREDICTION_DATE,
            window_days=90,
        )
    )

    assert "D" not in set(
        baseline[
            "customer"
        ]
    )


# ================================================================
# PRIOR WINDOW
# ================================================================

def test_prior_window_summary():

    df = make_leakage_data()

    prior = (
        build_prior_leakage_window_summary(
            df,
            PREDICTION_DATE,
            window_days=90,
        )
    )

    customer_a = (
        prior.loc[
            prior[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "prior_window_revenue"
        ]
        ==
        30.0
    )

    assert (
        customer_a[
            "prior_window_orders"
        ]
        ==
        2
    )


# ================================================================
# CUSTOMER HISTORY
# ================================================================

def test_leakage_customer_history():

    df = make_leakage_data()

    history = (
        build_leakage_customer_history(
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

    # Future A5 is excluded.
    assert (
        customer_a[
            "historical_revenue"
        ]
        ==
        100.0
    )

    assert (
        customer_a[
            "historical_order_count"
        ]
        ==
        4
    )


# ================================================================
# FINAL FEATURE TABLE
# ================================================================

def test_leakage_features_one_row_per_customer():

    df = make_leakage_data()

    features = (
        build_leakage_features(
            df,
            PREDICTION_DATE,
        )
    )

    assert features[
        "customer"
    ].is_unique


def test_leakage_scoring_population_requires_positive_baseline():

    df = make_leakage_data()

    features = (
        build_leakage_features(
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
    assert "C" not in customers
    assert "D" not in customers


def test_revenue_change_amount():

    df = make_leakage_data()

    features = (
        build_leakage_features(
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
            "revenue_change_amount"
        ]
        ==
        40.0
    )


def test_revenue_trend_ratio():

    df = make_leakage_data()

    features = (
        build_leakage_features(
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
            "revenue_trend_ratio"
        ]
        ==
        pytest.approx(
            70.0 / 30.0
        )
    )


def test_prediction_metadata():

    df = make_leakage_data()

    features = (
        build_leakage_features(
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


def test_no_future_leakage():

    df = make_leakage_data()

    features = (
        build_leakage_features(
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


def test_required_feature_columns_exist():

    df = make_leakage_data()

    features = (
        build_leakage_features(
            df,
            PREDICTION_DATE,
        )
    )

    expected = {
        "customer",
        "prediction_date",
        "prediction_period",
        "baseline_revenue",
        "baseline_orders",
        "baseline_units",
        "baseline_unique_products",
        "baseline_average_order_value",
        "prior_window_revenue",
        "prior_window_orders",
        "prior_window_units",
        "revenue_change_amount",
        "revenue_trend_ratio",
        "historical_order_count",
        "historical_revenue",
        "historical_units",
        "days_since_last_order",
        "tenure_days",
    }

    assert expected.issubset(
        set(
            features.columns
        )
    )


def test_invalid_baseline_window_fails():

    df = make_leakage_data()

    with pytest.raises(
        ValueError
    ):
        build_leakage_features(
            df,
            PREDICTION_DATE,
            baseline_window_days=0,
        )