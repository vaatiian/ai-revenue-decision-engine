"""
Tests for the research-compatible Revenue Leakage feature adapter.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.leakage_model_features import (
    build_customer_daily_history,
    build_last_region,
    build_leakage_base_features,
    build_leakage_model_features,
    build_positive_merchandise_purchases,
    build_prior_leakage_history,
    historical_month_end_dates,
    summarise_customer_window,
    summarise_product_window,
    summarise_purchase_gaps,
)
from src.models.leakage_spec import (
    LEAKAGE_MODEL_FEATURES,
)


# ================================================================
# SYNTHETIC TRANSACTIONS
# ================================================================

def make_transactions() -> pd.DataFrame:
    """
    Build enough history to test the three research windows.

    Prediction point:
        2011-08-31

    Current 90-day window:
        (2011-06-02, 2011-08-31]

    Previous 90-day window:
        (2011-03-04, 2011-06-02]
    """

    rows = [
        {
            "date": "2011-03-15",
            "customer": "A",
            "invoice": "A1",
            "product": "P1",
            "region": "UK",
            "quantity": 1,
            "unit_price": 100.0,
            "net_merchandise_revenue": 100.0,
            "is_cancellation": False,
            "is_return": False,
        },
        {
            "date": "2011-05-15",
            "customer": "A",
            "invoice": "A2",
            "product": "P2",
            "region": "UK",
            "quantity": 2,
            "unit_price": 100.0,
            "net_merchandise_revenue": 200.0,
            "is_cancellation": False,
            "is_return": False,
        },
        {
            "date": "2011-06-15",
            "customer": "A",
            "invoice": "A3",
            "product": "P1",
            "region": "UK",
            "quantity": 1,
            "unit_price": 100.0,
            "net_merchandise_revenue": 100.0,
            "is_cancellation": False,
            "is_return": False,
        },
        {
            "date": "2011-07-15",
            "customer": "A",
            "invoice": "A4",
            "product": "P2",
            "region": "France",
            "quantity": 2,
            "unit_price": 100.0,
            "net_merchandise_revenue": 200.0,
            "is_cancellation": False,
            "is_return": False,
        },
        {
            "date": "2011-08-20",
            "customer": "A",
            "invoice": "A5",
            "product": "P3",
            "region": "Germany",
            "quantity": 3,
            "unit_price": 100.0,
            "net_merchandise_revenue": 300.0,
            "is_cancellation": False,
            "is_return": False,
        },
        {
            "date": "2011-07-01",
            "customer": "B",
            "invoice": "B1",
            "product": "P4",
            "region": "UK",
            "quantity": 1,
            "unit_price": 80.0,
            "net_merchandise_revenue": 80.0,
            "is_cancellation": False,
            "is_return": False,
        },
        {
            "date": "2010-12-01",
            "customer": "H",
            "invoice": "H1",
            "product": "PX",
            "region": "UK",
            "quantity": 1,
            "unit_price": 10.0,
            "net_merchandise_revenue": 10.0,
            "is_cancellation": False,
            "is_return": False,
        },
        {
            "date": "2011-09-15",
            "customer": "A",
            "invoice": "A6",
            "product": "FUTURE",
            "region": "USA",
            "quantity": 50,
            "unit_price": 100.0,
            "net_merchandise_revenue": 5000.0,
            "is_cancellation": False,
            "is_return": False,
        },
        {
            "date": "2011-08-25",
            "customer": "A",
            "invoice": "RET1",
            "product": "P1",
            "region": "Germany",
            "quantity": -1,
            "unit_price": 100.0,
            "net_merchandise_revenue": -100.0,
            "is_cancellation": False,
            "is_return": True,
        },
    ]

    return pd.DataFrame(
        rows
    )


# ================================================================
# POSITIVE PURCHASE HISTORY
# ================================================================

def test_positive_purchase_history_excludes_future():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    assert (
        history[
            "purchase_day"
        ].max()
        <=
        pd.Timestamp(
            "2011-08-31"
        )
    )

    assert (
        "FUTURE"
        not in set(
            history[
                "product"
            ]
        )
    )


def test_positive_purchase_history_excludes_return():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    assert (
        "RET1"
        not in set(
            history[
                "invoice"
            ]
        )
    )


# ================================================================
# CUSTOMER-DAY AGGREGATION
# ================================================================

def test_customer_daily_history():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    daily = (
        build_customer_daily_history(
            history
        )
    )

    row = (
        daily.loc[
            (
                daily[
                    "customer"
                ]
                ==
                "A"
            )
            &
            (
                daily[
                    "date"
                ]
                ==
                pd.Timestamp(
                    "2011-08-20"
                )
            )
        ]
        .iloc[0]
    )

    assert (
        row[
            "daily_revenue"
        ]
        ==
        300.0
    )

    assert (
        row[
            "daily_orders"
        ]
        ==
        1
    )

    assert (
        row[
            "daily_units"
        ]
        ==
        3
    )


# ================================================================
# WINDOW SUMMARIES
# ================================================================

def test_customer_window_uses_open_start_closed_end():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    daily = (
        build_customer_daily_history(
            history
        )
    )

    summary = (
        summarise_customer_window(
            daily_data=daily,
            customers=["A"],
            start_date="2011-06-02",
            end_date="2011-08-31",
            prefix="b90",
        )
    )

    row = summary.iloc[0]

    assert (
        row[
            "b90_revenue"
        ]
        ==
        600.0
    )

    assert (
        row[
            "b90_orders"
        ]
        ==
        3
    )

    assert (
        row[
            "b90_active_days"
        ]
        ==
        3
    )


def test_product_window():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    summary = (
        summarise_product_window(
            line_data=history,
            customers=["A"],
            start_date="2011-06-02",
            end_date="2011-08-31",
            prefix="b90",
        )
    )

    row = summary.iloc[0]

    assert (
        row[
            "b90_unique_products"
        ]
        ==
        3
    )

    assert (
        row[
            "b90_average_unit_price"
        ]
        ==
        pytest.approx(
            100.0
        )
    )

    assert (
        row[
            "b90_maximum_unit_price"
        ]
        ==
        100.0
    )


# ================================================================
# PURCHASE CADENCE
# ================================================================

def test_purchase_gap_summary():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    daily = (
        build_customer_daily_history(
            history
        )
    )

    gaps = (
        summarise_purchase_gaps(
            daily_data=daily,
            customers=["A"],
            start_date="2011-06-02",
            end_date="2011-08-31",
            prefix="b90",
        )
    )

    row = gaps.iloc[0]

    assert (
        row[
            "b90_average_gap_days"
        ]
        ==
        pytest.approx(
            33.0
        )
    )

    assert (
        row[
            "b90_maximum_gap_days"
        ]
        ==
        36
    )

    assert (
        row[
            "b90_minimum_gap_days"
        ]
        ==
        30
    )


# ================================================================
# LAST REGION
# ================================================================

def test_last_region():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    result = (
        build_last_region(
            history,
            customers=["A"],
            prediction_date="2011-08-31",
        )
    )

    assert (
        result.iloc[0][
            "last_region"
        ]
        ==
        "Germany"
    )


# ================================================================
# BASE FEATURES
# ================================================================

def test_base_feature_economics():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    features = (
        build_leakage_base_features(
            history,
            "2011-08-31",
        )
    )

    row = (
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
        row[
            "b90_revenue"
        ]
        ==
        600.0
    )

    assert (
        row[
            "b90_orders"
        ]
        ==
        3
    )

    assert (
        row[
            "b90_average_order_value"
        ]
        ==
        pytest.approx(
            200.0
        )
    )

    assert (
        row[
            "b90_units_per_order"
        ]
        ==
        pytest.approx(
            2.0
        )
    )

    assert (
        row[
            "b90_products_per_order"
        ]
        ==
        pytest.approx(
            1.0
        )
    )


def test_previous_window_structural_zero():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    features = (
        build_leakage_base_features(
            history,
            "2011-08-31",
        )
    )

    row = (
        features.loc[
            features[
                "customer"
            ]
            ==
            "B"
        ]
        .iloc[0]
    )

    assert (
        row[
            "p90_revenue"
        ]
        ==
        0
    )

    assert (
        row[
            "p90_orders"
        ]
        ==
        0
    )

    assert np.isnan(
        row[
            "p90_average_order_value"
        ]
    )


def test_revenue_trend_features():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    features = (
        build_leakage_base_features(
            history,
            "2011-08-31",
        )
    )

    row = (
        features.loc[
            features[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    # Previous 90-day window is:
    # (2011-03-04, 2011-06-02]
    #
    # Therefore both:
    # 2011-03-15 = 100
    # 2011-05-15 = 200
    #
    # are included.
    assert (
        row[
            "p90_revenue"
        ]
        ==
        300.0
    )

    assert (
        row[
            "revenue_change_vs_previous_90d"
        ]
        ==
        300.0
    )

    assert (
        row[
            "revenue_ratio_vs_previous_90d"
        ]
        ==
        pytest.approx(
            2.0
        )
    )

    assert (
        row[
            "log_revenue_ratio_vs_previous_90d"
        ]
        ==
        pytest.approx(
            np.log1p(
                600.0
            )
            -
            np.log1p(
                300.0
            )
        )
    )


def test_recent_30_day_momentum():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    features = (
        build_leakage_base_features(
            history,
            "2011-08-31",
        )
    )

    row = (
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
        row[
            "r30_revenue"
        ]
        ==
        300.0
    )

    assert (
        row[
            "recent30_revenue_share"
        ]
        ==
        pytest.approx(
            0.5
        )
    )

    assert (
        row[
            "recent30_vs_earlier60_revenue_rate"
        ]
        ==
        pytest.approx(
            2.0
        )
    )


def test_recency_frequency_and_calendar():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    features = (
        build_leakage_base_features(
            history,
            "2011-08-31",
        )
    )

    row = (
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
        row[
            "days_since_last_purchase"
        ]
        ==
        11
    )

    assert (
        row[
            "b90_orders_per_30_days"
        ]
        ==
        pytest.approx(
            1.0
        )
    )

    expected_sin = np.sin(
        2
        *
        np.pi
        *
        8
        /
        12
    )

    expected_cos = np.cos(
        2
        *
        np.pi
        *
        8
        /
        12
    )

    assert (
        row[
            "prediction_month_sin"
        ]
        ==
        pytest.approx(
            expected_sin
        )
    )

    assert (
        row[
            "prediction_month_cos"
        ]
        ==
        pytest.approx(
            expected_cos
        )
    )


def test_log_features():

    history = (
        build_positive_merchandise_purchases(
            make_transactions(),
            "2011-08-31",
        )
    )

    features = (
        build_leakage_base_features(
            history,
            "2011-08-31",
        )
    )

    row = (
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
        row[
            "log1p_b90_revenue"
        ]
        ==
        pytest.approx(
            np.log1p(
                600.0
            )
        )
    )

    assert (
        row[
            "log1p_b90_orders"
        ]
        ==
        pytest.approx(
            np.log1p(
                3.0
            )
        )
    )


# ================================================================
# PRIOR LEAKAGE HISTORY
# ================================================================

def test_prior_leakage_history_only_uses_matured_outcomes():

    historical = pd.DataFrame(
        {
            "customer": [
                "A",
                "A",
                "A",
            ],
            "prediction_date":
                pd.to_datetime(
                    [
                        "2011-01-31",
                        "2011-04-30",
                        "2011-07-31",
                    ]
                ),
            "outcome_end":
                pd.to_datetime(
                    [
                        "2011-05-01",
                        "2011-07-29",
                        "2011-10-29",
                    ]
                ),
            "revenue_leakage_event": [
                1,
                0,
                1,
            ],
        }
    )

    result = (
        build_prior_leakage_history(
            customers=["A"],
            historical_episodes=historical,
            current_prediction_date=(
                "2011-08-31"
            ),
        )
    )

    row = result.iloc[0]

    assert (
        row[
            "prior_completed_leakage_episodes"
        ]
        ==
        2
    )

    assert (
        row[
            "prior_leakage_events"
        ]
        ==
        1
    )

    assert (
        row[
            "prior_leakage_rate"
        ]
        ==
        pytest.approx(
            0.5
        )
    )

    assert (
        row[
            "has_prior_completed_leakage_episode"
        ]
        ==
        1
    )


def test_customer_without_prior_episode_gets_zero_history():

    result = (
        build_prior_leakage_history(
            customers=["B"],
            historical_episodes=pd.DataFrame(
                columns=[
                    "customer",
                    "prediction_date",
                    "outcome_end",
                    "revenue_leakage_event",
                ]
            ),
            current_prediction_date=(
                "2011-08-31"
            ),
        )
    )

    row = result.iloc[0]

    assert (
        row[
            "prior_completed_leakage_episodes"
        ]
        ==
        0
    )

    assert (
        row[
            "prior_leakage_events"
        ]
        ==
        0
    )

    assert np.isnan(
        row[
            "prior_leakage_rate"
        ]
    )

    assert (
        row[
            "has_prior_completed_leakage_episode"
        ]
        ==
        0
    )


# ================================================================
# HISTORICAL DATE GOVERNANCE
# ================================================================

def test_historical_prediction_dates_are_matured():

    current_prediction_date = (
        pd.Timestamp(
            "2011-08-31"
        )
    )

    dates = (
        historical_month_end_dates(
            source_start="2010-01-01",
            current_prediction_date=(
                current_prediction_date
            ),
        )
    )

    assert dates

    assert (
        dates[-1]
        ==
        pd.Timestamp(
            "2011-05-31"
        )
    )

    for date in dates:

        assert (
            date
            +
            pd.Timedelta(
                days=90
            )
            <=
            current_prediction_date
        )


# ================================================================
# FINAL FROZEN FEATURE MATRIX
# ================================================================

def test_exact_frozen_feature_order():

    features = (
        build_leakage_model_features(
            transactions=(
                make_transactions()
            ),
            prediction_date=(
                "2011-08-31"
            ),
        )
    )

    assert list(
        features.columns
    ) == [
        "customer",
        *LEAKAGE_MODEL_FEATURES,
    ]


def test_exact_frozen_feature_count():

    features = (
        build_leakage_model_features(
            transactions=(
                make_transactions()
            ),
            prediction_date=(
                "2011-08-31"
            ),
        )
    )

    assert (
        len(
            features.columns
        )
        ==
        46
    )


def test_model_feature_customers_are_unique():

    features = (
        build_leakage_model_features(
            transactions=(
                make_transactions()
            ),
            prediction_date=(
                "2011-08-31"
            ),
        )
    )

    assert not (
        features[
            "customer"
        ]
        .duplicated()
        .any()
    )


def test_non_month_end_prediction_fails():

    with pytest.raises(
        ValueError,
        match="month-end",
    ):

        build_leakage_model_features(
            transactions=(
                make_transactions()
            ),
            prediction_date=(
                "2011-08-15"
            ),
        )