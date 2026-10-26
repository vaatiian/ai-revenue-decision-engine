"""
Tests for the frozen Future Customer Value model feature adapter.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.fcv_model_features import (
    build_fcv_customer_daily_history,
    build_fcv_eligible_population,
    build_fcv_last_region,
    build_fcv_model_features,
    build_fcv_positive_purchase_history,
    summarise_fcv_customer_window,
    summarise_fcv_product_window,
    summarise_fcv_purchase_gaps,
)
from src.models.fcv_spec import (
    FCV_MODEL_FEATURES,
)


# ================================================================
# TEST DATA
# ================================================================

def make_transactions():

    return pd.DataFrame(
        [
            {
                "date": "2011-03-01",
                "customer": "A",
                "invoice": "I1",
                "product": "P1",
                "region": "North",
                "quantity": 2,
                "unit_price": 50.0,
                "net_merchandise_revenue": 100.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-03-15",
                "customer": "A",
                "invoice": "I2",
                "product": "P2",
                "region": "North",
                "quantity": 1,
                "unit_price": 200.0,
                "net_merchandise_revenue": 200.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-05-15",
                "customer": "A",
                "invoice": "I3",
                "product": "P1",
                "region": "South",
                "quantity": 3,
                "unit_price": 100.0,
                "net_merchandise_revenue": 300.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-07-15",
                "customer": "A",
                "invoice": "I4",
                "product": "P3",
                "region": "East",
                "quantity": 4,
                "unit_price": 100.0,
                "net_merchandise_revenue": 400.0,
                "is_cancellation": False,
                "is_return": False,
            },

            {
                "date": "2011-04-15",
                "customer": "B",
                "invoice": "I5",
                "product": "P4",
                "region": "West",
                "quantity": 1,
                "unit_price": 150.0,
                "net_merchandise_revenue": 150.0,
                "is_cancellation": False,
                "is_return": False,
            },

            {
                "date": "2011-08-10",
                "customer": "C",
                "invoice": "I6",
                "product": "P5",
                "region": "Central",
                "quantity": -1,
                "unit_price": 100.0,
                "net_merchandise_revenue": -100.0,
                "is_cancellation": False,
                "is_return": True,
            },

            {
                "date": "2011-09-15",
                "customer": "A",
                "invoice": "I7",
                "product": "P6",
                "region": "Future",
                "quantity": 1,
                "unit_price": 999.0,
                "net_merchandise_revenue": 999.0,
                "is_cancellation": False,
                "is_return": False,
            },
        ]
    )


# ================================================================
# QUALIFYING PURCHASE HISTORY
# ================================================================

def test_positive_purchase_history_excludes_future():

    history = (
        build_fcv_positive_purchase_history(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
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


def test_positive_purchase_history_excludes_return():

    history = (
        build_fcv_positive_purchase_history(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    assert "C" not in set(
        history[
            "customer"
        ]
    )


# ================================================================
# DAILY HISTORY
# ================================================================

def test_customer_daily_history():

    history = (
        build_fcv_positive_purchase_history(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    daily = (
        build_fcv_customer_daily_history(
            history
        )
    )

    row = (
        daily.loc[
            (
                daily["customer"]
                ==
                "A"
            )
            &
            (
                daily["date"]
                ==
                pd.Timestamp(
                    "2011-07-15"
                )
            )
        ]
        .iloc[0]
    )

    assert row["daily_revenue"] == 400.0

    assert row["daily_orders"] == 1

    assert row["daily_units"] == 4


# ================================================================
# WINDOW SEMANTICS
# ================================================================

def test_customer_window_uses_open_start_closed_end():

    history = (
        build_fcv_positive_purchase_history(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    daily = (
        build_fcv_customer_daily_history(
            history
        )
    )

    summary = (
        summarise_fcv_customer_window(
            daily_data=daily,
            customers=["A"],
            start_date="2011-05-15",
            end_date="2011-07-15",
            prefix="test",
        )
    )

    assert len(summary) == 1

    assert (
        summary.iloc[0][
            "test_revenue"
        ]
        ==
        400.0
    )


def test_product_window():

    history = (
        build_fcv_positive_purchase_history(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    summary = (
        summarise_fcv_product_window(
            line_data=history,
            customers=["A"],
            start_date="2011-03-04",
            end_date="2011-08-31",
            prefix="h180",
        )
    )

    row = summary.iloc[0]

    assert row["h180_unique_products"] == 3

    assert np.isclose(
        row["h180_average_unit_price"],
        133.33333333333334,
    )

    assert row["h180_maximum_unit_price"] == 200.0


# ================================================================
# PURCHASE GAPS
# ================================================================

def test_purchase_gap_summary():

    history = (
        build_fcv_positive_purchase_history(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    daily = (
        build_fcv_customer_daily_history(
            history
        )
    )

    summary = (
        summarise_fcv_purchase_gaps(
            daily_data=daily,
            customers=["A"],
            start_date="2011-03-04",
            end_date="2011-08-31",
            prefix="h180",
        )
    )

    row = summary.iloc[0]

    expected_gaps = np.array(
        [
            61.0,
            61.0,
        ]
    )

    assert np.isclose(
        row["h180_average_gap_days"],
        expected_gaps.mean(),
    )

    assert (
        row["h180_maximum_gap_days"]
        ==
        expected_gaps.max()
    )

    assert (
        row["h180_minimum_gap_days"]
        ==
        expected_gaps.min()
    )


# ================================================================
# REGION
# ================================================================

def test_last_region():

    history = (
        build_fcv_positive_purchase_history(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    region = (
        build_fcv_last_region(
            line_data=history,
            customers=["A"],
            prediction_date="2011-08-31",
        )
    )

    assert (
        region.iloc[0][
            "last_region"
        ]
        ==
        "East"
    )


# ================================================================
# ELIGIBILITY
# ================================================================

def test_eligible_population_uses_prior_180_days():

    history = (
        build_fcv_positive_purchase_history(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    daily = (
        build_fcv_customer_daily_history(
            history
        )
    )

    eligible = (
        build_fcv_eligible_population(
            daily_data=daily,
            prediction_date="2011-08-31",
        )
    )

    assert set(
        eligible[
            "customer"
        ]
    ) == {
        "A",
        "B",
    }


# ================================================================
# FINAL ECONOMICS
# ================================================================

def test_h180_economics():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
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

    assert row["h180_revenue"] == 900.0

    assert row["h180_orders"] == 3

    assert row["h180_active_days"] == 3

    assert row["h180_units"] == 8

    assert row["h180_unique_products"] == 3

    assert row["h180_average_order_value"] == 300.0

    assert row["h180_revenue_per_active_day"] == 300.0

    assert np.isclose(
        row["h180_units_per_order"],
        8 / 3,
    )

    assert row["h180_products_per_order"] == 1.0


def test_recent_and_previous_90d_economics():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
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

    assert row["r90_revenue"] == 400.0

    assert row["r90_orders"] == 1

    assert row["p90_revenue"] == 500.0

    assert row["p90_orders"] == 2

    assert row["r90_average_order_value"] == 400.0

    assert row["p90_average_order_value"] == 250.0


def test_recent_30_day_structural_zero():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
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

    assert row["r30_revenue"] == 0

    assert row["r30_orders"] == 0

    assert row["active_in_last_30d"] == 0


def test_previous_window_missing_average_order_value_is_nan():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
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

    assert np.isnan(
        row[
            "r90_average_order_value"
        ]
    )


# ================================================================
# MOMENTUM
# ================================================================

def test_revenue_momentum():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
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
            "r90_minus_p90_revenue"
        ]
        ==
        -100.0
    )

    assert np.isclose(
        row[
            "r90_to_p90_revenue_ratio"
        ],
        0.8,
    )

    assert np.isclose(
        row[
            "log_revenue_momentum"
        ],
        (
            np.log1p(400.0)
            -
            np.log1p(500.0)
        ),
    )


def test_recent_30_day_momentum():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
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

    assert row["recent30_revenue_share"] == 0.0

    assert row["recent30_order_share"] == 0.0


# ================================================================
# RECENCY / FREQUENCY / BREADTH
# ================================================================

def test_recency_activity_frequency_and_breadth():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
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

    assert row["days_since_last_purchase"] == 47

    assert row["active_in_last_30d"] == 0

    assert row["active_in_last_90d"] == 1

    assert row["h180_orders_per_30_days"] == 0.5

    assert np.isclose(
        row["h180_active_day_rate"],
        3 / 180,
    )

    assert np.isclose(
        row["r90_active_day_rate"],
        1 / 90,
    )

    assert np.isclose(
        row["p90_active_day_rate"],
        2 / 90,
    )

    assert (
        row[
            "product_breadth_change"
        ]
        ==
        -1
    )


# ================================================================
# TRANSFORMS / CALENDAR
# ================================================================

def test_log_features():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
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

    assert np.isclose(
        row["log1p_h180_revenue"],
        np.log1p(900.0),
    )

    assert np.isclose(
        row["log1p_r90_revenue"],
        np.log1p(400.0),
    )

    assert np.isclose(
        row["log1p_p90_revenue"],
        np.log1p(500.0),
    )

    assert np.isclose(
        row["log1p_h180_orders"],
        np.log1p(3),
    )


def test_prediction_calendar_features():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    row = features.iloc[0]

    month = 8

    assert np.isclose(
        row[
            "prediction_month_sin"
        ],
        np.sin(
            2
            *
            np.pi
            *
            month
            /
            12
        ),
    )

    assert np.isclose(
        row[
            "prediction_month_cos"
        ],
        np.cos(
            2
            *
            np.pi
            *
            month
            /
            12
        ),
    )


# ================================================================
# FROZEN REGISTER
# ================================================================

def test_exact_frozen_feature_order():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    assert list(
        features.columns
    ) == [
        "customer",
        *FCV_MODEL_FEATURES,
    ]


def test_exact_frozen_feature_count():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    assert (
        len(
            features.columns
        )
        ==
        51
    )

    assert (
        len(
            FCV_MODEL_FEATURES
        )
        ==
        50
    )


def test_model_feature_customers_are_unique():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    assert (
        features[
            "customer"
        ]
        .is_unique
    )


def test_no_future_customer_enters_model_features():

    features = (
        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-31",
        )
    )

    assert "Future" not in set(
        features[
            "last_region"
        ]
    )


def test_non_month_end_prediction_fails():

    with pytest.raises(
        ValueError,
        match="month-end",
    ):

        build_fcv_model_features(
            transactions=make_transactions(),
            prediction_date="2011-08-15",
        )