import numpy as np
import pandas as pd
import pytest

from src.features.reactivation_model_features import (
    build_reactivation_cadence_features,
    build_reactivation_last_order_features,
    build_reactivation_model_features,
    build_reactivation_order_history,
)
from src.models.reactivation_spec import (
    REACTIVATION_MODEL_FEATURES,
)


PREDICTION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


def make_data():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-01-01",
                    "2011-02-01",
                    "2011-03-01",
                    "2011-04-01",
                    "2011-05-01",
                    "2011-09-10",
                ]
            ),

            "customer": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "A",
            ],

            "invoice": [
                "A1",
                "A1",
                "A2",
                "A3",
                "B1",
                "B2",
                "A4",
            ],

            "product": [
                "P1",
                "P2",
                "P1",
                "P3",
                "P1",
                "P2",
                "P4",
            ],

            "region": [
                "UK",
                "UK",
                "UK",
                "France",
                "Germany",
                "Germany",
                "UK",
            ],

            "quantity": [
                1,
                2,
                1,
                3,
                1,
                1,
                1,
            ],

            "unit_price": [
                10.0,
                20.0,
                15.0,
                10.0,
                12.0,
                14.0,
                50.0,
            ],

            "net_merchandise_revenue": [
                10.0,
                40.0,
                15.0,
                30.0,
                12.0,
                14.0,
                50.0,
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


def test_order_history_collapses_invoice_lines():

    orders = build_reactivation_order_history(
        make_data(),
        PREDICTION_DATE,
    )

    a1 = orders.loc[
        orders[
            "invoice"
        ]
        ==
        "A1"
    ].iloc[0]

    assert a1["order_value"] == 50.0
    assert a1["order_units"] == 3
    assert a1["order_product_count"] == 2


def test_order_history_excludes_future():

    orders = build_reactivation_order_history(
        make_data(),
        PREDICTION_DATE,
    )

    assert "A4" not in set(
        orders[
            "invoice"
        ]
    )


def test_cadence_features():

    cadence = build_reactivation_cadence_features(
        make_data(),
        PREDICTION_DATE,
    )

    customer_a = cadence.loc[
        cadence[
            "customer"
        ]
        ==
        "A"
    ].iloc[0]

    assert (
        customer_a[
            "historical_average_gap_days"
        ]
        >
        0
    )

    assert (
        customer_a[
            "historical_max_gap_days"
        ]
        >=
        customer_a[
            "historical_min_gap_days"
        ]
    )


def test_has_prior_purchase_gap():

    cadence = build_reactivation_cadence_features(
        make_data(),
        PREDICTION_DATE,
    )

    customer_a = cadence.loc[
        cadence[
            "customer"
        ]
        ==
        "A"
    ].iloc[0]

    assert (
        customer_a[
            "has_prior_purchase_gap"
        ]
        ==
        1
    )


def test_last_order_features():

    last = build_reactivation_last_order_features(
        make_data(),
        PREDICTION_DATE,
    )

    customer_a = last.loc[
        last[
            "customer"
        ]
        ==
        "A"
    ].iloc[0]

    assert customer_a[
        "last_order_value"
    ] == 30.0

    assert customer_a[
        "last_order_units"
    ] == 3

    assert customer_a[
        "last_order_product_count"
    ] == 1

    assert customer_a[
        "last_order_region"
    ] == "France"


def test_exact_frozen_feature_order():

    features = build_reactivation_model_features(
        make_data(),
        PREDICTION_DATE,
    )

    assert list(
        features.columns[
            1:
        ]
    ) == list(
        REACTIVATION_MODEL_FEATURES
    )


def test_feature_count():

    features = build_reactivation_model_features(
        make_data(),
        PREDICTION_DATE,
    )

    assert (
        len(
            features.columns
        )
        ==
        30
    )


def test_model_features_unique_customer():

    features = build_reactivation_model_features(
        make_data(),
        PREDICTION_DATE,
    )

    assert not features[
        "customer"
    ].duplicated().any()


def test_log_features():

    features = build_reactivation_model_features(
        make_data(),
        PREDICTION_DATE,
    )

    row = features.iloc[
        0
    ]

    assert (
        row[
            "log1p_historical_order_count"
        ]
        ==
        pytest.approx(
            np.log1p(
                row[
                    "historical_order_count"
                ]
            )
        )
    )


def test_prediction_calendar_features():

    features = build_reactivation_model_features(
        make_data(),
        PREDICTION_DATE,
    )

    assert (
        features[
            "prediction_month"
        ]
        ==
        8
    ).all()

    assert (
        features[
            "prediction_weekday"
        ]
        ==
        PREDICTION_DATE.dayofweek
    ).all()


def test_history_flags_are_binary():

    features = build_reactivation_model_features(
        make_data(),
        PREDICTION_DATE,
    )

    for column in [
        "has_prior_purchase_gap",
        "has_prior_inactivity_episode",
        "has_prior_successful_reactivation",
    ]:

        assert set(
            features[
                column
            ]
            .dropna()
            .unique()
        ).issubset(
            {
                0,
                1,
            }
        )


def test_inactivity_ratio_exists():

    features = build_reactivation_model_features(
        make_data(),
        PREDICTION_DATE,
    )

    assert (
        "inactivity_to_average_gap_ratio"
        in features.columns
    )


def test_invalid_inactivity_threshold():

    with pytest.raises(
        ValueError
    ):
        build_reactivation_model_features(
            make_data(),
            PREDICTION_DATE,
            inactivity_threshold_days=0,
        )