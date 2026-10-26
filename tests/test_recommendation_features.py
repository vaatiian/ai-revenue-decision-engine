import pandas as pd
import pytest

from src.features.recommendation_features import (
    add_recency_weighted_affinity,
    build_candidate_products,
    build_customer_product_features,
    build_customer_product_history,
    build_customer_product_profile,
    build_product_popularity,
    build_recommendation_features,
)


PREDICTION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


def make_recommendation_data():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-02-01",
                    "2011-07-15",
                    "2011-08-15",
                    "2011-08-20",
                    "2011-08-25",
                    "2011-09-15",
                    "2011-08-10",
                ]
            ),

            "customer": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "C",
                "A",
                "D",
            ],

            "invoice": [
                "A1",
                "A2",
                "A3",
                "B1",
                "B2",
                "C1",
                "A4",
                "D1",
            ],

            "product": [
                "P1",
                "P1",
                "P2",
                "P1",
                "P3",
                "P2",
                "P4",
                "P5",
            ],

            "region": [
                "UK",
                "UK",
                "UK",
                "France",
                "France",
                "Germany",
                "UK",
                "UK",
            ],

            "quantity": [
                1,
                2,
                1,
                1,
                3,
                1,
                1,
                -1,
            ],

            "unit_price": [
                10.0,
                10.0,
                20.0,
                10.0,
                15.0,
                20.0,
                50.0,
                10.0,
            ],

            "net_merchandise_revenue": [
                10.0,
                20.0,
                20.0,
                10.0,
                45.0,
                20.0,
                50.0,
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


def test_customer_product_history():

    df = make_recommendation_data()

    history = build_customer_product_history(
        df,
        PREDICTION_DATE,
    )

    a_p1 = (
        history.loc[
            (history["customer"] == "A")
            &
            (history["product"] == "P1")
        ]
        .iloc[0]
    )

    assert a_p1["customer_product_orders"] == 2
    assert a_p1["customer_product_revenue"] == 30.0
    assert a_p1["customer_product_units"] == 3


def test_customer_product_history_excludes_future_purchase():

    df = make_recommendation_data()

    history = build_customer_product_history(
        df,
        PREDICTION_DATE,
    )

    assert "P4" not in set(
        history.loc[
            history["customer"] == "A",
            "product",
        ]
    )


def test_customer_product_history_excludes_return():

    df = make_recommendation_data()

    history = build_customer_product_history(
        df,
        PREDICTION_DATE,
    )

    assert "D" not in set(
        history["customer"]
    )


def test_customer_product_profile():

    df = make_recommendation_data()

    profile = build_customer_product_profile(
        df,
        PREDICTION_DATE,
    )

    customer_a = (
        profile.loc[
            profile["customer"] == "A"
        ]
        .iloc[0]
    )

    assert customer_a["customer_total_orders"] == 3
    assert customer_a["customer_total_revenue"] == 50.0
    assert customer_a["customer_unique_products"] == 2


def test_product_popularity():

    df = make_recommendation_data()

    popularity = build_product_popularity(
        df,
        PREDICTION_DATE,
    )

    p1 = (
        popularity.loc[
            popularity["product"] == "P1"
        ]
        .iloc[0]
    )

    assert p1["product_total_orders"] == 3
    assert p1["product_total_revenue"] == 40.0
    assert p1["product_unique_customers"] == 2


def test_recency_weighted_affinity():

    df = make_recommendation_data()

    history = build_customer_product_history(
        df,
        PREDICTION_DATE,
    )

    result = add_recency_weighted_affinity(
        history,
        half_life_days=90,
    )

    assert "recency_decay" in result.columns
    assert "recency_weighted_affinity" in result.columns

    assert (
        result["recency_weighted_affinity"]
        >=
        0
    ).all()


def test_invalid_half_life_fails():

    df = make_recommendation_data()

    history = build_customer_product_history(
        df,
        PREDICTION_DATE,
    )

    with pytest.raises(
        ValueError
    ):
        add_recency_weighted_affinity(
            history,
            half_life_days=0,
        )


def test_customer_product_features_unique_pairs():

    df = make_recommendation_data()

    features = build_customer_product_features(
        df,
        PREDICTION_DATE,
    )

    assert not features.duplicated(
        subset=[
            "customer",
            "product",
        ]
    ).any()


def test_customer_product_revenue_share():

    df = make_recommendation_data()

    features = build_customer_product_features(
        df,
        PREDICTION_DATE,
    )

    a_p1 = (
        features.loc[
            (features["customer"] == "A")
            &
            (features["product"] == "P1")
        ]
        .iloc[0]
    )

    assert (
        a_p1["customer_product_revenue_share"]
        ==
        pytest.approx(
            30.0 / 50.0
        )
    )


def test_candidate_products_created():

    df = make_recommendation_data()

    candidates = build_candidate_products(
        df,
        PREDICTION_DATE,
        top_n_products=2,
    )

    assert not candidates.empty

    assert (
        candidates.groupby("customer")[
            "product"
        ]
        .nunique()
        .max()
        <=
        2
    )


def test_candidate_generation_excludes_future_product():

    df = make_recommendation_data()

    candidates = build_candidate_products(
        df,
        PREDICTION_DATE,
        top_n_products=10,
    )

    assert "P4" not in set(
        candidates["product"]
    )


def test_cross_sell_mode_excludes_previously_purchased():

    df = make_recommendation_data()

    candidates = build_candidate_products(
        df,
        PREDICTION_DATE,
        top_n_products=3,
        exclude_previously_purchased=True,
    )

    purchased = {
        ("A", "P1"),
        ("A", "P2"),
    }

    actual = set(
        zip(
            candidates["customer"],
            candidates["product"],
        )
    )

    for pair in purchased:
        assert pair not in actual


def test_invalid_top_n_fails():

    df = make_recommendation_data()

    with pytest.raises(
        ValueError
    ):
        build_candidate_products(
            df,
            PREDICTION_DATE,
            top_n_products=0,
        )


def test_recommendation_features_unique_pairs():

    df = make_recommendation_data()

    features = build_recommendation_features(
        df,
        PREDICTION_DATE,
        top_n_products=3,
    )

    assert not features.duplicated(
        subset=[
            "customer",
            "product",
        ]
    ).any()


def test_recommendation_features_include_unseen_pairs():

    df = make_recommendation_data()

    features = build_recommendation_features(
        df,
        PREDICTION_DATE,
        top_n_products=3,
    )

    b_p2 = features.loc[
        (features["customer"] == "B")
        &
        (features["product"] == "P2")
    ]

    assert len(b_p2) == 1

    row = b_p2.iloc[0]

    assert row["customer_product_orders"] == 0
    assert row["customer_product_revenue"] == 0


def test_prediction_metadata():

    df = make_recommendation_data()

    features = build_recommendation_features(
        df,
        PREDICTION_DATE,
        top_n_products=3,
    )

    assert (
        features["prediction_date"]
        ==
        PREDICTION_DATE
    ).all()

    assert (
        features["prediction_period"]
        ==
        "2011-08"
    ).all()


def test_required_recommendation_columns():

    df = make_recommendation_data()

    features = build_recommendation_features(
        df,
        PREDICTION_DATE,
        top_n_products=3,
    )

    required = {
        "customer",
        "product",
        "prediction_date",
        "prediction_period",
        "product_popularity_rank",
        "customer_product_orders",
        "customer_product_revenue",
        "customer_product_units",
        "recency_weighted_affinity",
        "customer_product_order_share",
        "customer_product_revenue_share",
        "customer_total_orders",
        "customer_total_revenue",
        "customer_unique_products",
        "product_total_orders",
        "product_total_revenue",
        "product_unique_customers",
    }

    assert required.issubset(
        set(
            features.columns
        )
    )