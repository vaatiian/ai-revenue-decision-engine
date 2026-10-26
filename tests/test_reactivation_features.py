import pandas as pd
import pytest

from src.features.reactivation_features import (
    build_current_reactivation_eligibility,
    build_historical_inactivity_episodes,
    build_prior_reactivation_features,
    build_reactivation_features,
)


PREDICTION_DATE = pd.Timestamp(
    "2011-12-31 23:59:59"
)


def make_reactivation_data():
    """
    Customer A:
    Jan 1 -> May 1 = long gap.
    Eligibility after Jan order = Apr 1.
    May 1 is within 90 days after Apr 1 -> successful prior reactivation.

    May 1 -> Dec 31 prediction date:
    currently inactive > 90 days -> currently eligible.

    Customer B:
    Sep 15 last order -> not yet 90 days inactive on Dec 1 example,
    but eligible by Dec 31 depending exact dates.

    Customer C:
    recent purchase -> not eligible.
    """

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-05-01",
                    "2011-09-15",
                    "2011-12-15",
                ]
            ),

            "customer": [
                "A",
                "A",
                "B",
                "C",
            ],

            "invoice": [
                "A1",
                "A2",
                "B1",
                "C1",
            ],

            "product": [
                "P1",
                "P2",
                "P3",
                "P4",
            ],

            "region": [
                "UK",
                "UK",
                "France",
                "Germany",
            ],

            "quantity": [
                1,
                2,
                1,
                1,
            ],

            "unit_price": [
                10.0,
                15.0,
                20.0,
                30.0,
            ],

            "net_merchandise_revenue": [
                10.0,
                30.0,
                20.0,
                30.0,
            ],

            "is_cancellation": [
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
            ],
        }
    )


# ================================================================
# CURRENT ELIGIBILITY
# ================================================================

def test_current_reactivation_eligibility():

    df = make_reactivation_data()

    eligible = (
        build_current_reactivation_eligibility(
            df,
            PREDICTION_DATE,
            inactivity_threshold_days=90,
        )
    )

    customers = set(
        eligible[
            "customer"
        ]
    )

    assert "A" in customers
    assert "B" in customers
    assert "C" not in customers


def test_eligibility_date_is_last_order_plus_threshold():

    df = make_reactivation_data()

    eligible = (
        build_current_reactivation_eligibility(
            df,
            PREDICTION_DATE,
            inactivity_threshold_days=90,
        )
    )

    customer_a = (
        eligible.loc[
            eligible[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    expected = (
        pd.Timestamp("2011-05-01")
        +
        pd.Timedelta(days=90)
    )

    assert (
        customer_a[
            "eligibility_date"
        ]
        ==
        expected
    )


# ================================================================
# HISTORICAL EPISODES
# ================================================================

def test_historical_successful_reactivation_episode():

    df = make_reactivation_data()

    episodes = (
        build_historical_inactivity_episodes(
            df,
            PREDICTION_DATE,
            inactivity_threshold_days=90,
            reactivation_horizon_days=90,
        )
    )

    customer_a = (
        episodes.loc[
            episodes[
                "customer"
            ]
            ==
            "A"
        ]
    )

    assert len(
        customer_a
    ) >= 1

    first_episode = (
        customer_a
        .sort_values(
            "eligibility_date"
        )
        .iloc[0]
    )

    assert bool(
        first_episode[
            "reactivated_within_horizon"
        ]
    ) is True


def test_only_matured_episodes_are_prior_history():

    df = make_reactivation_data()

    prediction_date = pd.Timestamp(
        "2011-06-01"
    )

    episodes = (
        build_historical_inactivity_episodes(
            df,
            prediction_date,
            inactivity_threshold_days=90,
            reactivation_horizon_days=90,
        )
    )

    # Jan 1 eligibility = Apr 1.
    # Outcome end is approximately Jun 30.
    # On Jun 1 that episode is not yet fully matured.
    assert episodes.empty


# ================================================================
# PRIOR FEATURES
# ================================================================

def test_prior_reactivation_features():

    df = make_reactivation_data()

    prior = (
        build_prior_reactivation_features(
            df,
            PREDICTION_DATE,
            inactivity_threshold_days=90,
            reactivation_horizon_days=90,
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
            "prior_inactivity_episode_count"
        ]
        >=
        1
    )

    assert (
        customer_a[
            "prior_reactivation_count"
        ]
        >=
        1
    )


# ================================================================
# FINAL FEATURE TABLE
# ================================================================

def test_reactivation_features_one_row_per_eligible_customer():

    df = make_reactivation_data()

    features = (
        build_reactivation_features(
            df,
            PREDICTION_DATE,
        )
    )

    assert features[
        "customer"
    ].is_unique


def test_reactivation_feature_table_excludes_recent_customer():

    df = make_reactivation_data()

    features = (
        build_reactivation_features(
            df,
            PREDICTION_DATE,
        )
    )

    assert "C" not in set(
        features[
            "customer"
        ]
    )


def test_reactivation_features_have_required_model_fields():

    df = make_reactivation_data()

    features = (
        build_reactivation_features(
            df,
            PREDICTION_DATE,
        )
    )

    expected_columns = {
        "customer",
        "prediction_date",
        "prediction_period",
        "inactivity_episode_number",
        "historical_order_count",
        "historical_revenue",
        "historical_units",
        "historical_average_order_value",
        "tenure_at_prediction_days",
        "historical_average_gap_days",
        "prior_inactivity_episode_count",
        "prior_reactivation_count",
        "prior_reactivation_rate",
        "last_order_value",
        "last_order_region",
    }

    assert expected_columns.issubset(
        set(
            features.columns
        )
    )


def test_no_future_eligibility():

    df = make_reactivation_data()

    features = (
        build_reactivation_features(
            df,
            PREDICTION_DATE,
        )
    )

    assert (
        features[
            "eligibility_date"
        ]
        <=
        PREDICTION_DATE
    ).all()


def test_invalid_inactivity_threshold_fails():

    df = make_reactivation_data()

    with pytest.raises(
        ValueError
    ):
        build_current_reactivation_eligibility(
            df,
            PREDICTION_DATE,
            inactivity_threshold_days=0,
        )