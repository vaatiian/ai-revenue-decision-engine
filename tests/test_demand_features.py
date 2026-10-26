import pandas as pd
import pytest

from src.features.demand_features import (
    add_calendar_features,
    add_demand_trend_features,
    add_lag_features,
    add_rolling_features,
    build_daily_demand,
    build_demand_features,
)


CUTOFF_DATE = pd.Timestamp(
    "2011-01-31 23:59:59"
)


def make_demand_data():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-01-01",
                    "2011-01-03",
                    "2011-01-10",
                    "2011-01-20",
                    "2011-01-31",

                    # Future transaction
                    "2011-02-05",

                    # Return
                    "2011-01-15",
                ]
            ),

            "customer": [
                "A",
                "B",
                "A",
                "C",
                "D",
                "A",
                "A",
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
                "France",
                "Germany",
                "UK",
                "UK",
                "UK",
            ],

            "quantity": [
                1,
                2,
                3,
                1,
                4,
                2,
                5,
                -1,
            ],

            "unit_price": [
                10.0,
                5.0,
                10.0,
                20.0,
                5.0,
                15.0,
                10.0,
                10.0,
            ],

            "net_merchandise_revenue": [
                10.0,
                10.0,
                30.0,
                20.0,
                20.0,
                30.0,
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


# ================================================================
# DAILY DEMAND
# ================================================================

def test_daily_demand_aggregation():

    df = make_demand_data()

    daily = build_daily_demand(
        df,
        CUTOFF_DATE,
    )

    jan_1 = (
        daily.loc[
            daily[
                "date"
            ]
            ==
            pd.Timestamp(
                "2011-01-01"
            )
        ]
        .iloc[0]
    )

    assert jan_1["revenue"] == 20.0
    assert jan_1["units"] == 3
    assert jan_1["orders"] == 2
    assert jan_1["customers"] == 2


def test_daily_demand_fills_missing_dates():

    df = make_demand_data()

    daily = build_daily_demand(
        df,
        CUTOFF_DATE,
    )

    jan_2 = (
        daily.loc[
            daily[
                "date"
            ]
            ==
            pd.Timestamp(
                "2011-01-02"
            )
        ]
        .iloc[0]
    )

    assert jan_2["revenue"] == 0
    assert jan_2["units"] == 0
    assert jan_2["orders"] == 0


def test_daily_demand_excludes_future_transaction():

    df = make_demand_data()

    daily = build_daily_demand(
        df,
        CUTOFF_DATE,
    )

    assert (
        daily[
            "date"
        ]
        <=
        CUTOFF_DATE.normalize()
    ).all()

    assert pd.Timestamp(
        "2011-02-05"
    ) not in set(
        daily[
            "date"
        ]
    )


def test_daily_demand_excludes_return():

    df = make_demand_data()

    daily = build_daily_demand(
        df,
        CUTOFF_DATE,
    )

    jan_15 = (
        daily.loc[
            daily[
                "date"
            ]
            ==
            pd.Timestamp(
                "2011-01-15"
            )
        ]
        .iloc[0]
    )

    assert jan_15["revenue"] == 0
    assert jan_15["units"] == 0


# ================================================================
# CALENDAR FEATURES
# ================================================================

def test_calendar_features():

    df = make_demand_data()

    daily = build_daily_demand(
        df,
        CUTOFF_DATE,
    )

    result = add_calendar_features(
        daily
    )

    row = (
        result.loc[
            result[
                "date"
            ]
            ==
            pd.Timestamp(
                "2011-01-01"
            )
        ]
        .iloc[0]
    )

    assert row["year"] == 2011
    assert row["month"] == 1
    assert row["quarter"] == 1
    assert row["is_weekend"] == 1


# ================================================================
# LAGS
# ================================================================

def test_lag_feature_uses_prior_day():

    df = make_demand_data()

    daily = build_daily_demand(
        df,
        CUTOFF_DATE,
    )

    result = add_lag_features(
        daily,
        lags=(1,),
    )

    jan_2 = (
        result.loc[
            result[
                "date"
            ]
            ==
            pd.Timestamp(
                "2011-01-02"
            )
        ]
        .iloc[0]
    )

    assert jan_2["revenue_lag_1"] == 20.0


def test_lag_feature_does_not_use_current_day():

    df = make_demand_data()

    daily = build_daily_demand(
        df,
        CUTOFF_DATE,
    )

    result = add_lag_features(
        daily,
        lags=(1,),
    )

    jan_3 = (
        result.loc[
            result[
                "date"
            ]
            ==
            pd.Timestamp(
                "2011-01-03"
            )
        ]
        .iloc[0]
    )

    # Jan 2 had zero revenue.
    # Jan 3 itself has revenue 30.
    assert jan_3["revenue_lag_1"] == 0
    assert jan_3["revenue"] == 30.0


def test_invalid_lag_fails():

    df = make_demand_data()

    daily = build_daily_demand(
        df,
        CUTOFF_DATE,
    )

    with pytest.raises(
        ValueError
    ):
        add_lag_features(
            daily,
            lags=(0,),
        )


# ================================================================
# ROLLING FEATURES
# ================================================================

def test_rolling_mean_is_shifted():

    daily = pd.DataFrame(
        {
            "date": pd.date_range(
                "2011-01-01",
                periods=4,
                freq="D",
            ),

            "revenue": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],

            "units": [
                1,
                2,
                3,
                4,
            ],

            "orders": [
                1,
                1,
                1,
                1,
            ],

            "customers": [
                1,
                1,
                1,
                1,
            ],
        }
    )

    result = add_rolling_features(
        daily,
        rolling_windows=(2,),
    )

    # For Jan 3 the rolling history is Jan 1 + Jan 2:
    # mean = (10 + 20) / 2 = 15.
    assert (
        result.loc[
            2,
            "revenue_rolling_mean_2"
        ]
        ==
        15.0
    )


def test_rolling_mean_excludes_current_target():

    daily = pd.DataFrame(
        {
            "date": pd.date_range(
                "2011-01-01",
                periods=3,
                freq="D",
            ),

            "revenue": [
                10.0,
                20.0,
                1000.0,
            ],

            "units": [
                1,
                2,
                100,
            ],

            "orders": [
                1,
                1,
                50,
            ],

            "customers": [
                1,
                1,
                40,
            ],
        }
    )

    result = add_rolling_features(
        daily,
        rolling_windows=(2,),
    )

    assert (
        result.loc[
            2,
            "revenue_rolling_mean_2"
        ]
        ==
        15.0
    )


def test_invalid_rolling_window_fails():

    df = make_demand_data()

    daily = build_daily_demand(
        df,
        CUTOFF_DATE,
    )

    with pytest.raises(
        ValueError
    ):
        add_rolling_features(
            daily,
            rolling_windows=(0,),
        )


# ================================================================
# TREND FEATURES
# ================================================================

def test_demand_trend_features():

    daily = pd.DataFrame(
        {
            "date": pd.date_range(
                "2011-01-01",
                periods=2,
            ),

            "revenue_rolling_mean_7": [
                10.0,
                20.0,
            ],

            "revenue_rolling_mean_28": [
                5.0,
                10.0,
            ],
        }
    )

    result = add_demand_trend_features(
        daily
    )

    assert (
        result.loc[
            0,
            "revenue_short_vs_long_ratio"
        ]
        ==
        2.0
    )

    assert (
        result.loc[
            1,
            "revenue_short_vs_long_change"
        ]
        ==
        10.0
    )


def test_trend_requires_rolling_columns():

    daily = pd.DataFrame(
        {
            "date": pd.date_range(
                "2011-01-01",
                periods=2,
            )
        }
    )

    with pytest.raises(
        ValueError
    ):
        add_demand_trend_features(
            daily
        )


# ================================================================
# FINAL FEATURE TABLE
# ================================================================

def test_demand_features_unique_dates():

    df = make_demand_data()

    features = build_demand_features(
        df,
        CUTOFF_DATE,
    )

    assert features[
        "date"
    ].is_unique


def test_demand_features_are_chronological():

    df = make_demand_data()

    features = build_demand_features(
        df,
        CUTOFF_DATE,
    )

    assert features[
        "date"
    ].is_monotonic_increasing


def test_demand_features_have_required_columns():

    df = make_demand_data()

    features = build_demand_features(
        df,
        CUTOFF_DATE,
    )

    required = {
        "date",
        "feature_cutoff_date",
        "feature_period",
        "revenue",
        "units",
        "orders",
        "customers",
        "month",
        "quarter",
        "day_of_week",
        "is_weekend",
        "revenue_lag_1",
        "revenue_lag_7",
        "revenue_lag_28",
        "revenue_rolling_mean_7",
        "revenue_rolling_mean_28",
        "revenue_rolling_std_7",
        "revenue_short_vs_long_ratio",
        "revenue_short_vs_long_change",
    }

    assert required.issubset(
        set(
            features.columns
        )
    )


def test_drop_incomplete_history():

    df = make_demand_data()

    features = build_demand_features(
        df,
        CUTOFF_DATE,
        lags=(
            1,
            7,
        ),
        rolling_windows=(
            7,
        ),
        drop_incomplete_history=True,
    )

    assert (
        features[
            "revenue_lag_7"
        ]
        .notna()
        .all()
    )


def test_feature_cutoff_metadata():

    df = make_demand_data()

    features = build_demand_features(
        df,
        CUTOFF_DATE,
    )

    assert (
        features[
            "feature_cutoff_date"
        ]
        ==
        CUTOFF_DATE
    ).all()


def test_no_future_demand_rows():

    df = make_demand_data()

    features = build_demand_features(
        df,
        CUTOFF_DATE,
    )

    assert (
        features[
            "date"
        ]
        <=
        CUTOFF_DATE.normalize()
    ).all()