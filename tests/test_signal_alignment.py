import numpy as np
import pandas as pd
import pytest

from src.decision_engine.signal_alignment import (
    DECISION_SIGNAL_COLUMNS,
    FCV_TIER_BINS,
    FCV_TIER_LABELS,
    LEAKAGE_MINIMUM_EVENT_LOSS_RATE,
    RECOMMENDATION_CURRENTNESS_SECONDS,
    align_decision_signals,
    align_fcv_signals,
    align_leakage_signals,
    align_reactivation_signals,
    align_recommendation_signals,
)


DECISION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


# ================================================================
# FIXTURES
# ================================================================


def make_decision_snapshot():

    return pd.DataFrame(
        {
            "customer": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
            ],

            "decision_date": [
                DECISION_DATE
            ] * 10,

            "last_order_date": pd.to_datetime(
                [
                    "2011-08-31 10:00:00",
                    "2011-08-10 10:00:00",
                    "2011-06-10 10:00:00",
                    "2011-04-01 10:00:00",
                    "2011-08-25 10:00:00",
                    "2011-08-20 10:00:00",
                    "2011-07-20 10:00:00",
                    "2011-05-01 10:00:00",
                    "2011-08-15 10:00:00",
                    "2011-02-01 10:00:00",
                ]
            ),

            "customer_state": [
                "Active",
                "Active",
                "At Risk",
                "Dormant",
                "Active",
                "Active",
                "At Risk",
                "Dormant",
                "Active",
                "Long Dormant",
            ],
        }
    )


def make_leakage_scores():

    return pd.DataFrame(
        {
            "customer": list(
                "ABCDEFGHIJ"
            ),

            "revenue_leakage_probability": [
                0.05,
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
            ],
        }
    )


def make_leakage_context():

    return pd.DataFrame(
        {
            "customer": list(
                "ABCDEFGHIJ"
            ),

            "baseline_revenue": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                1000.0,
            ],
        }
    )


def make_fcv_scores():

    return pd.DataFrame(
        {
            "customer": list(
                "ABCDEFGHIJ"
            ),

            "fcv_history_180d_revenue": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                1000.0,
            ],

            "future_active_probability": [
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
                0.95,
            ],

            "future_customer_value_90d": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
            ],
        }
    )


def make_reactivation_scores():

    return pd.DataFrame(
        {
            "customer": [
                "D",
                "H",
                "J",
            ],

            "reactivation_probability": [
                0.25,
                0.55,
                0.75,
            ],
        }
    )


def make_reactivation_features():

    return pd.DataFrame(
        {
            "customer": [
                "D",
                "H",
                "J",
            ],

            "prediction_date": pd.to_datetime(
                [
                    "2011-08-31 23:59:59",
                    "2011-08-01 23:59:59",
                    "2011-07-01 23:59:59",
                ]
            ),
        }
    )


def make_recommendation_scores():

    return pd.DataFrame(
        {
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

            "prediction_date": pd.to_datetime(
                [
                    "2011-07-01 10:00:00",
                    "2011-08-31 10:00:00",
                    "2011-08-10 10:00:30",
                    "2011-07-20 10:02:00",
                ]
            ),

            "repeat_top1_product": [
                "OLD-A",
                "NEW-A",
                "B-REPEAT",
                "C-REPEAT",
            ],

            "discovery_top1_product": [
                "OLD-A-D",
                "NEW-A-D",
                "B-DISCOVERY",
                "C-DISCOVERY",
            ],
        }
    )


# ================================================================
# CONSTANTS / CONTRACT
# ================================================================


def test_leakage_minimum_event_loss_rate_is_fifty_percent():

    assert (
        LEAKAGE_MINIMUM_EVENT_LOSS_RATE
        ==
        0.50
    )


def test_recommendation_currentness_is_sixty_seconds():

    assert (
        RECOMMENDATION_CURRENTNESS_SECONDS
        ==
        60
    )


def test_fcv_tier_bins_match_research():

    assert (
        FCV_TIER_BINS
        ==
        (
            0.00,
            0.50,
            0.80,
            0.90,
            0.95,
            1.00,
        )
    )


def test_fcv_tier_labels_match_research():

    assert (
        FCV_TIER_LABELS
        ==
        (
            "Core",
            "Growth",
            "High Value",
            "Priority",
            "Premier",
        )
    )


def test_decision_signal_contract_is_unique():

    assert len(
        DECISION_SIGNAL_COLUMNS
    ) == len(
        set(
            DECISION_SIGNAL_COLUMNS
        )
    )


# ================================================================
# LEAKAGE
# ================================================================


def test_leakage_native_probability_is_normalized():

    result = (
        align_leakage_signals(
            leakage_scores=(
                make_leakage_scores()
            ),
            leakage_context=(
                make_leakage_context()
            ),
        )
    )

    assert (
        "leakage_probability"
        in
        result.columns
    )

    assert (
        "revenue_leakage_probability"
        not in
        result.columns
    )


def test_leakage_probability_percentile_uses_research_rank():

    result = (
        align_leakage_signals(
            leakage_scores=(
                make_leakage_scores()
            ),
            leakage_context=(
                make_leakage_context()
            ),
        )
    )

    assert result.loc[
        result[
            "customer"
        ]
        ==
        "A",
        "leakage_probability_percentile",
    ].iloc[0] == pytest.approx(
        0.10
    )

    assert result.loc[
        result[
            "customer"
        ]
        ==
        "J",
        "leakage_probability_percentile",
    ].iloc[0] == pytest.approx(
        1.00
    )


def test_leakage_exposure_formula_matches_research():

    result = (
        align_leakage_signals(
            leakage_scores=(
                make_leakage_scores()
            ),
            leakage_context=(
                make_leakage_context()
            ),
        )
    )

    row = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "J"
        ]
        .iloc[0]
    )

    expected = (
        0.90
        *
        0.50
        *
        1000.0
    )

    assert (
        row[
            "leakage_exposure_score"
        ]
        ==
        pytest.approx(
            expected
        )
    )


def test_leakage_exposure_percentile_uses_rank_pct():

    result = (
        align_leakage_signals(
            leakage_scores=(
                make_leakage_scores()
            ),
            leakage_context=(
                make_leakage_context()
            ),
        )
    )

    assert result[
        "leakage_exposure_percentile"
    ].max() == pytest.approx(
        1.0
    )


def test_leakage_economic_priority_has_p1_and_p5():

    result = (
        align_leakage_signals(
            leakage_scores=(
                make_leakage_scores()
            ),
            leakage_context=(
                make_leakage_context()
            ),
        )
    )

    labels = set(
        result[
            "economic_priority_band"
        ]
        .astype(str)
    )

    assert (
        "P1 — Highest Exposure"
        in labels
    )

    assert (
        "P5 — Lowest Exposure"
        in labels
    )


def test_leakage_risk_bands_match_research_labels():

    result = (
        align_leakage_signals(
            leakage_scores=(
                make_leakage_scores()
            ),
            leakage_context=(
                make_leakage_context()
            ),
        )
    )

    labels = set(
        result[
            "risk_band"
        ]
        .astype(str)
    )

    assert labels == {
        "Low Risk",
        "Medium Risk",
        "High Risk",
    }


def test_leakage_value_bands_match_research_labels():

    result = (
        align_leakage_signals(
            leakage_scores=(
                make_leakage_scores()
            ),
            leakage_context=(
                make_leakage_context()
            ),
        )
    )

    labels = set(
        result[
            "value_band"
        ]
        .astype(str)
    )

    assert labels == {
        "Low Value",
        "Medium Value",
        "High Value",
    }


def test_leakage_rejects_probability_outside_unit_interval():

    broken = (
        make_leakage_scores()
    )

    broken.loc[
        0,
        "revenue_leakage_probability",
    ] = 1.20

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        align_leakage_signals(
            leakage_scores=broken,
            leakage_context=(
                make_leakage_context()
            ),
        )


def test_leakage_rejects_negative_baseline_revenue():

    broken = (
        make_leakage_context()
    )

    broken.loc[
        0,
        "baseline_revenue",
    ] = -1.0

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        align_leakage_signals(
            leakage_scores=(
                make_leakage_scores()
            ),
            leakage_context=broken,
        )


# ================================================================
# FCV
# ================================================================


def test_fcv_percentile_uses_average_rank():

    scores = (
        make_fcv_scores()
    )

    scores.loc[
        scores[
            "customer"
        ]
        .isin(
            [
                "A",
                "B",
            ]
        ),
        "future_customer_value_90d",
    ] = 10.0

    result = (
        align_fcv_signals(
            scores
        )
    )

    expected_tied_percentile = (
        1.5
        /
        10.0
    )

    observed = (
        result.loc[
            result[
                "customer"
            ]
            .isin(
                [
                    "A",
                    "B",
                ]
            ),
            "fcv_percentile",
        ]
    )

    assert np.allclose(
        observed.to_numpy(),
        expected_tied_percentile,
    )


def test_fcv_highest_customer_is_premier():

    result = (
        align_fcv_signals(
            make_fcv_scores()
        )
    )

    tier = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "J",
            "fcv_tier",
        ]
        .astype(str)
        .iloc[0]
    )

    assert tier == "Premier"


def test_fcv_tier_boundaries_follow_research_cut():

    scores = pd.DataFrame(
        {
            "customer": [
                f"C{i:02d}"
                for i in range(
                    1,
                    21,
                )
            ],

            "fcv_history_180d_revenue": [
                100.0
            ] * 20,

            "future_active_probability": [
                0.50
            ] * 20,

            "future_customer_value_90d": [
                float(
                    value
                )
                for value in range(
                    1,
                    21,
                )
            ],
        }
    )

    result = (
        align_fcv_signals(
            scores
        )
    )

    tier_by_customer = dict(
        zip(
            result[
                "customer"
            ],
            result[
                "fcv_tier"
            ].astype(str),
        )
    )

    assert (
        tier_by_customer[
            "C10"
        ]
        ==
        "Core"
    )

    assert (
        tier_by_customer[
            "C16"
        ]
        ==
        "Growth"
    )

    assert (
        tier_by_customer[
            "C18"
        ]
        ==
        "High Value"
    )

    assert (
        tier_by_customer[
            "C19"
        ]
        ==
        "Priority"
    )

    assert (
        tier_by_customer[
            "C20"
        ]
        ==
        "Premier"
    )


def test_fcv_rejects_negative_value():

    broken = (
        make_fcv_scores()
    )

    broken.loc[
        0,
        "future_customer_value_90d",
    ] = -5.0

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        align_fcv_signals(
            broken
        )


# ================================================================
# REACTIVATION
# ================================================================


def test_reactivation_prediction_date_restored_from_features():

    result = (
        align_reactivation_signals(
            reactivation_scores=(
                make_reactivation_scores()
            ),
            reactivation_features=(
                make_reactivation_features()
            ),
            decision_date=DECISION_DATE,
        )
    )

    observed = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "D",
            "reactivation_prediction_date",
        ]
        .iloc[0]
    )

    assert (
        observed
        ==
        DECISION_DATE
    )


def test_reactivation_score_age_is_calculated_from_prediction_date():

    result = (
        align_reactivation_signals(
            reactivation_scores=(
                make_reactivation_scores()
            ),
            reactivation_features=(
                make_reactivation_features()
            ),
            decision_date=DECISION_DATE,
        )
    )

    observed = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "H",
            "reactivation_score_age_days",
        ]
        .iloc[0]
    )

    expected = (
        (
            DECISION_DATE
            -
            pd.Timestamp(
                "2011-08-01 23:59:59"
            )
        )
        .total_seconds()
        /
        86400
    )

    assert observed == pytest.approx(
        expected
    )


def test_reactivation_rejects_missing_feature_provenance():

    features = (
        make_reactivation_features()
        .loc[
            lambda frame:
                frame[
                    "customer"
                ]
                !=
                "D"
        ]
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="provenance is missing",
    ):
        align_reactivation_signals(
            reactivation_scores=(
                make_reactivation_scores()
            ),
            reactivation_features=features,
            decision_date=DECISION_DATE,
        )


# ================================================================
# RECOMMENDATION
# ================================================================


def test_recommendation_selects_latest_episode_per_customer():

    result = (
        align_recommendation_signals(
            recommendation_scores=(
                make_recommendation_scores()
            ),
            decision_date=DECISION_DATE,
        )
    )

    customer_a = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "repeat_top1_product"
        ]
        ==
        "NEW-A"
    )

    assert (
        customer_a[
            "recommendation_prediction_date"
        ]
        ==
        pd.Timestamp(
            "2011-08-31 10:00:00"
        )
    )


def test_recommendation_excludes_future_episode():

    scores = (
        make_recommendation_scores()
    )

    future = pd.DataFrame(
        {
            "customer": [
                "A"
            ],
            "invoice": [
                "A3"
            ],
            "prediction_date": pd.to_datetime(
                [
                    "2011-09-01 10:00:00"
                ]
            ),
            "repeat_top1_product": [
                "FUTURE-A"
            ],
            "discovery_top1_product": [
                "FUTURE-A-D"
            ],
        }
    )

    scores = pd.concat(
        [
            scores,
            future,
        ],
        ignore_index=True,
    )

    result = (
        align_recommendation_signals(
            recommendation_scores=scores,
            decision_date=DECISION_DATE,
        )
    )

    customer_a = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "A"
        ]
        .iloc[0]
    )

    assert (
        customer_a[
            "repeat_top1_product"
        ]
        ==
        "NEW-A"
    )


# ================================================================
# COMMON ALIGNMENT
# ================================================================


def test_common_alignment_preserves_customer_population():

    snapshot = (
        make_decision_snapshot()
    )

    result = (
        align_decision_signals(
            snapshot,
            leakage_scores=(
                make_leakage_scores()
            ),
            leakage_context=(
                make_leakage_context()
            ),
            fcv_scores=(
                make_fcv_scores()
            ),
            reactivation_scores=(
                make_reactivation_scores()
            ),
            reactivation_features=(
                make_reactivation_features()
            ),
            recommendation_scores=(
                make_recommendation_scores()
            ),
        )
    )

    assert len(
        result
    ) == len(
        snapshot
    )

    assert (
        result[
            "customer"
        ].is_unique
    )


def test_common_alignment_contains_all_signal_contract_columns():

    result = (
        align_decision_signals(
            make_decision_snapshot()
        )
    )

    assert set(
        DECISION_SIGNAL_COLUMNS
    ).issubset(
        result.columns
    )


def test_leakage_operational_validity_is_state_aware():

    result = (
        align_decision_signals(
            make_decision_snapshot(),
            leakage_scores=(
                make_leakage_scores()
            ),
            leakage_context=(
                make_leakage_context()
            ),
        )
    )

    assert bool(
        result.loc[
            result[
                "customer"
            ]
            ==
            "A",
            "leakage_operationally_valid",
        ]
        .iloc[0]
    )

    assert not bool(
        result.loc[
            result[
                "customer"
            ]
            ==
            "D",
            "leakage_operationally_valid",
        ]
        .iloc[0]
    )


def test_reactivation_operationally_valid_for_dormant_customer():

    result = (
        align_decision_signals(
            make_decision_snapshot(),
            reactivation_scores=(
                make_reactivation_scores()
            ),
            reactivation_features=(
                make_reactivation_features()
            ),
        )
    )

    assert bool(
        result.loc[
            result[
                "customer"
            ]
            ==
            "D",
            "reactivation_operationally_valid",
        ]
        .iloc[0]
    )


def test_reactivation_operationally_valid_for_long_dormant_signal():

    result = (
        align_decision_signals(
            make_decision_snapshot(),
            reactivation_scores=(
                make_reactivation_scores()
            ),
            reactivation_features=(
                make_reactivation_features()
            ),
        )
    )

    assert bool(
        result.loc[
            result[
                "customer"
            ]
            ==
            "J",
            "reactivation_operationally_valid",
        ]
        .iloc[0]
    )


def test_reactivation_future_score_is_not_operationally_valid():

    features = (
        make_reactivation_features()
    )

    features.loc[
        features[
            "customer"
        ]
        ==
        "D",
        "prediction_date",
    ] = (
        DECISION_DATE
        +
        pd.Timedelta(
            days=1
        )
    )

    result = (
        align_decision_signals(
            make_decision_snapshot(),
            reactivation_scores=(
                make_reactivation_scores()
            ),
            reactivation_features=features,
        )
    )

    assert not bool(
        result.loc[
            result[
                "customer"
            ]
            ==
            "D",
            "reactivation_operationally_valid",
        ]
        .iloc[0]
    )


def test_recommendation_current_at_exact_sixty_seconds():

    scores = (
        make_recommendation_scores()
    )

    scores.loc[
        scores[
            "customer"
        ]
        ==
        "B",
        "prediction_date",
    ] = pd.Timestamp(
        "2011-08-10 10:01:00"
    )

    result = (
        align_decision_signals(
            make_decision_snapshot(),
            recommendation_scores=scores,
        )
    )

    assert bool(
        result.loc[
            result[
                "customer"
            ]
            ==
            "B",
            "recommendation_is_current",
        ]
        .iloc[0]
    )


def test_recommendation_invalid_at_more_than_sixty_seconds():

    result = (
        align_decision_signals(
            make_decision_snapshot(),
            recommendation_scores=(
                make_recommendation_scores()
            ),
        )
    )

    assert not bool(
        result.loc[
            result[
                "customer"
            ]
            ==
            "C",
            "recommendation_is_current",
        ]
        .iloc[0]
    )


def test_recommendation_operational_validity_requires_active_or_at_risk():

    scores = pd.DataFrame(
        {
            "customer": [
                "D"
            ],
            "invoice": [
                "D1"
            ],
            "prediction_date": pd.to_datetime(
                [
                    "2011-04-01 10:00:00"
                ]
            ),
            "repeat_top1_product": [
                "D-REPEAT"
            ],
            "discovery_top1_product": [
                "D-DISCOVERY"
            ],
        }
    )

    result = (
        align_decision_signals(
            make_decision_snapshot(),
            recommendation_scores=scores,
        )
    )

    assert bool(
        result.loc[
            result[
                "customer"
            ]
            ==
            "D",
            "recommendation_is_current",
        ]
        .iloc[0]
    )

    assert not bool(
        result.loc[
            result[
                "customer"
            ]
            ==
            "D",
            "recommendation_operationally_valid",
        ]
        .iloc[0]
    )


def test_fcv_signal_available_when_score_exists():

    result = (
        align_decision_signals(
            make_decision_snapshot(),
            fcv_scores=(
                make_fcv_scores()
            ),
        )
    )

    assert (
        result[
            "fcv_signal_available"
        ]
        .all()
    )


def test_missing_model_families_remain_unavailable():

    result = (
        align_decision_signals(
            make_decision_snapshot()
        )
    )

    assert not (
        result[
            "leakage_signal_available"
        ]
        .any()
    )

    assert not (
        result[
            "fcv_signal_available"
        ]
        .any()
    )

    assert not (
        result[
            "reactivation_signal_available"
        ]
        .any()
    )

    assert not (
        result[
            "recommendation_signal_available"
        ]
        .any()
    )


def test_leakage_scores_require_context():

    with pytest.raises(
        ValueError,
        match="Leakage context is required",
    ):
        align_decision_signals(
            make_decision_snapshot(),
            leakage_scores=(
                make_leakage_scores()
            ),
        )


def test_reactivation_scores_require_feature_provenance():

    with pytest.raises(
        ValueError,
        match="Reactivation features are required",
    ):
        align_decision_signals(
            make_decision_snapshot(),
            reactivation_scores=(
                make_reactivation_scores()
            ),
        )