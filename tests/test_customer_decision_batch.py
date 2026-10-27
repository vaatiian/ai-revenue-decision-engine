from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pandas as pd
import pytest

import src.deployment.customer_batch as customer_batch_module

from src.deployment.customer_batch import (
    CustomerDecisionBatchResult,
    run_customer_decision_batch,
)


DECISION_DATE = pd.Timestamp(
    "2011-08-31"
)

FIXED_SCORED_AT = datetime(
    2011,
    8,
    31,
    23,
    59,
    tzinfo=timezone.utc,
)


# ================================================================
# TEST FIXTURES
# ================================================================


def make_transactions():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-08-31 10:00:00",
                ]
            ),
            "customer": [
                "A",
            ],
            "invoice": [
                "A1",
            ],
            "product": [
                "P1",
            ],
            "region": [
                "United Kingdom",
            ],
            "quantity": [
                1,
            ],
            "unit_price": [
                10.0,
            ],
            "net_merchandise_revenue": [
                10.0,
            ],
        }
    )


def make_recommendation_episodes():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "invoice": [
                "A1",
            ],
            "order_date": pd.to_datetime(
                [
                    "2011-08-31 10:00:00",
                ]
            ),
            "prediction_period": [
                pd.Period(
                    "2011-08",
                    freq="M",
                )
            ],
        }
    )


def make_reactivation_features():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "reactivation_feature": [
                1.0,
            ],
        }
    )


def make_leakage_features():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "b90_revenue": [
                100.0,
            ],
            "leakage_feature": [
                1.0,
            ],
        }
    )


def make_fcv_features():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "fcv_feature": [
                1.0,
            ],
        }
    )


def make_reactivation_scores():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "reactivation_probability": [
                0.70,
            ],
        }
    )


def make_leakage_scores():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "revenue_leakage_probability": [
                0.60,
            ],
        }
    )


def make_fcv_scores():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "expected_fcv90": [
                50.0,
            ],
        }
    )


def make_recommendation_scores():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "invoice": [
                "A1",
            ],
            "prediction_date": [
                DECISION_DATE,
            ],
        }
    )


def make_decisions():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "primary_action": [
                "Grow",
            ],
        }
    )


# ================================================================
# VALIDATION
# ================================================================


def test_customer_batch_requires_dataframe_transactions():

    with pytest.raises(
        TypeError,
        match=(
            "transactions must be a pandas DataFrame"
        ),
    ):
        run_customer_decision_batch(
            transactions=[],
            decision_date=DECISION_DATE,
        )


def test_customer_batch_requires_month_end_decision_date():

    with pytest.raises(
        ValueError,
        match=(
            "requires a month-end decision date"
        ),
    ):
        run_customer_decision_batch(
            transactions=make_transactions(),
            decision_date=pd.Timestamp(
                "2011-08-30"
            ),
        )


def test_customer_batch_recommendation_episodes_must_be_dataframe_or_none():

    with pytest.raises(
        TypeError,
        match=(
            "recommendation_episodes must be "
            "a pandas DataFrame or None"
        ),
    ):
        run_customer_decision_batch(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
            recommendation_episodes=[
                {
                    "customer": "A",
                }
            ],
        )


# ================================================================
# FULL ORCHESTRATION PATH
# ================================================================


def test_customer_batch_builds_scores_and_calls_decision_engine(
    monkeypatch,
):

    transactions = make_transactions()

    reactivation_features = (
        make_reactivation_features()
    )

    leakage_features = (
        make_leakage_features()
    )

    fcv_features = (
        make_fcv_features()
    )

    reactivation_scores = (
        make_reactivation_scores()
    )

    leakage_scores = (
        make_leakage_scores()
    )

    fcv_scores = (
        make_fcv_scores()
    )

    recommendation_scores = (
        make_recommendation_scores()
    )

    decisions = make_decisions()

    captured = {}

    monkeypatch.setattr(
        customer_batch_module,
        "build_reactivation_model_features",
        lambda **kwargs: (
            reactivation_features.copy()
        ),
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_leakage_model_features",
        lambda **kwargs: (
            leakage_features.copy()
        ),
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_fcv_model_features",
        lambda **kwargs: (
            fcv_features.copy()
        ),
    )

    class FakeReactivationScorer:

        def score(
            self,
            features,
        ):
            captured[
                "reactivation_score_input"
            ] = features.copy()

            return (
                reactivation_scores.copy()
            )

    class FakeLeakageScorer:

        def score(
            self,
            features,
        ):
            captured[
                "leakage_score_input"
            ] = features.copy()

            return (
                leakage_scores.copy()
            )

    class FakeFCVScorer:

        def score(
            self,
            features,
        ):
            captured[
                "fcv_score_input"
            ] = features.copy()

            return (
                fcv_scores.copy()
            )

    class FakeRecommendationScorer:

        def score(
            self,
            transactions,
            episodes,
            scored_at=None,
        ):
            captured[
                "recommendation_transactions"
            ] = transactions.copy()

            captured[
                "recommendation_episodes"
            ] = episodes.copy()

            captured[
                "recommendation_scored_at"
            ] = scored_at

            return (
                recommendation_scores.copy()
            )

    monkeypatch.setattr(
        customer_batch_module,
        "ReactivationScorer",
        FakeReactivationScorer,
    )

    monkeypatch.setattr(
        customer_batch_module,
        "LeakageScorer",
        FakeLeakageScorer,
    )

    monkeypatch.setattr(
        customer_batch_module,
        "FCVScorer",
        FakeFCVScorer,
    )

    monkeypatch.setattr(
        customer_batch_module,
        "RecommendationScorer",
        FakeRecommendationScorer,
    )

    def fake_build_customer_decisions(
        **kwargs,
    ):
        captured[
            "decision_kwargs"
        ] = kwargs

        return decisions.copy()

    monkeypatch.setattr(
        customer_batch_module,
        "build_customer_decisions",
        fake_build_customer_decisions,
    )

    episodes = (
        make_recommendation_episodes()
    )

    result = run_customer_decision_batch(
        transactions=transactions,
        decision_date=DECISION_DATE,
        recommendation_episodes=episodes,
        recommendation_scored_at=(
            FIXED_SCORED_AT
        ),
    )

    assert isinstance(
        result,
        CustomerDecisionBatchResult,
    )

    assert (
        result.decision_date
        ==
        DECISION_DATE
    )

    pd.testing.assert_frame_equal(
        result.reactivation_features,
        reactivation_features,
    )

    pd.testing.assert_frame_equal(
        result.reactivation_scores,
        reactivation_scores,
    )

    pd.testing.assert_frame_equal(
        result.leakage_features,
        leakage_features,
    )

    pd.testing.assert_frame_equal(
        result.leakage_scores,
        leakage_scores,
    )

    pd.testing.assert_frame_equal(
        result.fcv_features,
        fcv_features,
    )

    pd.testing.assert_frame_equal(
        result.fcv_scores,
        fcv_scores,
    )

    pd.testing.assert_frame_equal(
        result.recommendation_scores,
        recommendation_scores,
    )

    pd.testing.assert_frame_equal(
        result.decisions,
        decisions,
    )

    # Frozen model inputs remain untouched.

    pd.testing.assert_frame_equal(
        captured[
            "reactivation_score_input"
        ],
        reactivation_features,
    )

    pd.testing.assert_frame_equal(
        captured[
            "leakage_score_input"
        ],
        leakage_features,
    )

    pd.testing.assert_frame_equal(
        captured[
            "fcv_score_input"
        ],
        fcv_features,
    )

    pd.testing.assert_frame_equal(
        captured[
            "recommendation_transactions"
        ],
        transactions,
    )

    pd.testing.assert_frame_equal(
        captured[
            "recommendation_episodes"
        ],
        episodes,
    )

    assert (
        captured[
            "recommendation_scored_at"
        ]
        ==
        FIXED_SCORED_AT
    )

    # Reactivation provenance is restored only for Decision Engine context.

    expected_reactivation_context = pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "prediction_date": [
                DECISION_DATE,
            ],
        }
    )

    # Leakage economic context maps the frozen b90_revenue feature
    # into the Decision Engine's semantic baseline_revenue name.

    expected_leakage_context = pd.DataFrame(
        {
            "customer": [
                "A",
            ],
            "baseline_revenue": [
                100.0,
            ],
        }
    )

    pd.testing.assert_frame_equal(
        result.leakage_context,
        expected_leakage_context,
    )

    decision_kwargs = captured[
        "decision_kwargs"
    ]

    assert (
        decision_kwargs[
            "decision_date"
        ]
        ==
        DECISION_DATE
    )

    pd.testing.assert_frame_equal(
        decision_kwargs[
            "leakage_scores"
        ],
        leakage_scores,
    )

    pd.testing.assert_frame_equal(
        decision_kwargs[
            "leakage_context"
        ],
        expected_leakage_context,
    )

    pd.testing.assert_frame_equal(
        decision_kwargs[
            "fcv_scores"
        ],
        fcv_scores,
    )

    pd.testing.assert_frame_equal(
        decision_kwargs[
            "reactivation_scores"
        ],
        reactivation_scores,
    )

    pd.testing.assert_frame_equal(
        decision_kwargs[
            "reactivation_features"
        ],
        expected_reactivation_context,
    )

    pd.testing.assert_frame_equal(
        decision_kwargs[
            "recommendation_scores"
        ],
        recommendation_scores,
    )


# ================================================================
# OPTIONAL RECOMMENDATION PATH
# ================================================================


def test_customer_batch_does_not_fabricate_recommendation_episodes(
    monkeypatch,
):

    monkeypatch.setattr(
        customer_batch_module,
        "build_reactivation_model_features",
        lambda **kwargs: pd.DataFrame(),
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_leakage_model_features",
        lambda **kwargs: pd.DataFrame(),
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_fcv_model_features",
        lambda **kwargs: pd.DataFrame(),
    )

    class RecommendationScorerMustNotRun:

        def __init__(
            self,
            *args,
            **kwargs,
        ):
            raise AssertionError(
                "RecommendationScorer must not run "
                "without governed episodes."
            )

    monkeypatch.setattr(
        customer_batch_module,
        "RecommendationScorer",
        RecommendationScorerMustNotRun,
    )

    captured = {}

    def fake_build_customer_decisions(
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        return make_decisions()

    monkeypatch.setattr(
        customer_batch_module,
        "build_customer_decisions",
        fake_build_customer_decisions,
    )

    result = run_customer_decision_batch(
        transactions=make_transactions(),
        decision_date=DECISION_DATE,
        recommendation_episodes=None,
    )

    assert (
        result.recommendation_scores
        is None
    )

    assert (
        captured[
            "recommendation_scores"
        ]
        is None
    )


def test_empty_recommendation_episode_frame_skips_recommendation_scoring(
    monkeypatch,
):

    monkeypatch.setattr(
        customer_batch_module,
        "build_reactivation_model_features",
        lambda **kwargs: pd.DataFrame(),
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_leakage_model_features",
        lambda **kwargs: pd.DataFrame(),
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_fcv_model_features",
        lambda **kwargs: pd.DataFrame(),
    )

    class RecommendationScorerMustNotRun:

        def __init__(
            self,
            *args,
            **kwargs,
        ):
            raise AssertionError(
                "RecommendationScorer must not run "
                "for an empty episode population."
            )

    monkeypatch.setattr(
        customer_batch_module,
        "RecommendationScorer",
        RecommendationScorerMustNotRun,
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_customer_decisions",
        lambda **kwargs: make_decisions(),
    )

    result = run_customer_decision_batch(
        transactions=make_transactions(),
        decision_date=DECISION_DATE,
        recommendation_episodes=(
            pd.DataFrame()
        ),
    )

    assert (
        result.recommendation_scores
        is None
    )


# ================================================================
# EMPTY ELIGIBLE SCORE POPULATIONS
# ================================================================


def test_empty_feature_populations_do_not_invoke_scorers(
    monkeypatch,
):

    monkeypatch.setattr(
        customer_batch_module,
        "build_reactivation_model_features",
        lambda **kwargs: pd.DataFrame(),
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_leakage_model_features",
        lambda **kwargs: pd.DataFrame(),
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_fcv_model_features",
        lambda **kwargs: pd.DataFrame(),
    )

    class ScorerMustNotRun:

        def __init__(
            self,
            *args,
            **kwargs,
        ):
            raise AssertionError(
                "Scorer must not run for an empty eligible population."
            )

    monkeypatch.setattr(
        customer_batch_module,
        "ReactivationScorer",
        ScorerMustNotRun,
    )

    monkeypatch.setattr(
        customer_batch_module,
        "LeakageScorer",
        ScorerMustNotRun,
    )

    monkeypatch.setattr(
        customer_batch_module,
        "FCVScorer",
        ScorerMustNotRun,
    )

    captured = {}

    def fake_build_customer_decisions(
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        return make_decisions()

    monkeypatch.setattr(
        customer_batch_module,
        "build_customer_decisions",
        fake_build_customer_decisions,
    )

    result = run_customer_decision_batch(
        transactions=make_transactions(),
        decision_date=DECISION_DATE,
    )

    assert (
        result.reactivation_scores
        is None
    )

    assert (
        result.leakage_scores
        is None
    )

    assert (
        result.leakage_context
        is None
    )

    assert (
        result.fcv_scores
        is None
    )

    assert (
        captured[
            "reactivation_scores"
        ]
        is None
    )

    assert (
        captured[
            "reactivation_features"
        ]
        is None
    )

    assert (
        captured[
            "leakage_scores"
        ]
        is None
    )

    assert (
        captured[
            "leakage_context"
        ]
        is None
    )

    assert (
        captured[
            "fcv_scores"
        ]
        is None
    )


# ================================================================
# REACTIVATION PROVENANCE GOVERNANCE
# ================================================================


def test_nonempty_reactivation_features_require_customer(
    monkeypatch,
):

    monkeypatch.setattr(
        customer_batch_module,
        "build_reactivation_model_features",
        lambda **kwargs: pd.DataFrame(
            {
                "some_feature": [
                    1.0,
                ],
            }
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Reactivation feature frame is missing customer"
        ),
    ):
        run_customer_decision_batch(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )


# ================================================================
# LEAKAGE CONTEXT GOVERNANCE
# ================================================================


def test_nonempty_leakage_features_require_b90_revenue(
    monkeypatch,
):

    monkeypatch.setattr(
        customer_batch_module,
        "build_reactivation_model_features",
        lambda **kwargs: pd.DataFrame(),
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_leakage_model_features",
        lambda **kwargs: pd.DataFrame(
            {
                "customer": [
                    "A",
                ],
                "some_feature": [
                    1.0,
                ],
            }
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "b90_revenue required to construct "
            "Decision Engine baseline_revenue context"
        ),
    ):
        run_customer_decision_batch(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )


# ================================================================
# EMPTY TRANSACTION BATCH
# ================================================================


def test_empty_transactions_delegate_directly_to_decision_engine(
    monkeypatch,
):

    empty_transactions = (
        make_transactions()
        .iloc[
            0:0
        ]
        .copy()
    )

    class FeatureBuilderMustNotRun:

        def __call__(
            self,
            *args,
            **kwargs,
        ):
            raise AssertionError(
                "Feature builders must not run for an empty transaction batch."
            )

    must_not_run = (
        FeatureBuilderMustNotRun()
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_reactivation_model_features",
        must_not_run,
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_leakage_model_features",
        must_not_run,
    )

    monkeypatch.setattr(
        customer_batch_module,
        "build_fcv_model_features",
        must_not_run,
    )

    captured = {}

    def fake_build_customer_decisions(
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        return pd.DataFrame(
            columns=[
                "customer",
            ]
        )

    monkeypatch.setattr(
        customer_batch_module,
        "build_customer_decisions",
        fake_build_customer_decisions,
    )

    result = run_customer_decision_batch(
        transactions=empty_transactions,
        decision_date=DECISION_DATE,
    )

    assert result.decisions.empty

    assert (
        captured[
            "decision_date"
        ]
        ==
        DECISION_DATE
    )

    pd.testing.assert_frame_equal(
        captured[
            "transactions"
        ],
        empty_transactions,
    )


# ================================================================
# RESULT IMMUTABILITY
# ================================================================


def test_customer_decision_batch_result_is_frozen():

    result = CustomerDecisionBatchResult(
        decision_date=DECISION_DATE,
        reactivation_features=pd.DataFrame(),
        reactivation_scores=None,
        leakage_features=pd.DataFrame(),
        leakage_scores=None,
        leakage_context=None,
        fcv_features=pd.DataFrame(),
        fcv_scores=None,
        recommendation_scores=None,
        decisions=pd.DataFrame(),
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.decision_date = (
            pd.Timestamp(
                "2011-09-30"
            )
        )