from datetime import datetime, timezone

import pandas as pd
import pytest

from src.models.base import (
    BaseModelScorer,
    ModelMetadata,
    attach_score_metadata,
    select_model_features,
    validate_required_features,
    validate_unique_keys,
)


# ================================================================
# TEST SCORER
# ================================================================

class DummyScorer(
    BaseModelScorer
):

    metadata = ModelMetadata(
        model_name="dummy_model",
        model_version="1.0.0",
        model_type="classification",
        target_name="dummy_target",
    )

    feature_names = (
        "feature_a",
        "feature_b",
    )

    key_columns = (
        "customer",
    )

    def score(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:

        model_input = self.model_matrix(
            features
        )

        result = features[
            [
                "customer",
            ]
        ].copy()

        result[
            "dummy_score"
        ] = (
            model_input[
                "feature_a"
            ]
            +
            model_input[
                "feature_b"
            ]
        )

        return attach_score_metadata(
            result,
            self.metadata,
        )


# ================================================================
# TEST DATA
# ================================================================

def make_features():

    return pd.DataFrame(
        {
            "customer": [
                "A",
                "B",
            ],

            "feature_a": [
                1.0,
                2.0,
            ],

            "feature_b": [
                3.0,
                4.0,
            ],

            "unused_feature": [
                100,
                200,
            ],
        }
    )


# ================================================================
# REQUIRED FEATURES
# ================================================================

def test_required_features_pass():

    df = make_features()

    validate_required_features(
        df,
        [
            "feature_a",
            "feature_b",
        ],
    )


def test_missing_required_feature_fails():

    df = make_features().drop(
        columns=[
            "feature_b",
        ]
    )

    with pytest.raises(
        ValueError
    ):
        validate_required_features(
            df,
            [
                "feature_a",
                "feature_b",
            ],
        )


# ================================================================
# FEATURE ORDER
# ================================================================

def test_model_features_selected_in_exact_order():

    df = make_features()

    selected = select_model_features(
        df,
        [
            "feature_b",
            "feature_a",
        ],
    )

    assert list(
        selected.columns
    ) == [
        "feature_b",
        "feature_a",
    ]


def test_unused_features_are_not_sent_to_model():

    df = make_features()

    selected = select_model_features(
        df,
        [
            "feature_a",
            "feature_b",
        ],
    )

    assert "unused_feature" not in (
        selected.columns
    )


# ================================================================
# UNIQUE ENTITY KEYS
# ================================================================

def test_unique_keys_pass():

    df = make_features()

    validate_unique_keys(
        df,
        [
            "customer",
        ],
    )


def test_duplicate_keys_fail():

    df = pd.concat(
        [
            make_features(),
            make_features().iloc[
                [
                    0
                ]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError
    ):
        validate_unique_keys(
            df,
            [
                "customer",
            ],
        )


def test_missing_key_column_fails():

    df = (
        make_features()
        .drop(
            columns=[
                "customer",
            ]
        )
    )

    with pytest.raises(
        ValueError
    ):
        validate_unique_keys(
            df,
            [
                "customer",
            ],
        )


# ================================================================
# METADATA
# ================================================================

def test_score_metadata_added():

    scores = pd.DataFrame(
        {
            "customer": [
                "A",
            ],

            "score": [
                0.75,
            ],
        }
    )

    metadata = ModelMetadata(
        model_name="reactivation_model",
        model_version="1.0.0",
        model_type="classification",
        target_name="reactivated",
    )

    scored_at = datetime(
        2026,
        10,
        26,
        9,
        0,
        0,
        tzinfo=timezone.utc,
    )

    result = attach_score_metadata(
        scores,
        metadata,
        scored_at=scored_at,
    )

    assert (
        result.loc[
            0,
            "model_name"
        ]
        ==
        "reactivation_model"
    )

    assert (
        result.loc[
            0,
            "model_version"
        ]
        ==
        "1.0.0"
    )

    assert (
        result.loc[
            0,
            "target_name"
        ]
        ==
        "reactivated"
    )

    assert (
        result.loc[
            0,
            "scored_at_utc"
        ]
        ==
        pd.Timestamp(
            scored_at
        )
    )


# ================================================================
# BASE SCORER
# ================================================================

def test_base_scorer_model_matrix():

    scorer = DummyScorer()

    df = make_features()

    matrix = scorer.model_matrix(
        df
    )

    assert list(
        matrix.columns
    ) == [
        "feature_a",
        "feature_b",
    ]


def test_base_scorer_rejects_empty_input():

    scorer = DummyScorer()

    empty = pd.DataFrame(
        columns=[
            "customer",
            "feature_a",
            "feature_b",
        ]
    )

    with pytest.raises(
        ValueError
    ):
        scorer.model_matrix(
            empty
        )


def test_dummy_scorer_produces_scores():

    scorer = DummyScorer()

    result = scorer.score(
        make_features()
    )

    assert len(result) == 2

    assert (
        result.loc[
            0,
            "dummy_score"
        ]
        ==
        4.0
    )

    assert (
        result.loc[
            1,
            "dummy_score"
        ]
        ==
        6.0
    )


def test_dummy_score_contains_governance_metadata():

    scorer = DummyScorer()

    result = scorer.score(
        make_features()
    )

    required = {
        "model_name",
        "model_version",
        "model_type",
        "target_name",
        "scored_at_utc",
    }

    assert required.issubset(
        set(
            result.columns
        )
    )