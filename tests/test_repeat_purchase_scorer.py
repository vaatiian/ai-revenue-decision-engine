import numpy as np
import pandas as pd
import pytest

from src.features.repeat_purchase_features import (
    build_repeat_purchase_scoring_features,
)
from src.models.repeat_purchase_scorer import (
    REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS,
    RepeatPurchaseScorer,
    score_repeat_purchase,
    score_repeat_purchase_features,
)


class DummyModel:
    def predict_proba(
        self,
        X,
    ):
        n = len(
            X
        )

        positive = np.full(
            n,
            0.60,
            dtype=float,
        )

        negative = (
            1.0
            -
            positive
        )

        return np.column_stack(
            [
                negative,
                positive,
            ]
        )


class InvalidProbabilityModel:
    def predict_proba(
        self,
        X,
    ):
        n = len(
            X
        )

        return np.column_stack(
            [
                np.zeros(
                    n
                ),
                np.full(
                    n,
                    1.5
                ),
            ]
        )


class WrongLengthModel:
    def predict_proba(
        self,
        X,
    ):
        return np.array(
            [
                [
                    0.4,
                    0.6,
                ]
            ]
        )


class NoPredictProba:
    pass


def make_line(
    date,
    invoice,
    customer,
    product,
    quantity=1,
    unit_price=10.0,
    region="United Kingdom",
):
    return {
        "date": date,
        "invoice": invoice,
        "customer": customer,
        "product": product,
        "product_description": f"Product {product}",
        "region": region,
        "quantity": quantity,
        "unit_price": unit_price,
        "customer_is_identified": True,
        "is_merchandise": True,
        "is_cancellation": False,
        "net_merchandise_revenue": (
            quantity
            *
            unit_price
        ),
    }


def scoring_dataframe():
    return pd.DataFrame(
        [
            make_line(
                "2009-12-10",
                "A1",
                "A",
                "P1",
            ),
            make_line(
                "2010-01-10",
                "B1",
                "B",
                "P1",
                quantity=2,
                unit_price=5,
            ),
            make_line(
                "2010-01-10",
                "B1",
                "B",
                "P2",
                quantity=3,
                unit_price=10,
            ),
            make_line(
                "2010-03-01",
                "C1",
                "C",
                "P3",
            ),
        ]
    )


def test_score_output_columns_are_frozen():
    assert (
        REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS
        ==
        (
            "customer",
            "first_invoice",
            "first_order_date",
            "repeat_purchase_score",
            "model_name",
            "model_version",
            "model_type",
            "target_name",
            "governance_decision",
            "scored_at_utc",
        )
    )


def test_score_features_returns_probability():
    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
    )

    result = (
        score_repeat_purchase_features(
            features=features,
            model=DummyModel(),
        )
    )

    assert (
        result[
            "repeat_purchase_score"
        ]
        .eq(
            0.60
        )
        .all()
    )


def test_score_features_preserves_row_count():
    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
    )

    result = (
        score_repeat_purchase_features(
            features=features,
            model=DummyModel(),
        )
    )

    assert (
        len(
            result
        )
        ==
        len(
            features
        )
    )


def test_score_features_preserves_customer_identity():
    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
    )

    result = (
        score_repeat_purchase_features(
            features=features,
            model=DummyModel(),
        )
    )

    assert (
        result[
            "customer"
        ].tolist()
        ==
        features[
            "customer"
        ].tolist()
    )


def test_score_features_includes_model_metadata():
    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
    )

    result = (
        score_repeat_purchase_features(
            features=features,
            model=DummyModel(),
        )
    )

    assert (
        result[
            "model_name"
        ]
        .eq(
            "Drop-First Interpretable Logistic Regression"
        )
        .all()
    )

    assert (
        result[
            "model_version"
        ]
        .eq(
            "1.0.0"
        )
        .all()
    )

    assert (
        result[
            "model_type"
        ]
        .eq(
            "binary_classifier"
        )
        .all()
    )

    assert (
        result[
            "target_name"
        ]
        .eq(
            "repeat_within_90d"
        )
        .all()
    )

    assert (
        result[
            "governance_decision"
        ]
        .eq(
            "CONDITIONAL GO"
        )
        .all()
    )


def test_score_features_has_utc_timestamp():
    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
    )

    result = (
        score_repeat_purchase_features(
            features=features,
            model=DummyModel(),
        )
    )

    assert (
        result[
            "scored_at_utc"
        ]
        .notna()
        .all()
    )


def test_empty_features_return_empty_contract():
    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
        .iloc[
            0:0
        ]
        .copy()
    )

    result = (
        score_repeat_purchase_features(
            features=features,
            model=DummyModel(),
        )
    )

    assert result.empty

    assert (
        tuple(
            result.columns
        )
        ==
        REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS
    )


def test_missing_metadata_fails_closed():
    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
        .drop(
            columns=[
                "first_invoice",
            ]
        )
    )

    with pytest.raises(
        KeyError
    ):
        score_repeat_purchase_features(
            features=features,
            model=DummyModel(),
        )


def test_model_without_predict_proba_fails():
    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
    )

    with pytest.raises(
        TypeError
    ):
        score_repeat_purchase_features(
            features=features,
            model=NoPredictProba(),
        )


def test_invalid_probability_fails_closed():
    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        score_repeat_purchase_features(
            features=features,
            model=InvalidProbabilityModel(),
        )


def test_wrong_probability_length_fails_closed():
    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
    )

    if len(
        features
    ) <= 1:
        pytest.skip(
            "Fixture requires more than one scoring row."
        )

    with pytest.raises(
        RuntimeError
    ):
        score_repeat_purchase_features(
            features=features,
            model=WrongLengthModel(),
        )


def test_end_to_end_score_function():
    result = (
        score_repeat_purchase(
            data=scoring_dataframe(),
            model=DummyModel(),
        )
    )

    assert not result.empty

    assert (
        result[
            "repeat_purchase_score"
        ]
        .eq(
            0.60
        )
        .all()
    )


def test_opening_month_exclusion_is_respected():
    result = (
        score_repeat_purchase(
            data=scoring_dataframe(),
            model=DummyModel(),
            exclude_opening_month=True,
        )
    )

    assert "A" not in set(
        result[
            "customer"
        ]
    )


def test_opening_month_can_be_retained_for_live_use():
    result = (
        score_repeat_purchase(
            data=scoring_dataframe(),
            model=DummyModel(),
            exclude_opening_month=False,
        )
    )

    assert "A" in set(
        result[
            "customer"
        ]
    )


def test_scorer_accepts_injected_model():
    scorer = (
        RepeatPurchaseScorer(
            model=DummyModel()
        )
    )

    assert isinstance(
        scorer.model,
        DummyModel,
    )


def test_scorer_rejects_invalid_model():
    with pytest.raises(
        TypeError
    ):
        RepeatPurchaseScorer(
            model=NoPredictProba()
        )


def test_scorer_score_features():
    scorer = (
        RepeatPurchaseScorer(
            model=DummyModel()
        )
    )

    features = (
        build_repeat_purchase_scoring_features(
            scoring_dataframe()
        )
    )

    result = (
        scorer.score_features(
            features
        )
    )

    assert (
        result[
            "repeat_purchase_score"
        ]
        .eq(
            0.60
        )
        .all()
    )


def test_scorer_score_raw_data():
    scorer = (
        RepeatPurchaseScorer(
            model=DummyModel()
        )
    )

    result = (
        scorer.score(
            scoring_dataframe()
        )
    )

    assert not result.empty