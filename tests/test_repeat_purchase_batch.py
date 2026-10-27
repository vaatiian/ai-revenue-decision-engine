from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

import src.deployment.repeat_purchase_batch as repeat_batch_module

from src.deployment.repeat_purchase_batch import (
    RepeatPurchaseBatchResult,
    run_repeat_purchase_batch,
)
from src.models.repeat_purchase_scorer import (
    REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS,
)


MODELING_END_DATE = pd.Timestamp(
    "2011-11-30"
)


# ================================================================
# FIXTURES
# ================================================================


def make_transactions():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-01-10 10:00:00",
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


def make_scores():

    return pd.DataFrame(
        {
            column: pd.Series(
                dtype="object"
            )
            for column
            in REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS
        }
    )


# ================================================================
# INPUT VALIDATION
# ================================================================


def test_repeat_purchase_batch_requires_dataframe_transactions():

    with pytest.raises(
        TypeError,
        match=(
            "transactions must be a pandas DataFrame"
        ),
    ):
        run_repeat_purchase_batch(
            transactions=[],
            modeling_end_date=(
                MODELING_END_DATE
            ),
        )


def test_repeat_purchase_batch_requires_explicit_modeling_end_date():

    with pytest.raises(
        ValueError,
        match=(
            "requires an explicit modeling_end_date"
        ),
    ):
        run_repeat_purchase_batch(
            transactions=make_transactions(),
            modeling_end_date=None,
        )


def test_repeat_purchase_batch_rejects_invalid_modeling_end_date():

    with pytest.raises(
        ValueError,
        match=(
            "modeling_end_date is invalid"
        ),
    ):
        run_repeat_purchase_batch(
            transactions=make_transactions(),
            modeling_end_date=(
                "not-a-date"
            ),
        )


def test_repeat_purchase_batch_requires_boolean_opening_month_policy():

    with pytest.raises(
        TypeError,
        match=(
            "exclude_opening_month must be a boolean"
        ),
    ):
        run_repeat_purchase_batch(
            transactions=make_transactions(),
            modeling_end_date=(
                MODELING_END_DATE
            ),
            exclude_opening_month=1,
        )


# ================================================================
# DATE NORMALISATION
# ================================================================


def test_repeat_purchase_batch_normalises_modeling_end_date(
    monkeypatch,
):

    captured = {}

    scores = make_scores()

    class FakeScorer:

        def score(
            self,
            data,
            modeling_end_date=None,
            exclude_opening_month=True,
        ):

            captured[
                "modeling_end_date"
            ] = modeling_end_date

            captured[
                "exclude_opening_month"
            ] = exclude_opening_month

            return scores.copy()

    monkeypatch.setattr(
        repeat_batch_module,
        "RepeatPurchaseScorer",
        FakeScorer,
    )

    result = run_repeat_purchase_batch(
        transactions=make_transactions(),
        modeling_end_date=(
            pd.Timestamp(
                "2011-11-30 18:30:00",
                tz="UTC",
            )
        ),
    )

    assert (
        result.modeling_end_date
        ==
        pd.Timestamp(
            "2011-11-30"
        )
    )

    assert (
        captured[
            "modeling_end_date"
        ]
        ==
        pd.Timestamp(
            "2011-11-30"
        )
    )

    assert (
        captured[
            "exclude_opening_month"
        ]
        is True
    )


# ================================================================
# SCORER ORCHESTRATION
# ================================================================


def test_repeat_purchase_batch_calls_governed_scorer(
    monkeypatch,
):

    transactions = make_transactions()

    scores = make_scores()

    captured = {}

    class FakeScorer:

        def score(
            self,
            data,
            modeling_end_date=None,
            exclude_opening_month=True,
        ):

            captured[
                "data"
            ] = data.copy()

            captured[
                "modeling_end_date"
            ] = modeling_end_date

            captured[
                "exclude_opening_month"
            ] = exclude_opening_month

            return scores.copy()

    monkeypatch.setattr(
        repeat_batch_module,
        "RepeatPurchaseScorer",
        FakeScorer,
    )

    result = run_repeat_purchase_batch(
        transactions=transactions,
        modeling_end_date=(
            MODELING_END_DATE
        ),
        exclude_opening_month=False,
    )

    assert isinstance(
        result,
        RepeatPurchaseBatchResult,
    )

    pd.testing.assert_frame_equal(
        captured[
            "data"
        ],
        transactions,
    )

    assert (
        captured[
            "modeling_end_date"
        ]
        ==
        MODELING_END_DATE
    )

    assert (
        captured[
            "exclude_opening_month"
        ]
        is False
    )

    assert (
        result.exclude_opening_month
        is False
    )

    pd.testing.assert_frame_equal(
        result.scores,
        scores,
    )


def test_repeat_purchase_batch_preserves_opening_month_exclusion_by_default(
    monkeypatch,
):

    captured = {}

    class FakeScorer:

        def score(
            self,
            data,
            modeling_end_date=None,
            exclude_opening_month=True,
        ):

            captured[
                "exclude_opening_month"
            ] = exclude_opening_month

            return make_scores()

    monkeypatch.setattr(
        repeat_batch_module,
        "RepeatPurchaseScorer",
        FakeScorer,
    )

    result = run_repeat_purchase_batch(
        transactions=make_transactions(),
        modeling_end_date=(
            MODELING_END_DATE
        ),
    )

    assert (
        captured[
            "exclude_opening_month"
        ]
        is True
    )

    assert (
        result.exclude_opening_month
        is True
    )


# ================================================================
# FAIL-CLOSED OUTPUT CONTRACT
# ================================================================


def test_repeat_purchase_batch_rejects_score_schema_drift(
    monkeypatch,
):

    class FakeScorer:

        def score(
            self,
            data,
            modeling_end_date=None,
            exclude_opening_month=True,
        ):

            return pd.DataFrame(
                {
                    "customer": [
                        "A",
                    ],
                }
            )

    monkeypatch.setattr(
        repeat_batch_module,
        "RepeatPurchaseScorer",
        FakeScorer,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "does not match "
            "REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS"
        ),
    ):
        run_repeat_purchase_batch(
            transactions=make_transactions(),
            modeling_end_date=(
                MODELING_END_DATE
            ),
        )


# ================================================================
# RESULT IMMUTABILITY
# ================================================================


def test_repeat_purchase_batch_result_is_frozen():

    result = RepeatPurchaseBatchResult(
        modeling_end_date=(
            MODELING_END_DATE
        ),
        exclude_opening_month=True,
        scores=make_scores(),
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.modeling_end_date = (
            pd.Timestamp(
                "2011-12-31"
            )
        )