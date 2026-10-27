"""
Scheduled Repeat Purchase propensity batch.

This module provides the deployment boundary for the frozen
90-day Repeat Purchase production scorer.

Responsibilities
----------------
- require an explicit production observation boundary;
- normalize the modeling end date deterministically;
- preserve the governed opening-month exclusion policy by default;
- invoke the existing frozen production scorer;
- validate the returned score contract;
- return one immutable orchestration result.

Non-responsibilities
--------------------
This module does not:
- retrain the Repeat Purchase model;
- change model features or governance;
- feed Repeat Purchase into the Customer Decision Engine;
- write files;
- schedule execution;
- create delivery actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.repeat_purchase_scorer import (
    REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS,
    RepeatPurchaseScorer,
)


# ================================================================
# RESULT CONTRACT
# ================================================================


@dataclass(
    frozen=True,
)
class RepeatPurchaseBatchResult:
    """
    Immutable result from one governed Repeat Purchase scoring batch.
    """

    modeling_end_date: pd.Timestamp
    exclude_opening_month: bool
    scores: pd.DataFrame


# ================================================================
# VALIDATION
# ================================================================


def _validate_transactions(
    transactions: pd.DataFrame,
) -> None:
    """
    Validate the orchestration-level transaction input type.
    """

    if not isinstance(
        transactions,
        pd.DataFrame,
    ):
        raise TypeError(
            "transactions must be a pandas DataFrame."
        )


def _normalise_modeling_end_date(
    modeling_end_date,
) -> pd.Timestamp:
    """
    Resolve an explicit production observation boundary.

    The scheduled deployment layer intentionally does not inherit the
    research-era default modeling end date.
    """

    if modeling_end_date is None:
        raise ValueError(
            "Repeat Purchase batch requires an explicit modeling_end_date."
        )

    try:

        value = pd.Timestamp(
            modeling_end_date
        )

    except Exception as exc:

        raise ValueError(
            "Repeat Purchase modeling_end_date is invalid."
        ) from exc

    if pd.isna(
        value
    ):
        raise ValueError(
            "Repeat Purchase modeling_end_date is invalid."
        )

    if value.tzinfo is not None:
        value = value.tz_convert(
            None
        )

    return value.normalize()


def _validate_exclude_opening_month(
    exclude_opening_month,
) -> None:
    """
    Require an explicit boolean policy value.
    """

    if not isinstance(
        exclude_opening_month,
        bool,
    ):
        raise TypeError(
            "exclude_opening_month must be a boolean."
        )


def _validate_score_contract(
    scores: pd.DataFrame,
) -> None:
    """
    Fail closed if the frozen scorer output schema drifts.
    """

    if not isinstance(
        scores,
        pd.DataFrame,
    ):
        raise TypeError(
            "Repeat Purchase scorer must return a pandas DataFrame."
        )

    if tuple(
        scores.columns
    ) != REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS:
        raise RuntimeError(
            "Repeat Purchase scorer output does not match "
            "REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS."
        )


# ================================================================
# SCHEDULED BATCH
# ================================================================


def run_repeat_purchase_batch(
    transactions: pd.DataFrame,
    modeling_end_date,
    *,
    exclude_opening_month: bool = True,
) -> RepeatPurchaseBatchResult:
    """
    Execute one governed Repeat Purchase propensity batch.

    Parameters
    ----------
    transactions:
        Canonical retail transactions.

    modeling_end_date:
        Explicit point-in-time observation boundary. This argument is
        required by the deployment layer so scheduled inference cannot
        silently fall back to the frozen historical research date.

    exclude_opening_month:
        Preserve historical-parity handling by default.

        The frozen production feature implementation documents that a
        live integration may explicitly set this False only when the
        system has a trustworthy acquisition-history boundary.

    Returns
    -------
    RepeatPurchaseBatchResult
        Immutable orchestration result containing the resolved observation
        boundary, policy flag, and canonical production scores.
    """

    _validate_transactions(
        transactions
    )

    _validate_exclude_opening_month(
        exclude_opening_month
    )

    resolved_modeling_end_date = (
        _normalise_modeling_end_date(
            modeling_end_date
        )
    )

    scorer = RepeatPurchaseScorer()

    scores = scorer.score(
        data=transactions,
        modeling_end_date=(
            resolved_modeling_end_date
        ),
        exclude_opening_month=(
            exclude_opening_month
        ),
    )

    _validate_score_contract(
        scores
    )

    return RepeatPurchaseBatchResult(
        modeling_end_date=(
            resolved_modeling_end_date
        ),
        exclude_opening_month=(
            exclude_opening_month
        ),
        scores=(
            scores
        ),
    )