"""
Monthly customer-decision batch orchestrator.

This module connects the already-governed production feature builders,
scorers, and Customer Decision Engine into one scheduler-safe batch.

Responsibilities
----------------
- validate the monthly decision boundary;
- construct frozen production feature frames;
- score only eligible customer populations;
- restore Reactivation prediction-date provenance from the explicit
  batch boundary without changing the frozen model feature matrix;
- derive Revenue Leakage economic context from the same feature frame
  that was scored;
- optionally score governed recommendation episodes;
- pass native governed scorer outputs into the production Decision Engine;
- return one immutable orchestration result.

Non-responsibilities
--------------------
This module does not:
- retrain models;
- change model governance;
- change Decision Engine policy;
- modify frozen model feature schemas;
- invent recommendation episodes;
- reconstruct future-conditioned research evaluation populations;
- build delivery payloads;
- write files;
- append audit ledgers;
- schedule execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.decision_engine.engine import (
    build_customer_decisions,
)
from src.features.customer_snapshot import (
    normalise_decision_date,
)
from src.features.fcv_model_features import (
    build_fcv_model_features,
)
from src.features.leakage_model_features import (
    build_leakage_model_features,
)
from src.features.reactivation_model_features import (
    build_reactivation_model_features,
)
from src.models.fcv_scorer import (
    FCVScorer,
)
from src.models.leakage_scorer import (
    LeakageScorer,
)
from src.models.reactivation_scorer import (
    ReactivationScorer,
)
from src.models.recommendation_scorer import (
    RecommendationScorer,
)


# ================================================================
# RESULT CONTRACT
# ================================================================


@dataclass(
    frozen=True,
)
class CustomerDecisionBatchResult:
    """
    Immutable top-level result from one monthly customer-decision batch.

    DataFrame objects remain ordinary pandas objects, but the batch result's
    field references cannot be reassigned after construction.
    """

    decision_date: pd.Timestamp

    reactivation_features: pd.DataFrame
    reactivation_scores: pd.DataFrame | None

    leakage_features: pd.DataFrame
    leakage_scores: pd.DataFrame | None
    leakage_context: pd.DataFrame | None

    fcv_features: pd.DataFrame
    fcv_scores: pd.DataFrame | None

    recommendation_scores: pd.DataFrame | None

    decisions: pd.DataFrame


# ================================================================
# VALIDATION
# ================================================================


def _normalise_month_end_decision_date(
    decision_date,
) -> pd.Timestamp:
    """
    Normalize and validate the governed monthly decision boundary.

    Revenue Leakage and FCV both use frozen month-end prediction points.
    The integrated customer-decision batch therefore adopts month-end as
    its common execution boundary.
    """

    normalized = normalise_decision_date(
        decision_date
    )

    if not normalized.is_month_end:
        raise ValueError(
            "Customer decision batch requires a month-end decision date."
        )

    return normalized


def _validate_transactions(
    transactions: pd.DataFrame,
) -> None:
    """
    Validate the orchestration-level transaction input type.

    Detailed transaction-schema validation remains owned by the canonical
    loader and downstream production feature builders.
    """

    if not isinstance(
        transactions,
        pd.DataFrame,
    ):
        raise TypeError(
            "transactions must be a pandas DataFrame."
        )


def _validate_recommendation_episodes(
    recommendation_episodes: pd.DataFrame | None,
) -> None:
    """
    Validate only the orchestration-level recommendation input type.

    The RecommendationScorer owns the frozen episode schema validation.
    """

    if recommendation_episodes is None:
        return

    if not isinstance(
        recommendation_episodes,
        pd.DataFrame,
    ):
        raise TypeError(
            "recommendation_episodes must be a pandas DataFrame or None."
        )


# ================================================================
# EMPTY BATCH
# ================================================================


def _run_empty_customer_batch(
    transactions: pd.DataFrame,
    decision_date: pd.Timestamp,
) -> CustomerDecisionBatchResult:
    """
    Return a valid empty batch without invoking model scorers.

    The production Decision Engine already owns the exact canonical empty
    decision-output schema, so empty transactions are delegated directly
    to it.
    """

    decisions = build_customer_decisions(
        transactions=transactions,
        decision_date=decision_date,
    )

    return CustomerDecisionBatchResult(
        decision_date=decision_date,
        reactivation_features=pd.DataFrame(),
        reactivation_scores=None,
        leakage_features=pd.DataFrame(),
        leakage_scores=None,
        leakage_context=None,
        fcv_features=pd.DataFrame(),
        fcv_scores=None,
        recommendation_scores=None,
        decisions=decisions,
    )


# ================================================================
# MONTHLY CUSTOMER DECISION BATCH
# ================================================================


def run_customer_decision_batch(
    transactions: pd.DataFrame,
    decision_date,
    *,
    recommendation_episodes: pd.DataFrame | None = None,
    recommendation_scored_at: datetime | None = None,
) -> CustomerDecisionBatchResult:
    """
    Execute one governed month-end customer-decision batch.

    Parameters
    ----------
    transactions:
        Canonical validated retail transaction data.

    decision_date:
        Month-end point-in-time boundary used by the integrated customer
        decision run.

    recommendation_episodes:
        Optional already-observed recommendation episodes.

        The required production episode schema is owned by
        RecommendationScorer and contains:

            customer
            invoice
            order_date
            prediction_period

        This orchestrator never reconstructs Notebook 06's
        future-conditioned research evaluation population.

    recommendation_scored_at:
        Optional UTC-aware timestamp passed through to the deterministic
        recommendation scorer for reproducible scoring provenance.

    Returns
    -------
    CustomerDecisionBatchResult
        Frozen orchestration result containing feature frames, native
        governed scorer outputs, and the canonical Decision Engine output.
    """

    _validate_transactions(
        transactions
    )

    _validate_recommendation_episodes(
        recommendation_episodes
    )

    resolved_decision_date = (
        _normalise_month_end_decision_date(
            decision_date
        )
    )

    if transactions.empty:
        return _run_empty_customer_batch(
            transactions=transactions,
            decision_date=resolved_decision_date,
        )

    # ------------------------------------------------------------
    # REACTIVATION
    # ------------------------------------------------------------

    reactivation_features = (
        build_reactivation_model_features(
            transactions=transactions,
            prediction_date=(
                resolved_decision_date
            ),
        )
    )

    reactivation_scores = None
    reactivation_context = None

    if not reactivation_features.empty:

        if (
            "customer"
            not in reactivation_features.columns
        ):
            raise RuntimeError(
                "Reactivation feature frame is missing customer."
            )

        reactivation_scores = (
            ReactivationScorer()
            .score(
                reactivation_features
            )
        )

        reactivation_context = (
            reactivation_features[
                [
                    "customer",
                ]
            ]
            .copy()
        )

        reactivation_context[
            "prediction_date"
        ] = resolved_decision_date

    # ------------------------------------------------------------
    # REVENUE LEAKAGE
    # ------------------------------------------------------------

    leakage_features = (
        build_leakage_model_features(
            transactions=transactions,
            prediction_date=(
                resolved_decision_date
            ),
        )
    )

    leakage_scores = None
    leakage_context = None

    if not leakage_features.empty:

        if (
            "customer"
            not in leakage_features.columns
        ):
            raise RuntimeError(
                "Revenue Leakage feature frame is missing customer."
            )

        if (
            "b90_revenue"
            not in leakage_features.columns
        ):
            raise RuntimeError(
                "Revenue Leakage feature frame is missing "
                "b90_revenue required to construct "
                "Decision Engine baseline_revenue context."
            )

        leakage_scores = (
            LeakageScorer()
            .score(
                leakage_features
            )
        )

        leakage_context = (
            leakage_features[
                [
                    "customer",
                    "b90_revenue",
                ]
            ]
            .rename(
                columns={
                    "b90_revenue":
                        "baseline_revenue",
                }
            )
            .copy()
        )

    # ------------------------------------------------------------
    # FUTURE CUSTOMER VALUE
    # ------------------------------------------------------------

    fcv_features = (
        build_fcv_model_features(
            transactions=transactions,
            prediction_date=(
                resolved_decision_date
            ),
        )
    )

    fcv_scores = None

    if not fcv_features.empty:
        fcv_scores = (
            FCVScorer()
            .score(
                fcv_features
            )
        )

    # ------------------------------------------------------------
    # RECOMMENDATION
    # ------------------------------------------------------------

    recommendation_scores = None

    if (
        recommendation_episodes is not None
        and
        not recommendation_episodes.empty
    ):
        recommendation_scores = (
            RecommendationScorer()
            .score(
                transactions=transactions,
                episodes=(
                    recommendation_episodes
                ),
                scored_at=(
                    recommendation_scored_at
                ),
            )
        )

    # ------------------------------------------------------------
    # GOVERNED CUSTOMER DECISION ENGINE
    # ------------------------------------------------------------

    decisions = build_customer_decisions(
        transactions=transactions,
        decision_date=(
            resolved_decision_date
        ),
        leakage_scores=(
            leakage_scores
        ),
        leakage_context=(
            leakage_context
        ),
        fcv_scores=(
            fcv_scores
        ),
        reactivation_scores=(
            reactivation_scores
        ),
        reactivation_features=(
            reactivation_context
        ),
        recommendation_scores=(
            recommendation_scores
        ),
    )

    return CustomerDecisionBatchResult(
        decision_date=(
            resolved_decision_date
        ),
        reactivation_features=(
            reactivation_features
        ),
        reactivation_scores=(
            reactivation_scores
        ),
        leakage_features=(
            leakage_features
        ),
        leakage_scores=(
            leakage_scores
        ),
        leakage_context=(
            leakage_context
        ),
        fcv_features=(
            fcv_features
        ),
        fcv_scores=(
            fcv_scores
        ),
        recommendation_scores=(
            recommendation_scores
        ),
        decisions=(
            decisions
        ),
    )