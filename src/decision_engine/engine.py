"""
Production Customer Decision Engine.

This module is the thin integration layer that composes the governed Decision
Engine stages:

    canonical transactions
        -> customer decision snapshot
        -> model-signal alignment
        -> deterministic action orchestration
        -> priority / contact policy
        -> canonical customer decision output

No predictive model is loaded or scored here.

The engine accepts already-produced governed scorer outputs and preserves the
independence of each model-scoring interface.
"""

from __future__ import annotations

import pandas as pd

from src.decision_engine.action_orchestration import (
    build_action_orchestration,
)

from src.decision_engine.priority_policy import (
    apply_priority_policy,
)

from src.decision_engine.signal_alignment import (
    align_decision_signals,
)

from src.decision_engine.snapshot import (
    build_decision_snapshot,
    normalise_decision_date,
)

from src.decision_engine.spec import (
    ACTIONABLE_ACTIONS,
    ACTION_GROW,
    ACTION_REACTIVATE,
    ALLOWED_ACTIONS_BY_CUSTOMER_STATE,
    DECISION_OUTPUT_COLUMNS,
    NOT_PRIORITISED,
    PRIMARY_ACTIONS,
)


# ================================================================
# FINAL OUTPUT VALIDATION
# ================================================================


def validate_decision_engine_output(
    output: pd.DataFrame,
    decision_date=None,
) -> None:
    """
    Validate the final canonical Customer Decision Engine output.

    The validation is intentionally business-governance focused. Upstream
    layers remain responsible for their own feature, signal and scoring
    contracts.
    """

    if not isinstance(
        output,
        pd.DataFrame,
    ):
        raise TypeError(
            "Decision Engine output must be a pandas DataFrame."
        )

    if tuple(
        output.columns
    ) != DECISION_OUTPUT_COLUMNS:
        raise ValueError(
            "Decision Engine output columns do not match the canonical "
            "DECISION_OUTPUT_COLUMNS contract."
        )

    if output.empty:
        return

    # ------------------------------------------------------------
    # CUSTOMER KEY
    # ------------------------------------------------------------

    if output[
        "customer"
    ].isna().any():
        raise ValueError(
            "Decision Engine output contains missing customer IDs."
        )

    if not output[
        "customer"
    ].is_unique:
        raise ValueError(
            "Decision Engine output must contain one row per customer."
        )

    # ------------------------------------------------------------
    # DECISION DATE
    # ------------------------------------------------------------

    output_dates = pd.to_datetime(
        output[
            "decision_date"
        ],
        errors="coerce",
    )

    if output_dates.isna().any():
        raise ValueError(
            "Decision Engine output contains invalid decision dates."
        )

    if output_dates.nunique() != 1:
        raise ValueError(
            "Decision Engine output must contain exactly one decision date."
        )

    if decision_date is not None:

        expected_date = (
            normalise_decision_date(
                decision_date
            )
        )

        if not (
            output_dates
            ==
            expected_date
        ).all():
            raise ValueError(
                "Decision Engine output decision date does not match the "
                "requested decision snapshot."
            )

    # ------------------------------------------------------------
    # PRIMARY ACTION
    # ------------------------------------------------------------

    if output[
        "primary_action"
    ].isna().any():
        raise ValueError(
            "Every customer must receive a primary action."
        )

    invalid_actions = (
        set(
            output[
                "primary_action"
            ].astype(str)
        )
        -
        set(
            PRIMARY_ACTIONS
        )
    )

    if invalid_actions:
        raise ValueError(
            "Decision Engine output contains unknown primary actions: "
            f"{sorted(invalid_actions)}"
        )

    # ------------------------------------------------------------
    # STATE / ACTION GOVERNANCE
    # ------------------------------------------------------------

    allowed_actions = dict(
        ALLOWED_ACTIONS_BY_CUSTOMER_STATE
    )

    invalid_state_action_rows = []

    for row in output[
        [
            "customer",
            "customer_state",
            "primary_action",
        ]
    ].itertuples(
        index=False
    ):

        state_actions = (
            allowed_actions.get(
                row.customer_state
            )
        )

        if (
            state_actions is None
            or
            row.primary_action
            not in state_actions
        ):
            invalid_state_action_rows.append(
                (
                    row.customer,
                    row.customer_state,
                    row.primary_action,
                )
            )

    if invalid_state_action_rows:
        raise ValueError(
            "Decision Engine output violates frozen state/action governance: "
            f"{invalid_state_action_rows}"
        )

    # ------------------------------------------------------------
    # ACTIONABLE / NON-ACTIONABLE POLICY
    # ------------------------------------------------------------

    actionable_mask = (
        output[
            "primary_action"
        ]
        .isin(
            ACTIONABLE_ACTIONS
        )
    )

    non_actionable_mask = (
        ~actionable_mask
    )

    if not (
        output.loc[
            non_actionable_mask,
            "priority_tier",
        ]
        ==
        NOT_PRIORITISED
    ).all():
        raise ValueError(
            "Non-actionable customers must be Not Prioritised."
        )

    if output.loc[
        non_actionable_mask,
        "primary_action_score",
    ].notna().any():
        raise ValueError(
            "Non-actionable customers cannot have primary action scores."
        )

    if output.loc[
        non_actionable_mask,
        "action_priority_percentile",
    ].notna().any():
        raise ValueError(
            "Non-actionable customers cannot have action priority "
            "percentiles."
        )

    # ------------------------------------------------------------
    # ACTION-SPECIFIC FINAL GUARANTEES
    # ------------------------------------------------------------

    grow_mask = (
        output[
            "primary_action"
        ]
        ==
        ACTION_GROW
    )

    if output.loc[
        grow_mask,
        "recommended_product",
    ].isna().any():
        raise ValueError(
            "Grow customers must have a recommended product."
        )

    reactivate_mask = (
        output[
            "primary_action"
        ]
        ==
        ACTION_REACTIVATE
    )

    if output.loc[
        reactivate_mask,
        "reactivation_probability",
    ].isna().any():
        raise ValueError(
            "Reactivate customers must have a Reactivation probability."
        )


# ================================================================
# PRODUCTION ENGINE
# ================================================================


def build_customer_decisions(
    transactions: pd.DataFrame,
    decision_date,
    *,
    leakage_scores: pd.DataFrame | None = None,
    leakage_context: pd.DataFrame | None = None,
    fcv_scores: pd.DataFrame | None = None,
    reactivation_scores: pd.DataFrame | None = None,
    reactivation_features: pd.DataFrame | None = None,
    recommendation_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the canonical customer-level Decision Engine output.

    Parameters
    ----------
    transactions:
        Canonical retail transactions used to construct the point-in-time
        customer snapshot.

    decision_date:
        Decision snapshot timestamp.

    leakage_scores:
        Optional governed production Revenue Leakage scorer output.

    leakage_context:
        Required when Leakage scores are supplied. Must provide the baseline
        revenue context needed for research-parity economic prioritisation.

    fcv_scores:
        Optional governed production Future Customer Value scorer output.

    reactivation_scores:
        Optional governed production Reactivation scorer output.

    reactivation_features:
        Required when Reactivation scores are supplied. Used to restore the
        business prediction timestamp that is intentionally not emitted by the
        scorer wrapper.

    recommendation_scores:
        Optional governed production recommendation scorer output. Multiple
        historical recommendation episodes per customer are permitted.

    Returns
    -------
    pandas.DataFrame
        Exact DECISION_OUTPUT_COLUMNS schema with one row per customer.
    """

    # ------------------------------------------------------------
    # 6C — DECISION SNAPSHOT
    # ------------------------------------------------------------

    snapshot = (
        build_decision_snapshot(
            transactions=transactions,
            decision_date=decision_date,
        )
    )

    snapshot_population = len(
        snapshot
    )

    # ------------------------------------------------------------
    # 6D — SIGNAL ALIGNMENT
    # ------------------------------------------------------------

    aligned = (
        align_decision_signals(
            snapshot,
            leakage_scores=leakage_scores,
            leakage_context=leakage_context,
            fcv_scores=fcv_scores,
            reactivation_scores=reactivation_scores,
            reactivation_features=reactivation_features,
            recommendation_scores=recommendation_scores,
        )
    )

    if len(
        aligned
    ) != snapshot_population:
        raise RuntimeError(
            "Signal alignment changed the Decision Engine customer population."
        )

    # ------------------------------------------------------------
    # 6E — ACTION ORCHESTRATION
    # ------------------------------------------------------------

    orchestrated = (
        build_action_orchestration(
            aligned
        )
    )

    if len(
        orchestrated
    ) != snapshot_population:
        raise RuntimeError(
            "Action orchestration changed the Decision Engine customer "
            "population."
        )

    # ------------------------------------------------------------
    # 6F — PRIORITY / CONTACT POLICY
    # ------------------------------------------------------------

    prioritised = (
        apply_priority_policy(
            orchestrated
        )
    )

    if len(
        prioritised
    ) != snapshot_population:
        raise RuntimeError(
            "Priority policy changed the Decision Engine customer population."
        )

    # ------------------------------------------------------------
    # FINAL CANONICAL OUTPUT
    # ------------------------------------------------------------

    missing_output_columns = [
        column
        for column in DECISION_OUTPUT_COLUMNS
        if column not in prioritised.columns
    ]

    if missing_output_columns:
        raise RuntimeError(
            "Integrated Decision Engine is missing canonical output columns: "
            + ", ".join(
                missing_output_columns
            )
        )

    output = (
        prioritised.loc[
            :,
            list(
                DECISION_OUTPUT_COLUMNS
            ),
        ]
        .copy()
    )

    validate_decision_engine_output(
        output,
        decision_date=decision_date,
    )

    return output