"""
Interactive portfolio adapter for the governed Decision Engine policy.

This module does not implement a second decision policy.

It deliberately calls the real production functions:

    src.decision_engine.action_orchestration.build_action_orchestration
    src.decision_engine.priority_policy.apply_priority_policy

The public demo operates at the post-signal-alignment layer. It therefore
does not load model binaries, raw transactions, or production audit state.

Priority percentiles in production are cohort-relative within primary action.
A deterministic reference cohort is used so the interactive demo can show
meaningful Priority 1–4 behaviour instead of the trivial single-row 100th
percentile outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.decision_engine.action_orchestration import (
    build_action_orchestration,
)

from src.decision_engine.priority_policy import (
    apply_priority_policy,
)

from src.decision_engine.spec import (
    ACTIONABLE_ACTIONS,
    ACTION_GROW,
    ACTION_PROTECT,
    ACTION_REACTIVATE,
    ACTION_RETAIN,
    CUSTOMER_STATE_ACTIVE,
    CUSTOMER_STATE_AT_RISK,
    CUSTOMER_STATE_DORMANT,
    CUSTOMER_STATE_LONG_DORMANT,
    CUSTOMER_STATE_UNCLASSIFIED,
    CUSTOMER_STATES,
)


DEMO_CUSTOMER = "DEMO-CUSTOMER"

REFERENCE_SCORE_LEVELS = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)


ECONOMIC_PRIORITY_BANDS = (
    "P1 — Highest Exposure",
    "P2",
    "P3",
    "P4",
    "P5 — Lowest Exposure",
)


@dataclass(frozen=True)
class DecisionDemoInputs:
    """
    Controlled post-alignment inputs for one simulated customer.

    Availability/validity toggles govern whether the corresponding
    production-policy signal is considered usable.
    """

    customer_state: str = CUSTOMER_STATE_ACTIVE

    economic_priority_band: str = "P3"

    leakage_operationally_valid: bool = False
    leakage_exposure_percentile: float = 0.50
    leakage_probability: float = 0.50

    fcv_signal_available: bool = True
    fcv_percentile: float = 0.50
    future_active_probability: float = 0.50

    reactivation_operationally_valid: bool = False
    reactivation_probability: float = 0.50

    recommendation_operationally_valid: bool = False

    repeat_product: str = "PROD-DEMO-REPEAT"
    discovery_product: str = "PROD-DEMO-DISCOVERY"


def _validate_probability(
    value: float,
    name: str,
) -> float:
    """
    Validate one bounded 0–1 demo input.
    """

    numeric = float(
        value
    )

    if not (
        0.0
        <=
        numeric
        <=
        1.0
    ):
        raise ValueError(
            f"{name} must lie within [0, 1]."
        )

    return numeric


def validate_demo_inputs(
    inputs: DecisionDemoInputs,
) -> None:
    """
    Fail closed on unsupported interactive demo inputs.
    """

    if inputs.customer_state not in CUSTOMER_STATES:
        raise ValueError(
            "Unsupported customer state: "
            f"{inputs.customer_state!r}"
        )

    if (
        inputs.economic_priority_band
        not in
        ECONOMIC_PRIORITY_BANDS
    ):
        raise ValueError(
            "Unsupported economic priority band: "
            f"{inputs.economic_priority_band!r}"
        )

    boolean_fields = (
        (
            "leakage_operationally_valid",
            inputs.leakage_operationally_valid,
        ),
        (
            "fcv_signal_available",
            inputs.fcv_signal_available,
        ),
        (
            "reactivation_operationally_valid",
            inputs.reactivation_operationally_valid,
        ),
        (
            "recommendation_operationally_valid",
            inputs.recommendation_operationally_valid,
        ),
    )

    for name, value in boolean_fields:

        if not isinstance(
            value,
            bool,
        ):
            raise TypeError(
                f"{name} must be a bool."
            )

    probability_fields = (
        (
            "leakage_exposure_percentile",
            inputs.leakage_exposure_percentile,
        ),
        (
            "leakage_probability",
            inputs.leakage_probability,
        ),
        (
            "fcv_percentile",
            inputs.fcv_percentile,
        ),
        (
            "future_active_probability",
            inputs.future_active_probability,
        ),
        (
            "reactivation_probability",
            inputs.reactivation_probability,
        ),
    )

    for name, value in probability_fields:
        _validate_probability(
            value,
            name,
        )

    if not str(
        inputs.repeat_product
    ).strip():
        raise ValueError(
            "repeat_product must not be blank."
        )

    if not str(
        inputs.discovery_product
    ).strip():
        raise ValueError(
            "discovery_product must not be blank."
        )


def build_demo_aligned_input(
    inputs: DecisionDemoInputs,
) -> pd.DataFrame:
    """
    Build one production-compatible aligned-policy row.

    Signal values become missing when their governed availability or
    operational-validity flag is false. This mirrors the meaning of the
    production alignment layer instead of allowing internally contradictory
    demo inputs.
    """

    validate_demo_inputs(
        inputs
    )

    leakage_probability = (
        float(
            inputs.leakage_probability
        )
        if inputs.leakage_operationally_valid
        else np.nan
    )

    leakage_exposure_percentile = (
        float(
            inputs.leakage_exposure_percentile
        )
        if inputs.leakage_operationally_valid
        else np.nan
    )

    fcv_percentile = (
        float(
            inputs.fcv_percentile
        )
        if inputs.fcv_signal_available
        else np.nan
    )

    future_active_probability = (
        float(
            inputs.future_active_probability
        )
        if inputs.fcv_signal_available
        else np.nan
    )

    reactivation_probability = (
        float(
            inputs.reactivation_probability
        )
        if inputs.reactivation_operationally_valid
        else np.nan
    )

    repeat_product = (
        str(
            inputs.repeat_product
        )
        if inputs.recommendation_operationally_valid
        else pd.NA
    )

    discovery_product = (
        str(
            inputs.discovery_product
        )
        if inputs.recommendation_operationally_valid
        else pd.NA
    )

    return pd.DataFrame(
        [
            {
                "customer":
                    DEMO_CUSTOMER,

                "customer_state":
                    inputs.customer_state,

                "economic_priority_band":
                    inputs.economic_priority_band,

                "leakage_operationally_valid":
                    inputs.leakage_operationally_valid,

                "fcv_signal_available":
                    inputs.fcv_signal_available,

                "reactivation_operationally_valid":
                    inputs.reactivation_operationally_valid,

                "recommendation_operationally_valid":
                    inputs.recommendation_operationally_valid,

                "leakage_exposure_percentile":
                    leakage_exposure_percentile,

                "leakage_probability":
                    leakage_probability,

                "fcv_percentile":
                    fcv_percentile,

                "future_active_probability":
                    future_active_probability,

                "reactivation_probability":
                    reactivation_probability,

                "repeat_top1_product":
                    repeat_product,

                "discovery_top1_product":
                    discovery_product,
            }
        ]
    )


def _reference_row(
    *,
    customer: str,
    action: str,
    score_level: float,
) -> dict[str, Any]:
    """
    Build one deterministic priority-ranking anchor.

    Setting every action-relevant score input to the same level means the
    production weighted score also equals that level because each frozen
    action-specific weight set sums to 1.0.

    The portfolio does not calculate those weights itself; production
    apply_priority_policy still calculates the score.
    """

    recommendation_valid = (
        action
        ==
        ACTION_GROW
    )

    return {
        "customer":
            customer,

        "primary_action":
            action,

        "recommendation_operationally_valid":
            recommendation_valid,

        "leakage_exposure_percentile":
            float(
                score_level
            ),

        "leakage_probability":
            float(
                score_level
            ),

        "fcv_percentile":
            float(
                score_level
            ),

        "future_active_probability":
            float(
                score_level
            ),

        "reactivation_probability":
            float(
                score_level
            ),

        "repeat_top1_product":
            (
                f"{customer}-REPEAT"
                if recommendation_valid
                else pd.NA
            ),

        "discovery_top1_product":
            (
                f"{customer}-DISCOVERY"
                if recommendation_valid
                else pd.NA
            ),
    }


def build_priority_reference_cohort(
    user_row: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the deterministic action-relative ranking cohort.

    Non-actionable decisions do not require ranking anchors because
    production correctly marks them Not Prioritised.
    """

    if len(
        user_row
    ) != 1:
        raise ValueError(
            "Decision demo requires exactly one user row."
        )

    action = user_row[
        "primary_action"
    ].iloc[
        0
    ]

    required_columns = (
        "customer",
        "primary_action",
        "recommendation_operationally_valid",
        "leakage_exposure_percentile",
        "leakage_probability",
        "fcv_percentile",
        "future_active_probability",
        "reactivation_probability",
        "repeat_top1_product",
        "discovery_top1_product",
    )

    user_policy_row = user_row.loc[
        :,
        list(
            required_columns
        ),
    ].copy()

    if action not in ACTIONABLE_ACTIONS:
        return user_policy_row

    reference_rows = [
        _reference_row(
            customer=(
                f"REFERENCE-{index:02d}"
            ),
            action=action,
            score_level=score_level,
        )
        for index, score_level
        in enumerate(
            REFERENCE_SCORE_LEVELS,
            start=1,
        )
    ]

    reference_frame = pd.DataFrame(
        reference_rows
    )

    return pd.concat(
        [
            user_policy_row,
            reference_frame,
        ],
        ignore_index=True,
    )


def run_decision_demo(
    inputs: DecisionDemoInputs,
) -> dict[str, Any]:
    """
    Run the real governed production action + priority policy.

    Returns only the simulated customer's result. Reference cohort rows
    exist solely to reproduce action-relative percentile ranking.
    """

    aligned = build_demo_aligned_input(
        inputs
    )

    orchestrated = build_action_orchestration(
        aligned
    )

    if len(
        orchestrated
    ) != 1:
        raise RuntimeError(
            "Decision demo action orchestration changed the user population."
        )

    primary_action = orchestrated[
        "primary_action"
    ].iloc[
        0
    ]

    policy_cohort = build_priority_reference_cohort(
        orchestrated
    )

    prioritised = apply_priority_policy(
        policy_cohort
    )

    user_result = (
        prioritised.loc[
            prioritised[
                "customer"
            ]
            ==
            DEMO_CUSTOMER
        ]
        .copy()
    )

    if len(
        user_result
    ) != 1:
        raise RuntimeError(
            "Decision demo could not isolate exactly one simulated customer."
        )

    row = user_result.iloc[
        0
    ]

    eligibility = {
        "high_leakage_exposure":
            bool(
                orchestrated[
                    "high_leakage_exposure"
                ].iloc[
                    0
                ]
            ),

        "protect_eligible":
            bool(
                orchestrated[
                    "protect_eligible"
                ].iloc[
                    0
                ]
            ),

        "grow_eligible":
            bool(
                orchestrated[
                    "grow_eligible"
                ].iloc[
                    0
                ]
            ),

        "retain_eligible":
            bool(
                orchestrated[
                    "retain_eligible"
                ].iloc[
                    0
                ]
            ),

        "reactivate_eligible":
            bool(
                orchestrated[
                    "reactivate_eligible"
                ].iloc[
                    0
                ]
            ),
    }

    def value_or_none(
        value,
    ):
        if pd.isna(
            value
        ):
            return None

        if hasattr(
            value,
            "item",
        ):
            try:
                return value.item()

            except (
                ValueError,
                TypeError,
                AttributeError,
            ):
                pass

        return value

    return {
        "customer_state":
            inputs.customer_state,

        "primary_action":
            value_or_none(
                row[
                    "primary_action"
                ]
            ),

        "primary_action_score":
            value_or_none(
                row[
                    "primary_action_score"
                ]
            ),

        "action_priority_percentile":
            value_or_none(
                row[
                    "action_priority_percentile"
                ]
            ),

        "priority_tier":
            value_or_none(
                row[
                    "priority_tier"
                ]
            ),

        "contact_intensity":
            value_or_none(
                row[
                    "contact_intensity"
                ]
            ),

        "recommended_product_channel":
            value_or_none(
                row[
                    "recommended_product_channel"
                ]
            ),

        "recommended_product":
            value_or_none(
                row[
                    "recommended_product"
                ]
            ),

        "decision_reason":
            value_or_none(
                row[
                    "decision_reason"
                ]
            ),

        "business_action":
            value_or_none(
                row[
                    "business_action"
                ]
            ),

        "experiment_requirement":
            value_or_none(
                row[
                    "experiment_requirement"
                ]
            ),

        "eligibility":
            eligibility,

        "reference_cohort_size":
            int(
                len(
                    policy_cohort
                )
            ),

        "policy_source":
            (
                "src.decision_engine.action_orchestration."
                "build_action_orchestration + "
                "src.decision_engine.priority_policy."
                "apply_priority_policy"
            ),
    }


def demo_presets() -> dict[str, DecisionDemoInputs]:
    """
    Provide safe starting scenarios for the interactive page.

    Presets supply inputs only. Outputs are always produced by the real
    production policy functions.
    """

    return {
        "Active · Protect":
            DecisionDemoInputs(
                customer_state=(
                    CUSTOMER_STATE_ACTIVE
                ),
                economic_priority_band=(
                    "P1 — Highest Exposure"
                ),
                leakage_operationally_valid=True,
                leakage_exposure_percentile=0.95,
                leakage_probability=0.85,
                fcv_signal_available=True,
                fcv_percentile=0.85,
                future_active_probability=0.80,
                recommendation_operationally_valid=True,
            ),

        "Active · Grow":
            DecisionDemoInputs(
                customer_state=(
                    CUSTOMER_STATE_ACTIVE
                ),
                economic_priority_band="P3",
                leakage_operationally_valid=True,
                leakage_exposure_percentile=0.45,
                leakage_probability=0.40,
                fcv_signal_available=True,
                fcv_percentile=0.90,
                future_active_probability=0.90,
                recommendation_operationally_valid=True,
            ),

        "At Risk · Retain":
            DecisionDemoInputs(
                customer_state=(
                    CUSTOMER_STATE_AT_RISK
                ),
                economic_priority_band="P2",
                leakage_operationally_valid=True,
                leakage_exposure_percentile=0.85,
                leakage_probability=0.80,
                fcv_signal_available=True,
                fcv_percentile=0.75,
                future_active_probability=0.65,
                recommendation_operationally_valid=True,
            ),

        "Dormant · Reactivate":
            DecisionDemoInputs(
                customer_state=(
                    CUSTOMER_STATE_DORMANT
                ),
                economic_priority_band="P5 — Lowest Exposure",
                leakage_operationally_valid=False,
                fcv_signal_available=True,
                fcv_percentile=0.75,
                future_active_probability=0.40,
                reactivation_operationally_valid=True,
                reactivation_probability=0.85,
                recommendation_operationally_valid=False,
            ),

        "Dormant · No Score":
            DecisionDemoInputs(
                customer_state=(
                    CUSTOMER_STATE_DORMANT
                ),
                economic_priority_band="P5 — Lowest Exposure",
                leakage_operationally_valid=False,
                fcv_signal_available=False,
                reactivation_operationally_valid=False,
                recommendation_operationally_valid=False,
            ),

        "Long Dormant":
            DecisionDemoInputs(
                customer_state=(
                    CUSTOMER_STATE_LONG_DORMANT
                ),
                economic_priority_band="P1 — Highest Exposure",
                leakage_operationally_valid=True,
                leakage_exposure_percentile=1.0,
                leakage_probability=1.0,
                fcv_signal_available=True,
                fcv_percentile=1.0,
                future_active_probability=1.0,
                reactivation_operationally_valid=True,
                reactivation_probability=1.0,
                recommendation_operationally_valid=True,
            ),

        "Unclassified":
            DecisionDemoInputs(
                customer_state=(
                    CUSTOMER_STATE_UNCLASSIFIED
                ),
                economic_priority_band="P3",
                leakage_operationally_valid=False,
                fcv_signal_available=False,
                reactivation_operationally_valid=False,
                recommendation_operationally_valid=False,
            ),
    }