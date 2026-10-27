"""
Frozen production specification for the Customer Decision Engine.

The Customer Decision Engine is a deterministic orchestration layer.

It does not train another predictive model. It combines governed model
signals with current customer lifecycle state to determine one primary
business action per customer.

The specification in this module is derived from Notebook 08 and freezes:

- lifecycle-state thresholds;
- canonical actions;
- action-specific priority policy;
- priority tiers;
- contact-intensity policy;
- recommendation-channel policy;
- experimentation requirements;
- governance status;
- and the final decision-output contract.

No scoring, dataframe construction, model loading, or feature engineering
occurs in this module.
"""

from __future__ import annotations


# ================================================================
# ENGINE METADATA
# ================================================================


DECISION_ENGINE_NAME = (
    "customer_decision_engine"
)

DECISION_ENGINE_VERSION = (
    "1.0.0"
)

DECISION_ENGINE_TYPE = (
    "deterministic_orchestration"
)

DECISION_ENGINE_TARGET_NAME = (
    "customer_primary_action"
)

DECISION_ENGINE_PREDICTION_UNIT = (
    "one identified customer at one decision snapshot"
)

DECISION_ENGINE_FINAL_DECISION = (
    "CONDITIONAL GO"
)

DECISION_ENGINE_APPROVED_USE = (
    "monitored decision-support and controlled commercial pilots"
)

DECISION_ENGINE_AUTONOMOUS_OPTIMIZATION_APPROVED = (
    False
)


# ================================================================
# LIFECYCLE STATE CONTRACT
# ================================================================


ACTIVE_MAX_DAYS_SINCE_LAST_ORDER = 30

AT_RISK_MAX_DAYS_SINCE_LAST_ORDER = 90

DORMANT_MAX_DAYS_SINCE_LAST_ORDER = 180


CUSTOMER_STATE_ACTIVE = (
    "Active"
)

CUSTOMER_STATE_AT_RISK = (
    "At Risk"
)

CUSTOMER_STATE_DORMANT = (
    "Dormant"
)

CUSTOMER_STATE_LONG_DORMANT = (
    "Long Dormant"
)

CUSTOMER_STATE_UNCLASSIFIED = (
    "Unclassified"
)


CUSTOMER_STATES = (
    CUSTOMER_STATE_ACTIVE,
    CUSTOMER_STATE_AT_RISK,
    CUSTOMER_STATE_DORMANT,
    CUSTOMER_STATE_LONG_DORMANT,
    CUSTOMER_STATE_UNCLASSIFIED,
)


# ================================================================
# PRIMARY ACTION CONTRACT
# ================================================================


ACTION_PROTECT = (
    "Protect"
)

ACTION_RETAIN = (
    "Retain"
)

ACTION_REACTIVATE = (
    "Reactivate"
)

ACTION_GROW = (
    "Grow"
)

ACTION_MONITOR = (
    "Monitor"
)

ACTION_MONITOR_NO_REACTIVATION_SCORE = (
    "Monitor — No Reactivation Score"
)

ACTION_UNSCORED_LONG_DORMANT = (
    "Unscored — Long Dormant"
)

ACTION_UNCLASSIFIED = (
    "Unclassified"
)


ACTIONABLE_ACTIONS = (
    ACTION_PROTECT,
    ACTION_RETAIN,
    ACTION_REACTIVATE,
    ACTION_GROW,
)


NON_ACTIONABLE_ACTIONS = (
    ACTION_MONITOR,
    ACTION_MONITOR_NO_REACTIVATION_SCORE,
    ACTION_UNSCORED_LONG_DORMANT,
    ACTION_UNCLASSIFIED,
)


PRIMARY_ACTIONS = (
    *ACTIONABLE_ACTIONS,
    *NON_ACTIONABLE_ACTIONS,
)


# ================================================================
# STATE / ACTION GOVERNANCE
# ================================================================


ALLOWED_ACTIONS_BY_CUSTOMER_STATE = (
    (
        CUSTOMER_STATE_ACTIVE,
        (
            ACTION_PROTECT,
            ACTION_GROW,
            ACTION_MONITOR,
        ),
    ),
    (
        CUSTOMER_STATE_AT_RISK,
        (
            ACTION_RETAIN,
            ACTION_MONITOR,
        ),
    ),
    (
        CUSTOMER_STATE_DORMANT,
        (
            ACTION_REACTIVATE,
            ACTION_MONITOR_NO_REACTIVATION_SCORE,
        ),
    ),
    (
        CUSTOMER_STATE_LONG_DORMANT,
        (
            ACTION_UNSCORED_LONG_DORMANT,
        ),
    ),
    (
        CUSTOMER_STATE_UNCLASSIFIED,
        (
            ACTION_UNCLASSIFIED,
        ),
    ),
)


# ================================================================
# TEMPORAL SIGNAL POLICY
# ================================================================


REACTIVATION_MAX_SIGNAL_AGE_DAYS = 90

REPEAT_PURCHASE_CURRENT_SIGNAL_HORIZON_DAYS = 90


REPEAT_PURCHASE_CURRENT_STATE_ORCHESTRATION_ENABLED = (
    False
)

DEMAND_FORECAST_CUSTOMER_ACTION_ORCHESTRATION_ENABLED = (
    False
)


# ================================================================
# PROTECT POLICY
# ================================================================


HIGH_LEAKAGE_EXPOSURE_PREFIXES = (
    "P1",
    "P2",
)


PROTECT_PRIORITY_WEIGHTS = (
    (
        "leakage_exposure_percentile",
        0.60,
    ),
    (
        "leakage_probability",
        0.25,
    ),
    (
        "fcv_percentile",
        0.15,
    ),
)


# ================================================================
# RETAIN POLICY
# ================================================================


RETAIN_PRIORITY_WEIGHTS = (
    (
        "leakage_exposure_percentile",
        0.50,
    ),
    (
        "leakage_probability",
        0.30,
    ),
    (
        "fcv_percentile",
        0.20,
    ),
)


# ================================================================
# REACTIVATION POLICY
# ================================================================


REACTIVATE_PRIORITY_WEIGHTS = (
    (
        "reactivation_probability",
        0.65,
    ),
    (
        "fcv_percentile",
        0.35,
    ),
)


# ================================================================
# GROW POLICY
# ================================================================


GROW_PRIORITY_WEIGHTS = (
    (
        "fcv_percentile",
        0.55,
    ),
    (
        "future_active_probability",
        0.45,
    ),
)


# ================================================================
# PRIORITY-TIER POLICY
# ================================================================


PRIORITY_1 = (
    "Priority 1"
)

PRIORITY_2 = (
    "Priority 2"
)

PRIORITY_3 = (
    "Priority 3"
)

PRIORITY_4 = (
    "Priority 4"
)

NOT_PRIORITISED = (
    "Not Prioritised"
)


PRIORITY_TIER_THRESHOLDS = (
    (
        PRIORITY_1,
        0.90,
    ),
    (
        PRIORITY_2,
        0.70,
    ),
    (
        PRIORITY_3,
        0.40,
    ),
)


PRIORITY_TIERS = (
    PRIORITY_1,
    PRIORITY_2,
    PRIORITY_3,
    PRIORITY_4,
    NOT_PRIORITISED,
)


# ================================================================
# CONTACT POLICY
# ================================================================


CONTACT_HIGH_TOUCH = (
    "High Touch"
)

CONTACT_CAMPAIGN = (
    "Campaign"
)

CONTACT_AUTOMATED_LOW_COST = (
    "Automated / Low Cost"
)

CONTACT_HOLD_CONTROL = (
    "Hold / Control"
)

CONTACT_NO_MODEL_LED = (
    "No Model-Led Contact"
)


CONTACT_INTENSITY_BY_PRIORITY_TIER = (
    (
        PRIORITY_1,
        CONTACT_HIGH_TOUCH,
    ),
    (
        PRIORITY_2,
        CONTACT_CAMPAIGN,
    ),
    (
        PRIORITY_3,
        CONTACT_AUTOMATED_LOW_COST,
    ),
    (
        PRIORITY_4,
        CONTACT_HOLD_CONTROL,
    ),
)


CONTACT_INTENSITIES = (
    CONTACT_HIGH_TOUCH,
    CONTACT_CAMPAIGN,
    CONTACT_AUTOMATED_LOW_COST,
    CONTACT_HOLD_CONTROL,
    CONTACT_NO_MODEL_LED,
)


# ================================================================
# PRODUCT RECOMMENDATION POLICY
# ================================================================


RECOMMENDATION_CHANNEL_REPEAT = (
    "Repeat"
)

RECOMMENDATION_CHANNEL_DISCOVERY = (
    "Discovery"
)


RECOMMENDATION_CHANNEL_BY_ACTION = (
    (
        ACTION_PROTECT,
        RECOMMENDATION_CHANNEL_REPEAT,
    ),
    (
        ACTION_RETAIN,
        RECOMMENDATION_CHANNEL_REPEAT,
    ),
    (
        ACTION_GROW,
        RECOMMENDATION_CHANNEL_DISCOVERY,
    ),
)


# ================================================================
# EXPERIMENTATION POLICY
# ================================================================


EXPERIMENT_REQUIRED = (
    "Required"
)

EXPERIMENT_REQUIRED_DISCOVERY = (
    "Required — Discovery channel is CONDITIONAL GO"
)

EXPERIMENT_RECOMMENDED = (
    "Recommended"
)

EXPERIMENT_NOT_APPLICABLE = (
    "Not Applicable"
)


EXPERIMENT_REQUIREMENT_BY_ACTION = (
    (
        ACTION_REACTIVATE,
        EXPERIMENT_REQUIRED,
    ),
    (
        ACTION_GROW,
        EXPERIMENT_REQUIRED_DISCOVERY,
    ),
    (
        ACTION_RETAIN,
        EXPERIMENT_RECOMMENDED,
    ),
    (
        ACTION_PROTECT,
        EXPERIMENT_RECOMMENDED,
    ),
)


# ================================================================
# CANONICAL DECISION OUTPUT
# ================================================================


DECISION_OUTPUT_COLUMNS = (
    "customer",
    "decision_date",

    # Lifecycle
    "customer_state",
    "days_since_last_order",
    "customer_tenure_days",
    "lifetime_orders",
    "lifetime_revenue",
    "orders_prior_30d",
    "orders_prior_90d",
    "revenue_prior_90d",
    "revenue_prior_180d",

    # Decision
    "primary_action",
    "primary_action_score",
    "action_priority_percentile",
    "priority_tier",
    "contact_intensity",
    "business_action",
    "decision_reason",
    "experiment_requirement",

    # Leakage
    "leakage_probability",
    "leakage_exposure_score",
    "leakage_exposure_percentile",
    "economic_priority_band",

    # FCV
    "future_active_probability",
    "future_customer_value_90d",
    "fcv_percentile",
    "fcv_tier",

    # Reactivation
    "reactivation_probability",
    "reactivation_prediction_date",

    # Product recommendation
    "recommended_product_channel",
    "recommended_product",
    "repeat_top1_product",
    "discovery_top1_product",
)


# ================================================================
# SPECIFICATION VALIDATION
# ================================================================


def _weight_sum(
    weights: tuple[
        tuple[
            str,
            float,
        ],
        ...,
    ],
) -> float:
    """
    Return the sum of weights in one priority-score policy.
    """

    return sum(
        weight
        for _,
        weight
        in weights
    )


def validate_decision_engine_specification() -> None:
    """
    Fail closed if the frozen decision policy becomes inconsistent.
    """

    # ------------------------------------------------------------
    # Lifecycle thresholds
    # ------------------------------------------------------------

    if not (
        0
        <
        ACTIVE_MAX_DAYS_SINCE_LAST_ORDER
        <
        AT_RISK_MAX_DAYS_SINCE_LAST_ORDER
        <
        DORMANT_MAX_DAYS_SINCE_LAST_ORDER
    ):
        raise ValueError(
            "Lifecycle thresholds must be strictly increasing."
        )

    if len(
        set(
            CUSTOMER_STATES
        )
    ) != len(
        CUSTOMER_STATES
    ):
        raise ValueError(
            "Customer-state labels must be unique."
        )

    # ------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------

    if len(
        set(
            PRIMARY_ACTIONS
        )
    ) != len(
        PRIMARY_ACTIONS
    ):
        raise ValueError(
            "Primary-action labels must be unique."
        )

    if set(
        ACTIONABLE_ACTIONS
    ).intersection(
        NON_ACTIONABLE_ACTIONS
    ):
        raise ValueError(
            "Actionable and non-actionable action sets must not overlap."
        )

    # ------------------------------------------------------------
    # State/action map
    # ------------------------------------------------------------

    mapped_states = tuple(
        state
        for state,
        _
        in ALLOWED_ACTIONS_BY_CUSTOMER_STATE
    )

    if mapped_states != CUSTOMER_STATES:
        raise ValueError(
            "State/action policy must cover canonical customer states "
            "in canonical order."
        )

    for state, actions in (
        ALLOWED_ACTIONS_BY_CUSTOMER_STATE
    ):
        if not actions:
            raise ValueError(
                f"Customer state {state!r} must allow at least one action."
            )

        unknown_actions = (
            set(
                actions
            )
            -
            set(
                PRIMARY_ACTIONS
            )
        )

        if unknown_actions:
            raise ValueError(
                "State/action policy contains unknown actions: "
                f"{sorted(unknown_actions)}"
            )

    # ------------------------------------------------------------
    # Temporal policy
    # ------------------------------------------------------------

    if REACTIVATION_MAX_SIGNAL_AGE_DAYS != 90:
        raise ValueError(
            "Reactivation maximum signal age must remain 90 days."
        )

    if (
        REPEAT_PURCHASE_CURRENT_SIGNAL_HORIZON_DAYS
        !=
        90
    ):
        raise ValueError(
            "Repeat Purchase current-signal horizon must remain 90 days."
        )

    if (
        REPEAT_PURCHASE_CURRENT_STATE_ORCHESTRATION_ENABLED
        is not False
    ):
        raise ValueError(
            "Repeat Purchase must remain excluded from the mature "
            "current-state orchestration contract."
        )

    if (
        DEMAND_FORECAST_CUSTOMER_ACTION_ORCHESTRATION_ENABLED
        is not False
    ):
        raise ValueError(
            "Demand Forecast must not enter customer-level primary-action "
            "assignment."
        )

    # ------------------------------------------------------------
    # Priority weights
    # ------------------------------------------------------------

    priority_policies = (
        PROTECT_PRIORITY_WEIGHTS,
        RETAIN_PRIORITY_WEIGHTS,
        REACTIVATE_PRIORITY_WEIGHTS,
        GROW_PRIORITY_WEIGHTS,
    )

    for weights in priority_policies:
        if not weights:
            raise ValueError(
                "Priority-score policy must contain at least one weight."
            )

        if len(
            {
                field
                for field,
                _
                in weights
            }
        ) != len(
            weights
        ):
            raise ValueError(
                "Priority-score policy contains duplicate fields."
            )

        if any(
            weight < 0
            for _,
            weight
            in weights
        ):
            raise ValueError(
                "Priority-score weights must be non-negative."
            )

        if abs(
            _weight_sum(
                weights
            )
            -
            1.0
        ) > 1e-12:
            raise ValueError(
                "Priority-score weights must sum to 1.0."
            )

    # ------------------------------------------------------------
    # Priority tiers
    # ------------------------------------------------------------

    threshold_values = tuple(
        threshold
        for _,
        threshold
        in PRIORITY_TIER_THRESHOLDS
    )

    if threshold_values != (
        0.90,
        0.70,
        0.40,
    ):
        raise ValueError(
            "Priority-tier thresholds have drifted from Notebook 08."
        )

    if any(
        threshold < 0
        or threshold > 1
        for threshold
        in threshold_values
    ):
        raise ValueError(
            "Priority-tier thresholds must be bounded between 0 and 1."
        )

    # ------------------------------------------------------------
    # Contact policy
    # ------------------------------------------------------------

    contact_tiers = tuple(
        tier
        for tier,
        _
        in CONTACT_INTENSITY_BY_PRIORITY_TIER
    )

    if contact_tiers != (
        PRIORITY_1,
        PRIORITY_2,
        PRIORITY_3,
        PRIORITY_4,
    ):
        raise ValueError(
            "Contact policy must cover Priority 1 through Priority 4 "
            "in canonical order."
        )

    # ------------------------------------------------------------
    # Recommendation policy
    # ------------------------------------------------------------

    recommendation_actions = tuple(
        action
        for action,
        _
        in RECOMMENDATION_CHANNEL_BY_ACTION
    )

    if recommendation_actions != (
        ACTION_PROTECT,
        ACTION_RETAIN,
        ACTION_GROW,
    ):
        raise ValueError(
            "Recommendation-channel action policy has drifted."
        )

    # ------------------------------------------------------------
    # Experimentation policy
    # ------------------------------------------------------------

    experiment_actions = {
        action
        for action,
        _
        in EXPERIMENT_REQUIREMENT_BY_ACTION
    }

    if experiment_actions != set(
        ACTIONABLE_ACTIONS
    ):
        raise ValueError(
            "Every actionable decision must have an experimentation policy."
        )

    # ------------------------------------------------------------
    # Output contract
    # ------------------------------------------------------------

    if len(
        set(
            DECISION_OUTPUT_COLUMNS
        )
    ) != len(
        DECISION_OUTPUT_COLUMNS
    ):
        raise ValueError(
            "Decision output columns must be unique."
        )

    if DECISION_OUTPUT_COLUMNS[
        :2
    ] != (
        "customer",
        "decision_date",
    ):
        raise ValueError(
            "Decision output must begin with customer and decision_date."
        )

    required_output_columns = {
        "customer_state",
        "primary_action",
        "primary_action_score",
        "action_priority_percentile",
        "priority_tier",
        "contact_intensity",
        "business_action",
        "decision_reason",
        "experiment_requirement",
        "leakage_probability",
        "future_active_probability",
        "future_customer_value_90d",
        "reactivation_probability",
        "recommended_product",
    }

    missing_output_columns = (
        required_output_columns
        -
        set(
            DECISION_OUTPUT_COLUMNS
        )
    )

    if missing_output_columns:
        raise ValueError(
            "Decision output is missing required columns: "
            f"{sorted(missing_output_columns)}"
        )

    # ------------------------------------------------------------
    # Governance
    # ------------------------------------------------------------

    if (
        DECISION_ENGINE_FINAL_DECISION
        !=
        "CONDITIONAL GO"
    ):
        raise ValueError(
            "Decision Engine governance decision must remain "
            "CONDITIONAL GO."
        )

    if (
        DECISION_ENGINE_AUTONOMOUS_OPTIMIZATION_APPROVED
        is not False
    ):
        raise ValueError(
            "Decision Engine must not be approved for autonomous "
            "optimisation."
        )


validate_decision_engine_specification()