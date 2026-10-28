"""
Streamlit page for the Customer Decision Explorer.

Presentation only.

All decisions displayed here already exist in the governed production
decision output.
"""

from __future__ import annotations

import math
from html import escape
from typing import Any

from portfolio.components import (
    governance_badge,
    render_claims_note,
    render_section_heading,
)

from portfolio.customer_explorer import (
    build_customer_dossier,
    build_model_evidence,
    filter_customer_candidates,
    get_customer_record,
    governance_lookup,
)


def _number(
    value: Any,
    *,
    decimals: int = 2,
) -> str:
    """
    Format one optional number.
    """

    if value is None:
        return "Not Available"

    try:
        numeric = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return "Not Available"

    if math.isnan(
        numeric
    ):
        return "Not Available"

    return f"{numeric:,.{decimals}f}"


def _integer(
    value: Any,
) -> str:
    """
    Format one optional integer-like value.
    """

    if value is None:
        return "Not Available"

    try:
        return f"{int(round(float(value))):,}"

    except (
        TypeError,
        ValueError,
    ):
        return "Not Available"


def _percentage(
    value: Any,
    *,
    decimals: int = 1,
) -> str:
    """
    Format a 0–1 score as a human-readable percentage.
    """

    if value is None:
        return "Not Available"

    try:
        numeric = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return "Not Available"

    if math.isnan(
        numeric
    ):
        return "Not Available"

    return f"{numeric * 100:.{decimals}f}%"


def _progress_value(
    value: Any,
) -> float | None:
    """
    Return a Streamlit-safe 0–1 progress value.
    """

    if value is None:
        return None

    try:
        numeric = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if math.isnan(
        numeric
    ):
        return None

    return min(
        max(
            numeric,
            0.0,
        ),
        1.0,
    )


def render_text_card(
    st_module,
    *,
    label: str,
    value: Any,
    caption: str | None = None,
) -> None:
    """
    Render a text-first card for values that should never be truncated.
    """

    safe_label = escape(
        str(
            label
        )
    )

    display_value = (
        "Not Available"
        if value is None
        else str(
            value
        )
    )

    safe_value = escape(
        display_value
    )

    caption_html = ""

    if caption:

        caption_html = (
            '<div style="'
            "color:#71817D;"
            "font-size:0.72rem;"
            "line-height:1.45;"
            "margin-top:0.55rem;"
            '">'
            f"{escape(str(caption))}"
            "</div>"
        )

    st_module.markdown(
        (
            '<div class="portfolio-panel" style="min-height:120px;">'
            '<div style="'
            "color:#AAB7B4;"
            "font-size:0.74rem;"
            "margin-bottom:0.65rem;"
            '">'
            f"{safe_label}"
            "</div>"
            '<div style="'
            "color:#F4F2EC;"
            "font-size:1.15rem;"
            "font-weight:650;"
            "line-height:1.35;"
            "overflow-wrap:anywhere;"
            '">'
            f"{safe_value}"
            "</div>"
            f"{caption_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_probability_signal(
    st_module,
    *,
    title: str,
    probability: Any,
    supporting_label: str | None = None,
    supporting_value: str | None = None,
) -> None:
    """
    Render one model probability without implying causality.
    """

    st_module.markdown(
        f"**{title}**"
    )

    progress = _progress_value(
        probability
    )

    if progress is None:

        st_module.caption(
            "No governed score available for this customer."
        )

        return

    st_module.progress(
        progress
    )

    st_module.caption(
        _percentage(
            probability
        )
    )

    if (
        supporting_label
        and
        supporting_value
    ):

        st_module.caption(
            f"{supporting_label}: {supporting_value}"
        )


def render_customer_decision_explorer(
    st_module,
    bundle,
) -> None:
    """
    Render the complete single-customer decision dossier.
    """

    decisions = bundle[
        "customer_decisions"
    ]

    registry = bundle[
        "model_registry"
    ]

    render_section_heading(
        st_module,
        kicker="Customer Decision Explorer",
        title="Inspect the evidence behind one governed customer decision",
        copy=(
            "Every dossier is derived from the existing anonymised "
            "customer-decision output. The Explorer explains the decision; "
            "it does not rescore the customer or create a new action."
        ),
    )

    control_row = st_module.columns(
        3,
        gap="medium",
    )

    state_options = [
        "All",
        *sorted(
            decisions[
                "customer_state"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    ]

    action_options = [
        "All",
        *sorted(
            decisions[
                "primary_action"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    ]

    priority_options = [
        "All",
        *sorted(
            decisions[
                "priority_tier"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    ]

    with control_row[
        0
    ]:

        selected_state = st_module.selectbox(
            "State",
            options=state_options,
        )

    with control_row[
        1
    ]:

        selected_action = st_module.selectbox(
            "Action",
            options=action_options,
        )

    with control_row[
        2
    ]:

        selected_priority = st_module.selectbox(
            "Priority",
            options=priority_options,
        )

    candidates = filter_customer_candidates(
        decisions,
        state=selected_state,
        action=selected_action,
        priority=selected_priority,
    )

    if candidates.empty:

        st_module.warning(
            "No customers match the current Explorer filters."
        )

        return

    ids = candidates[
        "portfolio_customer_id"
    ].tolist()

    selected_customer = st_module.selectbox(
        "Customer",
        options=ids,
        help=(
            "Public anonymised portfolio identifier. "
            "Raw customer identifiers are not exposed."
        ),
    )

    record = get_customer_record(
        decisions,
        selected_customer,
    )

    dossier = build_customer_dossier(
        record
    )

    identity = dossier[
        "identity"
    ]

    behaviour = dossier[
        "behaviour"
    ]

    decision = dossier[
        "decision"
    ]

    leakage = dossier[
        "leakage"
    ]

    future_value = dossier[
        "future_value"
    ]

    reactivation = dossier[
        "reactivation"
    ]

    recommendation = dossier[
        "recommendation"
    ]

    st_module.markdown(
        "<div style='height:0.8rem'></div>",
        unsafe_allow_html=True,
    )

    headline_left, headline_right = st_module.columns(
        [
            2,
            1,
        ],
        gap="large",
    )

    with headline_left:

        st_module.markdown(
            f"## {identity['customer_id']}"
        )

        st_module.caption(
            (
                f"Decision date · {identity['decision_date']} "
                f"· Customer state · {identity['state']}"
            )
        )

        st_module.markdown(
            f"### {decision['action'] or 'No governed action'}"
        )

        if decision[
            "business_action"
        ]:

            st_module.write(
                decision[
                    "business_action"
                ]
            )

    with headline_right:

        render_text_card(
            st_module,
            label="Governed priority",
            value=(
                decision[
                    "priority_tier"
                ]
                or
                "Not Prioritised"
            ),
            caption=(
                "Priority percentile · "
                f"{_percentage(decision['priority_percentile'])}"
            ),
        )

    st_module.markdown(
        "---"
    )

    render_section_heading(
        st_module,
        kicker="Observed Behaviour",
        title="What the system knew at the decision point",
        copy=(
            "These fields describe historical customer behaviour available "
            "at the governed decision date."
        ),
    )

    metrics_one = st_module.columns(
        4,
        gap="medium",
    )

    metrics_one[
        0
    ].metric(
        "Days since last order",
        _integer(
            behaviour[
                "days_since_last_order"
            ]
        ),
    )

    metrics_one[
        1
    ].metric(
        "Customer tenure · days",
        _integer(
            behaviour[
                "customer_tenure_days"
            ]
        ),
    )

    metrics_one[
        2
    ].metric(
        "Lifetime orders",
        _integer(
            behaviour[
                "lifetime_orders"
            ]
        ),
    )

    metrics_one[
        3
    ].metric(
        "Lifetime revenue",
        _number(
            behaviour[
                "lifetime_revenue"
            ]
        ),
    )

    metrics_two = st_module.columns(
        4,
        gap="medium",
    )

    metrics_two[
        0
    ].metric(
        "Orders · prior 30d",
        _integer(
            behaviour[
                "orders_prior_30d"
            ]
        ),
    )

    metrics_two[
        1
    ].metric(
        "Orders · prior 90d",
        _integer(
            behaviour[
                "orders_prior_90d"
            ]
        ),
    )

    metrics_two[
        2
    ].metric(
        "Revenue · prior 90d",
        _number(
            behaviour[
                "revenue_prior_90d"
            ]
        ),
    )

    metrics_two[
        3
    ].metric(
        "Revenue · prior 180d",
        _number(
            behaviour[
                "revenue_prior_180d"
            ]
        ),
    )

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Decision",
        title="What the governed policy decided",
        copy=(
            "Model outputs do not independently determine the customer "
            "action. Existing Decision Engine policy aligns available "
            "signals with customer state and governance."
        ),
    )

    decision_columns = st_module.columns(
        4,
        gap="medium",
    )

    with decision_columns[
        0
    ]:

        render_text_card(
            st_module,
            label="Action",
            value=(
                decision[
                    "action"
                ]
                or
                "Not Available"
            ),
        )

    with decision_columns[
        1
    ]:

        render_text_card(
            st_module,
            label="Action score",
            value=_percentage(
                decision[
                    "action_score"
                ]
            ),
        )

    with decision_columns[
        2
    ]:

        render_text_card(
            st_module,
            label="Contact policy",
            value=(
                decision[
                    "contact_intensity"
                ]
                or
                "Not Available"
            ),
        )

    with decision_columns[
        3
    ]:

        render_text_card(
            st_module,
            label="Experiment",
            value=(
                decision[
                    "experiment_requirement"
                ]
                or
                "Not Available"
            ),
        )

    reason_col, action_col = st_module.columns(
        2,
        gap="large",
    )

    with reason_col:

        st_module.markdown(
            "**Why this decision**"
        )

        st_module.info(
            decision[
                "decision_reason"
            ]
            or
            "No decision explanation available."
        )

    with action_col:

        st_module.markdown(
            "**Operational instruction**"
        )

        st_module.success(
            decision[
                "business_action"
            ]
            or
            "No model-led operational instruction."
        )

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Model Evidence",
        title="Which intelligence was available for this customer",
        copy=(
            "A missing score remains explicitly missing. "
            "The Explorer does not manufacture substitute predictions."
        ),
    )

    signal_columns = st_module.columns(
        3,
        gap="large",
    )

    with signal_columns[
        0
    ]:

        render_probability_signal(
            st_module,
            title="Revenue leakage probability",
            probability=leakage[
                "probability"
            ],
            supporting_label="Exposure signal",
            supporting_value=_number(
                leakage[
                    "exposure_score"
                ]
            ),
        )

        if leakage[
            "economic_priority_band"
        ]:

            st_module.caption(
                (
                    "Economic band: "
                    f"{leakage['economic_priority_band']}"
                )
            )

    with signal_columns[
        1
    ]:

        render_probability_signal(
            st_module,
            title="Future active probability",
            probability=future_value[
                "future_active_probability"
            ],
            supporting_label="Modelled FCV90",
            supporting_value=_number(
                future_value[
                    "future_customer_value_90d"
                ]
            ),
        )

        if future_value[
            "fcv_tier"
        ]:

            st_module.caption(
                f"FCV tier: {future_value['fcv_tier']}"
            )

    with signal_columns[
        2
    ]:

        render_probability_signal(
            st_module,
            title="Reactivation probability",
            probability=reactivation[
                "probability"
            ],
            supporting_label="Prediction date",
            supporting_value=(
                reactivation[
                    "prediction_date"
                ]
                or
                "Not Available"
            ),
        )

    evidence = build_model_evidence(
        dossier
    )

    st_module.dataframe(
        evidence,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Intelligence":
                st_module.column_config.TextColumn(
                    "Intelligence",
                    width="medium",
                ),

            "Status":
                st_module.column_config.TextColumn(
                    "Status",
                    width="small",
                ),

            "Primary Signal":
                st_module.column_config.NumberColumn(
                    "Probability",
                    format="%.3f",
                ),

            "Context":
                st_module.column_config.TextColumn(
                    "Context",
                    width="medium",
                ),
        },
    )

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Recommendation",
        title="Product context where a governed recommendation exists",
        copy=(
            "Recommendation information is shown only when it exists "
            "in the production decision output."
        ),
    )

    recommendation_columns = st_module.columns(
        4,
        gap="medium",
    )

    recommendation_cards = [
        (
            "Channel",
            recommendation[
                "channel"
            ]
            or
            "Not Available",
        ),
        (
            "Recommended product",
            recommendation[
                "recommended_product"
            ]
            or
            "Not Available",
        ),
        (
            "Repeat candidate",
            recommendation[
                "repeat_top1_product"
            ]
            or
            "Not Available",
        ),
        (
            "Discovery candidate",
            recommendation[
                "discovery_top1_product"
            ]
            or
            "Not Available",
        ),
    ]

    for column, (
        label,
        value,
    ) in zip(
        recommendation_columns,
        recommendation_cards,
    ):

        with column:

            render_text_card(
                st_module,
                label=label,
                value=value,
            )

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Governance",
        title="The customer decision inherits model governance",
        copy=(
            "Technical availability is not the same as unrestricted "
            "permission to operationalise a model output."
        ),
    )

    governance = governance_lookup(
        registry
    )

    governance_columns = st_module.columns(
        3,
        gap="medium",
    )

    governance_order = [
        (
            "revenue_leakage",
            "Revenue Leakage",
        ),
        (
            "future_customer_value",
            "Future Customer Value",
        ),
        (
            "reactivation",
            "Reactivation",
        ),
    ]

    for column, (
        system_id,
        label,
    ) in zip(
        governance_columns,
        governance_order,
    ):

        system = governance[
            system_id
        ]

        with column:

            st_module.markdown(
                f"**{label}**"
            )

            st_module.markdown(
                governance_badge(
                    system[
                        "governance"
                    ]
                ),
                unsafe_allow_html=True,
            )

            st_module.caption(
                (
                    f"{system['model_type']} · "
                    f"v{system['model_version']}"
                )
            )

    recommendation_governance = governance[
        "recommendation"
    ]

    st_module.markdown(
        "**Recommendation governance**"
    )

    st_module.markdown(
        governance_badge(
            recommendation_governance[
                "governance"
            ]
        ),
        unsafe_allow_html=True,
    )

    render_claims_note(
        st_module,
        (
            "This dossier explains an existing governed decision. "
            "Probabilities are model outputs, not guarantees; modelled "
            "future value is not accounting revenue; and recommendation "
            "signals are not realised conversion."
        ),
    )