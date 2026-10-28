"""
Streamlit page for the Interactive Decision Demo.

This page exposes controlled, synthetic post-alignment inputs and sends
them through the real production action orchestration and priority policy.

No predictive model binaries are loaded.
No raw customer data are used.
No alternative portfolio decision rules are implemented.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from html import escape
from typing import Any

from portfolio.components import (
    render_claims_note,
    render_section_heading,
)

from portfolio.decision_demo import (
    ECONOMIC_PRIORITY_BANDS,
    DecisionDemoInputs,
    demo_presets,
    run_decision_demo,
)

from src.decision_engine.spec import (
    CUSTOMER_STATES,
)


def _percentage(
    value: Any,
    *,
    decimals: int = 1,
) -> str:
    """
    Format a 0–1 value as a percentage.
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

    return (
        f"{numeric * 100:.{decimals}f}%"
    )


def _score(
    value: Any,
    *,
    decimals: int = 3,
) -> str:
    """
    Format a model or policy score.
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

    return (
        f"{numeric:.{decimals}f}"
    )


def _boolean_label(
    value: bool,
) -> str:
    """
    Human-readable boolean label.
    """

    return (
        "Yes"
        if value
        else
        "No"
    )


def render_text_card(
    st_module,
    *,
    label: str,
    value: Any,
    caption: str | None = None,
) -> None:
    """
    Render long categorical values without Streamlit metric truncation.

    This is presentation logic only.
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


def _eligibility_html(
    eligibility: dict[str, bool],
) -> str:
    """
    Render production eligibility flags compactly.
    """

    labels = (
        (
            "Protect",
            "protect_eligible",
        ),
        (
            "Grow",
            "grow_eligible",
        ),
        (
            "Retain",
            "retain_eligible",
        ),
        (
            "Reactivate",
            "reactivate_eligible",
        ),
    )

    parts = []

    for label, key in labels:

        active = bool(
            eligibility.get(
                key,
                False,
            )
        )

        if active:

            parts.append(
                (
                    '<span class="badge badge-go">'
                    f"{escape(label)} · Eligible"
                    "</span>"
                )
            )

        else:

            parts.append(
                (
                    '<span class="badge badge-neutral">'
                    f"{escape(label)} · No"
                    "</span>"
                )
            )

    return (
        " ".join(
            parts
        )
    )


def _input_from_session(
    st_module,
) -> DecisionDemoInputs:
    """
    Build the frozen demo dataclass from current widget state.
    """

    return DecisionDemoInputs(
        customer_state=(
            st_module.session_state[
                "demo_customer_state"
            ]
        ),

        economic_priority_band=(
            st_module.session_state[
                "demo_economic_band"
            ]
        ),

        leakage_operationally_valid=bool(
            st_module.session_state[
                "demo_leakage_valid"
            ]
        ),

        leakage_exposure_percentile=float(
            st_module.session_state[
                "demo_leakage_exposure_pct"
            ]
        ),

        leakage_probability=float(
            st_module.session_state[
                "demo_leakage_probability"
            ]
        ),

        fcv_signal_available=bool(
            st_module.session_state[
                "demo_fcv_available"
            ]
        ),

        fcv_percentile=float(
            st_module.session_state[
                "demo_fcv_percentile"
            ]
        ),

        future_active_probability=float(
            st_module.session_state[
                "demo_future_active_probability"
            ]
        ),

        reactivation_operationally_valid=bool(
            st_module.session_state[
                "demo_reactivation_valid"
            ]
        ),

        reactivation_probability=float(
            st_module.session_state[
                "demo_reactivation_probability"
            ]
        ),

        recommendation_operationally_valid=bool(
            st_module.session_state[
                "demo_recommendation_valid"
            ]
        ),

        repeat_product=(
            st_module.session_state[
                "demo_repeat_product"
            ]
        ),

        discovery_product=(
            st_module.session_state[
                "demo_discovery_product"
            ]
        ),
    )


def _load_preset_into_session(
    st_module,
    preset_name: str,
) -> None:
    """
    Load one input-only scenario into widget session state.
    """

    presets = demo_presets()

    preset = presets[
        preset_name
    ]

    values = asdict(
        preset
    )

    mapping = {
        "customer_state":
            "demo_customer_state",

        "economic_priority_band":
            "demo_economic_band",

        "leakage_operationally_valid":
            "demo_leakage_valid",

        "leakage_exposure_percentile":
            "demo_leakage_exposure_pct",

        "leakage_probability":
            "demo_leakage_probability",

        "fcv_signal_available":
            "demo_fcv_available",

        "fcv_percentile":
            "demo_fcv_percentile",

        "future_active_probability":
            "demo_future_active_probability",

        "reactivation_operationally_valid":
            "demo_reactivation_valid",

        "reactivation_probability":
            "demo_reactivation_probability",

        "recommendation_operationally_valid":
            "demo_recommendation_valid",

        "repeat_product":
            "demo_repeat_product",

        "discovery_product":
            "demo_discovery_product",
    }

    for field, session_key in mapping.items():

        st_module.session_state[
            session_key
        ] = values[
            field
        ]


def _initialise_session(
    st_module,
) -> None:
    """
    Seed widget state exactly once.
    """

    if (
        "demo_customer_state"
        in
        st_module.session_state
    ):
        return

    _load_preset_into_session(
        st_module,
        "Active · Protect",
    )


def render_decision_demo(
    st_module,
) -> None:
    """
    Render the production-policy interactive simulator.
    """

    _initialise_session(
        st_module
    )

    render_section_heading(
        st_module,
        kicker="Interactive Decision Demo",
        title="Change the evidence. Watch the real governed policy respond.",
        copy=(
            "This simulator operates on synthetic post-alignment inputs. "
            "The output is produced by the same production action "
            "orchestration and priority-policy functions used by the "
            "Customer Decision Engine."
        ),
    )

    st_module.info(
        (
            "This is a policy simulation, not a customer prediction. "
            "No raw customer data or model binaries are loaded."
        )
    )

    preset_names = list(
        demo_presets().keys()
    )

    preset_row = st_module.columns(
        [
            2,
            1,
        ],
        gap="medium",
    )

    with preset_row[
        0
    ]:

        preset_name = st_module.selectbox(
            "Scenario preset",
            options=preset_names,
            key="demo_preset_name",
        )

    with preset_row[
        1
    ]:

        st_module.markdown(
            "<div style='height:1.75rem'></div>",
            unsafe_allow_html=True,
        )

        if st_module.button(
            "Load preset",
            use_container_width=True,
        ):

            _load_preset_into_session(
                st_module,
                preset_name,
            )

            st_module.rerun()

    st_module.markdown(
        "<div style='height:0.5rem'></div>",
        unsafe_allow_html=True,
    )

    with st_module.expander(
        "1 · Lifecycle & signal availability",
        expanded=True,
    ):

        row_one = st_module.columns(
            3,
            gap="medium",
        )

        with row_one[
            0
        ]:

            st_module.selectbox(
                "Customer state",
                options=list(
                    CUSTOMER_STATES
                ),
                key="demo_customer_state",
            )

        with row_one[
            1
        ]:

            st_module.selectbox(
                "Economic leakage band",
                options=list(
                    ECONOMIC_PRIORITY_BANDS
                ),
                key="demo_economic_band",
            )

        with row_one[
            2
        ]:

            st_module.toggle(
                "Valid leakage signal",
                key="demo_leakage_valid",
            )

        row_two = st_module.columns(
            3,
            gap="medium",
        )

        with row_two[
            0
        ]:

            st_module.toggle(
                "FCV signal available",
                key="demo_fcv_available",
            )

        with row_two[
            1
        ]:

            st_module.toggle(
                "Valid reactivation signal",
                key="demo_reactivation_valid",
            )

        with row_two[
            2
        ]:

            st_module.toggle(
                "Valid recommendation",
                key="demo_recommendation_valid",
            )

    with st_module.expander(
        "2 · Governed model signals",
        expanded=True,
    ):

        leakage_cols = st_module.columns(
            2,
            gap="medium",
        )

        with leakage_cols[
            0
        ]:

            st_module.slider(
                "Leakage probability",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="demo_leakage_probability",
                disabled=(
                    not
                    st_module.session_state[
                        "demo_leakage_valid"
                    ]
                ),
            )

        with leakage_cols[
            1
        ]:

            st_module.slider(
                "Leakage exposure percentile",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="demo_leakage_exposure_pct",
                disabled=(
                    not
                    st_module.session_state[
                        "demo_leakage_valid"
                    ]
                ),
            )

        fcv_cols = st_module.columns(
            2,
            gap="medium",
        )

        with fcv_cols[
            0
        ]:

            st_module.slider(
                "FCV percentile",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="demo_fcv_percentile",
                disabled=(
                    not
                    st_module.session_state[
                        "demo_fcv_available"
                    ]
                ),
            )

        with fcv_cols[
            1
        ]:

            st_module.slider(
                "Future active probability",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="demo_future_active_probability",
                disabled=(
                    not
                    st_module.session_state[
                        "demo_fcv_available"
                    ]
                ),
            )

        st_module.slider(
            "Reactivation probability",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="demo_reactivation_probability",
            disabled=(
                not
                st_module.session_state[
                    "demo_reactivation_valid"
                ]
            ),
        )

    with st_module.expander(
        "3 · Recommendation context",
        expanded=False,
    ):

        product_cols = st_module.columns(
            2,
            gap="medium",
        )

        with product_cols[
            0
        ]:

            st_module.text_input(
                "Repeat product",
                key="demo_repeat_product",
                disabled=(
                    not
                    st_module.session_state[
                        "demo_recommendation_valid"
                    ]
                ),
            )

        with product_cols[
            1
        ]:

            st_module.text_input(
                "Discovery product",
                key="demo_discovery_product",
                disabled=(
                    not
                    st_module.session_state[
                        "demo_recommendation_valid"
                    ]
                ),
            )

    inputs = _input_from_session(
        st_module
    )

    result = run_decision_demo(
        inputs
    )

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Governed Output",
        title="The production policy result",
        copy=(
            "The portfolio page displays the result; it does not "
            "calculate the business action itself."
        ),
    )

    outcome_cols = st_module.columns(
        4,
        gap="medium",
    )

    with outcome_cols[
        0
    ]:

        render_text_card(
            st_module,
            label="Primary action",
            value=result[
                "primary_action"
            ],
        )

    with outcome_cols[
        1
    ]:

        render_text_card(
            st_module,
            label="Priority tier",
            value=result[
                "priority_tier"
            ],
            caption=(
                "Action-relative percentile · "
                f"{_percentage(result['action_priority_percentile'])}"
            ),
        )

    with outcome_cols[
        2
    ]:

        render_text_card(
            st_module,
            label="Contact policy",
            value=result[
                "contact_intensity"
            ],
        )

    with outcome_cols[
        3
    ]:

        render_text_card(
            st_module,
            label="Experiment",
            value=result[
                "experiment_requirement"
            ],
        )

    score_cols = st_module.columns(
        3,
        gap="medium",
    )

    score_cols[
        0
    ].metric(
        "Action score",
        _score(
            result[
                "primary_action_score"
            ]
        ),
    )

    score_cols[
        1
    ].metric(
        "Priority percentile",
        _percentage(
            result[
                "action_priority_percentile"
            ]
        ),
    )

    score_cols[
        2
    ].metric(
        "Reference cohort",
        (
            f"{result['reference_cohort_size']}"
            " rows"
        ),
        help=(
            "Priority percentile is action-relative in production. "
            "The public simulator uses a deterministic synthetic "
            "reference cohort so the percentile is meaningful."
        ),
    )

    st_module.markdown(
        "**Eligibility gates evaluated by production orchestration**"
    )

    st_module.markdown(
        _eligibility_html(
            result[
                "eligibility"
            ]
        ),
        unsafe_allow_html=True,
    )

    if result[
        "eligibility"
    ][
        "high_leakage_exposure"
    ]:

        st_module.caption(
            "The production policy classified the selected leakage band "
            "as high economic exposure."
        )

    st_module.markdown(
        "<div style='height:0.8rem'></div>",
        unsafe_allow_html=True,
    )

    explanation_cols = st_module.columns(
        2,
        gap="large",
    )

    with explanation_cols[
        0
    ]:

        st_module.markdown(
            "**Why this action**"
        )

        st_module.info(
            result[
                "decision_reason"
            ]
        )

    with explanation_cols[
        1
    ]:

        st_module.markdown(
            "**Commercial instruction**"
        )

        st_module.success(
            result[
                "business_action"
            ]
        )

    recommendation_cols = st_module.columns(
        2,
        gap="medium",
    )

    with recommendation_cols[
        0
    ]:

        render_text_card(
            st_module,
            label="Recommendation channel",
            value=(
                result[
                    "recommended_product_channel"
                ]
                or
                "Not Available"
            ),
        )

    with recommendation_cols[
        1
    ]:

        render_text_card(
            st_module,
            label="Recommended product",
            value=(
                result[
                    "recommended_product"
                ]
                or
                "Not Available"
            ),
        )

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Policy Trace",
        title="What changes the result",
        copy=(
            "The simulator exposes the same lifecycle gates and signal "
            "availability used by the production orchestration layer."
        ),
    )

    trace = [
        {
            "Policy input":
                "Customer state",

            "Current value":
                inputs.customer_state,
        },
        {
            "Policy input":
                "Economic leakage band",

            "Current value":
                inputs.economic_priority_band,
        },
        {
            "Policy input":
                "Leakage operationally valid",

            "Current value":
                _boolean_label(
                    inputs.leakage_operationally_valid
                ),
        },
        {
            "Policy input":
                "FCV signal available",

            "Current value":
                _boolean_label(
                    inputs.fcv_signal_available
                ),
        },
        {
            "Policy input":
                "Reactivation operationally valid",

            "Current value":
                _boolean_label(
                    inputs.reactivation_operationally_valid
                ),
        },
        {
            "Policy input":
                "Recommendation operationally valid",

            "Current value":
                _boolean_label(
                    inputs.recommendation_operationally_valid
                ),
        },
    ]

    st_module.dataframe(
        trace,
        use_container_width=True,
        hide_index=True,
    )

    with st_module.expander(
        "Technical implementation",
        expanded=False,
    ):

        st_module.code(
            result[
                "policy_source"
            ],
            language="text",
        )

        st_module.markdown(
            (
                "The demo supplies synthetic, post-alignment inputs. "
                "Action assignment is performed by the production "
                "`build_action_orchestration()` function. Priority score, "
                "action-relative percentile, tier, contact policy, "
                "recommendation selection, decision reason, business action, "
                "and experimentation requirement are produced by the "
                "production `apply_priority_policy()` function."
            )
        )

    render_claims_note(
        st_module,
        (
            "This page demonstrates deterministic policy behaviour under "
            "synthetic inputs. It is not a prediction for a real customer, "
            "not realised commercial impact, and not an autonomous "
            "optimisation interface."
        ),
    )