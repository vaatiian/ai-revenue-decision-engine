"""
Streamlit ML / Model Explorer.

The page presents only curated public registry metadata.
Missing public metrics remain explicitly unavailable.
"""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

from portfolio.components import (
    governance_badge,
    render_claims_note,
    render_section_heading,
)

from portfolio.model_explorer import (
    approved_use_rows,
    architecture_rows,
    governance_label,
    limitation_rows,
    metric_rows,
    prediction_horizon_label,
    registry_summary,
    system_role,
    systems_by_id,
    validate_model_registry,
)


def _display_name(
    value: Any,
) -> str:
    """
    Human-readable field name.
    """

    return (
        str(
            value
        )
        .replace(
            "_",
            " ",
        )
        .replace(
            ".",
            " · ",
        )
        .title()
    )


def _format_metric_value(
    value: Any,
) -> str:
    """
    Format governed metrics without changing their scale.
    """

    if isinstance(
        value,
        float,
    ):
        return f"{value:.6f}".rstrip(
            "0"
        ).rstrip(
            "."
        )

    return str(
        value
    )


def render_metadata_card(
    st_module,
    *,
    label: str,
    value: Any,
) -> None:
    """
    Text-first metadata card.
    """

    display_value = (
        "Not Available"
        if value is None
        else str(
            value
        )
    )

    st_module.markdown(
        (
            '<div class="portfolio-panel" style="min-height:112px;">'
            '<div style="'
            "font-size:0.72rem;"
            "color:#AAB7B4;"
            "margin-bottom:0.6rem;"
            '">'
            f"{escape(str(label))}"
            "</div>"
            '<div style="'
            "font-size:1rem;"
            "font-weight:650;"
            "line-height:1.35;"
            "color:#F4F2EC;"
            "overflow-wrap:anywhere;"
            '">'
            f"{escape(display_value)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_model_explorer(
    st_module,
    bundle,
) -> None:
    """
    Render the complete six-system ML / Model Explorer.
    """

    registry = bundle[
        "model_registry"
    ]

    validate_model_registry(
        registry
    )

    lookup = systems_by_id(
        registry
    )

    summary = registry_summary(
        registry
    )

    render_section_heading(
        st_module,
        kicker="ML / Model Explorer",
        title="Six intelligence systems. One governed production architecture.",
        copy=(
            "Inspect what each system predicts, how it was validated, "
            "which public metrics are available, what governance applies, "
            "and where the system enters the production architecture."
        ),
    )

    summary_cols = st_module.columns(
        4,
        gap="medium",
    )

    summary_cols[
        0
    ].metric(
        "Intelligence systems",
        summary[
            "system_count"
        ],
    )

    summary_cols[
        1
    ].metric(
        "Decision Engine inputs",
        summary[
            "decision_engine_inputs"
        ],
    )

    summary_cols[
        2
    ].metric(
        "Standalone systems",
        summary[
            "standalone_systems"
        ],
    )

    summary_cols[
        3
    ].metric(
        "Systems with public metrics",
        summary[
            "systems_with_public_metrics"
        ],
    )

    render_claims_note(
        st_module,
        (
            "The public registry intentionally does not expose every "
            "research artifact or evaluation population. Missing metrics "
            "remain missing rather than being reconstructed from private "
            "research context."
        ),
    )

    system_options = {
        system[
            "display_name"
        ]:
            system[
                "system_id"
            ]
        for system in registry[
            "systems"
        ]
    }

    selected_display_name = st_module.selectbox(
        "Select intelligence system",
        options=list(
            system_options.keys()
        ),
    )

    selected_id = system_options[
        selected_display_name
    ]

    system = lookup[
        selected_id
    ]

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    header_left, header_right = st_module.columns(
        [
            2,
            1,
        ],
        gap="large",
    )

    with header_left:

        st_module.markdown(
            f"## {system['display_name']}"
        )

        st_module.caption(
            system_role(
                selected_id
            )
        )

        st_module.markdown(
            f"### {system['model_name']}"
        )

    with header_right:

        st_module.markdown(
            "**Governance**"
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
            governance_label(
                system[
                    "governance"
                ]
            )
        )

    metadata_cols = st_module.columns(
        4,
        gap="medium",
    )

    with metadata_cols[
        0
    ]:

        render_metadata_card(
            st_module,
            label="Model type",
            value=system.get(
                "model_type"
            ),
        )

    with metadata_cols[
        1
    ]:

        render_metadata_card(
            st_module,
            label="Algorithm",
            value=(
                system.get(
                    "algorithm"
                )
                or
                "Deterministic architecture"
            ),
        )

    with metadata_cols[
        2
    ]:

        render_metadata_card(
            st_module,
            label="Feature count",
            value=system.get(
                "feature_count"
            ),
        )

    with metadata_cols[
        3
    ]:

        render_metadata_card(
            st_module,
            label="Version",
            value=system.get(
                "model_version"
            ),
        )

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Prediction Contract",
        title="What this intelligence system is designed to estimate",
        copy=(
            "The target and horizon below come directly from the "
            "curated production registry."
        ),
    )

    contract_cols = st_module.columns(
        3,
        gap="medium",
    )

    with contract_cols[
        0
    ]:

        render_metadata_card(
            st_module,
            label="Target",
            value=system.get(
                "target_name"
            ),
        )

    with contract_cols[
        1
    ]:

        render_metadata_card(
            st_module,
            label="Prediction horizon",
            value=prediction_horizon_label(
                system.get(
                    "prediction_horizon"
                )
            ),
        )

    with contract_cols[
        2
    ]:

        render_metadata_card(
            st_module,
            label="Validation",
            value=system.get(
                "validation_method"
            ),
        )

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Evaluation",
        title="Governed public evaluation evidence",
        copy=(
            "Metrics are shown exactly when they exist in the curated "
            "public registry. Their absence is treated as an explicit "
            "evidence boundary."
        ),
    )

    metrics = metric_rows(
        system
    )

    if not metrics:

        st_module.info(
            (
                "No governed evaluation metrics are exposed in the "
                "public registry for this system."
            )
        )

    else:

        metric_frame = pd.DataFrame(
            [
                {
                    "Metric":
                        _display_name(
                            row[
                                "metric"
                            ]
                        ),

                    "Value":
                        _format_metric_value(
                            row[
                                "value"
                            ]
                        ),
                }
                for row in metrics
            ]
        )

        st_module.dataframe(
            metric_frame,
            use_container_width=True,
            hide_index=True,
        )

    architecture = architecture_rows(
        system
    )

    if architecture:

        st_module.markdown(
            "<div style='height:1rem'></div>",
            unsafe_allow_html=True,
        )

        render_section_heading(
            st_module,
            kicker="Architecture",
            title="How the system is constructed",
            copy=(
                "Only architecture metadata explicitly published in the "
                "curated registry is shown."
            ),
        )

        architecture_frame = pd.DataFrame(
            [
                {
                    "Component":
                        row[
                            "component"
                        ],

                    "Value":
                        row[
                            "value"
                        ],
                }
                for row in architecture
            ]
        )

        st_module.dataframe(
            architecture_frame,
            use_container_width=True,
            hide_index=True,
        )

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    usage_left, limitation_right = st_module.columns(
        2,
        gap="large",
    )

    with usage_left:

        render_section_heading(
            st_module,
            kicker="Approved Use",
            title="Where this system may be used",
            copy=(
                "Governance constrains operational use independently "
                "from technical availability."
            ),
        )

        approved = approved_use_rows(
            system
        )

        if approved:

            for item in approved:

                st_module.success(
                    item
                )

        else:

            st_module.info(
                "No approved-use text is exposed in the public registry."
            )

    with limitation_right:

        render_section_heading(
            st_module,
            kicker="Limitations",
            title="What the system should not be assumed to guarantee",
            copy=(
                "These are explicit registry limitations, not newly "
                "invented portfolio warnings."
            ),
        )

        limitations = limitation_rows(
            system
        )

        if limitations:

            for item in limitations:

                st_module.warning(
                    item
                )

        else:

            st_module.info(
                "No explicit limitations are exposed in the public registry."
            )

    st_module.markdown(
        "<div style='height:1rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Production Role",
        title="Where this system enters the architecture",
        copy=(
            "Not every model belongs inside customer-level action "
            "orchestration."
        ),
    )

    if selected_id in {
        "reactivation",
        "revenue_leakage",
        "future_customer_value",
        "recommendation",
    }:

        st_module.success(
            (
                "This system feeds the governed customer Decision Engine."
            )
        )

    elif selected_id == "repeat_purchase":

        st_module.info(
            (
                "Repeat Purchase remains a standalone propensity product. "
                "It is intentionally excluded from mature current-state "
                "customer action orchestration."
            )
        )

    elif selected_id == "demand_forecast":

        st_module.info(
            (
                "Demand Forecast runs as a separate weekly planning job. "
                "It does not enter customer-level primary-action assignment."
            )
        )

    render_claims_note(
        st_module,
        (
            "Model metrics describe historical evaluation evidence. "
            "They are not causal uplift, guaranteed future performance, "
            "or realised commercial impact."
        ),
    )