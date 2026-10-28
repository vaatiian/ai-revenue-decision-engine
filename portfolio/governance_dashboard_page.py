"""
Streamlit Governance Dashboard.

The page visualises existing public governance evidence.
It does not create or modify governance policy.
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

from portfolio.governance_dashboard import (
    claims_boundary_rows,
    governance_summary,
    governance_surface_rows,
    runtime_availability_label,
    system_governance_detail,
    system_governance_rows,
    validate_governance_sources,
)


def render_boundary_card(
    st_module,
    *,
    title: str,
    evidence: str,
    meaning: str,
) -> None:
    """
    Render one governance boundary card.
    """

    st_module.markdown(
        (
            '<div class="portfolio-panel" style="min-height:190px;">'
            '<div style="'
            "font-size:0.72rem;"
            "color:#52D6FF;"
            "text-transform:uppercase;"
            "letter-spacing:0.08em;"
            "margin-bottom:0.7rem;"
            '">'
            f"{escape(title)}"
            "</div>"
            '<div style="'
            "font-size:1rem;"
            "font-weight:650;"
            "color:#F4F2EC;"
            "line-height:1.45;"
            "margin-bottom:0.75rem;"
            '">'
            f"{escape(evidence)}"
            "</div>"
            '<div style="'
            "font-size:0.78rem;"
            "color:#AAB7B4;"
            "line-height:1.55;"
            '">'
            f"{escape(meaning)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_governance_metric_card(
    st_module,
    *,
    label: str,
    value: str,
    caption: str | None = None,
) -> None:
    """
    Render governance KPIs without Streamlit metric truncation.
    """

    caption_html = ""

    if caption:

        caption_html = (
            '<div style="'
            "font-size:0.68rem;"
            "color:#71817D;"
            "line-height:1.4;"
            "margin-top:0.45rem;"
            '">'
            f"{escape(caption)}"
            "</div>"
        )

    st_module.markdown(
        (
            '<div class="portfolio-panel" '
            'style="min-height:118px;padding:1rem;">'
            '<div style="'
            "font-size:0.69rem;"
            "color:#AAB7B4;"
            "line-height:1.3;"
            "margin-bottom:0.55rem;"
            '">'
            f"{escape(label)}"
            "</div>"
            '<div style="'
            "font-size:1.65rem;"
            "font-weight:500;"
            "color:#F4F2EC;"
            "line-height:1.05;"
            "white-space:normal;"
            "overflow-wrap:anywhere;"
            '">'
            f"{escape(value)}"
            "</div>"
            f"{caption_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _display_system_matrix(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Produce a presentation-safe six-system governance matrix.
    """

    return pd.DataFrame(
        [
            {
                "System":
                    row[
                        "system"
                    ],

                "Runtime":
                    row[
                        "technical_availability"
                    ],

                "Governance":
                    row[
                        "governance"
                    ],

                "Decision Engine":
                    row[
                        "decision_engine_role"
                    ],

                "Validation":
                    row[
                        "validation"
                    ],
            }
            for row in rows
        ]
    )


def _display_permission_matrix(
    rows: list[dict[str, str]],
) -> pd.DataFrame:
    """
    Produce the system/channel permission matrix.
    """

    return pd.DataFrame(
        [
            {
                "System":
                    row[
                        "display_name"
                    ],

                "Surface":
                    row[
                        "governance_surface"
                    ],

                "Decision":
                    row[
                        "governance_decision"
                    ],

                "Approved Use":
                    row[
                        "approved_use"
                    ],
            }
            for row in rows
        ]
    )


def render_governance_dashboard(
    st_module,
    bundle,
) -> None:
    """
    Render governance as an operating-system control surface.
    """

    registry = bundle[
        "model_registry"
    ]

    summary = bundle[
        "summary"
    ]

    validate_governance_sources(
        registry,
        summary,
    )

    governance = governance_summary(
        registry,
        summary,
    )

    render_section_heading(
        st_module,
        kicker="Governance Dashboard",
        title="Available does not mean unrestricted.",
        copy=(
            "Technical runtime availability, governance permission, "
            "Decision Engine inclusion, validation evidence, and commercial "
            "claims are separate control questions. This dashboard exposes "
            "those boundaries from the curated production evidence."
        ),
    )

    metric_cols = st_module.columns(
        5,
        gap="medium",
    )

    with metric_cols[
        0
    ]:

        render_governance_metric_card(
            st_module,
            label="Runtime availability",
            value=runtime_availability_label(
                registry,
                summary,
            ),
        )

    with metric_cols[
        1
    ]:

        render_governance_metric_card(
            st_module,
            label="Governance surfaces",
            value=str(
                governance[
                    "governance_surfaces"
                ]
            ),
        )

    with metric_cols[
        2
    ]:

        render_governance_metric_card(
            st_module,
            label="GO surfaces",
            value=str(
                governance[
                    "go_surfaces"
                ]
            ),
        )

    with metric_cols[
        3
    ]:

        render_governance_metric_card(
            st_module,
            label="Conditional surfaces",
            value=str(
                governance[
                    "conditional_go_surfaces"
                ]
            ),
        )

    with metric_cols[
        4
    ]:

        render_governance_metric_card(
            st_module,
            label="Decision Engine inputs",
            value=(
                f"{governance['decision_engine_systems']}/"
                f"{governance['production_systems']}"
            ),
        )

    render_claims_note(
        st_module,
        summary[
            "claims_note"
        ],
    )

    st_module.markdown(
        "<div style='height:1.2rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Control Boundaries",
        title="Four questions that must not be collapsed into one",
        copy=(
            "The architecture deliberately separates whether a system runs, "
            "whether its use is permitted, whether it participates in "
            "customer action orchestration, and what can be claimed from "
            "the evidence."
        ),
    )

    boundary_rows = claims_boundary_rows(
        registry,
        summary,
    )

    first_row = st_module.columns(
        2,
        gap="medium",
    )

    second_row = st_module.columns(
        2,
        gap="medium",
    )

    for column, boundary in zip(
        first_row + second_row,
        boundary_rows,
    ):

        with column:

            render_boundary_card(
                st_module,
                title=boundary[
                    "boundary"
                ],
                evidence=boundary[
                    "evidence"
                ],
                meaning=boundary[
                    "meaning"
                ],
            )

    st_module.markdown(
        "<div style='height:1.4rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="System Matrix",
        title="Six production intelligence systems under one governance view",
        copy=(
            "Runtime availability is verified independently from governance "
            "permission. Decision Engine inclusion is shown separately."
        ),
    )

    system_rows = system_governance_rows(
        registry,
        summary,
    )

    st_module.dataframe(
        _display_system_matrix(
            system_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

    st_module.caption(
        (
            "Repeat Purchase and Weekly Demand Forecast remain production "
            "scoring systems but are intentionally standalone relative to "
            "customer-level action orchestration."
        )
    )

    st_module.markdown(
        "<div style='height:1.4rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Permission Matrix",
        title="GO and CONDITIONAL GO at the actual governed surface",
        copy=(
            "Next Best Product is intentionally split by channel because "
            "Repeat and Discovery do not share the same governance decision."
        ),
    )

    permission_rows = governance_surface_rows(
        registry
    )

    st_module.dataframe(
        _display_permission_matrix(
            permission_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

    permission_counts = (
        pd.DataFrame(
            permission_rows
        )[
            "governance_decision"
        ]
        .value_counts()
        .rename_axis(
            "Decision"
        )
        .reset_index(
            name="Surfaces"
        )
    )

    st_module.bar_chart(
        permission_counts,
        x="Decision",
        y="Surfaces",
        horizontal=True,
        height=220,
    )

    st_module.markdown(
        "<div style='height:1.4rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Governance Dossier",
        title="Inspect one intelligence system",
        copy=(
            "Approved uses and limitations are shown only when they are "
            "explicitly present in the curated registry."
        ),
    )

    systems = {
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

    selected_name = st_module.selectbox(
        "Select governed system",
        options=list(
            systems.keys()
        ),
        key="governance_dashboard_system",
    )

    selected_id = systems[
        selected_name
    ]

    detail = system_governance_detail(
        registry,
        selected_id,
    )

    detail_left, detail_right = st_module.columns(
        2,
        gap="large",
    )

    with detail_left:

        st_module.markdown(
            "### Governance decision"
        )

        raw_governance = next(
            system[
                "governance"
            ]
            for system in registry[
                "systems"
            ]
            if system[
                "system_id"
            ] == selected_id
        )

        st_module.markdown(
            governance_badge(
                raw_governance
            ),
            unsafe_allow_html=True,
        )

        st_module.write(
            detail[
                "governance"
            ]
        )

        st_module.markdown(
            "### Architecture role"
        )

        st_module.info(
            detail[
                "decision_engine_role"
            ]
        )

        st_module.markdown(
            "### Validation"
        )

        st_module.write(
            detail[
                "validation_method"
            ]
            or
            "Not exposed in public registry"
        )

    with detail_right:

        st_module.markdown(
            "### Approved use"
        )

        approved = detail[
            "approved_use"
        ]

        if approved:

            for item in approved:

                st_module.success(
                    item
                )

        else:

            st_module.info(
                "No approved-use text is exposed in the public registry."
            )

        st_module.markdown(
            "### Explicit limitations"
        )

        limitations = detail[
            "limitations"
        ]

        if limitations:

            for item in limitations:

                st_module.warning(
                    item
                )

        else:

            st_module.info(
                "No explicit limitation text is exposed in the public registry."
            )

    st_module.markdown(
        "<div style='height:1.4rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Evidence Coverage",
        title="What governance evidence is explicitly published",
        copy=(
            "Absence of public text is treated as an evidence boundary, "
            "not permission to infer missing policy."
        ),
    )

    evidence_cols = st_module.columns(
        3,
        gap="medium",
    )

    evidence_cols[
        0
    ].metric(
        "Validation methods",
        (
            f"{governance['systems_with_validation_method']}/"
            f"{governance['production_systems']}"
        ),
    )

    evidence_cols[
        1
    ].metric(
        "Explicit approved use",
        (
            f"{governance['systems_with_explicit_approved_use']}/"
            f"{governance['production_systems']}"
        ),
    )

    evidence_cols[
        2
    ].metric(
        "Explicit limitations",
        (
            f"{governance['systems_with_explicit_limitations']}/"
            f"{governance['production_systems']}"
        ),
    )

    render_claims_note(
        st_module,
        (
            "A blank public field is not interpreted as unrestricted use. "
            "The dashboard preserves missing governance evidence as missing."
        ),
    )