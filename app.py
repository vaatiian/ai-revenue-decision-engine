"""
AI Revenue Decision Engine — Portfolio Application.

Phase 13E establishes the visual system and portfolio shell.

The application consumes only curated, anonymised, governed portfolio
artifacts. Later Phase 13 subphases expand this shell into the full
Revenue Command Centre, Customer Decision Explorer, interactive
Decision Engine, Model Explorer, Governance, Production, Testing,
Architecture, and Case Study experiences.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from portfolio.components import (
    render_architecture_flow,
    render_claims_note,
    render_hero,
    render_proof_cards,
    render_section_heading,
    render_system_card,
)

from portfolio.data_loader import (
    load_portfolio_bundle,
)

from portfolio.theme import (
    apply_global_styles,
)


APP_TITLE = (
    "AI Revenue Decision Engine"
)

APP_ICON = "◆"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


apply_global_styles(
    st
)


@st.cache_data(
    show_spinner=False
)
def get_portfolio_bundle():
    """
    Cache only curated public portfolio artifacts.
    """

    return load_portfolio_bundle()


def render_sidebar() -> str:
    """
    Render navigation and portfolio context.

    Phase 13E uses a compact four-view shell.
    The full navigation system arrives in subsequent subphases.
    """

    st.sidebar.markdown(
        """
        <div class="portfolio-eyebrow">
            Revenue Intelligence
        </div>

        <h2 style="
            margin-top: 0;
            margin-bottom: 0.35rem;
        ">
            Decision Engine
        </h2>

        <div style="
            color: #71817D;
            font-size: 0.75rem;
            margin-bottom: 1.4rem;
        ">
            Portfolio · v1.0 build
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Explore",
        options=[
            "Overview",
            "Decision Intelligence",
            "Model Governance",
            "Production Evidence",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        "---"
    )

    st.sidebar.markdown(
        """
        <div style="
            color: #71817D;
            font-size: 0.72rem;
            line-height: 1.55;
        ">
            Curated public portfolio data only.<br>
            Raw transactions and operational audit state
            are not loaded by this application.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return page


def render_action_chart(
    decisions: pd.DataFrame,
) -> None:
    """
    Render governed action distribution.
    """

    counts = (
        decisions[
            "primary_action"
        ]
        .fillna(
            "Not Available"
        )
        .value_counts()
        .rename_axis(
            "Action"
        )
        .rename(
            "Customers"
        )
        .reset_index()
    )

    st.bar_chart(
        counts,
        x="Action",
        y="Customers",
        horizontal=True,
        height=330,
    )


def render_state_chart(
    decisions: pd.DataFrame,
) -> None:
    """
    Render customer-state distribution.
    """

    counts = (
        decisions[
            "customer_state"
        ]
        .fillna(
            "Not Available"
        )
        .value_counts()
        .rename_axis(
            "State"
        )
        .rename(
            "Customers"
        )
        .reset_index()
    )

    st.bar_chart(
        counts,
        x="State",
        y="Customers",
        height=330,
    )


def render_overview(
    bundle,
) -> None:
    """
    Executive-first project overview.
    """

    summary = bundle[
        "summary"
    ]

    decisions = bundle[
        "customer_decisions"
    ]

    model_registry = bundle[
        "model_registry"
    ]

    render_hero(
        st
    )

    st.markdown(
        "<div style='height: 1.2rem'></div>",
        unsafe_allow_html=True,
    )

    render_proof_cards(
        st,
        summary[
            "verified_evidence"
        ],
    )

    audience = st.segmented_control(
        "Perspective",
        options=[
            "Executive",
            "Technical",
        ],
        default="Executive",
        selection_mode="single",
    )

    st.markdown(
        "<div style='height: 1.3rem'></div>",
        unsafe_allow_html=True,
    )

    if audience == "Technical":

        render_section_heading(
            st,
            kicker="System View",
            title="A decision system, not six disconnected models",
            copy=(
                "The architecture separates predictive intelligence, "
                "governed policy, delivery, experimentation, monitoring, "
                "and scheduled execution. Temporal validation and "
                "production feature parity are explicit design constraints."
            ),
        )

        render_architecture_flow(
            st
        )

    else:

        render_section_heading(
            st,
            kicker="Commercial View",
            title="The system answers what the business should do next",
            copy=(
                "Historical customer behaviour is converted into governed "
                "actions — Protect, Retain, Reactivate, Grow, Monitor, "
                "or explicit non-actionable states — then mapped to "
                "priority, contact policy, and experimentation requirements."
            ),
        )

    st.markdown(
        "<div style='height: 2.2rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st,
        kicker="Decision Snapshot",
        title="5,220 governed customer outcomes",
        copy=(
            "The portfolio snapshot comes from the verified Docker "
            "customer-batch execution dated 31 August 2011."
        ),
    )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        st.markdown(
            '<div class="portfolio-panel-title">'
            "Customer state"
            "</div>",
            unsafe_allow_html=True,
        )

        render_state_chart(
            decisions
        )

    with right:
        st.markdown(
            '<div class="portfolio-panel-title">'
            "Primary governed action"
            "</div>",
            unsafe_allow_html=True,
        )

        render_action_chart(
            decisions
        )

    st.markdown(
        "<div style='height: 2.2rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st,
        kicker="Intelligence Layer",
        title="Six governed production intelligence systems",
        copy=(
            "Technical availability and permission to operationalise "
            "are deliberately separate. GO and CONDITIONAL GO remain "
            "visible throughout the portfolio."
        ),
    )

    systems = model_registry[
        "systems"
    ]

    for start in range(
        0,
        len(
            systems
        ),
        3,
    ):
        columns = st.columns(
            3,
            gap="medium",
        )

        row = systems[
            start:
            start + 3
        ]

        for column, system in zip(
            columns,
            row,
        ):
            with column:
                render_system_card(
                    st,
                    system,
                )

    render_claims_note(
        st,
        summary[
            "claims_note"
        ],
    )


def render_decision_intelligence(
    bundle,
) -> None:
    """
    Early decision-intelligence shell.

    Full interactive Customer Decision Explorer arrives in Phase 13G.
    """

    decisions = bundle[
        "customer_decisions"
    ]

    render_section_heading(
        st,
        kicker="Decision Intelligence",
        title="Prediction becomes useful only when policy turns it into action",
        copy=(
            "This layer aligns customer state, available model scores, "
            "economic context, action precedence, priority, contact "
            "intensity, and experimentation requirements."
        ),
    )

    metrics = st.columns(
        4
    )

    actionable = decisions[
        "priority_tier"
    ].ne(
        "Not Prioritised"
    ).sum()

    experiment_required = decisions[
        "experiment_requirement"
    ].eq(
        "Required"
    ).sum()

    high_touch = decisions[
        "contact_intensity"
    ].eq(
        "High Touch"
    ).sum()

    monitored = decisions[
        "primary_action"
    ].eq(
        "Monitor"
    ).sum()

    metrics[
        0
    ].metric(
        "Prioritised customers",
        f"{actionable:,}",
    )

    metrics[
        1
    ].metric(
        "Experiment required",
        f"{experiment_required:,}",
    )

    metrics[
        2
    ].metric(
        "High-touch contact",
        f"{high_touch:,}",
    )

    metrics[
        3
    ].metric(
        "Monitor",
        f"{monitored:,}",
    )

    st.markdown(
        "<div style='height: 1rem'></div>",
        unsafe_allow_html=True,
    )

    render_architecture_flow(
        st
    )

    st.info(
        "The full filterable Customer Decision Explorer and "
        "single-customer dossier are implemented in Phase 13G."
    )


def render_model_governance(
    bundle,
) -> None:
    """
    Governance-first model register.
    """

    registry = bundle[
        "model_registry"
    ]

    render_section_heading(
        st,
        kicker="Model Governance",
        title="Available does not mean unrestricted",
        copy=(
            "Every intelligence system keeps its governed research "
            "decision visible. Conditional systems remain constrained "
            "to their approved use."
        ),
    )

    for system in registry[
        "systems"
    ]:

        with st.expander(
            system[
                "display_name"
            ],
            expanded=False,
        ):

            left, right = st.columns(
                [
                    1,
                    1,
                ]
            )

            with left:

                st.markdown(
                    "**Model**"
                )

                st.write(
                    system.get(
                        "model_name"
                    )
                    or
                    "Deterministic architecture"
                )

                st.markdown(
                    "**Model type**"
                )

                st.write(
                    system.get(
                        "model_type"
                    )
                )

                st.markdown(
                    "**Target**"
                )

                st.write(
                    system.get(
                        "target_name"
                    )
                )

            with right:

                st.markdown(
                    "**Governance**"
                )

                st.write(
                    system.get(
                        "governance"
                    )
                )

                st.markdown(
                    "**Validation**"
                )

                st.write(
                    system.get(
                        "validation_method"
                    )
                    or
                    "See governed manifest"
                )

                st.markdown(
                    "**Prediction horizon**"
                )

                st.write(
                    system.get(
                        "prediction_horizon"
                    )
                )

            metrics = system.get(
                "metrics"
            )

            if metrics:

                st.markdown(
                    "**Governed evaluation metrics**"
                )

                st.json(
                    metrics,
                    expanded=False,
                )


def render_production_evidence(
    bundle,
) -> None:
    """
    Show production-readiness evidence without overstating impact.
    """

    summary = bundle[
        "summary"
    ]

    evidence = summary[
        "verified_evidence"
    ]

    render_section_heading(
        st,
        kicker="Production Evidence",
        title="The system was exercised beyond the notebooks",
        copy=(
            "Scoring, decisions, Docker execution, scheduling, "
            "delivery controls, monitoring, and regression testing "
            "were verified as part of the production engineering path."
        ),
    )

    render_proof_cards(
        st,
        evidence,
    )

    render_architecture_flow(
        st
    )

    st.markdown(
        "<div style='height: 1rem'></div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:
        st.success(
            "6 / 6 scoring systems available"
        )

    with col2:
        st.success(
            "Non-root Linux Docker runtime verified"
        )

    with col3:
        st.success(
            "Scheduled jobs fail closed on invalid boundaries"
        )

    render_claims_note(
        st,
        summary[
            "claims_note"
        ],
    )


def main() -> None:
    """
    Application entry point.
    """

    try:
        bundle = get_portfolio_bundle()

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:

        st.error(
            "The curated portfolio data layer failed validation."
        )

        st.code(
            str(
                exc
            )
        )

        st.stop()

    page = render_sidebar()

    if page == "Overview":
        render_overview(
            bundle
        )

    elif page == "Decision Intelligence":
        render_decision_intelligence(
            bundle
        )

    elif page == "Model Governance":
        render_model_governance(
            bundle
        )

    elif page == "Production Evidence":
        render_production_evidence(
            bundle
        )

    st.markdown(
        """
        <div class="portfolio-footer">
            AI Revenue Decision Engine ·
            curated portfolio demonstration ·
            governed system evidence, not realised commercial impact
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()