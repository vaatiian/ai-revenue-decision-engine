
"""
AI Revenue Decision Engine — Portfolio Application.

The application consumes only curated, anonymised, governed portfolio
artifacts.

Pages progressively expose the commercial, modelling, governance,
and production-engineering layers of the system.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from portfolio.charts import (
    categorical_bar_chart,
    risk_value_scatter_chart,
)
from portfolio.command_centre import (
    build_distribution,
    build_priority_queue,
    build_risk_value_view,
    filter_decisions,
    filter_values,
    summarise_command_centre,
)
from portfolio.components import (
    render_architecture_flow,
    render_claims_note,
    render_hero,
    render_proof_cards,
    render_section_heading,
    render_system_card,
)
from portfolio.customer_explorer_page import render_customer_decision_explorer
from portfolio.data_loader import load_portfolio_bundle
from portfolio.decision_demo_page import render_decision_demo
from portfolio.governance_dashboard_page import render_governance_dashboard
from portfolio.model_explorer_page import render_model_explorer
from portfolio.production_architecture_page import render_production_architecture
from portfolio.architecture_visuals_page import render_architecture_visuals
from portfolio.testing_reliability_page import render_testing_reliability
from portfolio.theme import apply_global_styles


APP_TITLE = "AI Revenue Decision Engine"
APP_ICON = "◆"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def get_portfolio_bundle():
    """Cache only curated public portfolio artifacts."""
    return load_portfolio_bundle()


def compact_number(value: float | int) -> str:
    """Render numeric values compactly without implying currency."""
    numeric = float(value)
    magnitude = abs(numeric)

    if magnitude >= 1_000_000:
        return f"{numeric / 1_000_000:,.2f}M"

    if magnitude >= 1_000:
        return f"{numeric / 1_000:,.1f}K"

    if numeric.is_integer():
        return f"{int(numeric):,}"

    return f"{numeric:,.2f}"


def render_sidebar() -> str:
    """
    Render grouped corporate navigation using native Streamlit controls.

    Session state preserves the active page across application reruns.
    All existing page labels are retained for dispatcher compatibility.
    """

    navigation_groups = [
        (
            "COMMAND CENTRE",
            [
                ("Overview", ":material/dashboard:"),
                ("Revenue Command Centre", ":material/monitoring:"),
                ("Customer Decision Explorer", ":material/groups:"),
            ],
        ),
        (
            "AI INTELLIGENCE",
            [
                ("Interactive Decision Demo", ":material/tune:"),
                ("Decision Intelligence", ":material/account_tree:"),
                ("ML / Model Explorer", ":material/neurology:"),
            ],
        ),
        (
            "GOVERNANCE & ENGINEERING",
            [
                ("Governance Dashboard", ":material/verified_user:"),
                ("Production Architecture", ":material/lan:"),
                ("Architecture Visuals", ":material/schema:"),
                ("Testing & Reliability", ":material/science:"),
                ("Production Evidence", ":material/fact_check:"),
            ],
        ),
    ]

    st.sidebar.markdown(
        (
            '<div class="revenue-sidebar-brand">'
            '<div class="revenue-sidebar-symbol">'
            '<svg width="24" height="24" viewBox="0 0 24 24" '
            'fill="none" stroke="currentColor" stroke-width="1.8" '
            'stroke-linecap="round" stroke-linejoin="round" '
            'aria-hidden="true">'
            '<path d="M3 17l6-6 4 4 8-9"/>'
            '<path d="M15 6h6v6"/>'
            '</svg>'
            '</div>'
            '<div class="revenue-sidebar-brand-copy">'
            '<div class="revenue-sidebar-product">REVENUE OS</div>'
            '<div class="revenue-sidebar-subtitle">'
            'AI DECISION INTELLIGENCE'
            '</div>'
            '</div>'
            '</div>'
            '<div class="revenue-sidebar-divider"></div>'
        ),
        unsafe_allow_html=True,
    )

    if "portfolio_active_page" not in st.session_state:
        st.session_state["portfolio_active_page"] = "Overview"

    def navigate_to(destination: str) -> None:
        st.session_state["portfolio_active_page"] = destination

    for group_label, entries in navigation_groups:
        st.sidebar.markdown(
            f'<div class="revenue-nav-heading">{group_label}</div>',
            unsafe_allow_html=True,
        )

        for label, icon in entries:
            active = (
                st.session_state["portfolio_active_page"] == label
            )

            st.sidebar.button(
                label,
                key=f"nav_{label}",
                icon=icon,
                type="primary" if active else "tertiary",
                use_container_width=True,
                on_click=navigate_to,
                args=(label,),
            )

    st.sidebar.markdown(
        (
            '<div class="revenue-sidebar-footer">'
            '<div class="revenue-sidebar-footer-label">'
            '<span class="revenue-sidebar-status-dot"></span>'
            'GOVERNED PORTFOLIO'
            '</div>'
            '<div class="revenue-sidebar-footer-copy">'
            'Curated and anonymised evidence.<br>'
            'No live customer interventions.'
            '</div>'
            '<div class="revenue-sidebar-version">'
            'AI REVENUE DECISION ENGINE'
            '<span>v1.0 BUILD</span>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    return st.session_state["portfolio_active_page"]


def render_horizontal_distribution(
    frame: pd.DataFrame,
    column: str,
    category_label: str,
    *,
    height: int = 300,
) -> None:
    """
    Render a governed horizontal categorical distribution.

    Data aggregation remains unchanged.
    Altair controls semantic colours and presentation.
    """

    distribution = (
        build_distribution(frame, column)
        .rename(
            columns={
                "category": category_label,
                "customers": "Customers",
            }
        )
    )

    chart = categorical_bar_chart(
        distribution,
        source_column=column,
        category_column=category_label,
        count_column="Customers",
        height=height,
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )


def render_overview(bundle) -> None:
    """Executive-first project overview."""

    summary = bundle["summary"]
    decisions = bundle["customer_decisions"]
    model_registry = bundle["model_registry"]

    render_hero(st)

    st.markdown(
        "<div style='height: 1.2rem'></div>",
        unsafe_allow_html=True,
    )

    render_proof_cards(
        st,
        summary["verified_evidence"],
    )

    audience = st.segmented_control(
        "Perspective",
        options=["Executive", "Technical"],
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
        render_architecture_flow(st)

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

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            '<div class="portfolio-panel-title">'
            "Customer state"
            "</div>",
            unsafe_allow_html=True,
        )

        render_horizontal_distribution(
            decisions,
            "customer_state",
            "State",
            height=320,
        )

    with right:
        st.markdown(
            '<div class="portfolio-panel-title">'
            "Primary governed action"
            "</div>",
            unsafe_allow_html=True,
        )

        render_horizontal_distribution(
            decisions,
            "primary_action",
            "Action",
            height=320,
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

    systems = model_registry["systems"]

    for start in range(0, len(systems), 3):
        columns = st.columns(3, gap="medium")
        row = systems[start:start + 3]

        for column, system in zip(columns, row):
            with column:
                render_system_card(st, system)

    render_claims_note(
        st,
        summary["claims_note"],
    )


def render_revenue_command_centre(bundle) -> None:
    """Render the executive Revenue Command Centre."""

    decisions = bundle["customer_decisions"]

    render_section_heading(
        st,
        kicker="Revenue Command Centre",
        title="5,220 customer decisions. One governed operating view.",
        copy=(
            "Filter the verified customer-decision snapshot by state, "
            "action, priority, contact policy, or experimentation status. "
            "All values come from the curated production smoke output."
        ),
    )

    with st.expander("Command Centre Filters", expanded=True):
        row_one = st.columns(3, gap="medium")

        with row_one[0]:
            selected_states = st.multiselect(
                "Customer state",
                options=filter_values(decisions, "customer_state"),
            )

        with row_one[1]:
            selected_actions = st.multiselect(
                "Primary action",
                options=filter_values(decisions, "primary_action"),
            )

        with row_one[2]:
            selected_priorities = st.multiselect(
                "Priority tier",
                options=filter_values(decisions, "priority_tier"),
            )

        row_two = st.columns(3, gap="medium")

        with row_two[0]:
            selected_contacts = st.multiselect(
                "Contact intensity",
                options=filter_values(decisions, "contact_intensity"),
            )

        with row_two[1]:
            selected_experiments = st.multiselect(
                "Experiment requirement",
                options=filter_values(
                    decisions,
                    "experiment_requirement",
                ),
            )

        with row_two[2]:
            customer_query = st.text_input(
                "Customer ID search",
                placeholder="e.g. CUST-00042",
            )

    filtered = filter_decisions(
        decisions,
        states=selected_states,
        actions=selected_actions,
        priorities=selected_priorities,
        contacts=selected_contacts,
        experiments=selected_experiments,
        customer_query=customer_query,
    )

    metrics = summarise_command_centre(filtered)

    st.caption(
        f"Showing {metrics['customer_count']:,} "
        f"of {len(decisions):,} governed customer decisions."
    )

    row_one = st.columns(4, gap="medium")

    row_one[0].metric(
        "Customers in view",
        f"{metrics['customer_count']:,}",
    )

    row_one[1].metric(
        "Prioritised",
        f"{metrics['prioritised_count']:,}",
    )

    row_one[2].metric(
        "Priority 1",
        f"{metrics['priority_1_count']:,}",
    )

    row_one[3].metric(
        "Experiment required",
        f"{metrics['experiment_required_count']:,}",
    )

    row_two = st.columns(4, gap="medium")

    row_two[0].metric(
        "Revenue · prior 90d",
        compact_number(metrics["revenue_prior_90d"]),
        help=(
            "Observed qualifying merchandise revenue in the "
            "90-day history window. Dataset units; not uplift."
        ),
    )

    row_two[1].metric(
        "Modelled FCV90",
        compact_number(metrics["future_customer_value_90d"]),
        help=(
            "Sum of modelled future-customer-value outputs "
            "for customers currently in view."
        ),
    )

    row_two[2].metric(
        "Leakage exposure signal",
        compact_number(metrics["leakage_exposure_score"]),
        help=(
            "Aggregate model-derived leakage exposure score. "
            "This is a prioritisation signal, not realised loss."
        ),
    )

    row_two[3].metric(
        "High-touch contact",
        f"{metrics['high_touch_count']:,}",
    )

    render_claims_note(
        st,
        (
            "Revenue, FCV, and leakage-exposure values describe "
            "observed history or governed model outputs. They are "
            "not measured incremental revenue or realised commercial impact."
        ),
    )

    if filtered.empty:
        st.warning(
            "No governed customer decisions match the current filters."
        )
        return

    st.markdown(
        "<div style='height: 1.5rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st,
        kicker="Decision Mix",
        title="Where attention is concentrated",
        copy="Every chart updates with the active Command Centre filters.",
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Primary action**")

        render_horizontal_distribution(
            filtered,
            "primary_action",
            "Action",
            height=330,
        )

    with right:
        st.markdown("**Customer state**")

        render_horizontal_distribution(
            filtered,
            "customer_state",
            "State",
            height=330,
        )

    priority_col, contact_col = st.columns(2, gap="large")

    with priority_col:
        st.markdown("**Priority tier**")

        render_horizontal_distribution(
            filtered,
            "priority_tier",
            "Priority",
            height=310,
        )

    with contact_col:
        st.markdown("**Contact policy**")

        render_horizontal_distribution(
            filtered,
            "contact_intensity",
            "Contact Policy",
            height=310,
        )

    st.markdown(
        "<div style='height: 1.5rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st,
        kicker="Risk × Value",
        title="Where modelled leakage risk meets future customer value",
        copy=(
            "Only customers with both a governed leakage probability "
            "and modelled FCV90 are included."
        ),
    )

    risk_value = build_risk_value_view(filtered)

    if risk_value.empty:
        st.info(
            "The current filters contain no customers with both "
            "leakage and FCV model outputs."
        )

    else:
        display_risk_value = risk_value.rename(
            columns={
                "portfolio_customer_id": "Customer",
                "customer_state": "Customer State",
                "primary_action": "Action",
                "priority_tier": "Priority",
                "leakage_probability": "Leakage Probability",
                "future_customer_value_90d": "Modelled FCV90",
                "lifetime_revenue": "Lifetime Revenue",
            }
        )

        risk_value_chart = risk_value_scatter_chart(
            display_risk_value,
            height=460,
        )

        st.altair_chart(
            risk_value_chart,
            use_container_width=True,
        )

        st.caption(
            "Bubble size represents observed lifetime revenue. "
            "FCV90 and leakage probability are governed model outputs."
        )

    st.markdown(
        "<div style='height: 1.5rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st,
        kicker="Action Queue",
        title="Highest-priority governed customer decisions",
        copy=(
            "The queue uses the existing Decision Engine priority "
            "percentile. The portfolio layer does not introduce "
            "a second ranking policy."
        ),
    )

    queue_limit = st.segmented_control(
        "Queue size",
        options=[10, 25, 50],
        default=25,
        selection_mode="single",
    )

    queue = build_priority_queue(
        filtered,
        limit=int(queue_limit or 25),
    )

    if queue.empty:
        st.info(
            "No prioritised customers exist in the current filtered view."
        )

    else:
        display_queue = (
            queue[
                [
                    "portfolio_customer_id",
                    "customer_state",
                    "primary_action",
                    "priority_tier",
                    "contact_intensity",
                    "experiment_requirement",
                    "action_priority_percentile",
                    "primary_action_score",
                    "revenue_prior_90d",
                    "future_customer_value_90d",
                ]
            ]
            .rename(
                columns={
                    "portfolio_customer_id": "Customer",
                    "customer_state": "State",
                    "primary_action": "Action",
                    "priority_tier": "Priority",
                    "contact_intensity": "Contact",
                    "experiment_requirement": "Experiment",
                    "action_priority_percentile": "Priority Percentile",
                    "primary_action_score": "Action Score",
                    "revenue_prior_90d": "Revenue 90d",
                    "future_customer_value_90d": "FCV90",
                }
            )
        )

        st.dataframe(
            display_queue,
            use_container_width=True,
            hide_index=True,
            height=520,
            column_config={
                "Priority Percentile": st.column_config.NumberColumn(
                    "Priority %",
                    format="%.1f%%",
                ),
                "Action Score": st.column_config.NumberColumn(
                    "Action Score",
                    format="%.3f",
                ),
                "Revenue 90d": st.column_config.NumberColumn(
                    "Revenue 90d",
                    format="%.2f",
                ),
                "FCV90": st.column_config.NumberColumn(
                    "FCV90",
                    format="%.2f",
                ),
            },
        )

        st.caption(
            "Detailed decision reasoning, model evidence, and "
            "customer-level recommendation context are available "
            "in the Customer Decision Explorer."
        )


def render_decision_intelligence(bundle) -> None:
    """Decision-system architecture shell."""

    decisions = bundle["customer_decisions"]

    render_section_heading(
        st,
        kicker="Decision Intelligence",
        title=(
            "Prediction becomes useful only when policy turns it into action"
        ),
        copy=(
            "This layer aligns customer state, available model scores, "
            "economic context, action precedence, priority, contact "
            "intensity, and experimentation requirements."
        ),
    )

    metrics = st.columns(4)

    actionable = decisions[
        "priority_tier"
    ].ne("Not Prioritised").sum()

    experiment_required = decisions[
        "experiment_requirement"
    ].eq("Required").sum()

    high_touch = decisions[
        "contact_intensity"
    ].eq("High Touch").sum()

    monitored = decisions[
        "primary_action"
    ].eq("Monitor").sum()

    metrics[0].metric(
        "Prioritised customers",
        f"{actionable:,}",
    )

    metrics[1].metric(
        "Experiment required",
        f"{experiment_required:,}",
    )

    metrics[2].metric(
        "High-touch contact",
        f"{high_touch:,}",
    )

    metrics[3].metric(
        "Monitor",
        f"{monitored:,}",
    )

    st.markdown(
        "<div style='height: 1rem'></div>",
        unsafe_allow_html=True,
    )

    render_architecture_flow(st)

    st.info(
        "Interactive production-policy simulation is available in "
        "the Interactive Decision Demo."
    )


def render_production_evidence(bundle) -> None:
    """Show production-readiness evidence without overstating impact."""

    summary = bundle["summary"]
    evidence = summary["verified_evidence"]

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

    render_proof_cards(st, evidence)
    render_architecture_flow(st)

    st.markdown(
        "<div style='height: 1rem'></div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("6 / 6 scoring systems available")

    with col2:
        st.success("Non-root Linux Docker runtime verified")

    with col3:
        st.success(
            "Scheduled jobs fail closed on invalid boundaries"
        )

    render_claims_note(
        st,
        summary["claims_note"],
    )


def main() -> None:
    """Application entry point."""

    # Apply the complete Revenue OS design system on every rerun.
    # The cloud deployment wrapper imports this module only once,
    # but invokes main() whenever Streamlit reruns the script.
    apply_global_styles(st)

    try:
        bundle = get_portfolio_bundle()

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ):
        st.error(
            "The portfolio demonstration is temporarily unavailable "
            "because its curated data could not be validated."
        )

        st.info(
            "No customer interventions have been initiated. "
            "Please contact the repository maintainer if this persists."
        )

        st.stop()

    page = render_sidebar()

    if page == "Overview":
        render_overview(bundle)

    elif page == "Revenue Command Centre":
        render_revenue_command_centre(bundle)

    elif page == "Customer Decision Explorer":
        render_customer_decision_explorer(st, bundle)

    elif page == "Interactive Decision Demo":
        render_decision_demo(st)

    elif page == "Decision Intelligence":
        render_decision_intelligence(bundle)

    elif page == "ML / Model Explorer":
        render_model_explorer(st, bundle)

    elif page == "Governance Dashboard":
        render_governance_dashboard(st, bundle)

    elif page == "Production Architecture":
        render_production_architecture(st, bundle)

    elif page == "Architecture Visuals":
        render_architecture_visuals(st, bundle)

    elif page == "Testing & Reliability":
        render_testing_reliability(st, bundle)

    elif page == "Production Evidence":
        render_production_evidence(bundle)

    st.markdown(
        (
            '<div class="portfolio-footer">'
            "AI Revenue Decision Engine · "
            "curated portfolio demonstration · "
            "governed system evidence, not realised commercial impact"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
