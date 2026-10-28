
"""
Architecture Visuals — Streamlit presentation.

Phase 13M.

Renders governed architecture specifications from the evidence model.
Does not execute production workloads or load model artifacts.
"""

from __future__ import annotations

from html import escape

import pandas as pd

from portfolio.architecture_visuals import (
    architecture_diagram,
    architecture_diagram_ids,
    architecture_visual_summary,
)

from portfolio.components import (
    render_claims_note,
    render_section_heading,
)


EDGE_LABELS = {
    "execution": "Execution",
    "input": "Governed input",
    "conceptual": "Conceptual relationship",
    "external": "External boundary",
}

EDGE_COLORS = {
    "execution": "#52D6FF",
    "input": "#66C69A",
    "conceptual": "#AAB7B4",
    "external": "#D9AC63",
}


def render_visual_metric(
    st_module,
    label: str,
    value: str,
    caption: str,
) -> None:
    """Render a compact, readable architecture proof card."""

    st_module.markdown(
        (
            '<div class="portfolio-panel" '
            'style="min-height:132px;">'
            '<div style="color:#AAB7B4;'
            'font-size:0.72rem;margin-bottom:0.55rem;">'
            f"{escape(label)}"
            "</div>"
            '<div style="color:#F4F2EC;'
            'font-size:1.75rem;font-weight:600;">'
            f"{escape(value)}"
            "</div>"
            '<div style="color:#71817D;'
            'font-size:0.69rem;margin-top:0.5rem;'
            'line-height:1.5;">'
            f"{escape(caption)}"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def render_visual_node(
    st_module,
    item: dict[str, str],
) -> None:
    """Render a governed architecture node."""

    layer = escape(item["layer"])
    label = escape(item["label"])
    description = escape(item["description"])

    st_module.markdown(
        (
            '<div class="portfolio-panel" '
            'style="min-height:116px;'
            'border-left:3px solid #52D6FF;">'
            '<div style="color:#52D6FF;'
            'font-size:0.68rem;'
            'letter-spacing:0.06em;'
            'text-transform:uppercase;">'
            f"{layer}"
            "</div>"
            '<div style="color:#F4F2EC;'
            'font-size:0.96rem;'
            'font-weight:650;'
            'margin-top:0.4rem;">'
            f"{label}"
            "</div>"
            '<div style="color:#AAB7B4;'
            'font-size:0.74rem;'
            'line-height:1.5;'
            'margin-top:0.45rem;">'
            f"{description}"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def render_visual_edge(
    st_module,
    relationship: dict[str, str],
) -> None:
    """Render an explicitly classified architecture relationship."""

    kind = relationship["kind"]
    color = EDGE_COLORS[kind]
    category = EDGE_LABELS[kind]

    st_module.markdown(
        (
            '<div style="text-align:center;'
            'padding:0.25rem 0 0.65rem;">'
            f'<div style="color:{color};font-size:1.2rem;">↓</div>'
            f'<div style="color:{color};font-size:0.64rem;">'
            f"{escape(category)}"
            "</div>"
            '<div style="color:#71817D;font-size:0.67rem;">'
            f"{escape(relationship['label'])}"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def diagram_node_rows(
    diagram: dict,
) -> pd.DataFrame:
    """Build a stable node evidence table."""

    return pd.DataFrame(
        [
            {
                "Node": item["label"],
                "Layer": item["layer"],
                "Purpose": item["description"],
            }
            for item in diagram["nodes"]
        ],
        columns=[
            "Node",
            "Layer",
            "Purpose",
        ],
    )


def diagram_relationship_rows(
    diagram: dict,
) -> pd.DataFrame:
    """Build a stable edge evidence table."""

    labels = {
        item["id"]: item["label"]
        for item in diagram["nodes"]
    }

    return pd.DataFrame(
        [
            {
                "From": labels[edge["source"]],
                "To": labels[edge["target"]],
                "Relationship": EDGE_LABELS[edge["kind"]],
                "Meaning": edge["label"],
            }
            for edge in diagram["edges"]
        ],
        columns=[
            "From",
            "To",
            "Relationship",
            "Meaning",
        ],
    )


def render_customer_execution(
    st_module,
    diagram: dict,
) -> None:
    """Render the verified customer execution sequence."""

    nodes = {
        item["id"]: item
        for item in diagram["nodes"]
    }

    order = [
        "transactions",
        "customer_batch",
        "decision_snapshot",
        "signal_alignment",
        "action_orchestration",
        "priority_policy",
        "canonical_decisions",
    ]

    relationships = {
        (item["source"], item["target"]): item
        for item in diagram["edges"]
    }

    for index, identifier in enumerate(order):
        left, centre, right = st_module.columns(
            [1, 3, 1],
            gap="small",
        )

        with centre:
            render_visual_node(
                st_module,
                nodes[identifier],
            )

            if index < len(order) - 1:
                following = order[index + 1]

                render_visual_edge(
                    st_module,
                    relationships[
                        (identifier, following)
                    ],
                )


def render_intelligence_branches(
    st_module,
    diagram: dict,
) -> None:
    """Render Decision Engine inputs separately from standalone systems."""

    nodes = {
        item["id"]: item
        for item in diagram["nodes"]
    }

    st_module.markdown(
        "#### Governed Customer Decision Inputs"
    )

    inputs = [
        "reactivation",
        "leakage",
        "fcv",
        "recommendation",
    ]

    for start in range(0, len(inputs), 2):
        columns = st_module.columns(
            2,
            gap="medium",
        )

        for column, identifier in zip(
            columns,
            inputs[start:start + 2],
        ):
            with column:
                render_visual_node(
                    st_module,
                    nodes[identifier],
                )

    st_module.markdown(
        '<div style="text-align:center;color:#66C69A;'
        'padding:0.75rem 0;font-size:0.85rem;">'
        "↓ Governed signals · conditional eligibility"
        "</div>",
        unsafe_allow_html=True,
    )

    render_visual_node(
        st_module,
        nodes["decision_engine"],
    )

    st_module.markdown(
        "#### Standalone Production Systems"
    )

    columns = st_module.columns(
        2,
        gap="medium",
    )

    for column, identifier in zip(
        columns,
        ["repeat_purchase", "demand_forecast"],
    ):
        with column:
            render_visual_node(
                st_module,
                nodes[identifier],
            )

    st_module.info(
        "Standalone production systems do not feed "
        "customer-level primary-action orchestration."
    )


def render_runtime_boundaries(
    st_module,
    diagram: dict,
) -> None:
    """Render runtime and public portfolio as distinct areas."""

    nodes = {
        item["id"]: item
        for item in diagram["nodes"]
    }

    st_module.markdown(
        "#### Production Execution Boundary"
    )

    for identifier in [
        "external_scheduler",
        "job_interface",
        "docker_runtime",
    ]:
        render_visual_node(
            st_module,
            nodes[identifier],
        )

    st_module.markdown(
        '<div style="text-align:center;color:#D9AC63;'
        'padding:0.65rem 0;font-size:0.75rem;">'
        "External invocation → governed one-shot execution"
        "</div>",
        unsafe_allow_html=True,
    )

    columns = st_module.columns(
        3,
        gap="medium",
    )

    for column, identifier in zip(
        columns,
        ["customer_job", "repeat_job", "demand_job"],
    ):
        with column:
            render_visual_node(
                st_module,
                nodes[identifier],
            )

    st_module.markdown(
        "#### Separate Public Evidence Boundary"
    )

    render_visual_node(
        st_module,
        nodes["curated_artifacts"],
    )

    st_module.markdown(
        '<div style="text-align:center;color:#66C69A;'
        'padding:0.4rem 0;">'
        "↓ Curated public data only"
        "</div>",
        unsafe_allow_html=True,
    )

    render_visual_node(
        st_module,
        nodes["streamlit"],
    )


def render_architecture_visuals(
    st_module,
    bundle,
) -> None:
    """Render the complete Phase 13M architecture explorer."""

    summary = bundle["summary"]
    evidence = architecture_visual_summary()

    render_section_heading(
        st_module,
        kicker="Architecture Visuals",
        title="Understand how the production system actually works.",
        copy=(
            "Three evidence-backed diagrams distinguish governed "
            "customer decision execution, standalone intelligence, "
            "and runtime/public portfolio boundaries."
        ),
    )

    metrics = [
        (
            "Architecture views",
            evidence["diagrams"],
            "Verified diagram specifications",
        ),
        (
            "Production systems",
            evidence["production_systems"],
            "Six governed intelligence systems",
        ),
        (
            "Decision inputs",
            evidence["decision_inputs"],
            "Four eligible signal families",
        ),
        (
            "Batch jobs",
            evidence["batch_jobs"],
            "Three distinct production contracts",
        ),
    ]

    columns = st_module.columns(
        4,
        gap="medium",
    )

    for column, metric in zip(columns, metrics):
        with column:
            render_visual_metric(
                st_module,
                metric[0],
                str(metric[1]),
                metric[2],
            )

    render_claims_note(
        st_module,
        summary["claims_note"],
    )

    options = {
        "Customer Decision Execution": "customer_execution",
        "Intelligence System Boundaries": "intelligence_branches",
        "Runtime & Public Boundaries": "runtime_boundaries",
    }

    selection = st_module.radio(
        "Architecture view",
        options=list(options),
        horizontal=True,
    )

    diagram_id = options[selection]

    if diagram_id not in architecture_diagram_ids():
        raise RuntimeError(
            "Architecture selection is not registered."
        )

    diagram = architecture_diagram(
        diagram_id
    )

    render_section_heading(
        st_module,
        kicker="Verified Architecture",
        title=diagram["title"],
        copy=diagram["subtitle"],
    )

    if diagram_id == "customer_execution":
        render_customer_execution(
            st_module,
            diagram,
        )

    elif diagram_id == "intelligence_branches":
        render_intelligence_branches(
            st_module,
            diagram,
        )

    elif diagram_id == "runtime_boundaries":
        render_runtime_boundaries(
            st_module,
            diagram,
        )

    for note in diagram["notes"]:
        st_module.caption(note)

    with st_module.expander(
        "Inspect architecture evidence",
        expanded=False,
    ):
        st_module.markdown("**Architecture nodes**")
        st_module.dataframe(
            diagram_node_rows(diagram),
            use_container_width=True,
            hide_index=True,
        )

        st_module.markdown("**Governed relationships**")
        st_module.dataframe(
            diagram_relationship_rows(diagram),
            use_container_width=True,
            hide_index=True,
        )

    render_claims_note(
        st_module,
        (
            "Architecture diagrams distinguish execution calls, "
            "governed inputs, conceptual relationships, and external "
            "boundaries. They do not claim autonomous deployment, "
            "automatic CRM contact, or realised revenue impact."
        ),
    )
