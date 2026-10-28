"""
Reusable Streamlit presentation components.

These helpers contain presentation logic only.
They do not make or modify governed customer decisions.
"""

from __future__ import annotations

from html import escape
from typing import Any


def _safe(
    value: Any,
) -> str:
    """
    HTML-escape display values.
    """

    if value is None:
        return "Not Available"

    return escape(
        str(
            value
        )
    )


def governance_badge(
    governance: Any,
) -> str:
    """
    Render one governance status as a small HTML badge.
    """

    if isinstance(
        governance,
        dict,
    ):
        repeat = governance.get(
            "repeat_channel",
            "Not Available",
        )

        discovery = governance.get(
            "discovery_channel",
            "Not Available",
        )

        return (
            '<span class="badge badge-go">'
            f"Repeat · {_safe(repeat)}"
            "</span>"
            "&nbsp;"
            '<span class="badge badge-conditional">'
            f"Discovery · {_safe(discovery)}"
            "</span>"
        )

    label = str(
        governance
        or
        "Not Available"
    )

    normalized = (
        label
        .upper()
        .strip()
    )

    if normalized == "GO":
        css_class = "badge-go"

    elif normalized == "CONDITIONAL GO":
        css_class = "badge-conditional"

    else:
        css_class = "badge-neutral"

    return (
        f'<span class="badge {css_class}">'
        f"{_safe(label)}"
        "</span>"
    )


def render_hero(
    st_module,
) -> None:
    """
    Render the portfolio hero.

    HTML is deliberately constructed without leading indentation because
    Markdown treats four leading spaces as a code block.
    """

    hero_html = (
        '<div class="portfolio-hero">'
        '<div class="portfolio-eyebrow">'
        "End-to-End Machine Learning · Decision Intelligence"
        "</div>"
        "<h1>"
        "AI Revenue<br>"
        "Decision Engine"
        "</h1>"
        '<div class="portfolio-hero-copy">'
        '<span class="portfolio-hero-tagline">'
        "From transactions to governed revenue decisions."
        "</span>"
        "<br><br>"
        "A production-oriented retail intelligence system that "
        "connects predictive modelling, recommendations, "
        "demand forecasting, governed decision policy, "
        "CRM delivery, monitoring, automated testing, "
        "and scheduled Docker execution."
        "</div>"
        "</div>"
    )

    st_module.markdown(
        hero_html,
        unsafe_allow_html=True,
    )


def render_section_heading(
    st_module,
    kicker: str,
    title: str,
    copy: str | None = None,
) -> None:
    """
    Render one portfolio section heading.
    """

    body = ""

    if copy:
        body = (
            '<div class="portfolio-section-copy">'
            f"{_safe(copy)}"
            "</div>"
        )

    st_module.markdown(
        (
            '<div class="portfolio-section-kicker">'
            f"{_safe(kicker)}"
            "</div>"
            '<div class="portfolio-section-title">'
            f"{_safe(title)}"
            "</div>"
            f"{body}"
        ),
        unsafe_allow_html=True,
    )


def render_proof_cards(
    st_module,
    evidence: dict[str, Any],
) -> None:
    """
    Render verified execution proof points.
    """

    horizons = evidence.get(
        "governed_demand_horizons",
        [],
    )

    horizon_label = (
        "/".join(
            str(value)
            for value in horizons
        )
        if horizons
        else
        "—"
    )

    cards = [
        (
            evidence.get(
                "production_intelligence_systems",
                "—",
            ),
            "Production intelligence systems",
            "Governed model and deterministic intelligence layer",
        ),
        (
            f"{evidence.get('automated_tests_passing', 0):,}",
            "Automated tests passing",
            "Current regression suite",
        ),
        (
            f"{evidence.get('customer_decisions_generated', 0):,}",
            "Customer decisions generated",
            "Verified Docker smoke execution",
        ),
        (
            f"{evidence.get('repeat_purchase_scores_generated', 0):,}",
            "Repeat-purchase scores",
            "Verified governed scoring execution",
        ),
        (
            horizon_label,
            "Demand horizons · weeks",
            "Governed weekly forecasts",
        ),
        (
            (
                f"{evidence.get('production_scoring_systems_available', 0)}"
                "/"
                f"{evidence.get('production_scoring_systems_total', 0)}"
            ),
            "Scoring systems available",
            "Verified runtime health",
        ),
    ]

    html = (
        '<div class="portfolio-proof-grid">'
    )

    for (
        value,
        label,
        caption,
    ) in cards:

        html += (
            '<div class="portfolio-proof-card">'
            '<div class="portfolio-proof-value">'
            f"{_safe(value)}"
            "</div>"
            '<div class="portfolio-proof-label">'
            f"{_safe(label)}"
            "</div>"
            '<div class="portfolio-proof-caption">'
            f"{_safe(caption)}"
            "</div>"
            "</div>"
        )

    html += "</div>"

    st_module.markdown(
        html,
        unsafe_allow_html=True,
    )


def render_system_card(
    st_module,
    system: dict[str, Any],
) -> None:
    """
    Render one production-intelligence system card.
    """

    model_type = system.get(
        "model_type",
        "Not Available",
    )

    model_version = system.get(
        "model_version",
        "Not Available",
    )

    target_name = system.get(
        "target_name",
        "Not Available",
    )

    badge = governance_badge(
        system.get(
            "governance"
        )
    )

    st_module.markdown(
        (
            '<div class="portfolio-system-card">'
            '<div class="portfolio-system-name">'
            f"{_safe(system.get('display_name'))}"
            "</div>"
            '<div class="portfolio-system-meta">'
            f"{_safe(model_type)} · v{_safe(model_version)}"
            "<br>"
            f"Target: {_safe(target_name)}"
            "</div>"
            f"{badge}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_architecture_flow(
    st_module,
) -> None:
    """
    Render the top-level production architecture.
    """

    nodes = [
        "Transactions",
        "Canonical Data",
        "Feature Pipelines",
        "Intelligence Systems",
        "Score Registry",
        "Decision Engine",
        "Priority & Contact Policy",
        "Experiment Governance",
        "CRM Delivery",
        "Audit",
        "Monitoring",
        "Scheduled Docker Runtime",
    ]

    html = (
        '<div class="portfolio-flow">'
    )

    for index, node in enumerate(
        nodes
    ):

        html += (
            '<div class="portfolio-flow-node">'
            f"{_safe(node)}"
            "</div>"
        )

        if index < len(
            nodes
        ) - 1:
            html += (
                '<div class="portfolio-flow-arrow">'
                "→"
                "</div>"
            )

    html += "</div>"

    st_module.markdown(
        html,
        unsafe_allow_html=True,
    )


def render_claims_note(
    st_module,
    text: str,
) -> None:
    """
    Render claims-governance notice.
    """

    st_module.markdown(
        (
            '<div class="portfolio-governance-note">'
            "<strong>Evidence discipline.</strong> "
            f"{_safe(text)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )