"""
Streamlit Testing & Reliability page.

The page explains regression breadth, source-test breadth, integration
coverage, and fail-closed reliability controls.
"""

from __future__ import annotations

from html import escape

import pandas as pd

from portfolio.components import (
    render_claims_note,
    render_section_heading,
)

from portfolio.testing_reliability import (
    fail_closed_rows,
    integration_layer_rows,
    reliability_domain_rows,
    reliability_summary,
    validate_testing_evidence,
)


def render_reliability_metric(
    st_module,
    *,
    label: str,
    value: str,
    caption: str,
) -> None:
    """
    Render reliability KPIs without truncation.
    """

    st_module.markdown(
        (
            '<div class="portfolio-panel" style="min-height:138px;">'
            '<div style="'
            "font-size:0.69rem;"
            "color:#AAB7B4;"
            "margin-bottom:0.55rem;"
            '">'
            f"{escape(label)}"
            "</div>"
            '<div style="'
            "font-size:1.75rem;"
            "font-weight:500;"
            "color:#F4F2EC;"
            "line-height:1.05;"
            '">'
            f"{escape(value)}"
            "</div>"
            '<div style="'
            "font-size:0.68rem;"
            "color:#71817D;"
            "line-height:1.45;"
            "margin-top:0.55rem;"
            '">'
            f"{escape(caption)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_testing_reliability(
    st_module,
    bundle,
) -> None:
    """
    Render testing and reliability evidence.
    """

    summary = bundle[
        "summary"
    ]

    validate_testing_evidence(
        summary
    )

    evidence = reliability_summary(
        summary
    )

    render_section_heading(
        st_module,
        kicker="Testing & Reliability",
        title="Reliability is designed into the boundaries, not added at the end.",
        copy=(
            "The regression suite tests predictive contracts, deterministic "
            "decision policy, delivery, experimentation, monitoring, "
            "deployment, Docker runtime, scheduled execution, and the "
            "public portfolio application."
        ),
    )

    first_row = st_module.columns(
        4,
        gap="medium",
    )

    first_metrics = (
        (
            "Executed regression cases",
            f"{evidence['executed_regression_cases']:,}",
            "Latest verified pytest result",
        ),
        (
            "Explicit test functions",
            f"{evidence['explicit_test_functions']:,}",
            "Source inventory of def test_ functions",
        ),
        (
            "Test modules",
            f"{evidence['test_modules']}",
            "Top-level test_*.py modules",
        ),
        (
            "Known warnings",
            f"{evidence['known_warnings']}",
            "Tracked non-fatal warning",
        ),
    )

    for column, metric in zip(
        first_row,
        first_metrics,
    ):

        with column:

            render_reliability_metric(
                st_module,
                label=metric[
                    0
                ],
                value=metric[
                    1
                ],
                caption=metric[
                    2
                ],
            )

    st_module.info(
        (
            f"Pytest executed {evidence['collection_expansion']:,} more "
            "cases than the number of explicit test-function definitions. "
            "This is reported as collection expansion, not as additional "
            "handwritten tests."
        )
    )

    render_claims_note(
        st_module,
        summary[
            "claims_note"
        ],
    )

    st_module.markdown(
        "<div style='height:1.3rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Reliability Domains",
        title="Seven layers of regression protection",
        copy=(
            "Testing is distributed across the production system rather "
            "than concentrated only around model accuracy."
        ),
    )

    domains = reliability_domain_rows()

    for start in range(
        0,
        len(
            domains
        ),
        2,
    ):

        columns = st_module.columns(
            2,
            gap="medium",
        )

        row = domains[
            start:
            start + 2
        ]

        for column, domain in zip(
            columns,
            row,
        ):

            with column:

                examples = " · ".join(
                    domain[
                        "examples"
                    ]
                )

                st_module.markdown(
                    (
                        '<div class="portfolio-panel" '
                        'style="min-height:205px;">'
                        '<div style="'
                        "font-size:0.78rem;"
                        "font-weight:650;"
                        "color:#52D6FF;"
                        "margin-bottom:0.55rem;"
                        '">'
                        f"{escape(domain['domain'])}"
                        "</div>"
                        '<div style="'
                        "font-size:0.77rem;"
                        "color:#F4F2EC;"
                        "line-height:1.5;"
                        "margin-bottom:0.7rem;"
                        '">'
                        f"{escape(domain['purpose'])}"
                        "</div>"
                        '<div style="'
                        "font-size:0.7rem;"
                        "color:#AAB7B4;"
                        "line-height:1.45;"
                        "margin-bottom:0.65rem;"
                        '">'
                        f"{escape(domain['evidence'])}"
                        "</div>"
                        '<div style="'
                        "font-size:0.67rem;"
                        "color:#71817D;"
                        "line-height:1.4;"
                        '">'
                        f"{escape(examples)}"
                        "</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

    st_module.markdown(
        "<div style='height:1.3rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Fail-Closed Design",
        title="Invalid production states should stop, not silently degrade",
        copy=(
            "The suite explicitly exercises malformed schemas, duplicate "
            "identities, unknown governed values, temporal violations, "
            "persistent-store corruption, and invalid delivery eligibility."
        ),
    )

    fail_closed_frame = pd.DataFrame(
        [
            {
                "Control":
                    row[
                        "control"
                    ],

                "Expected behaviour":
                    row[
                        "behaviour"
                    ],
            }
            for row in fail_closed_rows()
        ]
    )

    st_module.dataframe(
        fail_closed_frame,
        use_container_width=True,
        hide_index=True,
    )

    st_module.markdown(
        "<div style='height:1.3rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Integration Coverage",
        title="Cross-layer behaviour is tested, not only isolated functions",
        copy=(
            "Integration suites verify contracts across scoring, "
            "decisioning, delivery, experiments, monitoring, runtime, "
            "and deployment."
        ),
    )

    integration_frame = pd.DataFrame(
        {
            "Verified integration layer":
                integration_layer_rows()
        }
    )

    st_module.dataframe(
        integration_frame,
        use_container_width=True,
        hide_index=True,
    )

    integration_cols = st_module.columns(
        3,
        gap="medium",
    )

    integration_cols[
        0
    ].metric(
        "Integration layers",
        evidence[
            "integration_layers"
        ],
    )

    integration_cols[
        1
    ].metric(
        "Fail-closed control families",
        evidence[
            "fail_closed_controls"
        ],
    )

    integration_cols[
        2
    ].metric(
        "Reliability domains",
        evidence[
            "reliability_domains"
        ],
    )

    st_module.markdown(
        "<div style='height:1.3rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Regression Discipline",
        title="One authoritative suite, one published proof point",
        copy=(
            "The portfolio publishes the actual executed pytest count from "
            "the last green full regression. Source-level test-function "
            "inventory is shown separately to avoid inflating the claim."
        ),
    )

    st_module.code(
        "python -m pytest -q",
        language="bash",
    )

    st_module.success(
        (
            f"{evidence['executed_regression_cases']:,} tests passing "
            "in the latest verified full regression."
        )
    )

    st_module.warning(
        (
            "One known non-fatal pandas date-parsing warning remains in the "
            "intentionally invalid Repeat Purchase date test. It is visible "
            "rather than suppressed from the portfolio evidence."
        )
    )

    render_claims_note(
        st_module,
        (
            "Passing tests are engineering evidence, not proof of causal "
            "business impact. Reliability claims are limited to behaviours "
            "actually exercised by the regression suite."
        ),
    )