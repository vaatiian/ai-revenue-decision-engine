"""
Streamlit Production Architecture page.

The page explains the verified implementation topology without executing
production workloads.
"""

from __future__ import annotations

from html import escape

import pandas as pd

from portfolio.components import (
    render_claims_note,
    render_section_heading,
)

from portfolio.production_architecture import (
    architecture_principle_rows,
    architecture_summary,
    batch_job_rows,
    customer_pipeline_rows,
    runtime_control_rows,
    validate_architecture_evidence,
)


def render_architecture_metric_card(
    st_module,
    *,
    label: str,
    value: str,
    caption: str,
) -> None:
    """
    Render compact architecture proof points without metric truncation.
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
            "font-size:0.69rem;"
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


def render_pipeline(
    st_module,
    rows: list[dict[str, str]],
) -> None:
    """
    Render the ordered production customer pipeline.
    """

    for index, row in enumerate(
        rows
    ):

        left, right = st_module.columns(
            [
                1,
                4,
            ],
            gap="medium",
        )

        with left:

            st_module.markdown(
                (
                    '<div style="'
                    "color:#52D6FF;"
                    "font-size:0.72rem;"
                    "letter-spacing:0.08em;"
                    "text-transform:uppercase;"
                    "padding-top:0.6rem;"
                    '">'
                    f"{index + 1:02d} · {escape(row['layer'])}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        with right:

            st_module.markdown(
                (
                    '<div class="portfolio-panel" '
                    'style="margin-bottom:0.35rem;">'
                    '<div style="'
                    "font-size:1rem;"
                    "font-weight:650;"
                    "color:#F4F2EC;"
                    "margin-bottom:0.35rem;"
                    '">'
                    f"{escape(row['stage'])}"
                    "</div>"
                    '<div style="'
                    "font-size:0.77rem;"
                    "color:#AAB7B4;"
                    "line-height:1.5;"
                    '">'
                    f"{escape(row['purpose'])}"
                    "</div>"
                    '<div style="'
                    "font-size:0.68rem;"
                    "color:#71817D;"
                    "margin-top:0.55rem;"
                    '">'
                    f"{escape(row['source'])}"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_production_architecture(
    st_module,
    bundle,
) -> None:
    """
    Render the verified production architecture.
    """

    summary = bundle[
        "summary"
    ]

    validate_architecture_evidence(
        summary
    )

    evidence = architecture_summary(
        summary
    )

    render_section_heading(
        st_module,
        kicker="Production Architecture",
        title="From transaction evidence to governed commercial action.",
        copy=(
            "The production system separates data, feature construction, "
            "scoring, signal alignment, deterministic decision policy, "
            "delivery, experimentation, monitoring, and scheduled runtime."
        ),
    )

    metric_cols = st_module.columns(
        5,
        gap="medium",
    )

    metrics = (
        (
            "Production systems",
            str(
                evidence[
                    "production_systems"
                ]
            ),
            "Governed intelligence systems",
        ),
        (
            "Runtime availability",
            (
                f"{evidence['scoring_systems_available']}/"
                f"{evidence['production_systems']}"
            ),
            "Production scoring interfaces available",
        ),
        (
            "Batch jobs",
            str(
                evidence[
                    "batch_jobs"
                ]
            ),
            "Customer, Repeat Purchase, Demand",
        ),
        (
            "Customer decisions",
            f"{evidence['customer_decisions']:,}",
            "Verified customer-batch output",
        ),
        (
            "Regression tests",
            f"{evidence['automated_tests']:,}",
            "Latest verified portfolio proof point",
        ),
    )

    for column, metric in zip(
        metric_cols,
        metrics,
    ):

        with column:

            render_architecture_metric_card(
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
        kicker="Customer Decision Pipeline",
        title="The governed path from transactions to action",
        copy=(
            "Prediction and business policy are separate layers. "
            "Delivery, experiments, and monitoring consume governed outputs "
            "rather than silently redefining the customer action."
        ),
    )

    render_pipeline(
        st_module,
        customer_pipeline_rows(),
    )

    st_module.markdown(
        "<div style='height:1.4rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Deployment Jobs",
        title="Three production batch contracts",
        copy=(
            "Customer decisions, Repeat Purchase scoring, and Demand Forecasting "
            "have separate deployment entry points and operational scope."
        ),
    )

    batch_frame = pd.DataFrame(
        [
            {
                "Job":
                    row[
                        "job"
                    ],

                "Entrypoint":
                    row[
                        "entrypoint"
                    ],

                "Module":
                    row[
                        "module"
                    ],

                "Scope":
                    row[
                        "scope"
                    ],

                "Decision Engine":
                    (
                        "Yes"
                        if row[
                            "decision_engine"
                        ]
                        else
                        "No"
                    ),
            }
            for row in batch_job_rows()
        ]
    )

    st_module.dataframe(
        batch_frame,
        use_container_width=True,
        hide_index=True,
    )

    st_module.info(
        (
            "Repeat Purchase and Demand Forecast are production systems, "
            "but they intentionally remain outside mature customer-level "
            "primary-action orchestration."
        )
    )

    st_module.markdown(
        "<div style='height:1.4rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Runtime & Control Plane",
        title="A one-shot, non-root Docker execution contract",
        copy=(
            "The image packages the production source and governed manifests. "
            "An external scheduler invokes the governed one-shot job interface; "
            "the repository does not claim an in-repo scheduler implementation."
        ),
    )

    runtime_frame = pd.DataFrame(
        [
            {
                "Control":
                    row[
                        "control"
                    ],

                "Evidence":
                    row[
                        "evidence"
                    ],

                "Source":
                    row[
                        "source"
                    ],
            }
            for row in runtime_control_rows()
        ]
    )

    st_module.dataframe(
        runtime_frame,
        use_container_width=True,
        hide_index=True,
    )

    st_module.markdown(
        "<div style='height:1.4rem'></div>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        st_module,
        kicker="Production Design Principles",
        title="What the implementation is designed to protect",
        copy=(
            "These principles explain why the system is layered rather than "
            "collapsing models, business policy, delivery, and monitoring "
            "into one opaque scoring script."
        ),
    )

    principle_rows = architecture_principle_rows()

    for start in range(
        0,
        len(
            principle_rows
        ),
        2,
    ):

        columns = st_module.columns(
            2,
            gap="medium",
        )

        row = principle_rows[
            start:
            start + 2
        ]

        for column, principle in zip(
            columns,
            row,
        ):

            with column:

                st_module.markdown(
                    (
                        '<div class="portfolio-panel" '
                        'style="min-height:145px;">'
                        '<div style="'
                        "font-size:0.76rem;"
                        "font-weight:650;"
                        "color:#52D6FF;"
                        "margin-bottom:0.55rem;"
                        '">'
                        f"{escape(principle['principle'])}"
                        "</div>"
                        '<div style="'
                        "font-size:0.77rem;"
                        "color:#AAB7B4;"
                        "line-height:1.55;"
                        '">'
                        f"{escape(principle['meaning'])}"
                        "</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

    render_claims_note(
        st_module,
        (
            "Architecture evidence demonstrates implemented production "
            "boundaries and verified runtime behaviour. It does not imply "
            "a managed cloud platform or external scheduler that is not "
            "present in this repository."
        ),
    )