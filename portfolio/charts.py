
"""
Premium governed visualisations for the AI Revenue Decision Engine.

Presentation only:
- Preserve source data and governed decisions.
- Use deterministic semantic colours.
- Display readable numeric axes.
- Provide a visible action legend.
- Preserve unknown categories with neutral fallbacks.
"""

from __future__ import annotations

import altair as alt
import pandas as pd


BACKGROUND = "#101B27"
TEXT = "#B1C3CD"
GRID = "#344956"
TEAL = "#13B8A6"
NEUTRAL = "#7898AC"

ACTION_COLORS = {
    "Protect": "#E66E78",
    "Retain": "#E6B85C",
    "Reactivate": "#A9A0F5",
    "Grow": "#46D5A0",
    "Monitor": "#7898AC",
    "Unscored": "#718395",
    "Other": "#7898AC",
}

STATE_COLORS = {
    "Active": "#46D5A0",
    "At Risk": "#E6B85C",
    "Dormant": "#A9A0F5",
    "Long Dormant": "#718395",
}

PRIORITY_COLORS = {
    "Priority 1": "#E66E78",
    "Priority 2": "#E6B85C",
    "Priority 3": "#13B8A6",
    "Priority 4": "#7898AC",
    "Not Prioritised": "#718395",
}

CONTACT_COLORS = {
    "High Touch": "#E66E78",
    "Campaign": "#E6B85C",
    "Automated / Low Cost": "#13B8A6",
    "Hold / Control": "#7898AC",
    "No Model-Led Contact": "#718395",
}

CATEGORY_PALETTES = {
    "primary_action": ACTION_COLORS,
    "customer_state": STATE_COLORS,
    "priority_tier": PRIORITY_COLORS,
    "contact_intensity": CONTACT_COLORS,
}


def action_group(value: object) -> str:
    """Normalise action labels for presentation, not for decisions."""

    label = str(value)

    if label in {
        "Protect",
        "Retain",
        "Reactivate",
        "Grow",
        "Monitor",
    }:
        return label

    if label.startswith("Unscored"):
        return "Unscored"

    return "Other"


def colour_for_category(
    column: str,
    value: object,
) -> str:
    """Resolve a governed category to its semantic presentation colour."""

    label = str(value)

    if column == "primary_action":
        group = action_group(value)
        return ACTION_COLORS[group]

    return CATEGORY_PALETTES.get(
        column,
        {},
    ).get(label, NEUTRAL)


def apply_chart_style(
    chart: alt.Chart,
) -> alt.Chart:
    """Apply consistent corporate chart typography and grid styling."""

    return (
        chart
        .configure_view(
            stroke=None,
        )
        .configure_axis(
            labelColor=TEXT,
            titleColor=TEXT,
            gridColor=GRID,
            domainColor=GRID,
            tickColor=GRID,
            labelFontSize=12,
            titleFontSize=12,
            labelPadding=7,
        )
        .configure_legend(
            labelColor=TEXT,
            titleColor=TEXT,
            labelFontSize=12,
            titleFontSize=12,
            symbolSize=130,
            padding=12,
        )
    )


def categorical_bar_chart(
    distribution: pd.DataFrame,
    *,
    source_column: str,
    category_column: str,
    count_column: str = "Customers",
    height: int = 310,
) -> alt.Chart:
    """
    Build a semantic horizontal distribution chart.

    The input is aggregated by the existing portfolio logic.
    This function never changes that aggregation.
    """

    required = {
        category_column,
        count_column,
    }

    if not required.issubset(distribution.columns):
        raise ValueError(
            "Categorical chart is missing required columns."
        )

    frame = distribution[
        [
            category_column,
            count_column,
        ]
    ].copy()

    frame[category_column] = (
        frame[category_column].astype(str)
    )

    frame["_semantic_color"] = (
        frame[category_column].map(
            lambda value: colour_for_category(
                source_column,
                value,
            )
        )
    )

    chart = (
        alt.Chart(frame)
        .mark_bar(
            cornerRadiusEnd=2,
            size=19,
        )
        .encode(
            x=alt.X(
                f"{count_column}:Q",
                title="Customers",
                axis=alt.Axis(
                    format=",.0f",
                    tickMinStep=1,
                ),
                scale=alt.Scale(
                    zero=True,
                ),
            ),
            y=alt.Y(
                f"{category_column}:N",
                title=None,
                sort="-x",
                axis=alt.Axis(
                    labelLimit=260,
                    labelPadding=10,
                ),
            ),
            color=alt.Color(
                "_semantic_color:N",
                scale=None,
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(
                    f"{category_column}:N",
                    title=category_column,
                ),
                alt.Tooltip(
                    f"{count_column}:Q",
                    title="Customers",
                    format=",",
                ),
            ],
        )
        .properties(
            height=height,
        )
    )

    return apply_chart_style(chart)


def risk_value_scatter_chart(
    frame: pd.DataFrame,
    *,
    height: int = 460,
) -> alt.Chart:
    """
    Plot governed leakage probability against modelled FCV90.

    Bubble size represents observed lifetime revenue.

    The visible legend is derived from existing action labels.
    Unknown and unscored states remain neutral.
    """

    required = {
        "Leakage Probability",
        "Modelled FCV90",
        "Lifetime Revenue",
        "Action",
    }

    if not required.issubset(frame.columns):
        raise ValueError(
            "Risk-value chart is missing required columns."
        )

    chart_data = frame.copy()

    chart_data["_action_group"] = (
        chart_data["Action"].map(action_group)
    )

    action_order = list(ACTION_COLORS.keys())

    action_range = [
        ACTION_COLORS[action]
        for action in action_order
    ]

    tooltips = [
        alt.Tooltip(
            "Action:N",
            title="Governed action",
        ),
        alt.Tooltip(
            "Leakage Probability:Q",
            title="Leakage probability",
            format=".2f",
        ),
        alt.Tooltip(
            "Modelled FCV90:Q",
            title="Modelled FCV90",
            format=",.2f",
        ),
        alt.Tooltip(
            "Lifetime Revenue:Q",
            title="Observed lifetime revenue",
            format=",.2f",
        ),
    ]

    if "Customer" in chart_data.columns:
        tooltips.insert(
            0,
            alt.Tooltip(
                "Customer:N",
                title="Portfolio customer",
            ),
        )

    chart = (
        alt.Chart(chart_data)
        .mark_circle(
            opacity=0.68,
            stroke=BACKGROUND,
            strokeWidth=0.5,
        )
        .encode(
            x=alt.X(
                "Leakage Probability:Q",
                title="Leakage probability",
                scale=alt.Scale(
                    domain=[0, 1],
                ),
                axis=alt.Axis(
                    format=".2f",
                ),
            ),
            y=alt.Y(
                "Modelled FCV90:Q",
                title=(
                    "Modelled future customer value (90 days)"
                ),
                axis=alt.Axis(
                    format=",.0f",
                ),
            ),
            size=alt.Size(
                "Lifetime Revenue:Q",
                legend=None,
                scale=alt.Scale(
                    range=[25, 650],
                ),
            ),
            color=alt.Color(
                "_action_group:N",
                title="Governed action",
                scale=alt.Scale(
                    domain=action_order,
                    range=action_range,
                ),
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    columns=4,
                ),
            ),
            tooltip=tooltips,
        )
        .properties(
            height=height,
        )
    )

    return apply_chart_style(chart)
