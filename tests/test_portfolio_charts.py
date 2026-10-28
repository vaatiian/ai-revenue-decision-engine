
"""
Tests for premium semantic data visualisations.

These tests protect:
- Locked colour mappings
- Unknown-category fallbacks
- Input-data immutability
- Readable numeric axes
- Visible governed-action legends
"""

import pandas as pd
import pytest

from portfolio.charts import (
    ACTION_COLORS,
    action_group,
    categorical_bar_chart,
    colour_for_category,
    risk_value_scatter_chart,
)


def test_action_colours_are_locked():
    assert ACTION_COLORS["Protect"] == "#E66E78"
    assert ACTION_COLORS["Retain"] == "#E6B85C"
    assert ACTION_COLORS["Reactivate"] == "#A9A0F5"
    assert ACTION_COLORS["Grow"] == "#46D5A0"
    assert ACTION_COLORS["Monitor"] == "#7898AC"


def test_unscored_and_unknown_actions_are_neutral():
    assert colour_for_category(
        "primary_action",
        "Unscored — Long Dormant",
    ) == "#718395"

    assert colour_for_category(
        "primary_action",
        "Unscored",
    ) == "#718395"

    assert colour_for_category(
        "primary_action",
        "Unknown Action",
    ) == "#7898AC"

    assert action_group(
        "Unscored — Long Dormant"
    ) == "Unscored"

    assert action_group(
        "Unknown Action"
    ) == "Other"


def test_customer_state_colours():
    assert colour_for_category(
        "customer_state",
        "Active",
    ) == "#46D5A0"

    assert colour_for_category(
        "customer_state",
        "At Risk",
    ) == "#E6B85C"


def test_categorical_chart_preserves_categories():
    frame = pd.DataFrame(
        {
            "Action": [
                "Protect",
                "Retain",
                "Unscored — Long Dormant",
            ],
            "Customers": [
                45,
                80,
                120,
            ],
        }
    )

    original = frame.copy(deep=True)

    chart = categorical_bar_chart(
        frame,
        source_column="primary_action",
        category_column="Action",
    )

    spec = chart.to_dict()

    assert spec["mark"]["type"] == "bar"

    assert spec["encoding"]["color"]["scale"] is None

    assert (
        spec["encoding"]["x"]["axis"]["format"]
        == ",.0f"
    )

    assert len(spec["datasets"]) == 1

    assert frame.equals(original)


def test_categorical_chart_rejects_missing_columns():
    frame = pd.DataFrame(
        {
            "Action": ["Protect"],
        }
    )

    with pytest.raises(ValueError):
        categorical_bar_chart(
            frame,
            source_column="primary_action",
            category_column="Action",
        )


def test_risk_value_chart_preserves_model_outputs():
    frame = pd.DataFrame(
        {
            "Customer": ["CUST-00001"],
            "Action": ["Protect"],
            "Leakage Probability": [0.76],
            "Modelled FCV90": [250.0],
            "Lifetime Revenue": [1250.0],
        }
    )

    original = frame.copy(deep=True)

    chart = risk_value_scatter_chart(frame)

    spec = chart.to_dict()

    assert spec["mark"]["type"] == "circle"

    color = spec["encoding"]["color"]

    assert color["field"] == "_action_group"

    assert color["legend"]["orient"] == "bottom"

    assert (
        color["scale"]["domain"]
        == list(ACTION_COLORS.keys())
    )

    assert (
        color["scale"]["range"]
        == list(ACTION_COLORS.values())
    )

    assert (
        spec["encoding"]["y"]["axis"]["format"]
        == ",.0f"
    )

    assert frame.equals(original)


def test_risk_value_chart_rejects_missing_columns():
    with pytest.raises(ValueError):
        risk_value_scatter_chart(
            pd.DataFrame(
                {
                    "Action": ["Protect"],
                }
            )
        )
