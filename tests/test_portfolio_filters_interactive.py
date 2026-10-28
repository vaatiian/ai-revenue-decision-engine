
"""
Revenue OS interactive filter acceptance tests.

Verifies Streamlit widget interactions against the
existing governed, curated customer-decision population.

No production models or private transaction data are loaded.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from portfolio.command_centre import filter_decisions
from portfolio.data_loader import load_customer_decisions


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
TIMEOUT = 40

FILTERS = [
    ("Customer state", "customer_state", "states"),
    ("Primary action", "primary_action", "actions"),
    ("Priority tier", "priority_tier", "priorities"),
    ("Contact intensity", "contact_intensity", "contacts"),
    (
        "Experiment requirement",
        "experiment_requirement",
        "experiments",
    ),
]


@pytest.fixture(scope="module")
def decisions():
    return load_customer_decisions()


@pytest.fixture
def command_centre():
    app = AppTest.from_file(
        APP_PATH,
        default_timeout=TIMEOUT,
    ).run()

    assert len(app.exception) == 0

    app.button(
        key="nav_Revenue Command Centre"
    ).click().run()

    assert len(app.exception) == 0
    assert (
        app.session_state["portfolio_active_page"]
        == "Revenue Command Centre"
    )

    return app


def customer_metric(app):
    matches = [
        metric
        for metric in app.metric
        if metric.label == "Customers in view"
    ]

    assert len(matches) == 1

    return int(
        str(matches[0].value).replace(",", "")
    )


def assert_clean(app):
    assert len(app.exception) == 0


def test_command_centre_starts_with_complete_population(
    command_centre,
    decisions,
):
    assert customer_metric(command_centre) == len(decisions)
    assert len(decisions) == 5220


def test_all_five_multiselect_controls_are_visible(
    command_centre,
):
    actual = {
        widget.label
        for widget in command_centre.multiselect
    }

    expected = {
        label
        for label, _, _ in FILTERS
    }

    assert expected.issubset(actual)


def test_customer_id_search_is_visible(command_centre):
    matches = [
        widget
        for widget in command_centre.text_input
        if widget.label == "Customer ID search"
    ]

    assert len(matches) == 1


@pytest.mark.parametrize(
    "label,column,argument",
    FILTERS,
)
def test_each_filter_updates_visible_customer_count(
    command_centre,
    decisions,
    label,
    column,
    argument,
):
    values = sorted(
        decisions[column].dropna().unique().tolist()
    )

    assert values, f"No filter values for {column}"

    selected = values[0]

    expected = filter_decisions(
        decisions,
        **{argument: [selected]},
    )

    widget = next(
        widget
        for widget in command_centre.multiselect
        if widget.label == label
    )

    widget.set_value([selected]).run()

    assert_clean(command_centre)

    assert customer_metric(command_centre) == len(
        expected
    )


def test_clearing_filter_restores_full_population(
    command_centre,
    decisions,
):
    widget = next(
        widget
        for widget in command_centre.multiselect
        if widget.label == "Customer state"
    )

    selected = widget.options[0]

    widget.set_value([selected]).run()

    assert_clean(command_centre)

    widget = next(
        item
        for item in command_centre.multiselect
        if item.label == "Customer state"
    )

    widget.set_value([]).run()

    assert_clean(command_centre)

    assert customer_metric(command_centre) == len(
        decisions
    )


def test_exact_customer_search_finds_one_record(
    command_centre,
    decisions,
):
    customer_id = decisions[
        "portfolio_customer_id"
    ].iloc[0]

    search = next(
        widget
        for widget in command_centre.text_input
        if widget.label == "Customer ID search"
    )

    search.set_value(customer_id).run()

    assert_clean(command_centre)

    assert customer_metric(command_centre) == 1


def test_unmatched_customer_search_shows_empty_state(
    command_centre,
):
    search = next(
        widget
        for widget in command_centre.text_input
        if widget.label == "Customer ID search"
    )

    search.set_value("CUST-999999999").run()

    assert_clean(command_centre)

    assert customer_metric(command_centre) == 0

    messages = [
       warning.value
       for warning in command_centre.warning
    ]
    assert any(
        "No governed customer decisions match"
        in message
        for message in messages
    )


def test_empty_result_does_not_render_decision_charts(
    command_centre,
):
    search = next(
        widget
        for widget in command_centre.text_input
        if widget.label == "Customer ID search"
    )

    search.set_value("CUST-999999999").run()

    assert_clean(command_centre)
    assert customer_metric(command_centre) == 0

    assert len(command_centre.get("vega_lite_chart")) == 0


def test_queue_size_selector_is_present(
    command_centre,
):
    selectors = [
        widget
        for widget in command_centre.get("segmented_control")
        if widget.label == "Queue size"
    ]

    assert len(selectors) == 1


@pytest.mark.parametrize("limit", [10, 25, 50])
def test_queue_size_selection_does_not_break_page(
    command_centre,
    limit,
):
    selector = next(
        widget
        for widget in command_centre.get(
            "segmented_control"
        )
        if widget.label == "Queue size"
    )

    selector.set_value(limit).run()

    assert_clean(command_centre)

    assert (
        command_centre.session_state[
            "portfolio_active_page"
        ]
        == "Revenue Command Centre"
    )
