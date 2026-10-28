
"""
Interactive navigation tests for the Revenue OS portfolio.

Uses Streamlit's AppTest framework to verify:
- Initial application state
- Eleven sidebar buttons
- Navigation callbacks
- Session-state persistence
- Successful rendering of key application pages
"""

import pytest

from streamlit.testing.v1 import AppTest
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


TIMEOUT = 40

NAVIGATION_PAGES = [
    "Overview",
    "Revenue Command Centre",
    "Customer Decision Explorer",
    "Interactive Decision Demo",
    "Decision Intelligence",
    "ML / Model Explorer",
    "Governance Dashboard",
    "Production Architecture",
    "Architecture Visuals",
    "Testing & Reliability",
    "Production Evidence",
]


@pytest.fixture
def app():
    application = AppTest.from_file(
        APP_PATH,
        default_timeout=TIMEOUT,
    ).run()

    assert len(application.exception) == 0

    return application

    assert len(application.exception) == 0

    return application


def test_application_initialises_on_overview(app):
    assert (
        app.session_state["portfolio_active_page"]
        == "Overview"
    )


def test_all_navigation_buttons_are_present(app):
    labels = [button.label for button in app.button]

    assert len(labels) == 11

    assert set(labels) == set(NAVIGATION_PAGES)


@pytest.mark.parametrize(
    "destination",
    NAVIGATION_PAGES,
)
def test_navigation_click_opens_expected_page(
    app,
    destination,
):
    button = app.button(
        key=f"nav_{destination}"
    )

    button.click()

    app.run()

    assert len(app.exception) == 0

    assert (
        app.session_state["portfolio_active_page"]
        == destination
    )


def test_navigation_state_persists_after_rerun(app):
    destination = "Governance Dashboard"

    app.button(
        key=f"nav_{destination}"
    ).click()

    app.run()

    assert (
        app.session_state["portfolio_active_page"]
        == destination
    )

    app.run()

    assert (
        app.session_state["portfolio_active_page"]
        == destination
    )


def test_navigation_can_return_to_overview(app):
    app.button(
        key="nav_Revenue Command Centre"
    ).click()

    app.run()

    assert (
        app.session_state["portfolio_active_page"]
        == "Revenue Command Centre"
    )

    app.button(
        key="nav_Overview"
    ).click()

    app.run()

    assert (
        app.session_state["portfolio_active_page"]
        == "Overview"
    )

    assert len(app.exception) == 0
