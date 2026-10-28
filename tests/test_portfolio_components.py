
"""
Tests for shared portfolio presentation components.

The tests protect HTML rendering, executive narrative,
verified evidence, and the premium KPI hierarchy.
"""

from portfolio.components import (
    render_hero,
    render_proof_cards,
)


class MarkdownCapture:
    """Capture Streamlit Markdown without a live application."""

    def __init__(self):
        self.calls = []

    def markdown(self, body, **kwargs):
        self.calls.append((body, kwargs))


def test_hero_renders_html_without_markdown_code_indentation():
    fake_streamlit = MarkdownCapture()

    render_hero(fake_streamlit)

    assert len(fake_streamlit.calls) == 1

    body, kwargs = fake_streamlit.calls[0]

    assert kwargs["unsafe_allow_html"] is True

    assert body.startswith(
        '<div class="portfolio-hero">'
    )

    assert "<h1>" in body

    assert "From retail transactions to " in body

    assert "governed revenue decisions." in body

    assert "portfolio-hero-flow" in body

    assert "portfolio-hero-disclosure" in body

    for stage in [
        "TRANSACTIONS",
        "INTELLIGENCE",
        "DECISIONS",
        "GOVERNANCE",
    ]:
        assert stage in body

    assert "\n    <h1>" not in body

    assert body.endswith("</div>")


def test_proof_cards_preserve_verified_evidence():
    fake_streamlit = MarkdownCapture()

    evidence = {
        "customer_decisions_generated": 5220,
        "production_scoring_systems_available": 6,
        "production_scoring_systems_total": 6,
        "automated_tests_passing": 1840,
        "production_intelligence_systems": 6,
        "repeat_purchase_scores_generated": 4269,
        "governed_demand_horizons": [1, 2, 4],
    }

    render_proof_cards(fake_streamlit, evidence)

    assert len(fake_streamlit.calls) == 1

    body, kwargs = fake_streamlit.calls[0]

    assert kwargs["unsafe_allow_html"] is True

    assert "5,220" in body
    assert "6/6" in body
    assert "1,840" in body
    assert "4,269" in body
    assert "1 / 2 / 4" in body

    assert body.count(
        '<div class="portfolio-proof-card'
    ) == 6


def test_primary_kpis_follow_commercial_hierarchy():
    fake_streamlit = MarkdownCapture()

    evidence = {
        "customer_decisions_generated": 5220,
        "production_scoring_systems_available": 6,
        "production_scoring_systems_total": 6,
        "automated_tests_passing": 1840,
        "production_intelligence_systems": 6,
        "repeat_purchase_scores_generated": 4269,
        "governed_demand_horizons": [1, 2, 4],
    }

    render_proof_cards(fake_streamlit, evidence)

    body = fake_streamlit.calls[0][0]

    customer_position = body.index(
        "Customer decisions generated"
    )

    scoring_position = body.index(
        "Scoring systems available"
    )

    testing_position = body.index(
        "Automated tests passing"
    )

    secondary_position = body.index(
        "Production intelligence systems"
    )

    assert (
        customer_position
        < scoring_position
        < testing_position
        < secondary_position
    )

    assert "portfolio-proof-primary" in body
    assert "portfolio-proof-secondary" in body

    assert "proof-commercial" in body
    assert "proof-intelligence" in body
    assert "proof-reliability" in body


def test_proof_cards_do_not_invent_missing_evidence():
    fake_streamlit = MarkdownCapture()

    render_proof_cards(fake_streamlit, {})

    body = fake_streamlit.calls[0][0]

    assert "5,220" not in body
    assert "4,269" not in body
    assert "1,840" not in body

    assert "0/0" in body
    assert "—" in body

def test_global_styles_use_streamlit_html():
    """Ensure the Revenue OS stylesheet uses the HTML renderer."""
    from unittest.mock import Mock

    from portfolio.theme import GLOBAL_CSS, apply_global_styles

    streamlit_mock = Mock()

    apply_global_styles(streamlit_mock)

    streamlit_mock.html.assert_called_once_with(GLOBAL_CSS)
    streamlit_mock.markdown.assert_not_called()