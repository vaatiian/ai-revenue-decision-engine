
"""
Public Streamlit startup error-handling acceptance tests.

Verifies that a curated-data validation failure:
- Displays a safe public message
- Does not expose private exception details
- Does not render navigation or customer metrics
- Stops before displaying the portfolio

The test clears Streamlit's data cache to remain independent
of earlier AppTest executions in the same pytest process.

No real portfolio datasets are modified.
"""

from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"

SAFE_MESSAGE = (
    "The portfolio demonstration is temporarily unavailable "
    "because its curated data could not be validated."
)

PRIVATE_DETAIL = "PRIVATE_INTERNAL_DATA_PATH_SENTINEL"


def test_invalid_portfolio_data_fails_closed_without_leaking_details():
    # Earlier interactive tests may have populated the Streamlit cache.
    # Clear it so the injected loader failure is actually exercised.
    st.cache_data.clear()

    try:
        with patch(
            "portfolio.data_loader.load_portfolio_bundle",
            side_effect=FileNotFoundError(PRIVATE_DETAIL),
        ):
            app = AppTest.from_file(
                APP_PATH,
                default_timeout=40,
            ).run()

        assert len(app.exception) == 0

        errors = [
            element.value
            for element in app.error
        ]

        assert SAFE_MESSAGE in errors

        visible_messages = "\n".join(
            str(element.value)
            for element in app
            if hasattr(element, "value")
        )

        assert PRIVATE_DETAIL not in visible_messages

        assert len(app.button) == 0
        assert len(app.metric) == 0

    finally:
        # Prevent the test from leaving cached state that could
        # influence subsequent application tests.
        st.cache_data.clear()
