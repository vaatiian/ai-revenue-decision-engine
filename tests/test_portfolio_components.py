from portfolio.components import (
    render_hero,
)


class MarkdownCapture:
    """
    Minimal Streamlit markdown test double.
    """

    def __init__(
        self,
    ):
        self.calls = []

    def markdown(
        self,
        body,
        **kwargs,
    ):
        self.calls.append(
            (
                body,
                kwargs,
            )
        )


def test_hero_renders_html_without_markdown_code_indentation():
    fake_streamlit = MarkdownCapture()

    render_hero(
        fake_streamlit
    )

    assert len(
        fake_streamlit.calls
    ) == 1

    body, kwargs = (
        fake_streamlit.calls[
            0
        ]
    )

    assert kwargs[
        "unsafe_allow_html"
    ] is True

    assert body.startswith(
        '<div class="portfolio-hero">'
    )

    assert "<h1>" in body

    assert (
        "AI Revenue<br>Decision Engine"
        in
        body
    )

    assert "\n    <h1>" not in body

    assert body.endswith(
        "</div>"
    )