"""
Visual design system for the AI Revenue Decision Engine portfolio.

The interface intentionally avoids the default "notebook dashboard"
look. The design language is enterprise decision intelligence:
graphite surfaces, warm white typography, restrained cyan,
governance green, conditional amber, and risk red.
"""

from __future__ import annotations


COLORS = {
    "background":
        "#07100F",

    "background_elevated":
        "#0B1614",

    "panel":
        "#101D1B",

    "panel_soft":
        "#132320",

    "border":
        "#243633",

    "text_primary":
        "#F4F2EC",

    "text_secondary":
        "#AAB7B4",

    "text_muted":
        "#71817D",

    "accent":
        "#52D6FF",

    "accent_soft":
        "#17333A",

    "go":
        "#4DE3A3",

    "go_soft":
        "#12372A",

    "conditional":
        "#FFBD66",

    "conditional_soft":
        "#3D2A10",

    "risk":
        "#FF7070",

    "risk_soft":
        "#3B1919",

    "white":
        "#FFFFFF",
}


GLOBAL_CSS = """
<style>
:root {
    --bg: #07100F;
    --bg-elevated: #0B1614;
    --panel: #101D1B;
    --panel-soft: #132320;
    --border: #243633;
    --text: #F4F2EC;
    --text-secondary: #AAB7B4;
    --muted: #71817D;
    --accent: #52D6FF;
    --go: #4DE3A3;
    --conditional: #FFBD66;
    --risk: #FF7070;
}

html,
body,
[class*="css"] {
    font-family:
        Inter,
        ui-sans-serif,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 78% -10%,
            rgba(82, 214, 255, 0.08),
            transparent 32%
        ),
        #07100F;
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(7, 16, 15, 0.92);
    border-bottom: 1px solid rgba(36, 54, 51, 0.65);
}

[data-testid="stSidebar"] {
    background: #091311;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary);
}

.block-container {
    max-width: 1480px;
    padding-top: 2.8rem;
    padding-bottom: 5rem;
}

h1,
h2,
h3,
h4 {
    color: var(--text) !important;
    letter-spacing: -0.035em;
}

p,
li,
label {
    color: var(--text-secondary);
}

hr {
    border-color: var(--border) !important;
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 1rem 1.1rem;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary);
}

[data-testid="stMetricValue"] {
    color: var(--text);
}

[data-testid="stTabs"] button {
    color: var(--text-secondary);
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
}

[data-testid="stPlotlyChart"],
[data-testid="stVegaLiteChart"] {
    background: transparent;
}

button[kind="secondary"],
button[kind="primary"] {
    border-radius: 2px !important;
}

div[data-baseweb="select"] > div {
    background-color: var(--panel);
    border-color: var(--border);
}

.portfolio-eyebrow {
    color: var(--accent);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.portfolio-hero {
    padding:
        2.6rem
        2.7rem
        2.5rem
        2.7rem;
    background:
        linear-gradient(
            135deg,
            rgba(82, 214, 255, 0.085),
            transparent 42%
        ),
        var(--bg-elevated);
    border:
        1px solid
        var(--border);
    position: relative;
    overflow: hidden;
}

.portfolio-hero::after {
    content: "";
    position: absolute;
    right: -140px;
    top: -180px;
    width: 390px;
    height: 390px;
    border:
        1px solid
        rgba(82, 214, 255, 0.12);
    transform: rotate(33deg);
}

.portfolio-hero h1 {
    margin: 0;
    max-width: 1040px;
    font-size: clamp(2.7rem, 5.8vw, 5.6rem);
    line-height: 0.96;
    letter-spacing: -0.06em;
}

.portfolio-hero-copy {
    margin-top: 1.45rem;
    max-width: 790px;
    color: var(--text-secondary);
    font-size: 1.06rem;
    line-height: 1.7;
}

.portfolio-hero-tagline {
    color: var(--text);
    font-weight: 600;
}

.portfolio-proof-grid {
    display: grid;
    grid-template-columns:
        repeat(4, minmax(0, 1fr));
    gap: 0.8rem;
    margin-top: 1rem;
    margin-bottom: 2rem;
}

.portfolio-proof-card {
    min-height: 145px;
    padding: 1.15rem 1.2rem;
    background: var(--panel);
    border: 1px solid var(--border);
}

.portfolio-proof-value {
    color: var(--text);
    font-size: 2.25rem;
    font-weight: 750;
    letter-spacing: -0.055em;
    line-height: 1;
}

.portfolio-proof-label {
    color: var(--text-secondary);
    font-size: 0.84rem;
    line-height: 1.45;
    margin-top: 0.7rem;
}

.portfolio-proof-caption {
    color: var(--muted);
    font-size: 0.7rem;
    margin-top: 0.5rem;
}

.portfolio-section-kicker {
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.35rem;
}

.portfolio-section-title {
    color: var(--text);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.045em;
    margin: 0;
}

.portfolio-section-copy {
    color: var(--text-secondary);
    max-width: 760px;
    line-height: 1.65;
    margin-top: 0.65rem;
    margin-bottom: 1.25rem;
}

.portfolio-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 1.3rem;
    height: 100%;
}

.portfolio-panel-title {
    color: var(--text);
    font-size: 1rem;
    font-weight: 650;
    margin-bottom: 0.55rem;
}

.portfolio-panel-copy {
    color: var(--text-secondary);
    font-size: 0.84rem;
    line-height: 1.55;
}

.portfolio-system-card {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 1rem;
    min-height: 185px;
}

.portfolio-system-name {
    color: var(--text);
    font-weight: 650;
    font-size: 0.98rem;
}

.portfolio-system-meta {
    color: var(--muted);
    font-size: 0.75rem;
    margin-top: 0.35rem;
    line-height: 1.5;
}

.badge {
    display: inline-block;
    padding: 0.23rem 0.48rem;
    margin-top: 0.8rem;
    font-size: 0.67rem;
    font-weight: 750;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    border: 1px solid;
}

.badge-go {
    color: var(--go);
    border-color: rgba(77, 227, 163, 0.35);
    background: rgba(77, 227, 163, 0.07);
}

.badge-conditional {
    color: var(--conditional);
    border-color: rgba(255, 189, 102, 0.36);
    background: rgba(255, 189, 102, 0.07);
}

.badge-neutral {
    color: var(--text-secondary);
    border-color: var(--border);
}

.portfolio-flow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.48rem;
    margin-top: 1rem;
}

.portfolio-flow-node {
    padding: 0.56rem 0.72rem;
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--text);
    font-size: 0.76rem;
    font-weight: 600;
}

.portfolio-flow-arrow {
    color: var(--accent);
    font-size: 0.88rem;
}

.portfolio-governance-note {
    margin-top: 1rem;
    padding: 1rem 1.1rem;
    background: rgba(255, 189, 102, 0.055);
    border-left: 2px solid var(--conditional);
    color: var(--text-secondary);
    font-size: 0.82rem;
    line-height: 1.6;
}

.portfolio-footer {
    margin-top: 4rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.73rem;
}

@media (max-width: 1100px) {
    .portfolio-proof-grid {
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 680px) {
    .block-container {
        padding-top: 1.5rem;
    }

    .portfolio-hero {
        padding: 1.6rem;
    }

    .portfolio-proof-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def apply_global_styles(
    st_module,
) -> None:
    """
    Apply the portfolio visual system to a Streamlit application.
    """

    st_module.markdown(
        GLOBAL_CSS,
        unsafe_allow_html=True,
    )