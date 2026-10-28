
"""
Premium corporate visual system for the AI Revenue Decision Engine.

Design direction:
- Midnight navy and graphite foundations
- Electric teal for intelligence and interaction
- Emerald for governed positive / GO states
- Champagne amber for commercial emphasis and conditional states
- Coral for meaningful revenue risk
- Restrained borders, clear typography, and sharp geometry

Presentation only. No business rules or model outputs are changed.
"""

from __future__ import annotations


COLORS = {
    "background": "#101B27",
    "background_elevated": "#142330",
    "panel": "#192C39",
    "panel_soft": "#203542",
    "border": "#344956",
    "text_primary": "#F5F8FA",
    "text_secondary": "#B1C3CD",
    "text_muted": "#839BA9",
    "accent": "#13B8A6",
    "accent_soft": "#183D44",
    "go": "#46D5A0",
    "go_soft": "#173F35",
    "conditional": "#E6B85C",
    "conditional_soft": "#403521",
    "risk": "#E66E78",
    "risk_soft": "#422A36",
    "white": "#FFFFFF",
}


GLOBAL_CSS = """
<style>
:root {
    --bg: #101B27;
    --bg-elevated: #142330;
    --panel: #192C39;
    --panel-soft: #203542;
    --border: #344956;
    --text: #F5F8FA;
    --text-secondary: #B1C3CD;
    --muted: #839BA9;
    --accent: #13B8A6;
    --go: #46D5A0;
    --conditional: #E6B85C;
    --risk: #E66E78;
    --sidebar-bg: #101D2A;
}

html,
body,
[class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(16, 27, 39, 0.96);
    border-bottom: 1px solid var(--border);
}

[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary);
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.28rem;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 2px;
    transition:
        background-color 140ms ease,
        border-color 140ms ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: var(--panel-soft);
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: var(--accent-soft);
    border-left: 3px solid var(--accent);
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 4rem;
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
    padding: 1.1rem 1.25rem;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary);
}

[data-testid="stMetricValue"] {
    color: var(--text);
    font-weight: 650;
    letter-spacing: -0.045em;
}

[data-testid="stTabs"] button {
    color: var(--text-secondary);
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent);
    border-bottom-color: var(--accent);
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

/* Shared portfolio typography */

.portfolio-eyebrow {
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Executive hero */

.portfolio-hero {
    position: relative;
    overflow: hidden;
    padding: 2.5rem 2.65rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
}

.portfolio-hero::after {
    content: "";
    position: absolute;
    right: -100px;
    top: -100px;
    width: 290px;
    height: 290px;
    border: 1px solid rgba(19, 184, 166, 0.17);
    transform: rotate(35deg);
    pointer-events: none;
}

.portfolio-hero h1 {
    margin: 0;
    max-width: 1050px;
    font-size: clamp(2.5rem, 4.6vw, 4.5rem);
    line-height: 1.06;
    letter-spacing: -0.055em;
    font-weight: 700;
}

.portfolio-hero-copy {
    margin-top: 1.2rem;
    max-width: 800px;
    color: var(--text-secondary);
    font-size: 1rem;
    line-height: 1.7;
}

.portfolio-hero-tagline {
    color: var(--text);
    font-weight: 650;
}

/* Evidence cards */

.portfolio-proof-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1.1rem;
    margin-bottom: 2rem;
}

.portfolio-proof-card {
    min-height: 138px;
    padding: 1.2rem 1.3rem;
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid transparent;
}

.portfolio-proof-card:nth-child(3n + 1) {
    border-top-color: var(--accent);
}

.portfolio-proof-card:nth-child(3n + 2) {
    border-top-color: var(--conditional);
}

.portfolio-proof-card:nth-child(3n + 3) {
    border-top-color: #64879D;
}

.portfolio-proof-value {
    color: var(--text);
    font-size: 2.25rem;
    font-weight: 720;
    letter-spacing: -0.055em;
    line-height: 1.05;
    font-variant-numeric: tabular-nums;
}

.portfolio-proof-label {
    color: var(--text-secondary);
    font-size: 0.84rem;
    font-weight: 600;
    line-height: 1.4;
    margin-top: 0.65rem;
}

.portfolio-proof-caption {
    color: var(--muted);
    font-size: 0.72rem;
    line-height: 1.45;
    margin-top: 0.4rem;
}

/* Sections */

.portfolio-section-kicker {
    color: var(--accent);
    font-size: 0.71rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.4rem;
}

.portfolio-section-title {
    color: var(--text);
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1.2;
    margin: 0;
}

.portfolio-section-copy {
    color: var(--text-secondary);
    max-width: 830px;
    line-height: 1.65;
    margin-top: 0.65rem;
    margin-bottom: 1.3rem;
}

/* Panels */

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
    line-height: 1.6;
}

/* Production intelligence cards */

.portfolio-system-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    padding: 1.1rem;
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
    line-height: 1.55;
}

/* Governance badges */

.badge {
    display: inline-block;
    padding: 0.25rem 0.52rem;
    margin-top: 0.8rem;
    font-size: 0.67rem;
    font-weight: 750;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    border: 1px solid;
}

.badge-go {
    color: var(--go);
    border-color: rgba(70, 213, 160, 0.38);
    background: rgba(70, 213, 160, 0.08);
}

.badge-conditional {
    color: var(--conditional);
    border-color: rgba(230, 184, 92, 0.40);
    background: rgba(230, 184, 92, 0.08);
}

.badge-neutral {
    color: var(--text-secondary);
    border-color: var(--border);
    background: var(--panel-soft);
}

/* Architecture flow */

.portfolio-flow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.48rem;
    margin-top: 1rem;
}

.portfolio-flow-node {
    padding: 0.6rem 0.76rem;
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--text);
    font-size: 0.77rem;
    font-weight: 600;
}

.portfolio-flow-arrow {
    color: var(--accent);
    font-size: 0.9rem;
}

/* Governance disclosure — accessible, high-contrast */

.portfolio-governance-note {
    display: block;
    margin-top: 1.25rem;
    margin-bottom: 1.25rem;
    padding: 1.05rem 1.3rem;
    background: #263327;
    border: 1px solid #675836;
    border-left: 4px solid #E6B85C;
    border-radius: 0;
    color: #F5F8FA !important;
    font-size: 0.88rem;
    font-weight: 400;
    line-height: 1.75;
    letter-spacing: 0.005em;
}

.portfolio-governance-note strong {
    color: #E6B85C !important;
    font-weight: 750;
}

.portfolio-governance-note p,
.portfolio-governance-note span,
.portfolio-governance-note a {
    color: #F5F8FA !important;
}

.portfolio-governance-note a {
    text-decoration: underline;
    text-decoration-color: #E6B85C;
}

/* Premium corporate sidebar */

.revenue-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.5rem 0 1.1rem;
}

.revenue-sidebar-symbol {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 42px;
    height: 42px;
    color: #101B27;
    background: var(--accent);
    border: 1px solid var(--accent);
}

.revenue-sidebar-brand-copy {
    min-width: 0;
}

.revenue-sidebar-product {
    color: var(--text);
    font-size: 1.06rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: 0.045em;
}

.revenue-sidebar-subtitle {
    color: var(--muted);
    font-size: 0.60rem;
    font-weight: 650;
    letter-spacing: 0.085em;
    margin-top: 0.3rem;
}

.revenue-sidebar-divider {
    height: 1px;
    background: var(--border);
    margin-bottom: 1.2rem;
}

.revenue-sidebar-kicker {
    color: var(--muted);
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    margin-bottom: 0.7rem;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.22rem;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.52rem 0.65rem;
    border: 1px solid transparent;
    border-left: 3px solid transparent;
    min-height: 42px;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: var(--panel-soft);
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: var(--accent-soft);
    border-left-color: var(--accent);
}

[data-testid="stSidebar"] [role="radiogroup"] label p {
    font-size: 0.86rem;
    font-weight: 500;
    color: var(--text-secondary);
}

[data-testid="stSidebar"]
[role="radiogroup"]
label:has(input:checked) p {
    color: var(--text);
    font-weight: 650;
}

.revenue-sidebar-footer {
    margin-top: 1.6rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--border);
}

.revenue-sidebar-footer-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--go);
    font-size: 0.65rem;
    font-weight: 750;
    letter-spacing: 0.1em;
}

.revenue-sidebar-status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: var(--go);
    border-radius: 50%;
}

.revenue-sidebar-footer-copy {
    margin-top: 0.65rem;
    color: var(--text-secondary);
    font-size: 0.72rem;
    line-height: 1.55;
}

.revenue-sidebar-version {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    margin-top: 1.35rem;
    color: var(--muted);
    font-size: 0.57rem;
    letter-spacing: 0.045em;
}

.revenue-sidebar-version span {
    color: var(--accent);
    white-space: nowrap;
}

/* Grouped enterprise navigation */

.revenue-nav-heading {
    margin-top: 1.25rem;
    margin-bottom: 0.55rem;
    padding-left: 0.45rem;
    color: var(--muted);
    font-size: 0.65rem;
    font-weight: 750;
    letter-spacing: 0.13em;
}

[data-testid="stSidebar"] div.stButton {
    margin-bottom: 0.12rem;
}

[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    min-height: 39px;
    justify-content: flex-start;
    gap: 0.72rem;
    padding: 0.55rem 0.7rem;
    border-radius: 2px !important;
    border: 1px solid transparent;
    border-left: 3px solid transparent;
    box-shadow: none;
    font-size: 0.82rem;
    font-weight: 500;
    text-align: left;
    transition: background-color 130ms ease;
}

[data-testid="stSidebar"] div.stButton > button[kind="tertiary"] {
    background: transparent;
    color: var(--text-secondary);
}

[data-testid="stSidebar"] div.stButton > button[kind="tertiary"]:hover {
    background: var(--panel-soft);
    color: var(--text);
}

[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: var(--accent-soft);
    border-left-color: var(--accent);
    color: var(--text);
    font-weight: 700;
}

[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
    background: var(--accent-soft);
    color: var(--text);
}

[data-testid="stSidebar"] div.stButton > button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

[data-testid="stSidebar"] div.stButton > button svg {
    flex-shrink: 0;
}

/* Executive Overview — premium hero refinement */

.portfolio-hero {
    padding: 2rem 2.35rem 1.8rem;
    border-top: 2px solid var(--accent);
}

.portfolio-hero::after {
    width: 220px;
    height: 220px;
    right: -125px;
    top: -125px;
    opacity: 0.55;
}

.portfolio-hero-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.3rem;
}

.portfolio-hero-topline .portfolio-eyebrow {
    margin-bottom: 0;
}

.portfolio-hero-context {
    color: var(--muted);
    font-size: 0.67rem;
    font-weight: 650;
    letter-spacing: 0.12em;
}

.portfolio-hero h1 {
    max-width: 920px;
    font-size: clamp(2rem, 3.1vw, 3.15rem);
    line-height: 1.12;
    letter-spacing: -0.045em;
}

.portfolio-hero-highlight {
    color: var(--accent);
}

.portfolio-hero-copy {
    max-width: 800px;
    margin-top: 0.85rem;
    font-size: 0.94rem;
    line-height: 1.6;
}

.portfolio-hero-flow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.7rem;
    margin-top: 1.45rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
}

.portfolio-hero-step {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.55rem 0.7rem;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
    font-size: 0.67rem;
    font-weight: 650;
    letter-spacing: 0.055em;
}

.portfolio-hero-step-number {
    color: var(--accent);
    font-size: 0.66rem;
    font-variant-numeric: tabular-nums;
}

.portfolio-hero-flow-arrow {
    color: var(--accent);
    font-size: 1rem;
}

.portfolio-hero-disclosure {
    margin-top: 0.9rem;
    color: var(--muted);
    font-size: 0.69rem;
    line-height: 1.5;
}

@media (max-width: 680px) {
    .portfolio-hero {
        padding: 1.35rem;
    }

    .portfolio-hero h1 {
        font-size: 2rem;
    }

    .portfolio-hero-flow {
        gap: 0.4rem;
    }

    .portfolio-hero-step {
        font-size: 0.61rem;
    }

    .portfolio-hero-flow-arrow {
        display: none;
    }
}

/* Executive KPI hierarchy */

.portfolio-evidence-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-top: 0.3rem;
    margin-bottom: 0.8rem;
    flex-wrap: wrap;
}

.portfolio-evidence-heading .portfolio-section-kicker {
    margin-bottom: 0;
}

.portfolio-evidence-context {
    color: var(--muted);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.075em;
}

.portfolio-proof-primary {
    margin-top: 0;
    margin-bottom: 0.75rem;
}

.portfolio-proof-primary .portfolio-proof-card {
    min-height: 145px;
    padding: 1.3rem 1.4rem;
    border-top: 3px solid transparent;
}

.portfolio-proof-primary .proof-commercial {
    border-top-color: var(--accent);
}

.portfolio-proof-primary .proof-intelligence {
    border-top-color: var(--go);
}

.portfolio-proof-primary .proof-reliability {
    border-top-color: var(--conditional);
}

.portfolio-proof-primary .portfolio-proof-value {
    font-size: clamp(2rem, 2.8vw, 2.9rem);
}

.portfolio-proof-secondary {
    margin-top: 0;
    margin-bottom: 1.7rem;
}

.portfolio-proof-secondary .portfolio-proof-card {
    min-height: 115px;
    padding: 1rem 1.2rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
}

.portfolio-proof-secondary .portfolio-proof-value {
    font-size: 1.65rem;
}

.portfolio-proof-secondary .portfolio-proof-label {
    margin-top: 0.45rem;
}

.portfolio-proof-secondary .portfolio-proof-caption {
    font-size: 0.69rem;
}

@media (max-width: 680px) {
    .portfolio-evidence-context {
        display: none;
    }
}

/* Responsive */

@media (max-width: 1100px) {
    .portfolio-proof-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 680px) {
    .block-container {
        padding-top: 1.3rem;
    }

    .portfolio-hero {
        padding: 1.55rem;
    }

    .portfolio-hero h1 {
        font-size: 2.35rem;
    }

    .portfolio-proof-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def apply_global_styles(st_module) -> None:
    """Apply the shared portfolio design system."""

    st_module.html(GLOBAL_CSS)