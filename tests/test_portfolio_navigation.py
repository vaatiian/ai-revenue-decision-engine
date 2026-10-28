
"""
Application navigation contract tests.

Protects:
- All eleven portfolio routes
- Grouped sidebar hierarchy
- Session-state navigation persistence
- Active and inactive button styling
- Route dispatch completeness

Tests inspect the application AST without executing Streamlit
or loading private production artifacts.
"""

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"

EXPECTED_GROUPS = {
    "COMMAND CENTRE": [
        "Overview",
        "Revenue Command Centre",
        "Customer Decision Explorer",
    ],
    "AI INTELLIGENCE": [
        "Interactive Decision Demo",
        "Decision Intelligence",
        "ML / Model Explorer",
    ],
    "GOVERNANCE & ENGINEERING": [
        "Governance Dashboard",
        "Production Architecture",
        "Architecture Visuals",
        "Testing & Reliability",
        "Production Evidence",
    ],
}

EXPECTED_ROUTES = [
    label
    for labels in EXPECTED_GROUPS.values()
    for label in labels
]


def load_app_ast():
    source = APP_PATH.read_text(encoding="utf-8")
    return ast.parse(source)


def get_function(name):
    tree = load_app_ast()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node

    raise AssertionError(f"Missing application function: {name}")


def get_navigation_groups():
    function = get_function("render_sidebar")

    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == "navigation_groups"
                for target in node.targets
            ):
                return ast.literal_eval(node.value)

    raise AssertionError("Navigation groups not found")


def get_dispatched_routes():
    function = get_function("main")
    routes = []

    for node in ast.walk(function):
        if not isinstance(node, ast.Compare):
            continue

        if not isinstance(node.left, ast.Name):
            continue

        if node.left.id != "page":
            continue

        if len(node.ops) != 1 or not isinstance(
            node.ops[0], ast.Eq
        ):
            continue

        if len(node.comparators) != 1:
            continue

        value = node.comparators[0]

        if isinstance(value, ast.Constant) and isinstance(
            value.value, str
        ):
            routes.append(value.value)

    return routes


def test_navigation_contains_three_groups():
    groups = get_navigation_groups()

    assert len(groups) == 3
    assert [group[0] for group in groups] == list(
        EXPECTED_GROUPS
    )


@pytest.mark.parametrize(
    "group_label,expected_labels",
    list(EXPECTED_GROUPS.items()),
)
def test_navigation_group_contains_expected_pages(
    group_label,
    expected_labels,
):
    groups = dict(get_navigation_groups())

    actual_labels = [
        label
        for label, _icon in groups[group_label]
    ]

    assert actual_labels == expected_labels


def test_navigation_has_eleven_unique_pages():
    groups = get_navigation_groups()

    pages = [
        label
        for _, entries in groups
        for label, _ in entries
    ]

    assert len(pages) == 11
    assert len(set(pages)) == 11
    assert pages == EXPECTED_ROUTES


def test_navigation_uses_material_icons():
    groups = get_navigation_groups()

    for _, entries in groups:
        for label, icon in entries:
            assert label
            assert icon.startswith(":material/")
            assert icon.endswith(":")


def test_all_navigation_pages_have_dispatch_routes():
    routes = get_dispatched_routes()

    assert len(routes) == 11
    assert len(set(routes)) == 11
    assert set(routes) == set(EXPECTED_ROUTES)


@pytest.mark.parametrize("page", EXPECTED_ROUTES)
def test_each_page_is_dispatched(page):
    assert page in get_dispatched_routes()


def test_sidebar_uses_session_state():
    function = get_function("render_sidebar")
    source = ast.unparse(function)

    assert "portfolio_active_page" in source
    assert "st.session_state" in source


def test_sidebar_defaults_to_overview():
    function = get_function("render_sidebar")
    source = ast.unparse(function)

    assert "'Overview'" in source
    assert "portfolio_active_page" in source


def test_sidebar_registers_navigation_callback():
    function = get_function("render_sidebar")
    source = ast.unparse(function)

    assert "on_click=navigate_to" in source
    assert "args=(label,)" in source


def test_sidebar_distinguishes_active_buttons():
    function = get_function("render_sidebar")
    source = ast.unparse(function)

    assert "'primary' if active else 'tertiary'" in source


def test_navigation_does_not_use_legacy_radio():
    function = get_function("render_sidebar")
    source = ast.unparse(function)

    assert "st.sidebar.radio" not in source
