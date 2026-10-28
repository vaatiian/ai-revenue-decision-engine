
"""
Revenue OS public deployment configuration tests.

Verify that the dedicated Streamlit deployment entry point
uses the existing portfolio application and keeps production
batch operations separate from public hosting.
"""

import ast
from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINT = ROOT / "streamlit_app" / "app.py"
PORTFOLIO_REQUIREMENTS = ROOT / "streamlit_app" / "requirements.txt"
SOURCE_REQUIREMENTS = ROOT / "requirements-portfolio.txt"


def test_deployment_entry_point_exists_and_parses():
    assert ENTRY_POINT.is_file()
    ast.parse(ENTRY_POINT.read_text(encoding="utf-8"))


def test_deployment_entry_point_imports_existing_app():
    tree = ast.parse(ENTRY_POINT.read_text(encoding="utf-8"))

    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "app"
    ]

    assert len(imports) == 1
    assert any(
        alias.name == "main"
        for alias in imports[0].names
    )


def test_deployment_entry_point_invokes_main():
    source = ENTRY_POINT.read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":' in source
    assert "main()" in source


def test_deployment_requirements_match_portfolio_specification():
    assert PORTFOLIO_REQUIREMENTS.is_file()
    assert SOURCE_REQUIREMENTS.is_file()

    assert (
        PORTFOLIO_REQUIREMENTS.read_text(encoding="utf-8")
        == SOURCE_REQUIREMENTS.read_text(encoding="utf-8")
    )


def test_deployment_requirements_are_portfolio_only():
    lines = PORTFOLIO_REQUIREMENTS.read_text(
        encoding="utf-8"
    ).splitlines()

    requirements = [
        Requirement(line.strip())
        for line in lines
        if line.strip()
        and not line.lstrip().startswith("#")
    ]

    names = {
        requirement.name.lower()
        for requirement in requirements
    }

    assert names == {
        "streamlit",
        "altair",
        "pandas",
        "pyarrow",
        "numpy",
    }


def test_public_entry_point_does_not_run_production_cli():
    source = ENTRY_POINT.read_text(encoding="utf-8")

    assert "src.cli" not in source
    assert "subprocess" not in source
    assert "joblib.load" not in source
    assert "customer-batch" not in source


def test_local_portfolio_environment_is_gitignored():
    gitignore = (ROOT / ".gitignore").read_text(
        encoding="utf-8"
    )

    assert ".venv-portfolio/" in gitignore
