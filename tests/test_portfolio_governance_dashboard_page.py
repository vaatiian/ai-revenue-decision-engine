import ast
from pathlib import Path

from portfolio.data_loader import (
    load_model_registry,
    load_portfolio_summary,
)

from portfolio.governance_dashboard import (
    governance_surface_rows,
    system_governance_rows,
)

from portfolio.governance_dashboard_page import (
    _display_permission_matrix,
    _display_system_matrix,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[
    1
]


def test_system_matrix_has_six_rows():
    frame = _display_system_matrix(
        system_governance_rows(
            load_model_registry(),
            load_portfolio_summary(),
        )
    )

    assert len(
        frame
    ) == 6


def test_system_matrix_columns_are_stable():
    frame = _display_system_matrix(
        system_governance_rows(
            load_model_registry(),
            load_portfolio_summary(),
        )
    )

    assert list(
        frame.columns
    ) == [
        "System",
        "Runtime",
        "Governance",
        "Decision Engine",
        "Validation",
    ]


def test_permission_matrix_has_seven_rows():
    frame = _display_permission_matrix(
        governance_surface_rows(
            load_model_registry()
        )
    )

    assert len(
        frame
    ) == 7


def test_permission_matrix_columns_are_stable():
    frame = _display_permission_matrix(
        governance_surface_rows(
            load_model_registry()
        )
    )

    assert list(
        frame.columns
    ) == [
        "System",
        "Surface",
        "Decision",
        "Approved Use",
    ]


def test_page_file_parses():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "governance_dashboard_page.py"
    ).read_text(
        encoding="utf-8"
    )

    ast.parse(
        source
    )


def test_page_uses_curated_bundle_only():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "governance_dashboard_page.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        'bundle[\n        "model_registry"\n    ]'
        in
        source
    )

    assert (
        'bundle[\n        "summary"\n    ]'
        in
        source
    )


def test_page_does_not_load_private_model_artifacts():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "governance_dashboard_page.py"
    ).read_text(
        encoding="utf-8"
    )

    forbidden = [
        "joblib.load",
        "pickle.load",
        "cleaned_online_retail.parquet",
        "artifacts/models",
    ]

    for token in forbidden:

        assert token not in source


def test_page_preserves_missing_governance_evidence():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "governance_dashboard_page.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "A blank public field is not interpreted as unrestricted use."
        in
        source
    )