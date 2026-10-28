import ast
from pathlib import Path

from portfolio.model_explorer_page import (
    _display_name,
    _format_metric_value,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[
    1
]


def test_metric_display_name():
    assert (
        _display_name(
            "stage_a_pr_auc"
        )
        ==
        "Stage A Pr Auc"
    )


def test_nested_metric_display_name():
    assert (
        _display_name(
            "approx_horizon_wape.4"
        )
        ==
        "Approx Horizon Wape · 4"
    )


def test_metric_float_format():
    assert (
        _format_metric_value(
            0.822148
        )
        ==
        "0.822148"
    )


def test_metric_format_removes_unnecessary_trailing_zeroes():
    assert (
        _format_metric_value(
            0.560000
        )
        ==
        "0.56"
    )


def test_page_file_parses():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "model_explorer_page.py"
    ).read_text(
        encoding="utf-8"
    )

    ast.parse(
        source
    )


def test_page_uses_public_registry_only():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "model_explorer_page.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        'bundle[\n        "model_registry"\n    ]'
        in
        source
    )


def test_page_does_not_load_model_binaries():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "model_explorer_page.py"
    ).read_text(
        encoding="utf-8"
    )

    forbidden = [
        "joblib.load",
        "pickle.load",
        "artifacts/models",
        "cleaned_online_retail.parquet",
    ]

    for token in forbidden:

        assert token not in source


def test_page_explicitly_preserves_missing_metrics():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "model_explorer_page.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "No governed evaluation metrics are exposed"
        in
        source
    )