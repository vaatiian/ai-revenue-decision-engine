import ast
from pathlib import Path

from portfolio.decision_demo import (
    demo_presets,
    run_decision_demo,
)

from portfolio.decision_demo_page import (
    _boolean_label,
    _percentage,
    _score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_percentage_format():
    assert (
        _percentage(
            0.856
        )
        ==
        "85.6%"
    )


def test_percentage_missing():
    assert (
        _percentage(
            None
        )
        ==
        "Not Available"
    )


def test_score_format():
    assert (
        _score(
            0.85642
        )
        ==
        "0.856"
    )


def test_boolean_labels():
    assert (
        _boolean_label(
            True
        )
        ==
        "Yes"
    )

    assert (
        _boolean_label(
            False
        )
        ==
        "No"
    )


def test_demo_page_file_parses():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "decision_demo_page.py"
    ).read_text(
        encoding="utf-8"
    )

    ast.parse(
        source
    )


def test_page_imports_real_demo_adapter():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "decision_demo_page.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "run_decision_demo"
        in
        source
    )


def test_page_does_not_reimplement_orchestration():
    source = (
        PROJECT_ROOT
        /
        "portfolio"
        /
        "decision_demo_page.py"
    ).read_text(
        encoding="utf-8"
    )

    forbidden = [
        "protect_eligible =",
        "grow_eligible =",
        "retain_eligible =",
        "reactivate_eligible =",
        "PRIORITY_TIER_THRESHOLDS",
        "PROTECT_PRIORITY_WEIGHTS",
        "RETAIN_PRIORITY_WEIGHTS",
        "REACTIVATE_PRIORITY_WEIGHTS",
        "GROW_PRIORITY_WEIGHTS",
    ]

    for token in forbidden:

        assert token not in source


def test_every_preset_executes():
    for inputs in demo_presets().values():

        result = run_decision_demo(
            inputs
        )

        assert (
            result[
                "primary_action"
            ]
            is not None
        )

        assert (
            result[
                "priority_tier"
            ]
            is not None
        )

        assert (
            result[
                "contact_intensity"
            ]
            is not None
        )

        assert (
            result[
                "experiment_requirement"
            ]
            is not None
        )