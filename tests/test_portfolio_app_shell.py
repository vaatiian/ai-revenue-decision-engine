import ast
from pathlib import Path

from portfolio.data_loader import (
    load_customer_decisions,
    load_demand_forecasts,
    load_model_registry,
    load_portfolio_bundle,
    load_portfolio_summary,
    load_repeat_purchase,
)

from portfolio.theme import (
    COLORS,
    GLOBAL_CSS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_customer_data_is_public_safe():
    frame = load_customer_decisions()

    assert len(
        frame
    ) == 5220

    assert "portfolio_customer_id" in frame.columns

    assert "customer" not in frame.columns

    assert "recommended_product" not in frame.columns

    assert frame[
        "portfolio_customer_id"
    ].str.match(
        r"^CUST-\d{5}$"
    ).all()


def test_repeat_purchase_data_is_public_safe():
    frame = load_repeat_purchase()

    assert len(
        frame
    ) == 4269

    assert "customer" not in frame.columns

    assert "first_invoice" not in frame.columns

    assert frame[
        "portfolio_first_order_id"
    ].str.match(
        r"^ORD-\d{6}$"
    ).all()


def test_demand_data_contains_governed_horizons():
    frame = load_demand_forecasts()

    assert frame[
        "horizon_weeks"
    ].tolist() == [
        1,
        2,
        4,
    ]

    assert set(
        frame[
            "governance_decision"
        ]
    ) == {
        "GO",
    }


def test_model_registry_contains_six_systems():
    registry = load_model_registry()

    assert registry[
        "system_count"
    ] == 6

    ids = {
        system[
            "system_id"
        ]
        for system in registry[
            "systems"
        ]
    }

    assert ids == {
        "reactivation",
        "revenue_leakage",
        "future_customer_value",
        "repeat_purchase",
        "recommendation",
        "demand_forecast",
    }


def test_portfolio_summary_contains_verified_evidence():
    summary = load_portfolio_summary()

    evidence = summary[
        "verified_evidence"
    ]

    assert evidence[
        "automated_tests_passing"
    ] == 1711

    assert evidence[
        "customer_decisions_generated"
    ] == 5220

    assert evidence[
        "repeat_purchase_scores_generated"
    ] == 4269

    assert evidence[
        "production_intelligence_systems"
    ] == 6

    assert evidence[
        "production_scoring_systems_available"
    ] == 6

    assert evidence[
        "production_scoring_systems_total"
    ] == 6

    assert evidence[
        "governed_demand_horizons"
    ] == [
        1,
        2,
        4,
    ]


def test_portfolio_bundle_loads_all_public_layers():
    bundle = load_portfolio_bundle()

    assert set(
        bundle
    ) == {
        "customer_decisions",
        "repeat_purchase",
        "demand_forecasts",
        "model_registry",
        "summary",
    }


def test_visual_system_contains_locked_brand_tokens():
    assert COLORS[
        "background"
    ] == "#07100F"

    assert COLORS[
        "accent"
    ] == "#52D6FF"

    assert COLORS[
        "go"
    ] == "#4DE3A3"

    assert COLORS[
        "conditional"
    ] == "#FFBD66"

    assert COLORS[
        "risk"
    ] == "#FF7070"


def test_visual_system_contains_core_components():
    required = [
        ".portfolio-hero",
        ".portfolio-proof-grid",
        ".portfolio-system-card",
        ".portfolio-flow",
        ".badge-go",
        ".badge-conditional",
    ]

    for token in required:
        assert token in GLOBAL_CSS


def test_app_file_parses_as_python():
    app_path = (
        PROJECT_ROOT
        /
        "app.py"
    )

    source = app_path.read_text(
        encoding="utf-8"
    )

    ast.parse(
        source
    )


def test_app_uses_curated_portfolio_loader():
    app_source = (
        PROJECT_ROOT
        /
        "app.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "load_portfolio_bundle"
        in
        app_source
    )

    assert (
        "cleaned_online_retail.parquet"
        not in
        app_source
    )

    assert (
        "online_retail_II.xlsx"
        not in
        app_source
    )


def test_app_does_not_load_model_binaries():
    app_source = (
        PROJECT_ROOT
        /
        "app.py"
    ).read_text(
        encoding="utf-8"
    )

    assert ".joblib" not in app_source

    assert "joblib.load" not in app_source