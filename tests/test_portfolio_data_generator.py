import json
from pathlib import Path

import pandas as pd
import pytest

from portfolio.generate_portfolio_data import (
    CUSTOMER_DECISIONS_PORTFOLIO_COLUMNS,
    CUSTOMER_DECISIONS_SOURCE_COLUMNS,
    DEMAND_FORECAST_PORTFOLIO_COLUMNS,
    DEMAND_FORECAST_SOURCE_COLUMNS,
    MODEL_MANIFESTS,
    REPEAT_PURCHASE_PORTFOLIO_COLUMNS,
    REPEAT_PURCHASE_SOURCE_COLUMNS,
    build_customer_decisions_portfolio,
    build_demand_forecasts_portfolio,
    build_repeat_purchase_portfolio,
    generate_portfolio_data,
)


DECISION_DATE = "2011-08-31"

SCORED_AT = "2026-10-27T19:40:15+00:00"


def make_customer_source():
    return pd.DataFrame(
        [
            {
                "customer": 100,
                "decision_date": DECISION_DATE,
                "customer_state": "Active",
                "days_since_last_order": 20.0,
                "customer_tenure_days": 300.0,
                "lifetime_orders": 5,
                "lifetime_revenue": 1000.0,
                "orders_prior_30d": 1.0,
                "orders_prior_90d": 2.0,
                "revenue_prior_90d": 400.0,
                "revenue_prior_180d": 800.0,
                "primary_action": "Protect",
                "primary_action_score": 0.8,
                "action_priority_percentile": 0.9,
                "priority_tier": "Priority 1",
                "contact_intensity": "High Touch",
                "business_action": "Protect customer value",
                "decision_reason": "High leakage exposure",
                "experiment_requirement": "Recommended",
                "leakage_probability": 0.7,
                "leakage_exposure_score": 140.0,
                "leakage_exposure_percentile": 0.9,
                "economic_priority_band": "P1",
                "future_active_probability": 0.8,
                "future_customer_value_90d": 300.0,
                "fcv_percentile": 0.8,
                "fcv_tier": "High Value",
                "reactivation_probability": None,
                "reactivation_prediction_date": None,
                "recommended_product_channel": "repeat",
                "recommended_product": "A",
                "repeat_top1_product": "A",
                "discovery_top1_product": "B",
            },
            {
                "customer": 200,
                "decision_date": DECISION_DATE,
                "customer_state": "Long Dormant",
                "days_since_last_order": 220.0,
                "customer_tenure_days": 500.0,
                "lifetime_orders": 2,
                "lifetime_revenue": 500.0,
                "orders_prior_30d": 0.0,
                "orders_prior_90d": 0.0,
                "revenue_prior_90d": 0.0,
                "revenue_prior_180d": 0.0,
                "primary_action": "Unscored ▒ Long Dormant",
                "primary_action_score": None,
                "action_priority_percentile": None,
                "priority_tier": "Not Prioritised",
                "contact_intensity": "No Model-Led Contact",
                "business_action": "Outside current model coverage",
                "decision_reason": "Outside model coverage",
                "experiment_requirement": "Not Applicable",
                "leakage_probability": None,
                "leakage_exposure_score": None,
                "leakage_exposure_percentile": None,
                "economic_priority_band": None,
                "future_active_probability": None,
                "future_customer_value_90d": None,
                "fcv_percentile": None,
                "fcv_tier": None,
                "reactivation_probability": 0.3,
                "reactivation_prediction_date": DECISION_DATE,
                "recommended_product_channel": None,
                "recommended_product": None,
                "repeat_top1_product": None,
                "discovery_top1_product": None,
            },
        ],
        columns=CUSTOMER_DECISIONS_SOURCE_COLUMNS,
    )


def make_repeat_source():
    return pd.DataFrame(
        [
            {
                "customer": 100,
                "first_invoice": 9001,
                "first_order_date": "2010-01-04 10:00:00",
                "repeat_purchase_score": 0.55,
                "model_name": "Logistic Regression",
                "model_version": "1.0.0",
                "model_type": "binary_classifier",
                "target_name": "repeat_within_90d",
                "governance_decision": "CONDITIONAL GO",
                "scored_at_utc": SCORED_AT,
            },
            {
                "customer": 300,
                "first_invoice": 9002,
                "first_order_date": "2010-01-05 11:00:00",
                "repeat_purchase_score": 0.40,
                "model_name": "Logistic Regression",
                "model_version": "1.0.0",
                "model_type": "binary_classifier",
                "target_name": "repeat_within_90d",
                "governance_decision": "CONDITIONAL GO",
                "scored_at_utc": SCORED_AT,
            },
        ],
        columns=REPEAT_PURCHASE_SOURCE_COLUMNS,
    )


def make_demand_source():
    return pd.DataFrame(
        [
            {
                "forecast_origin_week": "2011-11-21",
                "target_week": "2011-11-28",
                "horizon_weeks": 1,
                "seasonal_analogue_week": "2010-11-29",
                "seasonal_analogue_revenue": 100.0,
                "recent_four_week_average": 110.0,
                "forecast_revenue": 101.0,
                "seasonal_weight": 0.9,
                "recent_weight": 0.1,
                "model_name": "Seasonal Blend",
                "model_version": "1.0.0",
                "model_type": "deterministic_forecast",
                "target_name": "weekly_positive_merchandise_revenue",
                "governance_decision": "GO",
                "scored_at_utc": SCORED_AT,
            },
            {
                "forecast_origin_week": "2011-11-21",
                "target_week": "2011-12-05",
                "horizon_weeks": 2,
                "seasonal_analogue_week": "2010-12-06",
                "seasonal_analogue_revenue": 120.0,
                "recent_four_week_average": 110.0,
                "forecast_revenue": 119.0,
                "seasonal_weight": 0.9,
                "recent_weight": 0.1,
                "model_name": "Seasonal Blend",
                "model_version": "1.0.0",
                "model_type": "deterministic_forecast",
                "target_name": "weekly_positive_merchandise_revenue",
                "governance_decision": "GO",
                "scored_at_utc": SCORED_AT,
            },
            {
                "forecast_origin_week": "2011-11-21",
                "target_week": "2011-12-19",
                "horizon_weeks": 4,
                "seasonal_analogue_week": "2010-12-20",
                "seasonal_analogue_revenue": 80.0,
                "recent_four_week_average": 110.0,
                "forecast_revenue": 83.0,
                "seasonal_weight": 0.9,
                "recent_weight": 0.1,
                "model_name": "Seasonal Blend",
                "model_version": "1.0.0",
                "model_type": "deterministic_forecast",
                "target_name": "weekly_positive_merchandise_revenue",
                "governance_decision": "GO",
                "scored_at_utc": SCORED_AT,
            },
        ],
        columns=DEMAND_FORECAST_SOURCE_COLUMNS,
    )


def make_customer_mapping():
    return {
        "100": "CUST-00001",
        "200": "CUST-00002",
        "300": "CUST-00003",
    }


def make_product_mapping():
    return {
        "A": "PROD-00001",
        "B": "PROD-00002",
    }


def make_order_mapping():
    return {
        "9001": "ORD-000001",
        "9002": "ORD-000002",
    }


def write_manifest(
    root: Path,
    relative_path: Path,
    payload: dict,
):
    path = (
        root
        /
        relative_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )


def write_minimal_manifests(
    root: Path,
):
    for (
        system_id,
        relative_path,
    ) in MODEL_MANIFESTS.items():

        payload = {
            "model_name": f"{system_id} model",
            "model_version": "1.0.0",
            "model_type": "test",
            "target_name": f"{system_id}_target",
            "feature_count": 10,
            "algorithm": "TestAlgorithm",
        }

        if system_id in {
            "reactivation",
            "revenue_leakage",
            "future_customer_value",
        }:
            payload[
                "research_final_decision"
            ] = "GO"

            payload[
                "research_validation_design"
            ] = "chronological"

        if system_id == "reactivation":
            payload[
                "target_horizon_days"
            ] = 90

        if system_id in {
            "revenue_leakage",
            "future_customer_value",
        }:
            payload[
                "outcome_window_days"
            ] = 90

        if system_id == "future_customer_value":
            payload[
                "research_test_metrics"
            ] = {
                "stage_a_pr_auc":
                    0.82
            }

            payload[
                "stage_b_model_name"
            ] = "Historical Run Rate"

            payload[
                "complete_model_formula"
            ] = "probability * value"

        if system_id == "repeat_purchase":
            payload[
                "governance_decision"
            ] = "CONDITIONAL GO"

            payload[
                "prediction_horizon_days"
            ] = 90

            payload[
                "reported_final_test_metrics"
            ] = {
                "pr_auc":
                    0.50
            }

            payload[
                "recommended_use"
            ] = "controlled experimentation"

            payload[
                "prohibited_interpretations"
            ] = [
                "guaranteed probability",
            ]

        if system_id == "recommendation":
            payload[
                "outcome_window_days"
            ] = 90

            payload[
                "repeat_channel"
            ] = {
                "research_decision":
                    "GO",

                "approved_use":
                    "repeat recommendation",
            }

            payload[
                "discovery_channel"
            ] = {
                "research_decision":
                    "CONDITIONAL GO",

                "approved_use":
                    "discovery experimentation",

                "prohibited_interpretation":
                    "guaranteed purchase",
            }

            payload[
                "architecture"
            ] = {
                "repeat_channel":
                    "Repeat",

                "discovery_channel":
                    "Discovery",
            }

            payload[
                "final_holdout_performance"
            ] = {
                "repeat_hit_rate_at_10":
                    0.70
            }

        if system_id == "demand_forecast":
            payload[
                "forecast_horizons_weeks"
            ] = [
                1,
                2,
                4,
            ]

            payload[
                "governance"
            ] = {
                "decision":
                    "GO",

                "decision_description":
                    "GO for planning",
            }

            payload[
                "architecture"
            ] = {
                "formula":
                    "0.9 seasonal + 0.1 recent"
            }

            payload[
                "reported_historical_performance"
            ] = {
                "approx_horizon_wape": {
                    "1":
                        0.16,
                }
            }

            payload[
                "validation"
            ] = {
                "method":
                    "Temporal backtesting"
            }

            payload[
                "planning_ranges"
            ] = {
                "interpretation":
                    "Historical planning range"
            }

        write_manifest(
            root,
            relative_path,
            payload,
        )


def write_source_files(
    root: Path,
):
    customer_path = (
        root
        /
        "outputs/phase12_smoke/customer/"
        "customer-batch/2011-08-31/"
        "customer_decisions.csv"
    )

    repeat_path = (
        root
        /
        "outputs/phase12_smoke/repeat/"
        "repeat-purchase/2011-08-31/"
        "repeat_purchase_scores.csv"
    )

    demand_path = (
        root
        /
        "outputs/phase12_smoke/demand/"
        "demand/2011-11-21/"
        "demand_forecasts.csv"
    )

    customer_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    repeat_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    demand_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    make_customer_source().to_csv(
        customer_path,
        index=False,
    )

    make_repeat_source().to_csv(
        repeat_path,
        index=False,
    )

    make_demand_source().to_csv(
        demand_path,
        index=False,
    )


def test_customer_portfolio_anonymises_raw_identifiers():
    result = (
        build_customer_decisions_portfolio(
            source=make_customer_source(),
            customer_mapping=(
                make_customer_mapping()
            ),
            product_mapping=(
                make_product_mapping()
            ),
        )
    )

    assert tuple(
        result.columns
    ) == (
        CUSTOMER_DECISIONS_PORTFOLIO_COLUMNS
    )

    assert "customer" not in result.columns
    assert "recommended_product" not in result.columns
    assert "repeat_top1_product" not in result.columns
    assert "discovery_top1_product" not in result.columns

    assert result[
        "portfolio_customer_id"
    ].tolist() == [
        "CUST-00001",
        "CUST-00002",
    ]

    assert (
        result.loc[
            0,
            "portfolio_recommended_product",
        ]
        ==
        "PROD-00001"
    )


def test_customer_portfolio_normalises_display_encoding():
    result = (
        build_customer_decisions_portfolio(
            source=make_customer_source(),
            customer_mapping=(
                make_customer_mapping()
            ),
            product_mapping=(
                make_product_mapping()
            ),
        )
    )

    assert (
        result.loc[
            1,
            "primary_action",
        ]
        ==
        "Unscored — Long Dormant"
    )


def test_repeat_purchase_uses_shared_customer_mapping():
    result = (
        build_repeat_purchase_portfolio(
            source=make_repeat_source(),
            customer_mapping=(
                make_customer_mapping()
            ),
            order_mapping=(
                make_order_mapping()
            ),
        )
    )

    assert tuple(
        result.columns
    ) == (
        REPEAT_PURCHASE_PORTFOLIO_COLUMNS
    )

    assert (
        result.loc[
            0,
            "portfolio_customer_id",
        ]
        ==
        "CUST-00001"
    )

    assert (
        result.loc[
            0,
            "portfolio_first_order_id",
        ]
        ==
        "ORD-000001"
    )

    assert "customer" not in result.columns
    assert "first_invoice" not in result.columns


def test_demand_portfolio_preserves_governed_schema():
    result = (
        build_demand_forecasts_portfolio(
            make_demand_source()
        )
    )

    assert tuple(
        result.columns
    ) == (
        DEMAND_FORECAST_PORTFOLIO_COLUMNS
    )

    assert result[
        "horizon_weeks"
    ].tolist() == [
        1,
        2,
        4,
    ]


def test_customer_schema_drift_fails_closed():
    source = (
        make_customer_source()
        .drop(
            columns=[
                "decision_reason",
            ]
        )
    )

    with pytest.raises(
        RuntimeError,
        match="schema drift",
    ):
        build_customer_decisions_portfolio(
            source=source,
            customer_mapping=(
                make_customer_mapping()
            ),
            product_mapping=(
                make_product_mapping()
            ),
        )


def test_repeat_purchase_schema_drift_fails_closed():
    source = (
        make_repeat_source()
        .rename(
            columns={
                "target_name":
                    "unexpected_target_name",
            }
        )
    )

    with pytest.raises(
        RuntimeError,
        match="schema drift",
    ):
        build_repeat_purchase_portfolio(
            source=source,
            customer_mapping=(
                make_customer_mapping()
            ),
            order_mapping=(
                make_order_mapping()
            ),
        )


def test_demand_duplicate_horizon_fails_closed():
    source = make_demand_source()

    source.loc[
        1,
        "horizon_weeks",
    ] = 1

    with pytest.raises(
        RuntimeError,
        match="duplicate horizons",
    ):
        build_demand_forecasts_portfolio(
            source
        )


def test_end_to_end_generator_writes_only_public_artifacts(
    tmp_path,
):
    write_source_files(
        tmp_path
    )

    write_minimal_manifests(
        tmp_path
    )

    output_dir = (
        tmp_path
        /
        "portfolio_data"
    )

    outputs = generate_portfolio_data(
        project_root=tmp_path,
        output_dir=output_dir,
    )

    assert set(
        outputs
    ) == {
        "customer_decisions",
        "repeat_purchase",
        "demand_forecasts",
        "model_registry",
        "portfolio_summary",
    }

    for path in outputs.values():
        assert path.is_file()

    customer = pd.read_parquet(
        outputs[
            "customer_decisions"
        ]
    )

    repeat = pd.read_parquet(
        outputs[
            "repeat_purchase"
        ]
    )

    assert "customer" not in customer.columns
    assert "customer" not in repeat.columns
    assert "first_invoice" not in repeat.columns

    registry = json.loads(
        outputs[
            "model_registry"
        ].read_text(
            encoding="utf-8"
        )
    )

    assert registry[
        "system_count"
    ] == 6

    assert len(
        registry[
            "systems"
        ]
    ) == 6

    summary = json.loads(
        outputs[
            "portfolio_summary"
        ].read_text(
            encoding="utf-8"
        )
    )

    assert (
        summary[
            "verified_evidence"
        ][
            "customer_decisions_generated"
        ]
        ==
        2
    )

    assert (
        summary[
            "verified_evidence"
        ][
            "repeat_purchase_scores_generated"
        ]
        ==
        2
    )

    assert (
        summary[
            "verified_evidence"
        ][
            "governed_demand_horizons"
        ]
        ==
        [
            1,
            2,
            4,
        ]
    )


def test_end_to_end_generation_is_deterministic(
    tmp_path,
):
    write_source_files(
        tmp_path
    )

    write_minimal_manifests(
        tmp_path
    )

    first_dir = (
        tmp_path
        /
        "first"
    )

    second_dir = (
        tmp_path
        /
        "second"
    )

    first = generate_portfolio_data(
        project_root=tmp_path,
        output_dir=first_dir,
    )

    second = generate_portfolio_data(
        project_root=tmp_path,
        output_dir=second_dir,
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(
            first[
                "customer_decisions"
            ]
        ),
        pd.read_parquet(
            second[
                "customer_decisions"
            ]
        ),
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(
            first[
                "repeat_purchase"
            ]
        ),
        pd.read_parquet(
            second[
                "repeat_purchase"
            ]
        ),
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(
            first[
                "demand_forecasts"
            ]
        ),
        pd.read_parquet(
            second[
                "demand_forecasts"
            ]
        ),
    )

    assert (
        first[
            "model_registry"
        ].read_text(
            encoding="utf-8"
        )
        ==
        second[
            "model_registry"
        ].read_text(
            encoding="utf-8"
        )
    )

    assert (
        first[
            "portfolio_summary"
        ].read_text(
            encoding="utf-8"
        )
        ==
        second[
            "portfolio_summary"
        ].read_text(
            encoding="utf-8"
        )
    )


def test_missing_required_source_fails_closed(
    tmp_path,
):
    write_minimal_manifests(
        tmp_path
    )

    with pytest.raises(
        FileNotFoundError,
        match="Required portfolio source",
    ):
        generate_portfolio_data(
            project_root=tmp_path,
            output_dir=(
                tmp_path
                /
                "portfolio_data"
            ),
        )