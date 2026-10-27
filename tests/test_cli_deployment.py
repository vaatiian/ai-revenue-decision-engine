from types import SimpleNamespace

import pandas as pd

import src.cli as cli_module


# ================================================================
# FIXTURES
# ================================================================


def make_transactions():

    return pd.DataFrame(
        {
            "customer": [
                "A",
            ],
        }
    )


# ================================================================
# REPEAT PURCHASE COMMAND
# ================================================================


def test_repeat_purchase_cli_routes_to_governed_batch(
    monkeypatch,
    tmp_path,
):

    observed = {}

    monkeypatch.setattr(
        cli_module,
        "load_transactions",
        lambda path: (
            make_transactions(),
            object(),
        ),
    )

    def fake_batch(
        *,
        transactions,
        modeling_end_date,
        exclude_opening_month,
    ):

        observed[
            "modeling_end_date"
        ] = modeling_end_date

        observed[
            "exclude_opening_month"
        ] = exclude_opening_month

        return SimpleNamespace(
            modeling_end_date=(
                pd.Timestamp(
                    "2011-11-30"
                )
            ),
            scores=pd.DataFrame(
                {
                    "customer": [
                        "A",
                    ],
                }
            ),
        )

    monkeypatch.setattr(
        cli_module,
        "run_repeat_purchase_batch",
        fake_batch,
    )

    monkeypatch.setattr(
        cli_module,
        "write_dataframe_csv",
        lambda dataframe, path: path,
    )

    exit_code = cli_module.main(
        [
            "repeat-purchase",
            "--data",
            "transactions.parquet",
            "--modeling-end-date",
            "2011-11-30",
            "--output-dir",
            str(
                tmp_path
            ),
        ]
    )

    assert exit_code == 0

    assert (
        observed[
            "modeling_end_date"
        ]
        ==
        "2011-11-30"
    )

    assert (
        observed[
            "exclude_opening_month"
        ]
        is True
    )


def test_repeat_purchase_cli_can_include_opening_month(
    monkeypatch,
    tmp_path,
):

    observed = {}

    monkeypatch.setattr(
        cli_module,
        "load_transactions",
        lambda path: (
            make_transactions(),
            object(),
        ),
    )

    def fake_batch(
        *,
        transactions,
        modeling_end_date,
        exclude_opening_month,
    ):

        observed[
            "exclude_opening_month"
        ] = exclude_opening_month

        return SimpleNamespace(
            modeling_end_date=(
                pd.Timestamp(
                    "2011-11-30"
                )
            ),
            scores=pd.DataFrame(),
        )

    monkeypatch.setattr(
        cli_module,
        "run_repeat_purchase_batch",
        fake_batch,
    )

    monkeypatch.setattr(
        cli_module,
        "write_dataframe_csv",
        lambda dataframe, path: path,
    )

    exit_code = cli_module.main(
        [
            "repeat-purchase",
            "--data",
            "transactions.parquet",
            "--modeling-end-date",
            "2011-11-30",
            "--output-dir",
            str(
                tmp_path
            ),
            "--include-opening-month",
        ]
    )

    assert exit_code == 0

    assert (
        observed[
            "exclude_opening_month"
        ]
        is False
    )


# ================================================================
# DEMAND COMMAND
# ================================================================


def test_demand_cli_forwards_horizons(
    monkeypatch,
    tmp_path,
):

    observed = {}

    monkeypatch.setattr(
        cli_module,
        "load_transactions",
        lambda path: (
            make_transactions(),
            object(),
        ),
    )

    def fake_batch(
        *,
        transactions,
        forecast_origin_week,
        horizons,
        scored_at_utc,
    ):

        observed[
            "horizons"
        ] = horizons

        return SimpleNamespace(
            forecast_origin_week=(
                pd.Timestamp(
                    "2011-01-03"
                )
            ),
            horizons=horizons,
            forecasts=pd.DataFrame(
                {
                    "horizon_weeks":
                        list(
                            horizons
                        ),
                }
            ),
        )

    monkeypatch.setattr(
        cli_module,
        "run_demand_batch",
        fake_batch,
    )

    monkeypatch.setattr(
        cli_module,
        "write_dataframe_csv",
        lambda dataframe, path: path,
    )

    exit_code = cli_module.main(
        [
            "demand",
            "--data",
            "transactions.parquet",
            "--forecast-origin-week",
            "2011-01-03",
            "--output-dir",
            str(
                tmp_path
            ),
            "--horizons",
            "1",
            "4",
        ]
    )

    assert exit_code == 0

    assert (
        observed[
            "horizons"
        ]
        ==
        (
            1,
            4,
        )
    )


# ================================================================
# CUSTOMER COMMAND
# ================================================================


def test_customer_cli_uses_deterministic_delivery_ids_and_audit(
    monkeypatch,
    tmp_path,
):

    observed = {}

    transactions = make_transactions()

    decisions = pd.DataFrame(
        {
            "customer": [
                "A",
            ],
        }
    )

    monkeypatch.setattr(
        cli_module,
        "load_transactions",
        lambda path: (
            transactions,
            object(),
        ),
    )

    monkeypatch.setattr(
        cli_module,
        "run_customer_decision_batch",
        lambda **kwargs:
            SimpleNamespace(
                decision_date=(
                    pd.Timestamp(
                        "2011-08-31"
                    )
                ),
                decisions=decisions,
            ),
    )

    monkeypatch.setattr(
        cli_module,
        "build_deterministic_delivery_ids",
        lambda **kwargs: {
            "A":
                "stable-delivery-id",
        },
    )

    def fake_preflight(
        *,
        delivery_ids,
        audit_path,
    ):

        observed[
            "preflight_ids"
        ] = delivery_ids

        observed[
            "audit_path"
        ] = audit_path

    monkeypatch.setattr(
        cli_module,
        "assert_delivery_ids_not_audited",
        fake_preflight,
    )

    def fake_builder(
        *,
        decisions,
        delivery_created_at_utc,
        delivery_ids,
    ):

        observed[
            "builder_ids"
        ] = delivery_ids

        return (
            "payload",
        )

    monkeypatch.setattr(
        cli_module,
        "build_delivery_batch",
        fake_builder,
    )

    monkeypatch.setattr(
        cli_module,
        "write_dataframe_csv",
        lambda dataframe, path: path,
    )

    monkeypatch.setattr(
        cli_module,
        "export_delivery_csv",
        lambda payloads, path: path,
    )

    monkeypatch.setattr(
        cli_module,
        "export_delivery_jsonl",
        lambda payloads, path: path,
    )

    def fake_append(
        *,
        payloads,
        path,
    ):

        observed[
            "audit_payloads"
        ] = payloads

    monkeypatch.setattr(
        cli_module,
        "append_delivery_batch_audit",
        fake_append,
    )

    audit_path = (
        tmp_path
        /
        "delivery_audit.jsonl"
    )

    exit_code = cli_module.main(
        [
            "customer-batch",
            "--data",
            "transactions.parquet",
            "--decision-date",
            "2011-08-31",
            "--output-dir",
            str(
                tmp_path
            ),
            "--delivery-audit",
            str(
                audit_path
            ),
        ]
    )

    assert exit_code == 0

    assert (
        observed[
            "preflight_ids"
        ]
        ==
        {
            "A":
                "stable-delivery-id",
        }
    )

    assert (
        observed[
            "builder_ids"
        ]
        ==
        {
            "A":
                "stable-delivery-id",
        }
    )

    assert (
        observed[
            "audit_payloads"
        ]
        ==
        (
            "payload",
        )
    )


def test_customer_cli_loads_optional_recommendation_episodes(
    monkeypatch,
    tmp_path,
):

    observed = {}

    transactions = make_transactions()

    episodes = pd.DataFrame(
        {
            "customer": [
                "A",
            ],
        }
    )

    monkeypatch.setattr(
        cli_module,
        "load_transactions",
        lambda path: (
            transactions,
            object(),
        ),
    )

    monkeypatch.setattr(
        cli_module,
        "load_recommendation_episodes",
        lambda path:
            episodes,
    )

    def fake_batch(
        **kwargs,
    ):

        observed[
            "episodes"
        ] = kwargs[
            "recommendation_episodes"
        ]

        return SimpleNamespace(
            decision_date=(
                pd.Timestamp(
                    "2011-08-31"
                )
            ),
            decisions=pd.DataFrame(
                columns=[
                    "customer",
                ]
            ),
        )

    monkeypatch.setattr(
        cli_module,
        "run_customer_decision_batch",
        fake_batch,
    )

    monkeypatch.setattr(
        cli_module,
        "build_deterministic_delivery_ids",
        lambda **kwargs: {},
    )

    monkeypatch.setattr(
        cli_module,
        "assert_delivery_ids_not_audited",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        cli_module,
        "build_delivery_batch",
        lambda **kwargs: (),
    )

    monkeypatch.setattr(
        cli_module,
        "write_dataframe_csv",
        lambda dataframe, path: path,
    )

    monkeypatch.setattr(
        cli_module,
        "export_delivery_csv",
        lambda payloads, path: path,
    )

    monkeypatch.setattr(
        cli_module,
        "export_delivery_jsonl",
        lambda payloads, path: path,
    )

    monkeypatch.setattr(
        cli_module,
        "append_delivery_batch_audit",
        lambda **kwargs: None,
    )

    exit_code = cli_module.main(
        [
            "customer-batch",
            "--data",
            "transactions.parquet",
            "--decision-date",
            "2011-08-31",
            "--output-dir",
            str(
                tmp_path
            ),
            "--recommendation-episodes",
            "episodes.parquet",
        ]
    )

    assert exit_code == 0

    pd.testing.assert_frame_equal(
        observed[
            "episodes"
        ],
        episodes,
    )


# ================================================================
# GOVERNED EXECUTION FAILURE
# ================================================================


def test_batch_execution_failure_returns_one(
    monkeypatch,
    capsys,
    tmp_path,
):

    monkeypatch.setattr(
        cli_module,
        "load_transactions",
        lambda path: (
            make_transactions(),
            object(),
        ),
    )

    monkeypatch.setattr(
        cli_module,
        "run_repeat_purchase_batch",
        lambda **kwargs:
            (_ for _ in ())
            .throw(
                ValueError(
                    "governed failure"
                )
            ),
    )

    exit_code = cli_module.main(
        [
            "repeat-purchase",
            "--data",
            "transactions.parquet",
            "--modeling-end-date",
            "2011-11-30",
            "--output-dir",
            str(
                tmp_path
            ),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1

    assert (
        "FAILED: governed failure"
        in
        captured.err
    )