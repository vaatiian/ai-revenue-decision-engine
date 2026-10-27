from types import SimpleNamespace

import pandas as pd
import pytest

import src.deployment.runtime as runtime_module

from src.deployment.runtime import (
    assert_delivery_ids_not_audited,
    build_deterministic_delivery_ids,
    customer_batch_output_paths,
    demand_output_paths,
    repeat_purchase_output_paths,
    write_dataframe_csv,
)


# ================================================================
# OUTPUT PATHS
# ================================================================


def test_customer_output_paths_are_run_specific(
    tmp_path,
):

    paths = customer_batch_output_paths(
        tmp_path,
        "2011-08-31",
    )

    assert paths.run_dir == (
        tmp_path
        /
        "customer-batch"
        /
        "2011-08-31"
    )

    assert (
        paths.decisions_csv.name
        ==
        "customer_decisions.csv"
    )

    assert (
        paths.delivery_csv.name
        ==
        "delivery_payloads.csv"
    )

    assert (
        paths.delivery_jsonl.name
        ==
        "delivery_payloads.jsonl"
    )


def test_repeat_purchase_output_paths_are_run_specific(
    tmp_path,
):

    paths = repeat_purchase_output_paths(
        tmp_path,
        "2011-11-30",
    )

    assert paths.run_dir == (
        tmp_path
        /
        "repeat-purchase"
        /
        "2011-11-30"
    )


def test_demand_output_paths_are_run_specific(
    tmp_path,
):

    paths = demand_output_paths(
        tmp_path,
        "2011-01-03",
    )

    assert paths.run_dir == (
        tmp_path
        /
        "demand"
        /
        "2011-01-03"
    )


# ================================================================
# CSV EXPORT
# ================================================================


def test_dataframe_csv_export_creates_parent_directory(
    tmp_path,
):

    path = (
        tmp_path
        /
        "nested"
        /
        "output.csv"
    )

    frame = pd.DataFrame(
        {
            "a": [
                1,
            ],
        }
    )

    result = write_dataframe_csv(
        frame,
        path,
    )

    assert result == path
    assert path.exists()


def test_dataframe_csv_export_fails_on_rerun(
    tmp_path,
):

    path = (
        tmp_path
        /
        "output.csv"
    )

    frame = pd.DataFrame(
        {
            "a": [
                1,
            ],
        }
    )

    write_dataframe_csv(
        frame,
        path,
    )

    with pytest.raises(
        FileExistsError,
    ):
        write_dataframe_csv(
            frame,
            path,
        )


# ================================================================
# DETERMINISTIC DELIVERY IDS
# ================================================================


def test_delivery_ids_are_deterministic():

    decisions = pd.DataFrame(
        {
            "customer": [
                "A",
                "B",
            ],
        }
    )

    first = (
        build_deterministic_delivery_ids(
            decisions,
            "2011-08-31",
        )
    )

    second = (
        build_deterministic_delivery_ids(
            decisions,
            "2011-08-31",
        )
    )

    assert first == second

    assert (
        first["A"]
        !=
        first["B"]
    )


def test_delivery_id_changes_across_decision_dates():

    decisions = pd.DataFrame(
        {
            "customer": [
                "A",
            ],
        }
    )

    first = (
        build_deterministic_delivery_ids(
            decisions,
            "2011-08-31",
        )
    )

    second = (
        build_deterministic_delivery_ids(
            decisions,
            "2011-09-30",
        )
    )

    assert (
        first["A"]
        !=
        second["A"]
    )


def test_delivery_ids_reject_duplicate_customers():

    decisions = pd.DataFrame(
        {
            "customer": [
                "A",
                "A",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="duplicate customers",
    ):
        build_deterministic_delivery_ids(
            decisions,
            "2011-08-31",
        )


# ================================================================
# AUDIT IDEMPOTENCY
# ================================================================


def test_delivery_audit_preflight_accepts_new_ids(
    monkeypatch,
    tmp_path,
):

    monkeypatch.setattr(
        runtime_module,
        "list_delivery_audit",
        lambda path: (),
    )

    assert_delivery_ids_not_audited(
        {
            "A": "delivery-a",
        },
        tmp_path
        /
        "audit.jsonl",
    )


def test_delivery_audit_preflight_rejects_rerun(
    monkeypatch,
    tmp_path,
):

    monkeypatch.setattr(
        runtime_module,
        "list_delivery_audit",
        lambda path: (
            SimpleNamespace(
                delivery_id=(
                    "delivery-a"
                )
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "already been audited"
        ),
    ):
        assert_delivery_ids_not_audited(
            {
                "A": "delivery-a",
            },
            tmp_path
            /
            "audit.jsonl",
        )