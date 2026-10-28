import pandas as pd
import pytest

from portfolio.customer_explorer import (
    build_customer_dossier,
    build_model_evidence,
    customer_ids,
    evidence_status,
    filter_customer_candidates,
    get_customer_record,
    governance_lookup,
    validate_explorer_frame,
)

from portfolio.data_loader import (
    load_customer_decisions,
    load_model_registry,
)


def decisions():
    return load_customer_decisions()


def test_real_explorer_frame_validates():
    validate_explorer_frame(
        decisions()
    )


def test_customer_ids_are_public_and_deterministic():
    values = customer_ids(
        decisions()
    )

    assert len(
        values
    ) == 5220

    assert values == sorted(
        values
    )

    assert values[
        0
    ].startswith(
        "CUST-"
    )


def test_customer_candidate_filter_preserves_existing_decisions():
    frame = decisions()

    dormant = filter_customer_candidates(
        frame,
        state="Dormant",
    )

    assert len(
        dormant
    ) == 857

    assert set(
        dormant[
            "customer_state"
        ]
    ) == {
        "Dormant",
    }


def test_customer_candidate_filter_by_action():
    frame = decisions()

    protect = filter_customer_candidates(
        frame,
        action="Protect",
    )

    assert len(
        protect
    ) == 493

    assert set(
        protect[
            "primary_action"
        ]
    ) == {
        "Protect",
    }


def test_customer_candidate_all_filters_preserve_full_population():
    frame = decisions()

    result = filter_customer_candidates(
        frame,
        state="All",
        action="All",
        priority="All",
    )

    assert len(
        result
    ) == 5220


def test_get_customer_record_returns_one_public_customer():
    frame = decisions()

    record = get_customer_record(
        frame,
        "CUST-00001",
    )

    assert (
        record[
            "portfolio_customer_id"
        ]
        ==
        "CUST-00001"
    )

    assert "customer" not in record.index


def test_unknown_customer_fails_closed():
    with pytest.raises(
        KeyError,
        match="Unknown portfolio customer",
    ):
        get_customer_record(
            decisions(),
            "CUST-99999",
        )


def test_customer_dossier_preserves_governed_action():
    frame = decisions()

    record = get_customer_record(
        frame,
        "CUST-00001",
    )

    dossier = build_customer_dossier(
        record
    )

    assert (
        dossier[
            "identity"
        ][
            "customer_id"
        ]
        ==
        "CUST-00001"
    )

    assert (
        dossier[
            "decision"
        ][
            "action"
        ]
        ==
        record[
            "primary_action"
        ]
    )

    assert (
        dossier[
            "decision"
        ][
            "priority_tier"
        ]
        ==
        record[
            "priority_tier"
        ]
    )


def test_dossier_contains_all_evidence_sections():
    record = get_customer_record(
        decisions(),
        "CUST-00001",
    )

    dossier = build_customer_dossier(
        record
    )

    assert set(
        dossier
    ) == {
        "identity",
        "behaviour",
        "decision",
        "leakage",
        "future_value",
        "reactivation",
        "recommendation",
    }


def test_model_evidence_has_four_intelligence_rows():
    record = get_customer_record(
        decisions(),
        "CUST-00001",
    )

    dossier = build_customer_dossier(
        record
    )

    evidence = build_model_evidence(
        dossier
    )

    assert len(
        evidence
    ) == 4

    assert set(
        evidence[
            "Intelligence"
        ]
    ) == {
        "Revenue Leakage",
        "Future Customer Value",
        "Reactivation",
        "Recommendation",
    }


def test_missing_signal_remains_not_available():
    assert evidence_status(
        None
    ) == "Not Available"

    assert evidence_status(
        float(
            "nan"
        )
    ) == "Not Available"


def test_present_signal_is_available():
    assert evidence_status(
        0.5
    ) == "Available"

    assert evidence_status(
        "PROD-00001"
    ) == "Available"


def test_registry_governance_lookup_contains_six_systems():
    result = governance_lookup(
        load_model_registry()
    )

    assert len(
        result
    ) == 6

    assert (
        result[
            "reactivation"
        ][
            "governance"
        ]
        ==
        "GO"
    )

    assert (
        result[
            "future_customer_value"
        ][
            "governance"
        ]
        ==
        "CONDITIONAL GO"
    )


def test_recommendation_governance_preserves_channel_split():
    result = governance_lookup(
        load_model_registry()
    )

    governance = result[
        "recommendation"
    ][
        "governance"
    ]

    assert governance == {
        "discovery_channel":
            "CONDITIONAL GO",

        "repeat_channel":
            "GO",
    }


def test_missing_required_explorer_column_fails_closed():
    frame = (
        decisions()
        .drop(
            columns=[
                "decision_reason",
            ]
        )
    )

    with pytest.raises(
        RuntimeError,
        match="incomplete",
    ):
        validate_explorer_frame(
            frame
        )


def test_explorer_never_requires_raw_customer_identifier():
    frame = decisions()

    assert "customer" not in frame.columns

    assert (
        "portfolio_customer_id"
        in
        frame.columns
    )