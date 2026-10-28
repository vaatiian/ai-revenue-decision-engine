
import pytest

from portfolio.data_loader import load_portfolio_summary

from portfolio.testing_reliability import (
    EXPLICIT_TEST_FUNCTION_COUNT,
    KNOWN_WARNING_COUNT,
    TEST_MODULE_COUNT,
    executed_case_expansion,
    fail_closed_rows,
    integration_layer_rows,
    reliability_domain_rows,
    reliability_summary,
    validate_testing_evidence,
)


def summary():
    return load_portfolio_summary()


def test_real_testing_evidence_validates():
    validate_testing_evidence(summary())


def test_executed_regression_evidence_is_preserved():
    evidence = reliability_summary(summary())

    assert evidence["executed_regression_cases"] == 1801
    assert evidence["explicit_test_functions"] == EXPLICIT_TEST_FUNCTION_COUNT
    assert evidence["test_modules"] == TEST_MODULE_COUNT
    assert evidence["known_warnings"] == KNOWN_WARNING_COUNT


def test_collection_expansion_is_consistent():
    evidence = reliability_summary(summary())

    assert executed_case_expansion(summary()) == (
        evidence["executed_regression_cases"]
        - evidence["explicit_test_functions"]
    )


def test_reliability_domains_have_required_fields():
    domains = reliability_domain_rows()

    assert len(domains) == 7

    for domain in domains:
        assert domain["domain"]
        assert domain["purpose"]
        assert domain["evidence"]
        assert domain["examples"]


def test_fail_closed_controls_have_required_fields():
    controls = fail_closed_rows()

    assert len(controls) == 6

    for control in controls:
        assert control["control"]
        assert control["behaviour"]


def test_integration_layers_are_unique():
    layers = integration_layer_rows()

    assert len(layers) == 9
    assert len(layers) == len(set(layers))


def test_missing_verified_evidence_fails_closed():
    with pytest.raises(RuntimeError):
        validate_testing_evidence({})


def test_invalid_regression_count_fails_closed():
    with pytest.raises(RuntimeError):
        validate_testing_evidence(
            {
                "verified_evidence": {
                    "automated_tests_passing": "1801"
                }
            }
        )


def test_regression_count_below_explicit_inventory_fails_closed():
    with pytest.raises(RuntimeError):
        validate_testing_evidence(
            {
                "verified_evidence": {
                    "automated_tests_passing": (
                        EXPLICIT_TEST_FUNCTION_COUNT - 1
                    )
                }
            }
        )
