import pandas as pd
import pytest

from src.data.schema import (
    REQUIRED_COLUMNS,
    assert_valid_transaction_schema,
    validate_transaction_schema,
)


# ================================================================
# TEST DATA FACTORY
# ================================================================

def make_valid_dataframe():
    return pd.DataFrame(
        {
            "date": [
                "2011-08-01 10:00:00",
                "2011-08-02 11:00:00",
            ],

            "customer": [
                "12345",
                None,
            ],

            "invoice": [
                "100001",
                "100002",
            ],

            "product": [
                "ABC123",
                "XYZ456",
            ],

            "region": [
                "United Kingdom",
                "France",
            ],

            "quantity": [
                2,
                1,
            ],

            "unit_price": [
                10.0,
                5.0,
            ],

            "net_merchandise_revenue": [
                20.0,
                5.0,
            ],

            "gross_merchandise_revenue": [
                20.0,
                5.0,
            ],

            "non_merchandise_revenue": [
                0.0,
                0.0,
            ],

            "is_cancellation": [
                False,
                False,
            ],

            "is_return": [
                False,
                False,
            ],
        }
    )


# ================================================================
# REQUIRED COLUMN TEST
# ================================================================

def test_required_columns_defined():
    assert len(REQUIRED_COLUMNS) > 0


# ================================================================
# VALID DATA TEST
# ================================================================

def test_valid_dataframe_passes():

    df = make_valid_dataframe()

    result = validate_transaction_schema(
        df
    )

    assert result.passed is True
    assert len(result.errors) == 0


# ================================================================
# MISSING CUSTOMER IDS ARE ALLOWED
# ================================================================

def test_missing_customer_is_warning_not_error():

    df = make_valid_dataframe()

    result = validate_transaction_schema(
        df
    )

    assert result.passed is True

    assert any(
        "missing customer" in warning.lower()
        for warning in result.warnings
    )


# ================================================================
# MISSING REQUIRED COLUMN FAILS
# ================================================================

def test_missing_required_column_fails():

    df = make_valid_dataframe().drop(
        columns=[
            "net_merchandise_revenue"
        ]
    )

    result = validate_transaction_schema(
        df
    )

    assert result.passed is False

    assert any(
        "missing required columns"
        in error.lower()
        for error in result.errors
    )


# ================================================================
# INVALID DATE FAILS
# ================================================================

def test_invalid_date_fails():

    df = make_valid_dataframe()

    df.loc[
        0,
        "date"
    ] = "not-a-date"

    result = validate_transaction_schema(
        df
    )

    assert result.passed is False


# ================================================================
# NON-NUMERIC REVENUE FAILS
# ================================================================

def test_non_numeric_revenue_fails():

    df = make_valid_dataframe()

    df[
        "net_merchandise_revenue"
    ] = (
        df[
            "net_merchandise_revenue"
        ]
        .astype("object")
    )

    df.loc[
        0,
        "net_merchandise_revenue"
    ] = "bad-value"

    result = validate_transaction_schema(
        df
    )

    assert result.passed is False

# ================================================================
# INVALID BOOLEAN FAILS
# ================================================================

def test_invalid_boolean_fails():

    df = make_valid_dataframe()

    df[
        "is_return"
    ] = (
        df[
            "is_return"
        ]
        .astype("object")
    )

    df.loc[
        0,
        "is_return"
    ] = "maybe"

    result = validate_transaction_schema(
        df
    )

    assert result.passed is False

# ================================================================
# STRICT VALIDATOR RAISES
# ================================================================

def test_strict_validator_raises_on_invalid_schema():

    df = make_valid_dataframe().drop(
        columns=[
            "invoice"
        ]
    )

    with pytest.raises(
        ValueError
    ):
        assert_valid_transaction_schema(
            df
        )


# ================================================================
# NEGATIVE QUANTITY IS WARNING
# ================================================================

def test_negative_quantity_is_warning():

    df = make_valid_dataframe()

    df.loc[
        0,
        "quantity"
    ] = -1

    result = validate_transaction_schema(
        df
    )

    assert result.passed is True

    assert any(
        "negative quantity"
        in warning.lower()
        for warning in result.warnings
    )