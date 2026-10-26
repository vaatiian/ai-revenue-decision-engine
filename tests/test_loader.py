import pandas as pd
import pytest

from src.data.loader import (
    load_transactions,
    standardise_transaction_types,
    validate_source_path,
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
                12345,
                None,
            ],

            "invoice": [
                100001,
                100002,
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
# TYPE STANDARDISATION
# ================================================================

def test_standardise_transaction_types():

    df = make_valid_dataframe()

    result = (
        standardise_transaction_types(
            df
        )
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["date"]
    )

    assert isinstance(
        result["customer"].dtype,
        pd.StringDtype,
    )

    assert isinstance(
        result["invoice"].dtype,
        pd.StringDtype,
    )

    assert str(
        result["is_return"].dtype
    ) == "boolean"


# ================================================================
# MISSING CUSTOMER REMAINS MISSING
# ================================================================

def test_missing_customer_remains_missing():

    df = make_valid_dataframe()

    result = (
        standardise_transaction_types(
            df
        )
    )

    assert (
        result["customer"]
        .isna()
        .sum()
        ==
        1
    )

    assert not (
        result["customer"]
        .fillna("")
        .eq("UNKNOWN")
        .any()
    )


# ================================================================
# PARQUET LOAD
# ================================================================

def test_load_valid_parquet(
    tmp_path,
):

    df = make_valid_dataframe()

    file_path = (
        tmp_path
        /
        "transactions.parquet"
    )

    df.to_parquet(
        file_path,
        index=False,
    )

    loaded, validation = (
        load_transactions(
            file_path
        )
    )

    assert validation.passed is True

    assert len(loaded) == len(df)


# ================================================================
# CSV LOAD
# ================================================================

def test_load_valid_csv(
    tmp_path,
):

    df = make_valid_dataframe()

    file_path = (
        tmp_path
        /
        "transactions.csv"
    )

    df.to_csv(
        file_path,
        index=False,
    )

    loaded, validation = (
        load_transactions(
            file_path
        )
    )

    assert validation.passed is True

    assert len(loaded) == len(df)


# ================================================================
# MISSING FILE
# ================================================================

def test_missing_file_fails(
    tmp_path,
):

    file_path = (
        tmp_path
        /
        "missing.parquet"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        validate_source_path(
            file_path
        )


# ================================================================
# UNSUPPORTED FILE TYPE
# ================================================================

def test_unsupported_file_type_fails(
    tmp_path,
):

    file_path = (
        tmp_path
        /
        "transactions.xlsx"
    )

    file_path.touch()

    with pytest.raises(
        ValueError
    ):
        validate_source_path(
            file_path
        )


# ================================================================
# INVALID SCHEMA FAILS CLOSED
# ================================================================

def test_invalid_schema_fails_closed(
    tmp_path,
):

    df = (
        make_valid_dataframe()
        .drop(
            columns=[
                "net_merchandise_revenue"
            ]
        )
    )

    file_path = (
        tmp_path
        /
        "invalid.parquet"
    )

    df.to_parquet(
        file_path,
        index=False,
    )

    with pytest.raises(
        ValueError
    ):
        load_transactions(
            file_path
        )