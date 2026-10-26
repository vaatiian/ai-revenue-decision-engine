import numpy as np
import pandas as pd
import pytest

from src.features.repeat_purchase_features import (
    REPEAT_PURCHASE_MODELING_END_DATE,
    add_first_order_temporal_features,
    add_repeat_purchase_target,
    apply_repeat_purchase_observation_boundary,
    build_first_observable_orders,
    build_positive_customer_purchases,
    build_repeat_purchase_customer_orders,
    build_repeat_purchase_historical_dataset,
    build_repeat_purchase_scoring_features,
    select_repeat_purchase_model_features,
    standardize_repeat_purchase_data,
    validate_repeat_purchase_input,
)
from src.models.repeat_purchase_spec import (
    REPEAT_PURCHASE_MODEL_FEATURES,
)


# ================================================================
# FIXTURES
# ================================================================

def make_line(
    date,
    invoice,
    customer,
    product,
    quantity=1,
    unit_price=10.0,
    revenue=None,
    region="United Kingdom",
    identified=True,
    merchandise=True,
    cancellation=False,
):
    if revenue is None:
        revenue = quantity * unit_price

    return {
        "date": date,
        "invoice": invoice,
        "customer": customer,
        "product": product,
        "product_description": f"Product {product}",
        "region": region,
        "quantity": quantity,
        "unit_price": unit_price,
        "customer_is_identified": identified,
        "is_merchandise": merchandise,
        "is_cancellation": cancellation,
        "net_merchandise_revenue": revenue,
    }


def base_dataframe():
    return pd.DataFrame(
        [
            # Opening-month customer: excluded historically.
            make_line(
                "2009-12-10 09:00:00",
                "A1",
                "A",
                "P1",
            ),

            # Customer B first order — two lines.
            make_line(
                "2010-01-10 10:00:00",
                "B1",
                "B",
                "P1",
                quantity=2,
                unit_price=5.0,
            ),
            make_line(
                "2010-01-10 10:05:00",
                "B1",
                "B",
                "P2",
                quantity=3,
                unit_price=10.0,
            ),

            # B repeats within 90 days.
            make_line(
                "2010-02-15 11:00:00",
                "B2",
                "B",
                "P3",
                quantity=1,
                unit_price=20.0,
            ),

            # Customer C first order.
            make_line(
                "2010-03-01 14:00:00",
                "C1",
                "C",
                "P4",
                quantity=4,
                unit_price=7.0,
            ),

            # C repeats after >90 days.
            make_line(
                "2010-07-01 14:00:00",
                "C2",
                "C",
                "P5",
                quantity=1,
                unit_price=5.0,
            ),
        ]
    )


# ================================================================
# INPUT CONTRACT
# ================================================================

def test_validate_input_accepts_required_columns():
    data = base_dataframe()

    validate_repeat_purchase_input(
        data
    )


def test_validate_input_rejects_missing_column():
    data = (
        base_dataframe()
        .drop(
            columns=[
                "is_merchandise",
            ]
        )
    )

    with pytest.raises(
        KeyError
    ):
        validate_repeat_purchase_input(
            data
        )


def test_validate_input_requires_dataframe():
    with pytest.raises(
        TypeError
    ):
        validate_repeat_purchase_input(
            []
        )


# ================================================================
# STANDARDISATION
# ================================================================

def test_standardization_does_not_mutate_input():
    data = base_dataframe()

    original = (
        data.copy(
            deep=True
        )
    )

    standardize_repeat_purchase_data(
        data
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_standardization_converts_dates():
    data = base_dataframe()

    result = (
        standardize_repeat_purchase_data(
            data
        )
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result[
            "date"
        ]
    )


def test_invalid_date_fails_closed():
    data = base_dataframe()

    data.loc[
        0,
        "date",
    ] = "not-a-date"

    with pytest.raises(
        ValueError
    ):
        standardize_repeat_purchase_data(
            data
        )


# ================================================================
# OBSERVATION BOUNDARY
# ================================================================

def test_modeling_end_date_is_frozen():
    assert (
        REPEAT_PURCHASE_MODELING_END_DATE
        ==
        pd.Timestamp(
            "2011-11-30 23:59:59"
        )
    )


def test_rows_after_boundary_are_removed():
    data = base_dataframe()

    future = pd.DataFrame(
        [
            make_line(
                "2011-12-01 00:00:00",
                "F1",
                "F",
                "P9",
            )
        ]
    )

    data = pd.concat(
        [
            data,
            future,
        ],
        ignore_index=True,
    )

    result = (
        apply_repeat_purchase_observation_boundary(
            data
        )
    )

    assert (
        result[
            "date"
        ].max()
        <=
        REPEAT_PURCHASE_MODELING_END_DATE
    )

    assert "F1" not in set(
        result[
            "invoice"
        ]
    )


def test_exact_boundary_is_retained():
    data = pd.DataFrame(
        [
            make_line(
                "2011-11-30 23:59:59",
                "X1",
                "X",
                "P1",
            )
        ]
    )

    result = (
        apply_repeat_purchase_observation_boundary(
            data
        )
    )

    assert len(
        result
    ) == 1


# ================================================================
# QUALIFYING PURCHASE FILTER
# ================================================================

@pytest.mark.parametrize(
    (
        "override",
        "should_survive",
    ),
    [
        (
            {},
            True,
        ),
        (
            {
                "identified": False,
            },
            False,
        ),
        (
            {
                "merchandise": False,
            },
            False,
        ),
        (
            {
                "quantity": 0,
            },
            False,
        ),
        (
            {
                "quantity": -1,
            },
            False,
        ),
        (
            {
                "unit_price": 0,
            },
            False,
        ),
        (
            {
                "unit_price": -1,
            },
            False,
        ),
        (
            {
                "cancellation": True,
            },
            False,
        ),
    ],
)
def test_exact_qualifying_purchase_filter(
    override,
    should_survive,
):
    kwargs = {
        "date": "2010-01-01",
        "invoice": "1",
        "customer": "C1",
        "product": "P1",
    }

    kwargs.update(
        override
    )

    data = pd.DataFrame(
        [
            make_line(
                **kwargs
            )
        ]
    )

    result = (
        build_positive_customer_purchases(
            data
        )
    )

    assert (
        len(
            result
        )
        ==
        int(
            should_survive
        )
    )


def test_no_explicit_positive_net_revenue_filter():
    data = pd.DataFrame(
        [
            make_line(
                "2010-01-01",
                "1",
                "C1",
                "P1",
                quantity=1,
                unit_price=10,
                revenue=-50,
            )
        ]
    )

    result = (
        build_positive_customer_purchases(
            data
        )
    )

    assert len(
        result
    ) == 1


# ================================================================
# ORDER CONSTRUCTION
# ================================================================

def test_customer_invoice_forms_order_grain():
    data = base_dataframe()

    result = (
        build_repeat_purchase_customer_orders(
            data
        )
    )

    b1 = (
        result.loc[
            result[
                "invoice"
            ]
            ==
            "B1"
        ]
        .iloc[
            0
        ]
    )

    assert (
        b1[
            "order_revenue"
        ]
        ==
        pytest.approx(
            40.0
        )
    )

    assert (
        b1[
            "order_units"
        ]
        ==
        5
    )

    assert (
        b1[
            "distinct_products"
        ]
        ==
        2
    )

    assert (
        b1[
            "average_unit_price"
        ]
        ==
        pytest.approx(
            7.5
        )
    )


def test_order_date_is_minimum_line_timestamp():
    data = base_dataframe()

    result = (
        build_repeat_purchase_customer_orders(
            data
        )
    )

    b1 = (
        result.loc[
            result[
                "invoice"
            ]
            ==
            "B1"
        ]
        .iloc[
            0
        ]
    )

    assert (
        b1[
            "order_date"
        ]
        ==
        pd.Timestamp(
            "2010-01-10 10:00:00"
        )
    )


def test_order_number_is_chronological():
    result = (
        build_repeat_purchase_customer_orders(
            base_dataframe()
        )
    )

    b = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "B"
        ]
        .sort_values(
            "order_number"
        )
    )

    assert (
        b[
            "invoice"
        ].tolist()
        ==
        [
            "B1",
            "B2",
        ]
    )


# ================================================================
# FIRST ORDER
# ================================================================

def test_first_observable_order_is_order_number_one():
    orders = (
        build_repeat_purchase_customer_orders(
            base_dataframe()
        )
    )

    result = (
        build_first_observable_orders(
            orders
        )
    )

    b = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "B"
        ]
        .iloc[
            0
        ]
    )

    assert (
        b[
            "first_invoice"
        ]
        ==
        "B1"
    )


# ================================================================
# TARGET
# ================================================================

def test_target_includes_exactly_90_days():
    first_orders = pd.DataFrame(
        {
            "first_order_date": [
                pd.Timestamp(
                    "2010-01-01"
                ),
            ],
            "second_order_date": [
                pd.Timestamp(
                    "2010-04-01"
                ),
            ],
            "days_to_second_purchase": [
                90,
            ],
        }
    )

    result = (
        add_repeat_purchase_target(
            first_orders,
            modeling_end_date=pd.Timestamp(
                "2010-12-31"
            ),
        )
    )

    assert (
        result[
            "repeat_within_90d"
        ].iloc[
            0
        ]
        ==
        1
    )


def test_target_excludes_91_days():
    first_orders = pd.DataFrame(
        {
            "first_order_date": [
                pd.Timestamp(
                    "2010-01-01"
                ),
            ],
            "second_order_date": [
                pd.Timestamp(
                    "2010-04-02"
                ),
            ],
            "days_to_second_purchase": [
                91,
            ],
        }
    )

    result = (
        add_repeat_purchase_target(
            first_orders,
            modeling_end_date=pd.Timestamp(
                "2010-12-31"
            ),
        )
    )

    assert (
        result[
            "repeat_within_90d"
        ].iloc[
            0
        ]
        ==
        0
    )


def test_missing_second_order_is_negative_target():
    first_orders = pd.DataFrame(
        {
            "first_order_date": [
                pd.Timestamp(
                    "2010-01-01"
                ),
            ],
            "second_order_date": [
                pd.NaT,
            ],
            "days_to_second_purchase": [
                np.nan,
            ],
        }
    )

    result = (
        add_repeat_purchase_target(
            first_orders,
            modeling_end_date=pd.Timestamp(
                "2010-12-31"
            ),
        )
    )

    assert (
        result[
            "repeat_within_90d"
        ].iloc[
            0
        ]
        ==
        0
    )


def test_incomplete_followup_is_excluded():
    first_orders = pd.DataFrame(
        {
            "first_order_date": [
                pd.Timestamp(
                    "2011-11-01"
                ),
            ],
            "second_order_date": [
                pd.NaT,
            ],
            "days_to_second_purchase": [
                np.nan,
            ],
        }
    )

    result = (
        add_repeat_purchase_target(
            first_orders,
            modeling_end_date=(
                REPEAT_PURCHASE_MODELING_END_DATE
            ),
            require_complete_followup=True,
        )
    )

    assert result.empty


# ================================================================
# TEMPORAL FEATURES
# ================================================================

def test_temporal_features_match_timestamp():
    data = pd.DataFrame(
        {
            "first_order_date": [
                pd.Timestamp(
                    "2010-01-09 14:30:00"
                )
            ]
        }
    )

    result = (
        add_first_order_temporal_features(
            data
        )
    )

    assert (
        result[
            "first_order_month_number"
        ].iloc[
            0
        ]
        ==
        1
    )

    assert (
        result[
            "first_order_weekday"
        ].iloc[
            0
        ]
        ==
        5
    )

    assert (
        result[
            "first_order_hour"
        ].iloc[
            0
        ]
        ==
        14
    )

    assert (
        result[
            "first_order_is_weekend"
        ].iloc[
            0
        ]
        ==
        1
    )


# ================================================================
# END-TO-END FEATURE DATASET
# ================================================================

def test_opening_month_customer_is_excluded():
    result = (
        build_repeat_purchase_historical_dataset(
            base_dataframe()
        )
    )

    assert "A" not in set(
        result[
            "customer"
        ]
    )


def test_repeat_customer_target_is_one():
    result = (
        build_repeat_purchase_historical_dataset(
            base_dataframe()
        )
    )

    b = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "B"
        ]
        .iloc[
            0
        ]
    )

    assert (
        b[
            "repeat_within_90d"
        ]
        ==
        1
    )


def test_late_repeat_customer_target_is_zero():
    result = (
        build_repeat_purchase_historical_dataset(
            base_dataframe()
        )
    )

    c = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "C"
        ]
        .iloc[
            0
        ]
    )

    assert (
        c[
            "repeat_within_90d"
        ]
        ==
        0
    )


def test_first_order_basket_features_exact():
    result = (
        build_repeat_purchase_historical_dataset(
            base_dataframe()
        )
    )

    b = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "B"
        ]
        .iloc[
            0
        ]
    )

    assert (
        b[
            "first_order_line_count"
        ]
        ==
        2
    )

    assert (
        b[
            "first_order_max_unit_price"
        ]
        ==
        pytest.approx(
            10.0
        )
    )

    assert (
        b[
            "first_order_min_unit_price"
        ]
        ==
        pytest.approx(
            5.0
        )
    )

    assert (
        b[
            "first_order_max_line_quantity"
        ]
        ==
        3
    )

    assert (
        b[
            "first_order_mean_line_quantity"
        ]
        ==
        pytest.approx(
            2.5
        )
    )

    assert (
        b[
            "units_per_product"
        ]
        ==
        pytest.approx(
            2.5
        )
    )

    assert (
        b[
            "revenue_per_product"
        ]
        ==
        pytest.approx(
            20.0
        )
    )

    assert (
        b[
            "revenue_per_unit"
        ]
        ==
        pytest.approx(
            8.0
        )
    )


def test_single_line_price_std_remains_missing():
    result = (
        build_repeat_purchase_historical_dataset(
            base_dataframe()
        )
    )

    c = (
        result.loc[
            result[
                "customer"
            ]
            ==
            "C"
        ]
        .iloc[
            0
        ]
    )

    assert pd.isna(
        c[
            "first_order_price_std"
        ]
    )


def test_scoring_features_do_not_include_target():
    result = (
        build_repeat_purchase_scoring_features(
            base_dataframe()
        )
    )

    assert (
        "repeat_within_90d"
        not in result.columns
    )

    assert (
        "second_order_date"
        not in result.columns
    )


def test_scoring_features_have_all_frozen_predictors():
    result = (
        build_repeat_purchase_scoring_features(
            base_dataframe()
        )
    )

    for feature in (
        REPEAT_PURCHASE_MODEL_FEATURES
    ):
        assert feature in result.columns


def test_select_model_features_uses_exact_order():
    result = (
        build_repeat_purchase_scoring_features(
            base_dataframe()
        )
    )

    selected = (
        select_repeat_purchase_model_features(
            result
        )
    )

    assert (
        tuple(
            selected.columns
        )
        ==
        REPEAT_PURCHASE_MODEL_FEATURES
    )


def test_select_model_features_fails_on_missing_feature():
    result = (
        build_repeat_purchase_scoring_features(
            base_dataframe()
        )
    )

    result = (
        result.drop(
            columns=[
                "first_order_value",
            ]
        )
    )

    with pytest.raises(
        KeyError
    ):
        select_repeat_purchase_model_features(
            result
        )