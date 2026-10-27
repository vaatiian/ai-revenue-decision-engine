import pandas as pd
import pytest

from src.decision_engine.snapshot import (
    DECISION_SNAPSHOT_COLUMNS,
    DECISION_SNAPSHOT_WINDOWS,
    build_decision_snapshot,
    validate_decision_snapshot,
)


# ================================================================
# TEST DATA
# ================================================================


DECISION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


def make_transactions():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    # Customer 1001
                    "2011-01-01 10:00:00",
                    "2011-08-15 10:00:00",

                    # Customer 1002
                    "2011-06-20 10:00:00",

                    # Customer 1003
                    "2011-04-01 10:00:00",

                    # Customer 1004
                    "2011-01-01 10:00:00",

                    # Future transaction - must be excluded
                    "2011-09-10 10:00:00",

                    # Return - must be excluded
                    "2011-08-20 10:00:00",

                    # Cancellation - must be excluded
                    "2011-08-21 10:00:00",

                    # Missing customer - must be excluded
                    "2011-08-22 10:00:00",
                ]
            ),

            "customer": [
                "1001",
                "1001",
                "1002",
                "1003",
                "1004",
                "1001",
                "1002",
                "1003",
                pd.NA,
            ],

            "invoice": [
                "I001",
                "I002",
                "I003",
                "I004",
                "I005",
                "I006",
                "I007",
                "I008",
                "I009",
            ],

            "quantity": [
                1,
                2,
                1,
                1,
                1,
                10,
                -1,
                1,
                1,
            ],

            "unit_price": [
                100.0,
                100.0,
                200.0,
                300.0,
                400.0,
                100.0,
                200.0,
                300.0,
                500.0,
            ],

            "net_merchandise_revenue": [
                100.0,
                200.0,
                200.0,
                300.0,
                400.0,
                1000.0,
                -200.0,
                300.0,
                500.0,
            ],

            "is_cancellation": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
            ],

            "is_return": [
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
            ],
        }
    )


# ================================================================
# CONTRACT
# ================================================================


def test_decision_snapshot_windows_are_frozen():

    assert (
        DECISION_SNAPSHOT_WINDOWS
        ==
        (
            30,
            90,
            180,
        )
    )


def test_decision_snapshot_columns_are_exact():

    assert (
        DECISION_SNAPSHOT_COLUMNS
        ==
        (
            "customer",
            "decision_date",
            "first_order_date",
            "last_order_date",
            "customer_state",
            "days_since_last_order",
            "customer_tenure_days",
            "lifetime_orders",
            "lifetime_revenue",
            "orders_prior_30d",
            "orders_prior_90d",
            "orders_prior_180d",
            "revenue_prior_30d",
            "revenue_prior_90d",
            "revenue_prior_180d",
        )
    )


# ================================================================
# BUILD
# ================================================================


def test_build_decision_snapshot_has_exact_schema():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    assert tuple(
        snapshot.columns
    ) == DECISION_SNAPSHOT_COLUMNS


def test_build_decision_snapshot_has_one_row_per_customer():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    assert (
        snapshot[
            "customer"
        ].is_unique
    )

    assert len(
        snapshot
    ) == 4


def test_missing_customer_is_excluded():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    assert (
        snapshot[
            "customer"
        ].notna().all()
    )


def test_future_transaction_is_excluded():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    customer = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "1001"
        ]
        .iloc[0]
    )

    assert (
        customer[
            "lifetime_revenue"
        ]
        ==
        pytest.approx(
            300.0
        )
    )

    assert (
        customer[
            "lifetime_orders"
        ]
        ==
        2
    )


def test_returns_and_cancellations_are_excluded():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    customer_1002 = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "1002"
        ]
        .iloc[0]
    )

    customer_1003 = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "1003"
        ]
        .iloc[0]
    )

    assert (
        customer_1002[
            "lifetime_revenue"
        ]
        ==
        pytest.approx(
            200.0
        )
    )

    assert (
        customer_1003[
            "lifetime_revenue"
        ]
        ==
        pytest.approx(
            300.0
        )
    )


def test_decision_date_is_constant():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    assert (
        snapshot[
            "decision_date"
        ]
        ==
        DECISION_DATE
    ).all()


# ================================================================
# LIFECYCLE
# ================================================================


def test_active_customer_state():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    state = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "1001",
            "customer_state",
        ]
        .iloc[0]
    )

    assert state == "Active"


def test_at_risk_customer_state():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    state = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "1002",
            "customer_state",
        ]
        .iloc[0]
    )

    assert state == "At Risk"


def test_dormant_customer_state():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    state = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "1003",
            "customer_state",
        ]
        .iloc[0]
    )

    assert state == "Dormant"


def test_long_dormant_customer_state():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    state = (
        snapshot.loc[
            snapshot[
                "customer"
            ]
            ==
            "1004",
            "customer_state",
        ]
        .iloc[0]
    )

    assert state == "Long Dormant"


# ================================================================
# WINDOWS
# ================================================================


def test_recent_window_features_are_present():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    required = {
        "orders_prior_30d",
        "orders_prior_90d",
        "orders_prior_180d",
        "revenue_prior_30d",
        "revenue_prior_90d",
        "revenue_prior_180d",
    }

    assert required.issubset(
        snapshot.columns
    )


def test_recent_window_features_are_non_negative():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    columns = [
        "orders_prior_30d",
        "orders_prior_90d",
        "orders_prior_180d",
        "revenue_prior_30d",
        "revenue_prior_90d",
        "revenue_prior_180d",
    ]

    assert (
        snapshot[
            columns
        ]
        >=
        0
    ).all().all()


# ================================================================
# EMPTY POPULATION
# ================================================================


def test_empty_population_preserves_exact_schema():

    transactions = (
        make_transactions()
        .copy()
    )

    snapshot = (
        build_decision_snapshot(
            transactions=transactions,
            decision_date=(
                pd.Timestamp(
                    "2010-01-01"
                )
            ),
        )
    )

    assert snapshot.empty

    assert tuple(
        snapshot.columns
    ) == DECISION_SNAPSHOT_COLUMNS


# ================================================================
# VALIDATION
# ================================================================


def test_validation_accepts_valid_snapshot():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    assert (
        validate_decision_snapshot(
            snapshot=snapshot,
            decision_date=DECISION_DATE,
        )
        is None
    )


def test_validation_rejects_duplicate_customers():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    duplicate = pd.concat(
        [
            snapshot,
            snapshot.iloc[
                [0]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="one row per customer",
    ):
        validate_decision_snapshot(
            snapshot=duplicate,
            decision_date=DECISION_DATE,
        )


def test_validation_rejects_future_last_order_date():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    broken = (
        snapshot.copy()
    )

    broken.loc[
        0,
        "last_order_date",
    ] = (
        DECISION_DATE
        +
        pd.Timedelta(
            days=1
        )
    )

    with pytest.raises(
        ValueError,
        match="Future transaction leakage",
    ):
        validate_decision_snapshot(
            snapshot=broken,
            decision_date=DECISION_DATE,
        )


def test_validation_rejects_unknown_customer_state():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    broken = (
        snapshot.copy()
    )

    broken.loc[
        0,
        "customer_state",
    ] = "Mystery State"

    with pytest.raises(
        ValueError,
        match="unknown customer states",
    ):
        validate_decision_snapshot(
            snapshot=broken,
            decision_date=DECISION_DATE,
        )


def test_validation_rejects_wrong_column_order():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    wrong_order = (
        snapshot[
            list(
                reversed(
                    snapshot.columns
                )
            )
        ]
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="columns do not match",
    ):
        validate_decision_snapshot(
            snapshot=wrong_order,
            decision_date=DECISION_DATE,
        )


# ================================================================
# LAYER SEPARATION
# ================================================================


def test_snapshot_contains_no_model_signal_columns():

    snapshot = (
        build_decision_snapshot(
            transactions=make_transactions(),
            decision_date=DECISION_DATE,
        )
    )

    forbidden = {
        "leakage_probability",
        "revenue_leakage_probability",
        "future_active_probability",
        "future_customer_value_90d",
        "reactivation_probability",
        "repeat_top1_product",
        "discovery_top1_product",
        "primary_action",
        "priority_tier",
        "contact_intensity",
    }

    assert not (
        forbidden
        &
        set(
            snapshot.columns
        )
    )