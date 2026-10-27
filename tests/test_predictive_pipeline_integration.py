"""
Integration tests for production predictive feature -> scorer pipelines.

These tests deliberately cross module boundaries.

They prove that raw production-style retail transactions can pass through
the production model-feature adapters and then through the real governed
production scoring interfaces without relying on research notebooks or
permanent local model artifacts.
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.features.fcv_model_features import (
    build_fcv_model_features,
)
from src.features.leakage_model_features import (
    build_leakage_model_features,
)
from src.features.reactivation_model_features import (
    build_reactivation_model_features,
)
from src.features.repeat_purchase_features import (
    build_repeat_purchase_scoring_features,
)

from src.models.fcv_scorer import (
    FCVScorer,
    sha256_file as fcv_sha256_file,
)
from src.models.fcv_spec import (
    FCV_COMPLETE_FORMULA,
    FCV_COMPLETE_MODEL_NAME,
    FCV_FINAL_DECISION,
    FCV_MODEL_FEATURES,
    FCV_MODEL_METADATA,
    FCV_STAGE_A_MODEL_METADATA,
    FCV_STAGE_A_RANDOM_FOREST_PARAMS,
    FCV_STAGE_B_FORMULA,
    FCV_STAGE_B_MODEL_NAME,
    FCV_STAGE_B_RUN_RATE_MULTIPLIER,
    build_frozen_fcv_stage_a_pipeline,
)

from src.models.leakage_scorer import (
    LeakageScorer,
    sha256_file as leakage_sha256_file,
)
from src.models.leakage_spec import (
    LEAKAGE_GRADIENT_BOOSTING_PARAMS,
    LEAKAGE_MODEL_FEATURES,
    LEAKAGE_MODEL_METADATA,
    build_frozen_leakage_pipeline,
)

from src.models.reactivation_scorer import (
    ReactivationScorer,
    sha256_file as reactivation_sha256_file,
)
from src.models.reactivation_spec import (
    REACTIVATION_MODEL_FEATURES,
    REACTIVATION_MODEL_METADATA,
    REACTIVATION_RANDOM_FOREST_PARAMS,
    build_frozen_reactivation_pipeline,
)

from src.models.repeat_purchase_scorer import (
    REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS,
    RepeatPurchaseScorer,
)
from src.models.repeat_purchase_spec import (
    REPEAT_PURCHASE_MODEL_FEATURES,
)


REACTIVATION_PREDICTION_DATE = pd.Timestamp(
    "2011-12-31 23:59:59"
)

CUSTOMER_PREDICTION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


# ================================================================
# SYNTHETIC FITTED ARTIFACTS
# ================================================================


def make_reactivation_training_data(
    n_rows: int = 120,
):
    """
    Deterministic training data using the exact frozen Reactivation register.
    """

    rng = np.random.default_rng(
        42
    )

    data = {}

    for feature in REACTIVATION_MODEL_FEATURES:

        if feature == "last_order_region":

            data[
                feature
            ] = rng.choice(
                [
                    "UK",
                    "France",
                    "Germany",
                ],
                size=n_rows,
            )

        elif feature in {
            "has_prior_purchase_gap",
            "has_prior_inactivity_episode",
            "has_prior_successful_reactivation",
            "prediction_is_weekend",
        }:

            data[
                feature
            ] = rng.integers(
                0,
                2,
                size=n_rows,
            )

        elif feature == "prediction_month":

            data[
                feature
            ] = rng.integers(
                1,
                13,
                size=n_rows,
            )

        elif feature == "prediction_weekday":

            data[
                feature
            ] = rng.integers(
                0,
                7,
                size=n_rows,
            )

        else:

            data[
                feature
            ] = (
                rng.random(
                    n_rows
                )
                *
                100
                +
                1
            )

    X = (
        pd.DataFrame(
            data
        )[
            list(
                REACTIVATION_MODEL_FEATURES
            )
        ]
    )

    y = pd.Series(
        [
            0,
            1,
        ]
        *
        (
            n_rows
            //
            2
        ),
        name="target",
    )

    return X, y


def make_leakage_training_data(
    n_rows: int = 80,
):
    """
    Deterministic training data using the exact Leakage feature register.
    """

    rng = np.random.default_rng(
        42
    )

    data = {}

    for feature in LEAKAGE_MODEL_FEATURES:

        if feature == "last_region":

            data[
                feature
            ] = rng.choice(
                [
                    "UK",
                    "France",
                    "Germany",
                ],
                size=n_rows,
            )

        elif feature == (
            "has_prior_completed_leakage_episode"
        ):

            data[
                feature
            ] = rng.integers(
                0,
                2,
                size=n_rows,
            )

        elif feature in {
            "prediction_month_sin",
            "prediction_month_cos",
        }:

            data[
                feature
            ] = rng.uniform(
                -1.0,
                1.0,
                size=n_rows,
            )

        elif feature == "prior_leakage_rate":

            values = rng.random(
                n_rows
            )

            values[
                ::10
            ] = np.nan

            data[
                feature
            ] = values

        else:

            data[
                feature
            ] = (
                rng.random(
                    n_rows
                )
                *
                100
                +
                1
            )

    X = (
        pd.DataFrame(
            data
        )[
            list(
                LEAKAGE_MODEL_FEATURES
            )
        ]
    )

    y = pd.Series(
        [
            0,
            1,
        ]
        *
        (
            n_rows
            //
            2
        ),
        name="target",
    )

    return X, y


def make_fcv_training_data(
    n_rows: int = 100,
):
    """
    Deterministic training data using the exact frozen FCV feature register.
    """

    rng = np.random.default_rng(
        42
    )

    data = {}

    binary_features = {
        "active_in_last_30d",
        "active_in_last_90d",
    }

    potentially_missing_features = {
        "r90_average_order_value",
        "p90_average_order_value",
        "r90_to_p90_revenue_ratio",
        "h180_average_gap_days",
        "h180_maximum_gap_days",
        "h180_minimum_gap_days",
    }

    for feature in FCV_MODEL_FEATURES:

        if feature == "last_region":

            data[
                feature
            ] = rng.choice(
                [
                    "UK",
                    "France",
                    "Germany",
                ],
                size=n_rows,
            )

        elif feature in binary_features:

            data[
                feature
            ] = rng.integers(
                0,
                2,
                size=n_rows,
            )

        elif feature in {
            "prediction_month_sin",
            "prediction_month_cos",
        }:

            data[
                feature
            ] = rng.uniform(
                -1.0,
                1.0,
                size=n_rows,
            )

        elif feature in potentially_missing_features:

            values = (
                rng.random(
                    n_rows
                )
                *
                100
                +
                1
            )

            values[
                ::10
            ] = np.nan

            data[
                feature
            ] = values

        else:

            data[
                feature
            ] = (
                rng.random(
                    n_rows
                )
                *
                100
                +
                1
            )

    X = (
        pd.DataFrame(
            data
        )[
            list(
                FCV_MODEL_FEATURES
            )
        ]
    )

    y = pd.Series(
        [
            0,
            1,
        ]
        *
        (
            n_rows
            //
            2
        ),
        name="target",
    )

    return X, y


def build_reactivation_test_artifact(
    tmp_path,
):

    X, y = (
        make_reactivation_training_data()
    )

    model = (
        build_frozen_reactivation_pipeline()
    )

    model.fit(
        X,
        y,
    )

    model_path = (
        tmp_path
        /
        "reactivation.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "reactivation_manifest.json"
    )

    manifest = {
        "model_name":
            REACTIVATION_MODEL_METADATA.model_name,

        "model_version":
            REACTIVATION_MODEL_METADATA.model_version,

        "feature_count":
            len(
                REACTIVATION_MODEL_FEATURES
            ),

        "features":
            list(
                REACTIVATION_MODEL_FEATURES
            ),

        "algorithm":
            "RandomForestClassifier",

        "hyperparameters":
            dict(
                REACTIVATION_RANDOM_FOREST_PARAMS
            ),

        "model_artifact_sha256":
            reactivation_sha256_file(
                model_path
            ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        model_path,
        manifest_path,
    )


def build_leakage_test_artifact(
    tmp_path,
):

    X, y = (
        make_leakage_training_data()
    )

    model = (
        build_frozen_leakage_pipeline()
    )

    model.fit(
        X,
        y,
    )

    model_path = (
        tmp_path
        /
        "leakage.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "leakage_manifest.json"
    )

    manifest = {
        "model_name":
            LEAKAGE_MODEL_METADATA.model_name,

        "model_version":
            LEAKAGE_MODEL_METADATA.model_version,

        "model_type":
            LEAKAGE_MODEL_METADATA.model_type,

        "target_name":
            LEAKAGE_MODEL_METADATA.target_name,

        "feature_count":
            len(
                LEAKAGE_MODEL_FEATURES
            ),

        "features":
            list(
                LEAKAGE_MODEL_FEATURES
            ),

        "algorithm":
            "GradientBoostingClassifier",

        "hyperparameters":
            dict(
                LEAKAGE_GRADIENT_BOOSTING_PARAMS
            ),

        "model_artifact_sha256":
            leakage_sha256_file(
                model_path
            ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        model_path,
        manifest_path,
    )


def build_fcv_test_artifact(
    tmp_path,
):

    X, y = (
        make_fcv_training_data()
    )

    model = (
        build_frozen_fcv_stage_a_pipeline()
    )

    model.fit(
        X,
        y,
    )

    model_path = (
        tmp_path
        /
        "fcv.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    manifest_path = (
        tmp_path
        /
        "fcv_manifest.json"
    )

    manifest = {
        "model_name":
            FCV_STAGE_A_MODEL_METADATA.model_name,

        "model_version":
            FCV_STAGE_A_MODEL_METADATA.model_version,

        "model_type":
            FCV_STAGE_A_MODEL_METADATA.model_type,

        "target_name":
            FCV_STAGE_A_MODEL_METADATA.target_name,

        "feature_count":
            len(
                FCV_MODEL_FEATURES
            ),

        "features":
            list(
                FCV_MODEL_FEATURES
            ),

        "algorithm":
            "RandomForestClassifier",

        "hyperparameters":
            dict(
                FCV_STAGE_A_RANDOM_FOREST_PARAMS
            ),

        "stage_b_model_name":
            FCV_STAGE_B_MODEL_NAME,

        "stage_b_formula":
            FCV_STAGE_B_FORMULA,

        "stage_b_multiplier":
            FCV_STAGE_B_RUN_RATE_MULTIPLIER,

        "complete_model_name":
            FCV_COMPLETE_MODEL_NAME,

        "complete_model_formula":
            FCV_COMPLETE_FORMULA,

        "research_final_decision":
            FCV_FINAL_DECISION,

        "model_artifact_sha256":
            fcv_sha256_file(
                model_path
            ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        model_path,
        manifest_path,
    )


# ================================================================
# RAW TRANSACTION FIXTURES
# ================================================================


def make_reactivation_transactions():

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-05-01",
                    "2011-09-15",
                    "2011-12-15",
                ]
            ),

            "customer": [
                "A",
                "A",
                "B",
                "C",
            ],

            "invoice": [
                "A1",
                "A2",
                "B1",
                "C1",
            ],

            "product": [
                "P1",
                "P2",
                "P3",
                "P4",
            ],

            "region": [
                "UK",
                "UK",
                "France",
                "Germany",
            ],

            "quantity": [
                1,
                2,
                1,
                1,
            ],

            "unit_price": [
                10.0,
                15.0,
                20.0,
                30.0,
            ],

            "net_merchandise_revenue": [
                10.0,
                30.0,
                20.0,
                30.0,
            ],

            "is_cancellation": [
                False,
                False,
                False,
                False,
            ],

            "is_return": [
                False,
                False,
                False,
                False,
            ],
        }
    )


def make_leakage_transactions():

    return pd.DataFrame(
        [
            {
                "date": "2011-03-15",
                "customer": "A",
                "invoice": "A1",
                "product": "P1",
                "region": "UK",
                "quantity": 1,
                "unit_price": 100.0,
                "net_merchandise_revenue": 100.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-05-15",
                "customer": "A",
                "invoice": "A2",
                "product": "P2",
                "region": "UK",
                "quantity": 2,
                "unit_price": 100.0,
                "net_merchandise_revenue": 200.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-06-15",
                "customer": "A",
                "invoice": "A3",
                "product": "P1",
                "region": "UK",
                "quantity": 1,
                "unit_price": 100.0,
                "net_merchandise_revenue": 100.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-07-15",
                "customer": "A",
                "invoice": "A4",
                "product": "P2",
                "region": "France",
                "quantity": 2,
                "unit_price": 100.0,
                "net_merchandise_revenue": 200.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-08-20",
                "customer": "A",
                "invoice": "A5",
                "product": "P3",
                "region": "Germany",
                "quantity": 3,
                "unit_price": 100.0,
                "net_merchandise_revenue": 300.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-07-01",
                "customer": "B",
                "invoice": "B1",
                "product": "P4",
                "region": "UK",
                "quantity": 1,
                "unit_price": 80.0,
                "net_merchandise_revenue": 80.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2010-12-01",
                "customer": "H",
                "invoice": "H1",
                "product": "PX",
                "region": "UK",
                "quantity": 1,
                "unit_price": 10.0,
                "net_merchandise_revenue": 10.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-09-15",
                "customer": "A",
                "invoice": "A6",
                "product": "FUTURE",
                "region": "USA",
                "quantity": 50,
                "unit_price": 100.0,
                "net_merchandise_revenue": 5000.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-08-25",
                "customer": "A",
                "invoice": "RET1",
                "product": "P1",
                "region": "Germany",
                "quantity": -1,
                "unit_price": 100.0,
                "net_merchandise_revenue": -100.0,
                "is_cancellation": False,
                "is_return": True,
            },
        ]
    )


def make_fcv_transactions():

    return pd.DataFrame(
        [
            {
                "date": "2011-03-01",
                "customer": "A",
                "invoice": "I1",
                "product": "P1",
                "region": "North",
                "quantity": 2,
                "unit_price": 50.0,
                "net_merchandise_revenue": 100.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-03-15",
                "customer": "A",
                "invoice": "I2",
                "product": "P2",
                "region": "North",
                "quantity": 1,
                "unit_price": 200.0,
                "net_merchandise_revenue": 200.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-05-15",
                "customer": "A",
                "invoice": "I3",
                "product": "P1",
                "region": "South",
                "quantity": 3,
                "unit_price": 100.0,
                "net_merchandise_revenue": 300.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-07-15",
                "customer": "A",
                "invoice": "I4",
                "product": "P3",
                "region": "East",
                "quantity": 4,
                "unit_price": 100.0,
                "net_merchandise_revenue": 400.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-04-15",
                "customer": "B",
                "invoice": "I5",
                "product": "P4",
                "region": "West",
                "quantity": 1,
                "unit_price": 150.0,
                "net_merchandise_revenue": 150.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-08-10",
                "customer": "C",
                "invoice": "I6",
                "product": "P5",
                "region": "Central",
                "quantity": 2,
                "unit_price": 75.0,
                "net_merchandise_revenue": 150.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2010-12-01",
                "customer": "OLD",
                "invoice": "OLD1",
                "product": "OLDP",
                "region": "North",
                "quantity": 1,
                "unit_price": 50.0,
                "net_merchandise_revenue": 50.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-09-15",
                "customer": "FUTURE",
                "invoice": "F1",
                "product": "PF",
                "region": "North",
                "quantity": 1,
                "unit_price": 500.0,
                "net_merchandise_revenue": 500.0,
                "is_cancellation": False,
                "is_return": False,
            },
            {
                "date": "2011-08-20",
                "customer": "RETURN",
                "invoice": "R1",
                "product": "PR",
                "region": "North",
                "quantity": -1,
                "unit_price": 100.0,
                "net_merchandise_revenue": -100.0,
                "is_cancellation": False,
                "is_return": True,
            },
        ]
    )


def make_repeat_purchase_line(
    date,
    invoice,
    customer,
    product,
    *,
    quantity=1,
    unit_price=10.0,
    region="United Kingdom",
):

    return {
        "date": date,
        "invoice": invoice,
        "customer": customer,
        "product": product,
        "product_description": (
            f"Product {product}"
        ),
        "region": region,
        "quantity": quantity,
        "unit_price": unit_price,
        "customer_is_identified": True,
        "is_merchandise": True,
        "is_cancellation": False,
        "net_merchandise_revenue": (
            quantity
            *
            unit_price
        ),
    }


def make_repeat_purchase_transactions():

    return pd.DataFrame(
        [
            make_repeat_purchase_line(
                "2009-12-10 09:00:00",
                "A1",
                "A",
                "P1",
            ),

            make_repeat_purchase_line(
                "2010-01-10 10:00:00",
                "B1",
                "B",
                "P1",
                quantity=2,
                unit_price=5.0,
            ),

            make_repeat_purchase_line(
                "2010-01-10 10:05:00",
                "B1",
                "B",
                "P2",
                quantity=3,
                unit_price=10.0,
            ),

            make_repeat_purchase_line(
                "2010-02-15 11:00:00",
                "B2",
                "B",
                "P3",
                quantity=1,
                unit_price=20.0,
            ),

            make_repeat_purchase_line(
                "2010-03-01 14:00:00",
                "C1",
                "C",
                "P4",
                quantity=4,
                unit_price=7.0,
            ),

            make_repeat_purchase_line(
                "2010-07-01 14:00:00",
                "C2",
                "C",
                "P5",
                quantity=1,
                unit_price=5.0,
            ),
        ]
    )


class RepeatPurchaseDummyModel:

    def predict_proba(
        self,
        X,
    ):

        positive = np.full(
            len(
                X
            ),
            0.60,
            dtype=float,
        )

        negative = (
            1.0
            -
            positive
        )

        return np.column_stack(
            [
                negative,
                positive,
            ]
        )


# ================================================================
# REACTIVATION INTEGRATION
# ================================================================


def test_reactivation_raw_transactions_build_exact_model_features():

    features = (
        build_reactivation_model_features(
            make_reactivation_transactions(),
            REACTIVATION_PREDICTION_DATE,
        )
    )

    assert not features.empty

    assert list(
        features.columns[
            1:
        ]
    ) == list(
        REACTIVATION_MODEL_FEATURES
    )

    assert set(
        features[
            "customer"
        ]
    ) == {
        "A",
        "B",
    }


def test_reactivation_model_features_to_real_scorer(
    tmp_path,
):

    model_path, manifest_path = (
        build_reactivation_test_artifact(
            tmp_path
        )
    )

    features = (
        build_reactivation_model_features(
            make_reactivation_transactions(),
            REACTIVATION_PREDICTION_DATE,
        )
    )

    scorer = (
        ReactivationScorer(
            artifact_path=model_path,
            manifest_path=manifest_path,
        )
    )

    scores = (
        scorer.score(
            features
        )
    )

    assert (
        scores[
            "customer"
        ].tolist()
        ==
        features[
            "customer"
        ].tolist()
    )

    assert (
        scores[
            "reactivation_probability"
        ]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
    )

    assert (
        scores[
            "model_name"
        ]
        .eq(
            REACTIVATION_MODEL_METADATA.model_name
        )
        .all()
    )


# ================================================================
# LEAKAGE INTEGRATION
# ================================================================


def test_leakage_raw_transactions_build_exact_model_features():

    features = (
        build_leakage_model_features(
            make_leakage_transactions(),
            CUSTOMER_PREDICTION_DATE,
        )
    )

    assert not features.empty

    assert set(
        LEAKAGE_MODEL_FEATURES
    ).issubset(
        set(
            features.columns
        )
    )

    assert set(
        features[
            "customer"
        ]
    ) == {
        "A",
        "B",
    }


def test_leakage_model_features_to_real_scorer(
    tmp_path,
):

    model_path, manifest_path = (
        build_leakage_test_artifact(
            tmp_path
        )
    )

    features = (
        build_leakage_model_features(
            make_leakage_transactions(),
            CUSTOMER_PREDICTION_DATE,
        )
    )

    scorer = (
        LeakageScorer(
            artifact_path=model_path,
            manifest_path=manifest_path,
        )
    )

    scores = (
        scorer.score(
            features
        )
    )

    assert (
        scores[
            "customer"
        ].tolist()
        ==
        features[
            "customer"
        ].tolist()
    )

    assert (
        scores[
            "revenue_leakage_probability"
        ]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
    )

    assert (
        scores[
            "model_name"
        ]
        .eq(
            LEAKAGE_MODEL_METADATA.model_name
        )
        .all()
    )


# ================================================================
# FCV INTEGRATION
# ================================================================


def test_fcv_raw_transactions_build_exact_model_features():

    features = (
        build_fcv_model_features(
            make_fcv_transactions(),
            CUSTOMER_PREDICTION_DATE,
        )
    )

    assert not features.empty

    assert set(
        FCV_MODEL_FEATURES
    ).issubset(
        set(
            features.columns
        )
    )

    assert "FUTURE" not in set(
        features[
            "customer"
        ]
    )

    assert "OLD" not in set(
        features[
            "customer"
        ]
    )

    assert "RETURN" not in set(
        features[
            "customer"
        ]
    )


def test_fcv_model_features_to_real_hurdle_scorer(
    tmp_path,
):

    model_path, manifest_path = (
        build_fcv_test_artifact(
            tmp_path
        )
    )

    features = (
        build_fcv_model_features(
            make_fcv_transactions(),
            CUSTOMER_PREDICTION_DATE,
        )
    )

    scorer = (
        FCVScorer(
            artifact_path=model_path,
            manifest_path=manifest_path,
        )
    )

    scores = (
        scorer.score(
            features
        )
    )

    assert (
        scores[
            "customer"
        ].tolist()
        ==
        features[
            "customer"
        ].tolist()
    )

    assert (
        scores[
            "future_active_probability"
        ]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
    )

    expected = (
        scores[
            "future_active_probability"
        ]
        *
        0.50
        *
        scores[
            "fcv_history_180d_revenue"
        ]
    )

    np.testing.assert_allclose(
        scores[
            "future_customer_value_90d"
        ],
        expected,
    )

    assert (
        scores[
            "model_name"
        ]
        .eq(
            FCV_MODEL_METADATA.model_name
        )
        .all()
    )


# ================================================================
# REPEAT PURCHASE INTEGRATION
# ================================================================


def test_repeat_purchase_raw_transactions_build_model_ready_features():

    features = (
        build_repeat_purchase_scoring_features(
            make_repeat_purchase_transactions()
        )
    )

    assert not features.empty

    assert set(
        REPEAT_PURCHASE_MODEL_FEATURES
    ).issubset(
        set(
            features.columns
        )
    )

    assert set(
        features[
            "customer"
        ]
    ) == {
        "B",
        "C",
    }


def test_repeat_purchase_raw_transactions_to_production_scorer():

    scorer = (
        RepeatPurchaseScorer(
            model=(
                RepeatPurchaseDummyModel()
            )
        )
    )

    scores = (
        scorer.score(
            make_repeat_purchase_transactions()
        )
    )

    assert tuple(
        scores.columns
    ) == REPEAT_PURCHASE_SCORE_OUTPUT_COLUMNS

    assert set(
        scores[
            "customer"
        ]
    ) == {
        "B",
        "C",
    }

    assert (
        scores[
            "repeat_purchase_score"
        ]
        .eq(
            0.60
        )
        .all()
    )

    assert (
        scores[
            "governance_decision"
        ]
        .notna()
        .all()
    )