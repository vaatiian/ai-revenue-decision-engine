"""
Export the exact frozen Repeat Purchase model fitted by
notebooks/02_ML_Repeat_Purchase_Model.ipynb.

Important governance rule:
This script does NOT rebuild, retrain, tune, or alter the model.

It:
1. hashes the original research notebook;
2. executes the notebook;
3. appends an export-only cell;
4. serializes the notebook's existing `final_refit_model`;
5. validates the fitted artifact against the frozen production spec;
6. writes a governance manifest;
7. saves the executed notebook as an audit artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


# ================================================================
# PROJECT ROOT / IMPORT PATH
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


# ================================================================
# THIRD-PARTY IMPORTS
# ================================================================

import joblib
import nbformat
import sklearn

from nbconvert.preprocessors import ExecutePreprocessor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ================================================================
# PROJECT IMPORTS
# ================================================================

from src.models.repeat_purchase_spec import (
    REPEAT_PURCHASE_BUSINESS_TARGET,
    REPEAT_PURCHASE_CALIBRATION,
    REPEAT_PURCHASE_CATEGORICAL_FEATURES,
    REPEAT_PURCHASE_CATEGORICAL_IMPUTER_STRATEGY,
    REPEAT_PURCHASE_COMMERCIAL_METRICS,
    REPEAT_PURCHASE_FINAL_DECISION,
    REPEAT_PURCHASE_FINAL_EVALUATION_POPULATION,
    REPEAT_PURCHASE_FINAL_TRAINING_POPULATION,
    REPEAT_PURCHASE_HORIZON_DAYS,
    REPEAT_PURCHASE_LOGISTIC_PARAMS,
    REPEAT_PURCHASE_MODEL_FEATURES,
    REPEAT_PURCHASE_MODEL_NAME,
    REPEAT_PURCHASE_MODEL_TYPE,
    REPEAT_PURCHASE_MODEL_VERSION,
    REPEAT_PURCHASE_NUMERIC_FEATURES,
    REPEAT_PURCHASE_NUMERIC_IMPUTER_STRATEGY,
    REPEAT_PURCHASE_NUMERIC_SCALER,
    REPEAT_PURCHASE_ONEHOT_DROP,
    REPEAT_PURCHASE_ONEHOT_HANDLE_UNKNOWN,
    REPEAT_PURCHASE_PREDICTION_POINT,
    REPEAT_PURCHASE_PREDICTION_UNIT,
    REPEAT_PURCHASE_PRIMARY_METRIC,
    REPEAT_PURCHASE_PROHIBITED_INTERPRETATIONS,
    REPEAT_PURCHASE_RECOMMENDED_USE,
    REPEAT_PURCHASE_REPORTED_CALIBRATION_ERROR_PP,
    REPEAT_PURCHASE_REPORTED_TEST_BRIER,
    REPEAT_PURCHASE_REPORTED_TEST_LIFT_10,
    REPEAT_PURCHASE_REPORTED_TEST_LIFT_20,
    REPEAT_PURCHASE_REPORTED_TEST_PR_AUC,
    REPEAT_PURCHASE_REPORTED_TEST_ROC_AUC,
    REPEAT_PURCHASE_TARGET_NAME,
    REPEAT_PURCHASE_TEST_SET_CONSUMED,
)


# ================================================================
# PATHS
# ================================================================

NOTEBOOK_PATH = (
    PROJECT_ROOT
    /
    "notebooks"
    /
    "02_ML_Repeat_Purchase_Model.ipynb"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "models"
    /
    "repeat_purchase"
)

MODEL_PATH = (
    ARTIFACT_DIR
    /
    "repeat_purchase_logistic_v1.0.0.joblib"
)

MANIFEST_PATH = (
    ARTIFACT_DIR
    /
    "repeat_purchase_logistic_v1.0.0.json"
)

EXECUTED_NOTEBOOK_PATH = (
    ARTIFACT_DIR
    /
    "executed_02_ML_Repeat_Purchase_Model.ipynb"
)

SPEC_PATH = (
    PROJECT_ROOT
    /
    "src"
    /
    "models"
    /
    "repeat_purchase_spec.py"
)

FEATURE_PATH = (
    PROJECT_ROOT
    /
    "src"
    /
    "features"
    /
    "repeat_purchase_features.py"
)

SCORER_PATH = (
    PROJECT_ROOT
    /
    "src"
    /
    "models"
    /
    "repeat_purchase_scorer.py"
)


# ================================================================
# HASHING
# ================================================================

def sha256_file(
    path: Path,
) -> str:
    """
    Calculate SHA-256 for a file.
    """

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


# ================================================================
# SOURCE VALIDATION
# ================================================================

def validate_source_files() -> None:
    """
    Fail closed if required research or production sources are missing.
    """

    required_paths = (
        NOTEBOOK_PATH,
        SPEC_PATH,
        FEATURE_PATH,
        SCORER_PATH,
    )

    missing = [
        str(
            path
        )
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required Repeat Purchase source files are missing: "
            f"{missing}"
        )


# ================================================================
# NOTEBOOK EXPORT CELL
# ================================================================

def build_export_cell() -> str:
    """
    Build an export-only cell.

    The cell does not fit or modify the model.
    It only serializes the already-fitted final_refit_model.
    """

    model_path_literal = repr(
        str(
            MODEL_PATH
        )
    )

    return f'''
# ================================================================
# PRODUCTION EXPORT — APPENDED BY export_repeat_purchase_model.py
# ================================================================

from pathlib import Path
import joblib


_export_model_path = Path(
    {model_path_literal}
)

_export_model_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


if "final_refit_model" not in globals():
    raise RuntimeError(
        "Notebook did not create final_refit_model."
    )


if not hasattr(
    final_refit_model,
    "predict_proba",
):
    raise RuntimeError(
        "final_refit_model does not expose predict_proba()."
    )


joblib.dump(
    final_refit_model,
    _export_model_path,
)


print(
    "REPEAT PURCHASE MODEL SERIALIZED"
)

print(
    _export_model_path
)
'''


# ================================================================
# ARTIFACT VALIDATION
# ================================================================

def validate_exported_model(
    model,
) -> None:
    """
    Validate the serialized artifact against the frozen Notebook 02
    specification.
    """

    if not isinstance(
        model,
        Pipeline,
    ):
        raise TypeError(
            "Repeat Purchase artifact must be an sklearn Pipeline."
        )

    if list(
        model.named_steps
    ) != [
        "preprocessing",
        "model",
    ]:
        raise ValueError(
            "Unexpected Repeat Purchase pipeline step structure."
        )

    if not hasattr(
        model,
        "feature_names_in_",
    ):
        raise ValueError(
            "Repeat Purchase artifact is not fitted or lacks "
            "feature_names_in_."
        )

    actual_features = tuple(
        model.feature_names_in_
    )

    if (
        actual_features
        !=
        REPEAT_PURCHASE_MODEL_FEATURES
    ):
        raise ValueError(
            "Repeat Purchase artifact feature schema does not match "
            "the frozen 18-feature contract."
        )

    estimator = (
        model.named_steps[
            "model"
        ]
    )

    if not isinstance(
        estimator,
        LogisticRegression,
    ):
        raise TypeError(
            "Repeat Purchase estimator must be LogisticRegression."
        )

    if (
        estimator.max_iter
        !=
        REPEAT_PURCHASE_LOGISTIC_PARAMS[
            "max_iter"
        ]
    ):
        raise ValueError(
            "Repeat Purchase LogisticRegression max_iter drift detected."
        )

    if (
        estimator.random_state
        !=
        REPEAT_PURCHASE_LOGISTIC_PARAMS[
            "random_state"
        ]
    ):
        raise ValueError(
            "Repeat Purchase LogisticRegression random_state drift detected."
        )

    if not hasattr(
        estimator,
        "classes_",
    ):
        raise ValueError(
            "Repeat Purchase estimator is not fitted."
        )

    if set(
        estimator.classes_
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Repeat Purchase classifier must contain classes {0, 1}."
        )

    preprocessor = (
        model.named_steps[
            "preprocessing"
        ]
    )

    if not isinstance(
        preprocessor,
        ColumnTransformer,
    ):
        raise TypeError(
            "Repeat Purchase preprocessing must be a ColumnTransformer."
        )

    numeric_transformer = (
        preprocessor.named_transformers_[
            "numeric"
        ]
    )

    categorical_transformer = (
        preprocessor.named_transformers_[
            "categorical"
        ]
    )

    if not isinstance(
        numeric_transformer,
        Pipeline,
    ):
        raise TypeError(
            "Repeat Purchase numeric transformer must be a Pipeline."
        )

    if not isinstance(
        categorical_transformer,
        Pipeline,
    ):
        raise TypeError(
            "Repeat Purchase categorical transformer must be a Pipeline."
        )

    numeric_imputer = (
        numeric_transformer.named_steps[
            "imputer"
        ]
    )

    if not isinstance(
        numeric_imputer,
        SimpleImputer,
    ):
        raise TypeError(
            "Repeat Purchase numeric imputer must be SimpleImputer."
        )

    if (
        numeric_imputer.strategy
        !=
        REPEAT_PURCHASE_NUMERIC_IMPUTER_STRATEGY
    ):
        raise ValueError(
            "Repeat Purchase numeric imputer strategy drift detected."
        )

    if bool(
        numeric_imputer.add_indicator
    ):
        raise ValueError(
            "Repeat Purchase numeric imputer must not add indicators."
        )

    numeric_scaler = (
        numeric_transformer.named_steps[
            "scaler"
        ]
    )

    if not isinstance(
        numeric_scaler,
        StandardScaler,
    ):
        raise TypeError(
            "Repeat Purchase numeric scaler must be StandardScaler."
        )

    if (
        type(
            numeric_scaler
        ).__name__
        !=
        REPEAT_PURCHASE_NUMERIC_SCALER
    ):
        raise ValueError(
            "Repeat Purchase numeric scaler drift detected."
        )

    categorical_imputer = (
        categorical_transformer.named_steps[
            "imputer"
        ]
    )

    if not isinstance(
        categorical_imputer,
        SimpleImputer,
    ):
        raise TypeError(
            "Repeat Purchase categorical imputer must be SimpleImputer."
        )

    if (
        categorical_imputer.strategy
        !=
        REPEAT_PURCHASE_CATEGORICAL_IMPUTER_STRATEGY
    ):
        raise ValueError(
            "Repeat Purchase categorical imputer strategy drift detected."
        )

    onehot = (
        categorical_transformer.named_steps[
            "onehot"
        ]
    )

    if not isinstance(
        onehot,
        OneHotEncoder,
    ):
        raise TypeError(
            "Repeat Purchase categorical encoder must be OneHotEncoder."
        )

    if (
        onehot.handle_unknown
        !=
        REPEAT_PURCHASE_ONEHOT_HANDLE_UNKNOWN
    ):
        raise ValueError(
            "Repeat Purchase unknown-category policy drift detected."
        )

    if (
        onehot.drop
        !=
        REPEAT_PURCHASE_ONEHOT_DROP
    ):
        raise ValueError(
            "Repeat Purchase drop-first encoding drift detected."
        )


# ================================================================
# MANIFEST
# ================================================================

def build_manifest(
    notebook_sha256: str,
    model_sha256: str,
) -> dict:
    """
    Build the production governance manifest.
    """

    return {
        "model_name":
            REPEAT_PURCHASE_MODEL_NAME,

        "model_version":
            REPEAT_PURCHASE_MODEL_VERSION,

        "model_type":
            REPEAT_PURCHASE_MODEL_TYPE,

        "target_name":
            REPEAT_PURCHASE_TARGET_NAME,

        "business_target":
            REPEAT_PURCHASE_BUSINESS_TARGET,

        "prediction_point":
            REPEAT_PURCHASE_PREDICTION_POINT,

        "prediction_unit":
            REPEAT_PURCHASE_PREDICTION_UNIT,

        "prediction_horizon_days":
            REPEAT_PURCHASE_HORIZON_DAYS,

        "feature_count":
            len(
                REPEAT_PURCHASE_MODEL_FEATURES
            ),

        "numeric_feature_count":
            len(
                REPEAT_PURCHASE_NUMERIC_FEATURES
            ),

        "categorical_feature_count":
            len(
                REPEAT_PURCHASE_CATEGORICAL_FEATURES
            ),

        "features":
            list(
                REPEAT_PURCHASE_MODEL_FEATURES
            ),

        "numeric_features":
            list(
                REPEAT_PURCHASE_NUMERIC_FEATURES
            ),

        "categorical_features":
            list(
                REPEAT_PURCHASE_CATEGORICAL_FEATURES
            ),

        "algorithm":
            "LogisticRegression",

        "hyperparameters":
            dict(
                REPEAT_PURCHASE_LOGISTIC_PARAMS
            ),

        "preprocessing": {
            "numeric_imputation":
                REPEAT_PURCHASE_NUMERIC_IMPUTER_STRATEGY,

            "numeric_scaling":
                REPEAT_PURCHASE_NUMERIC_SCALER,

            "categorical_imputation":
                REPEAT_PURCHASE_CATEGORICAL_IMPUTER_STRATEGY,

            "onehot_handle_unknown":
                REPEAT_PURCHASE_ONEHOT_HANDLE_UNKNOWN,

            "onehot_drop":
                REPEAT_PURCHASE_ONEHOT_DROP,
        },

        "calibration":
            REPEAT_PURCHASE_CALIBRATION,

        "training_population":
            REPEAT_PURCHASE_FINAL_TRAINING_POPULATION,

        "evaluation_population":
            REPEAT_PURCHASE_FINAL_EVALUATION_POPULATION,

        "primary_metric":
            REPEAT_PURCHASE_PRIMARY_METRIC,

        "commercial_metrics":
            list(
                REPEAT_PURCHASE_COMMERCIAL_METRICS
            ),

        "reported_final_test_metrics": {
            "pr_auc":
                REPEAT_PURCHASE_REPORTED_TEST_PR_AUC,

            "roc_auc":
                REPEAT_PURCHASE_REPORTED_TEST_ROC_AUC,

            "brier_score":
                REPEAT_PURCHASE_REPORTED_TEST_BRIER,

            "lift_at_10":
                REPEAT_PURCHASE_REPORTED_TEST_LIFT_10,

            "lift_at_20":
                REPEAT_PURCHASE_REPORTED_TEST_LIFT_20,

            "calibration_error_percentage_points":
                REPEAT_PURCHASE_REPORTED_CALIBRATION_ERROR_PP,
        },

        "governance_decision":
            REPEAT_PURCHASE_FINAL_DECISION,

        "recommended_use":
            REPEAT_PURCHASE_RECOMMENDED_USE,

        "prohibited_interpretations":
            list(
                REPEAT_PURCHASE_PROHIBITED_INTERPRETATIONS
            ),

        "test_set_consumed":
            REPEAT_PURCHASE_TEST_SET_CONSUMED,

        "post_test_tuning_allowed":
            False,

        "serialized_model_artifact_required":
            True,

        "artifact_filename":
            MODEL_PATH.name,

        "source_notebook":
            str(
                NOTEBOOK_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "source_notebook_sha256":
            notebook_sha256,

        "model_artifact_sha256":
            model_sha256,

        "production_spec_sha256":
            sha256_file(
                SPEC_PATH
            ),

        "production_feature_engine_sha256":
            sha256_file(
                FEATURE_PATH
            ),

        "production_scorer_sha256":
            sha256_file(
                SCORER_PATH
            ),

        "sklearn_version":
            sklearn.__version__,
    }


# ================================================================
# EXPORT
# ================================================================

def export_repeat_purchase_model() -> None:
    """
    Execute Notebook 02 and export the exact frozen final_refit_model.
    """

    validate_source_files()

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 72
    )

    print(
        "REPEAT PURCHASE MODEL ARTIFACT EXPORT"
    )

    print(
        "=" * 72
    )

    print(
        f"Research notebook:\n"
        f"{NOTEBOOK_PATH}"
    )

    print()

    notebook_hash = (
        sha256_file(
            NOTEBOOK_PATH
        )
    )

    print(
        "Original notebook SHA-256:"
    )

    print(
        notebook_hash
    )

    print()

    with NOTEBOOK_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:

        notebook = nbformat.read(
            handle,
            as_version=4,
        )

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            build_export_cell()
        )
    )

    print(
        "Executing research notebook..."
    )

    executor = ExecutePreprocessor(
        timeout=1800,
        kernel_name="python3",
        allow_errors=False,
    )

    executor.preprocess(
        notebook,
        {
            "metadata": {
                "path":
                    str(
                        PROJECT_ROOT
                    )
            }
        },
    )

    with EXECUTED_NOTEBOOK_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        nbformat.write(
            notebook,
            handle,
        )

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Repeat Purchase model artifact was not created."
        )

    model = joblib.load(
        MODEL_PATH
    )

    validate_exported_model(
        model
    )

    model_hash = (
        sha256_file(
            MODEL_PATH
        )
    )

    manifest = (
        build_manifest(
            notebook_sha256=notebook_hash,
            model_sha256=model_hash,
        )
    )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
        )

    print()

    print(
        "=" * 72
    )

    print(
        "REPEAT PURCHASE ARTIFACT EXPORT: PASS"
    )

    print(
        "=" * 72
    )

    print(
        f"Model artifact:\n"
        f"{MODEL_PATH}"
    )

    print()

    print(
        f"Manifest:\n"
        f"{MANIFEST_PATH}"
    )

    print()

    print(
        f"Executed audit notebook:\n"
        f"{EXECUTED_NOTEBOOK_PATH}"
    )

    print()

    print(
        f"Frozen features: "
        f"{len(REPEAT_PURCHASE_MODEL_FEATURES)}"
    )

    print(
        f"Model: "
        f"{REPEAT_PURCHASE_MODEL_NAME}"
    )

    print(
        f"Governance decision: "
        f"{REPEAT_PURCHASE_FINAL_DECISION}"
    )

    print()

    print(
        "Notebook SHA-256:"
    )

    print(
        notebook_hash
    )

    print()

    print(
        "Model SHA-256:"
    )

    print(
        model_hash
    )


if __name__ == "__main__":
    export_repeat_purchase_model()