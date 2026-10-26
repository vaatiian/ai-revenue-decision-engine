"""
Export the fitted frozen Revenue Leakage model from the original
research notebook.

This script intentionally does NOT recreate the training logic
independently.

Instead it:

1. loads Notebook 04;
2. appends a temporary export cell in memory;
3. executes the notebook from beginning to end;
4. retrieves final_leakage_refit_model;
5. serializes the exact trained sklearn Pipeline;
6. writes a governance manifest;
7. records SHA-256 hashes for provenance.

The original research notebook is never modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "04_ML_Revenue_Leakage_Risk.ipynb"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "leakage"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "revenue_leakage_full_gradient_boosting_v1.0.0.joblib"
)

MANIFEST_PATH = (
    ARTIFACT_DIR
    / "revenue_leakage_full_gradient_boosting_v1.0.0.json"
)

EXECUTED_NOTEBOOK_PATH = (
    ARTIFACT_DIR
    / "executed_04_ML_Revenue_Leakage_Risk.ipynb"
)


# ================================================================
# HASHING
# ================================================================

def sha256_file(
    path: Path,
) -> str:
    """
    Calculate the SHA-256 hash of a file.
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
    Confirm the research notebook exists and prepare the artifact
    directory.
    """

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            "Revenue Leakage research notebook not found: "
            f"{NOTEBOOK_PATH}"
        )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ================================================================
# EXPORT CELL
# ================================================================

def build_export_cell() -> str:
    """
    Create the temporary notebook cell that serializes the exact
    fitted frozen research model.
    """

    return f'''
from pathlib import Path
import json
import joblib
import sklearn

assert LEAKAGE_MODEL_FROZEN is True
assert FINAL_LEAKAGE_MODEL_NAME == "Full Gradient Boosting"
assert FINAL_LEAKAGE_FEATURES is not None
assert final_leakage_refit_model is not None
assert FINAL_LEAKAGE_DECISION == "GO"

artifact_dir = Path(
    r"{ARTIFACT_DIR.as_posix()}"
)

artifact_dir.mkdir(
    parents=True,
    exist_ok=True
)

model_path = Path(
    r"{MODEL_PATH.as_posix()}"
)

manifest_path = Path(
    r"{MANIFEST_PATH.as_posix()}"
)

joblib.dump(
    final_leakage_refit_model,
    model_path
)

gb_model = (
    final_leakage_refit_model
    .named_steps["model"]
)

manifest = {{
    "model_name":
        FINAL_LEAKAGE_MODEL_NAME,

    "model_version":
        "1.0.0",

    "model_type":
        "classification",

    "target_name":
        "revenue_leakage_event",

    "prediction_problem":
        "future customer revenue leakage",

    "prediction_unit":
        "one active customer at one month-end prediction point",

    "baseline_window_days":
        90,

    "outcome_window_days":
        90,

    "target_definition":
        "future revenue <= 50% of prior 90-day revenue",

    "training_population":
        "all label-matured pre-Test development episodes",

    "feature_count":
        len(
            FINAL_LEAKAGE_FEATURES
        ),

    "features":
        list(
            FINAL_LEAKAGE_FEATURES
        ),

    "algorithm":
        type(
            gb_model
        ).__name__,

    "hyperparameters": {{
        "n_estimators":
            gb_model.n_estimators,

        "learning_rate":
            gb_model.learning_rate,

        "max_depth":
            gb_model.max_depth,

        "min_samples_leaf":
            gb_model.min_samples_leaf,

        "random_state":
            gb_model.random_state,
    }},

    "numeric_preprocessing":
        "median imputation with missingness indicators",

    "categorical_preprocessing":
        "most-frequent imputation plus one-hot encoding",

    "calendar_representation":
        "sine/cosine month encoding",

    "calibration":
        "none",

    "primary_selection_metric":
        "PR-AUC",

    "commercial_metrics":
        [
            "Lift@10%",
            "Lift@20%",
            "Capture@10%",
            "Capture@20%"
        ],

    "research_validation_design":
        "purged chronological validation",

    "research_final_decision":
        FINAL_LEAKAGE_DECISION,

    "sklearn_version":
        sklearn.__version__,
}}

with manifest_path.open(
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2,
    )

print(
    "PRODUCTION REVENUE LEAKAGE ARTIFACT EXPORTED"
)

print(
    f"Model: {{model_path}}"
)

print(
    f"Manifest: {{manifest_path}}"
)

print(
    f"Feature count: {{len(FINAL_LEAKAGE_FEATURES)}}"
)
'''


# ================================================================
# EXPORT
# ================================================================

def export_leakage_model() -> None:
    """
    Execute Notebook 04 and export its exact final fitted model.
    """

    validate_source_files()

    print(
        "=" * 72
    )

    print(
        "REVENUE LEAKAGE MODEL ARTIFACT EXPORT"
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
            "Revenue Leakage model artifact was not created."
        )

    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            "Revenue Leakage manifest was not created."
        )

    model_hash = (
        sha256_file(
            MODEL_PATH
        )
    )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:

        manifest = json.load(
            handle
        )

    manifest[
        "source_notebook_sha256"
    ] = notebook_hash

    manifest[
        "model_artifact_sha256"
    ] = model_hash

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
        "EXPORT COMPLETE"
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
        "Model SHA-256:"
    )

    print(
        model_hash
    )


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    export_leakage_model()