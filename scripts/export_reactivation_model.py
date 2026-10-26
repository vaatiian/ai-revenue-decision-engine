"""
Export the fitted frozen Reactivation model from the original research notebook.

This script intentionally does NOT recreate the training logic independently.

Instead it:

1. loads Notebook 03;
2. appends a temporary export cell in memory;
3. executes the notebook from beginning to end;
4. retrieves the exact final_reactivation_refit_model produced by research;
5. serializes the trained sklearn Pipeline;
6. writes a governance manifest;
7. calculates SHA-256 hashes for provenance.

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "03_ML_Reactivation_Propensity.ipynb"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "reactivation"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "reactivation_full_random_forest_v1.0.0.joblib"
)

MANIFEST_PATH = (
    ARTIFACT_DIR
    / "reactivation_full_random_forest_v1.0.0.json"
)

EXECUTED_NOTEBOOK_PATH = (
    ARTIFACT_DIR
    / "executed_03_ML_Reactivation_Propensity.ipynb"
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
# VALIDATION
# ================================================================

def validate_source_files() -> None:
    """
    Validate required research inputs.
    """

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Research notebook not found: "
            f"{NOTEBOOK_PATH}"
        )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ================================================================
# TEMPORARY EXPORT CELL
# ================================================================

def build_export_cell() -> str:
    """
    Return a cell appended only to the in-memory notebook.

    This cell executes after the complete research notebook and
    serializes the final fitted research model.
    """

    return f'''
from pathlib import Path
import hashlib
import json
import joblib
import sklearn


# ------------------------------------------------------------
# Validate research state
# ------------------------------------------------------------

assert REACTIVATION_MODEL_FROZEN is True
assert FINAL_REACTIVATION_MODEL_NAME == "Full Random Forest"
assert FINAL_REACTIVATION_FEATURES is not None
assert final_reactivation_refit_model is not None


# ------------------------------------------------------------
# Artifact paths
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Export fitted model
# ------------------------------------------------------------

joblib.dump(
    final_reactivation_refit_model,
    model_path
)


# ------------------------------------------------------------
# Extract fitted model parameters
# ------------------------------------------------------------

rf_model = (
    final_reactivation_refit_model
    .named_steps["model"]
)

manifest = {{
    "model_name":
        FINAL_REACTIVATION_MODEL_NAME,

    "model_version":
        "1.0.0",

    "model_type":
        "classification",

    "target_name":
        "reactivation_within_next_90_days",

    "prediction_point":
        "exactly 90 days after previous qualifying purchase",

    "target_horizon_days":
        90,

    "training_population":
        "combined chronological Train + Validation development population",

    "feature_count":
        len(
            FINAL_REACTIVATION_FEATURES
        ),

    "features":
        list(
            FINAL_REACTIVATION_FEATURES
        ),

    "algorithm":
        type(
            rf_model
        ).__name__,

    "hyperparameters": {{
        "n_estimators":
            rf_model.n_estimators,

        "max_depth":
            rf_model.max_depth,

        "min_samples_leaf":
            rf_model.min_samples_leaf,

        "max_features":
            rf_model.max_features,

        "random_state":
            rf_model.random_state,

        "n_jobs":
            rf_model.n_jobs,
    }},

    "numeric_preprocessing":
        "median imputation with missingness indicators",

    "categorical_preprocessing":
        "most-frequent imputation plus one-hot encoding",

    "calibration":
        "none",

    "research_validation_design":
        "chronological out-of-time",

    "research_final_decision":
        globals().get(
            "FINAL_REACTIVATION_DECISION",
            "GO"
        ),

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
    "PRODUCTION REACTIVATION ARTIFACT EXPORTED"
)

print(
    f"Model: {{model_path}}"
)

print(
    f"Manifest: {{manifest_path}}"
)

print(
    f"Feature count: "
    f"{{len(FINAL_REACTIVATION_FEATURES)}}"
)
'''


# ================================================================
# EXECUTION
# ================================================================

def export_reactivation_model() -> None:
    """
    Execute Notebook 03 and serialize its final fitted model.
    """

    validate_source_files()

    print(
        "=" * 72
    )

    print(
        "REACTIVATION MODEL ARTIFACT EXPORT"
    )

    print(
        "=" * 72
    )

    print(
        f"Research notebook:\n"
        f"{NOTEBOOK_PATH}"
    )

    print()

    print(
        "Original notebook SHA-256:"
    )

    notebook_hash = sha256_file(
        NOTEBOOK_PATH
    )

    print(
        notebook_hash
    )

    print()

    # ------------------------------------------------------------
    # Read notebook
    # ------------------------------------------------------------

    with NOTEBOOK_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:

        notebook = nbformat.read(
            handle,
            as_version=4,
        )

    # ------------------------------------------------------------
    # Append export cell only to in-memory copy
    # ------------------------------------------------------------

    notebook.cells.append(
        nbformat.v4.new_code_cell(
            build_export_cell()
        )
    )

    # ------------------------------------------------------------
    # Execute entire research notebook
    # ------------------------------------------------------------

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
                "path": str(
                    PROJECT_ROOT
                )
            }
        },
    )

    # ------------------------------------------------------------
    # Save executed audit copy
    # ------------------------------------------------------------

    with EXECUTED_NOTEBOOK_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        nbformat.write(
            notebook,
            handle,
        )

    # ------------------------------------------------------------
    # Verify outputs
    # ------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model artifact was not created."
        )

    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            "Model manifest was not created."
        )

    model_hash = sha256_file(
        MODEL_PATH
    )

    # ------------------------------------------------------------
    # Add provenance hashes to manifest
    # ------------------------------------------------------------

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
    export_reactivation_model()