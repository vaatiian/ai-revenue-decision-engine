"""
Export the fitted frozen Future Customer Value Stage A model from the
original research notebook.

This script intentionally does NOT independently recreate the research
training process.

Instead it:

1. loads Notebook 05;
2. appends a temporary export cell in memory;
3. executes the notebook from beginning to end;
4. retrieves the final Stage A Random Forest fitted during the final
   chronological Test-month refit;
5. verifies that this fitted object corresponds to the latest Test
   prediction date;
6. serializes the exact trained sklearn Pipeline;
7. records the deterministic Stage B rule and complete FCV architecture
   in a governance manifest;
8. records SHA-256 hashes for notebook and model provenance.

Important
---------
Notebook 05 refits Stage A separately for each final Test month using
only historical episodes whose 90-day outcome had matured before that
prediction date.

The variable `final_stage_a_model` is overwritten on each chronological
Test iteration. After notebook completion it therefore represents the
fitted Stage A model for the latest Test prediction date.

Stage B is deterministic:

    0.50 × h180_revenue

and is therefore NOT serialized as a second model artifact.

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
    / "05_ML_Future_Customer_Value.ipynb"
)


ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "fcv"
)


MODEL_PATH = (
    ARTIFACT_DIR
    / "fcv_stage_a_full_random_forest_v1.0.0.joblib"
)


MANIFEST_PATH = (
    ARTIFACT_DIR
    / "fcv_stage_a_full_random_forest_v1.0.0.json"
)


EXECUTED_NOTEBOOK_PATH = (
    ARTIFACT_DIR
    / "executed_05_ML_Future_Customer_Value.ipynb"
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
    Confirm the FCV research notebook exists and prepare the artifact
    directory.
    """

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            "Future Customer Value research notebook not found: "
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
    Create the temporary notebook cell that serializes the exact fitted
    frozen Stage A FCV model.
    """

    return f'''
from pathlib import Path
import json
import joblib
import sklearn
import pandas as pd


# ------------------------------------------------
# Frozen architecture assertions
# ------------------------------------------------

assert FCV_STAGE_A_MODEL_FROZEN is True
assert FCV_STAGE_B_MODEL_FROZEN is True
assert FCV_COMPLETE_MODEL_FROZEN is True

assert FINAL_FCV_STAGE_A_MODEL_NAME == "Full Random Forest"

assert FINAL_FCV_STAGE_A_FEATURES is not None

assert len(
    FINAL_FCV_STAGE_A_FEATURES
) == 50

assert FINAL_FCV_STAGE_B_MODEL_NAME == (
    "180-Day Historical Run Rate"
)

assert FINAL_FCV_STAGE_B_MODEL == (
    "0.50 × h180_revenue"
)

assert FINAL_FCV_MODEL_NAME == (
    "Random Forest Activity × "
    "180-Day Historical Run Rate"
)

assert FINAL_FCV_FORMULA == (
    "P(Future Active) × 0.50 × h180_revenue"
)

assert FINAL_FCV_DECISION == "CONDITIONAL GO"

assert final_stage_a_model is not None

assert FCV_TEST_SET_OPENED is True
assert FCV_TEST_SET_CONSUMED is True


# ------------------------------------------------
# Verify final fitted object represents latest Test month
# ------------------------------------------------

final_test_prediction_date = pd.Timestamp(
    max(
        fcv_test_prediction_dates
    )
)

assert pd.Timestamp(
    test_prediction_date
) == final_test_prediction_date


# ------------------------------------------------
# Verify fitted sklearn pipeline
# ------------------------------------------------

assert hasattr(
    final_stage_a_model,
    "predict_proba"
)

assert hasattr(
    final_stage_a_model,
    "named_steps"
)

assert "preprocessing" in (
    final_stage_a_model.named_steps
)

assert "model" in (
    final_stage_a_model.named_steps
)

assert hasattr(
    final_stage_a_model,
    "feature_names_in_"
)

assert list(
    final_stage_a_model.feature_names_in_
) == list(
    FINAL_FCV_STAGE_A_FEATURES
)


# ------------------------------------------------
# Derive final refit governance metadata from notebook audit
# ------------------------------------------------

final_test_rows = (
    final_fcv_test_predictions.loc[
        pd.to_datetime(
            final_fcv_test_predictions[
                "prediction_date"
            ]
        )
        ==
        final_test_prediction_date
    ]
    .copy()
)

assert not final_test_rows.empty

development_episode_counts = (
    final_test_rows[
        "development_episodes"
    ]
    .dropna()
    .unique()
)

assert len(
    development_episode_counts
) == 1

final_development_episodes = int(
    development_episode_counts[0]
)

latest_development_outcome_values = (
    pd.to_datetime(
        final_test_rows[
            "latest_development_outcome_end"
        ]
    )
    .dropna()
    .unique()
)

assert len(
    latest_development_outcome_values
) == 1

latest_development_outcome_end = pd.Timestamp(
    latest_development_outcome_values[0]
)

assert (
    latest_development_outcome_end
    <
    final_test_prediction_date
)


# ------------------------------------------------
# Artifact paths
# ------------------------------------------------

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


# ------------------------------------------------
# Serialize exact fitted Stage A pipeline
# ------------------------------------------------

joblib.dump(
    final_stage_a_model,
    model_path
)


# ------------------------------------------------
# Estimator
# ------------------------------------------------

rf_model = (
    final_stage_a_model
    .named_steps["model"]
)


# ------------------------------------------------
# Governance manifest
# ------------------------------------------------

manifest = {{
    "model_name":
        FINAL_FCV_STAGE_A_MODEL_NAME,

    "model_version":
        "1.0.0",

    "model_type":
        "classification",

    "target_name":
        "future_active",

    "business_target":
        "Future 90-Day Customer Value (FCV90)",

    "prediction_problem":
        "probability customer generates positive qualifying "
        "merchandise revenue during the following 90 days",

    "prediction_unit":
        "one identified customer at one monthly prediction point",

    "eligibility_rule":
        "at least one qualifying merchandise purchase "
        "during the prior 180 days",

    "eligibility_lookback_days":
        180,

    "outcome_window_days":
        90,

    "stage_a_target_definition":
        "FCV90 > 0",

    "stage_a_probability_definition":
        "P(Future Active)",

    "feature_count":
        len(
            FINAL_FCV_STAGE_A_FEATURES
        ),

    "features":
        list(
            FINAL_FCV_STAGE_A_FEATURES
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

    "calendar_representation":
        "sine/cosine month encoding",

    "calibration":
        "none",

    "primary_selection_metric":
        "PR-AUC",

    "economic_metric":
        "future revenue captured by ranked customers",

    "research_validation_design":
        "purged chronological validation",

    "stage_b_model_name":
        FINAL_FCV_STAGE_B_MODEL_NAME,

    "stage_b_model_type":
        "deterministic rule",

    "stage_b_formula":
        FINAL_FCV_STAGE_B_MODEL,

    "stage_b_multiplier":
        0.50,

    "complete_model_name":
        FINAL_FCV_MODEL_NAME,

    "complete_model_formula":
        FINAL_FCV_FORMULA,

    "artifact_scope":
        "Stage A fitted sklearn Pipeline only",

    "artifact_refit_policy":
        "latest final-Test-month label-matured historical refit",

    "artifact_prediction_date":
        final_test_prediction_date.isoformat(),

    "artifact_development_episodes":
        final_development_episodes,

    "artifact_latest_matured_outcome_end":
        latest_development_outcome_end.isoformat(),

    "research_test_episodes":
        5534,

    "research_test_metrics": {{
        "stage_a_pr_auc":
            0.822148,

        "stage_a_roc_auc":
            0.749591,

        "complete_spearman":
            0.550464,

        "complete_capture_at_10":
            0.573750,

        "complete_capture_at_20":
            0.686870,

        "complete_capture_at_30":
            0.768946,

        "complete_predicted_to_actual_revenue":
            0.593481,
    }},

    "research_final_decision":
        FINAL_FCV_DECISION,

    "approved_use":
        [
            "customer ranking",
            "value segmentation",
            "CRM prioritisation",
            "high-value customer identification",
            "customer-service prioritisation",
            "controlled commercial decision support",
        ],

    "not_approved_for":
        [
            "accounting revenue forecasts",
            "financial budgets",
            "exact individual future-revenue expectations",
            "guaranteed future revenue interpretation",
        ],

    "primary_limitation":
        "absolute monetary calibration drift under changing "
        "economic regimes",

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
    "PRODUCTION FCV STAGE A ARTIFACT EXPORTED"
)

print(
    f"Model: {{model_path}}"
)

print(
    f"Manifest: {{manifest_path}}"
)

print(
    f"Feature count: {{len(FINAL_FCV_STAGE_A_FEATURES)}}"
)

print(
    f"Artifact prediction date: {{final_test_prediction_date}}"
)

print(
    f"Development episodes: {{final_development_episodes:,}}"
)

print(
    "Latest matured development outcome: "
    f"{{latest_development_outcome_end}}"
)
'''


# ================================================================
# EXPORT
# ================================================================

def export_fcv_model() -> None:
    """
    Execute Notebook 05 and export the exact final fitted Stage A model.
    """

    validate_source_files()

    print(
        "=" * 72
    )

    print(
        "FUTURE CUSTOMER VALUE STAGE A MODEL ARTIFACT EXPORT"
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
            "FCV Stage A model artifact was not created."
        )

    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            "FCV Stage A manifest was not created."
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
    export_fcv_model()