"""
End-to-end production validation for the frozen Future Customer Value model.

Validated path
--------------
canonical transactions
    |
    v
50 frozen FCV model features
    |
    v
fitted research Random Forest Stage A model
    |
    v
P(Future Active)
    |
    v
0.50 × h180_revenue
    |
    v
Expected FCV90

This script validates:
- canonical data loading;
- exact FCV feature parity and order;
- real artifact integrity;
- real manifest integrity;
- Stage A probability generation;
- deterministic Stage B calculation;
- complete hurdle formula;
- customer uniqueness;
- customer-order preservation;
- probability bounds;
- non-negative FCV outputs;
- governance metadata;
- output persistence.

No model retraining occurs here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ================================================================
# PROJECT IMPORTS
# ================================================================

from src.data.loader import (
    load_transactions,
)

from src.features.fcv_model_features import (
    build_fcv_model_features,
)

from src.models.fcv_scorer import (
    DEFAULT_FCV_ARTIFACT_PATH,
    DEFAULT_FCV_MANIFEST_PATH,
    FCVScorer,
    validate_fcv_artifact,
    validate_fcv_manifest,
)

from src.models.fcv_spec import (
    FCV_COMPLETE_FORMULA,
    FCV_MODEL_FEATURES,
    FCV_MODEL_METADATA,
    FCV_STAGE_B_RUN_RATE_MULTIPLIER,
)


# ================================================================
# PATHS
# ================================================================

DATA_PATH = (
    PROJECT_ROOT
    / "cleaned_online_retail.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "fcv_end_to_end_validation.parquet"
)


# ================================================================
# VALIDATION DATE
# ================================================================

PREDICTION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


# ================================================================
# DISPLAY HELPERS
# ================================================================

def print_header(
    title: str,
) -> None:
    """
    Print a consistent validation heading.
    """

    print("=" * 72)
    print(title)
    print("=" * 72)


# ================================================================
# LOADER RESULT NORMALISATION
# ================================================================

def extract_transaction_dataframe(
    load_result,
) -> pd.DataFrame:
    """
    Extract the canonical transaction DataFrame returned by
    load_transactions().
    """

    if isinstance(
        load_result,
        pd.DataFrame,
    ):
        return load_result

    if isinstance(
        load_result,
        tuple,
    ):
        if len(load_result) == 0:
            raise RuntimeError(
                "Canonical transaction loader returned an empty tuple."
            )

        transactions = load_result[0]

        if not isinstance(
            transactions,
            pd.DataFrame,
        ):
            raise TypeError(
                "The first value returned by load_transactions() "
                "is not a pandas DataFrame."
            )

        return transactions

    if isinstance(
        load_result,
        list,
    ):
        if len(load_result) == 0:
            raise RuntimeError(
                "Canonical transaction loader returned an empty list."
            )

        transactions = load_result[0]

        if not isinstance(
            transactions,
            pd.DataFrame,
        ):
            raise TypeError(
                "The first value returned by load_transactions() "
                "is not a pandas DataFrame."
            )

        return transactions

    raise TypeError(
        "Unsupported return type from load_transactions(): "
        f"{type(load_result).__name__}"
    )


# ================================================================
# FEATURE VALIDATION
# ================================================================

def validate_feature_schema(
    features: pd.DataFrame,
) -> None:
    """
    Confirm the exact frozen FCV feature schema.
    """

    expected_columns = [
        "customer",
        *FCV_MODEL_FEATURES,
    ]

    actual_columns = list(
        features.columns
    )

    if actual_columns != expected_columns:
        raise RuntimeError(
            "FCV feature schema does not match the frozen "
            "50-feature production contract."
        )

    if len(FCV_MODEL_FEATURES) != 50:
        raise RuntimeError(
            "Frozen FCV feature count is not 50."
        )

    if features[
        "customer"
    ].isna().any():
        raise RuntimeError(
            "FCV feature matrix contains missing customer identifiers."
        )

    if features[
        "customer"
    ].duplicated().any():
        raise RuntimeError(
            "FCV feature matrix contains duplicate customer identifiers."
        )

    if features[
        "h180_revenue"
    ].isna().any():
        raise RuntimeError(
            "FCV eligible population contains missing h180_revenue."
        )

    if (
        features[
            "h180_revenue"
        ]
        <
        0
    ).any():
        raise RuntimeError(
            "FCV eligible population contains negative h180_revenue."
        )


# ================================================================
# SCORE VALIDATION
# ================================================================

def validate_scores(
    scores: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """
    Validate complete FCV production scores.
    """

    required_columns = {
        "customer",
        "future_active_probability",
        "fcv_history_180d_revenue",
        "future_customer_value_90d",
        "model_name",
        "model_version",
        "model_type",
        "target_name",
        "scored_at_utc",
    }

    missing = (
        required_columns
        -
        set(scores.columns)
    )

    if missing:
        raise RuntimeError(
            "FCV score output is missing columns: "
            +
            ", ".join(
                sorted(missing)
            )
        )

    if len(scores) != len(features):
        raise RuntimeError(
            "FCV score count does not match feature population."
        )

    if (
        scores[
            "customer"
        ].tolist()
        !=
        features[
            "customer"
        ].tolist()
    ):
        raise RuntimeError(
            "FCV scoring changed customer order."
        )

    if scores[
        "customer"
    ].duplicated().any():
        raise RuntimeError(
            "FCV score output contains duplicate customers."
        )

    probabilities = (
        pd.to_numeric(
            scores[
                "future_active_probability"
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "FCV future-active probabilities contain non-finite values."
        )

    if (
        (probabilities < 0).any()
        or
        (probabilities > 1).any()
    ):
        raise RuntimeError(
            "FCV future-active probabilities are outside [0, 1]."
        )

    history_revenue = (
        pd.to_numeric(
            scores[
                "fcv_history_180d_revenue"
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        history_revenue
    ).all():
        raise RuntimeError(
            "FCV historical revenue contains non-finite values."
        )

    expected_history_revenue = (
        features[
            "h180_revenue"
        ]
        .to_numpy(
            dtype=float
        )
    )

    np.testing.assert_allclose(
        history_revenue,
        expected_history_revenue,
        rtol=0,
        atol=0,
    )

    stage_b_value = (
        FCV_STAGE_B_RUN_RATE_MULTIPLIER
        *
        history_revenue
    )

    expected_fcv = (
        probabilities
        *
        stage_b_value
    )

    actual_fcv = (
        pd.to_numeric(
            scores[
                "future_customer_value_90d"
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        actual_fcv
    ).all():
        raise RuntimeError(
            "FCV Expected FCV90 contains non-finite values."
        )

    if (
        actual_fcv
        <
        0
    ).any():
        raise RuntimeError(
            "FCV Expected FCV90 contains negative values."
        )

    np.testing.assert_allclose(
        actual_fcv,
        expected_fcv,
        rtol=1e-12,
        atol=1e-12,
    )

    if not scores[
        "model_name"
    ].eq(
        FCV_MODEL_METADATA.model_name
    ).all():
        raise RuntimeError(
            "FCV model_name metadata mismatch."
        )

    if not scores[
        "model_version"
    ].eq(
        FCV_MODEL_METADATA.model_version
    ).all():
        raise RuntimeError(
            "FCV model_version metadata mismatch."
        )

    if not scores[
        "model_type"
    ].eq(
        FCV_MODEL_METADATA.model_type
    ).all():
        raise RuntimeError(
            "FCV model_type metadata mismatch."
        )

    if not scores[
        "target_name"
    ].eq(
        FCV_MODEL_METADATA.target_name
    ).all():
        raise RuntimeError(
            "FCV target_name metadata mismatch."
        )

    if scores[
        "scored_at_utc"
    ].isna().any():
        raise RuntimeError(
            "FCV score metadata contains missing scored_at_utc values."
        )


# ================================================================
# DISTRIBUTION DISPLAY
# ================================================================

def print_score_distribution(
    scores: pd.DataFrame,
) -> None:
    """
    Print FCV score-distribution diagnostics.
    """

    probabilities = (
        scores[
            "future_active_probability"
        ]
    )

    values = (
        scores[
            "future_customer_value_90d"
        ]
    )

    history_revenue = (
        scores[
            "fcv_history_180d_revenue"
        ]
    )

    customers_scored = len(
        scores
    )

    minimum_probability = (
        probabilities.min()
    )

    p10_probability = (
        probabilities.quantile(
            0.10
        )
    )

    median_probability = (
        probabilities.median()
    )

    mean_probability = (
        probabilities.mean()
    )

    p90_probability = (
        probabilities.quantile(
            0.90
        )
    )

    maximum_probability = (
        probabilities.max()
    )

    total_h180_revenue = (
        history_revenue.sum()
    )

    total_stage_b_run_rate = (
        total_h180_revenue
        *
        FCV_STAGE_B_RUN_RATE_MULTIPLIER
    )

    total_expected_fcv90 = (
        values.sum()
    )

    median_expected_fcv90 = (
        values.median()
    )

    mean_expected_fcv90 = (
        values.mean()
    )

    p90_expected_fcv90 = (
        values.quantile(
            0.90
        )
    )

    maximum_expected_fcv90 = (
        values.max()
    )

    print(
        "   customers_scored: "
        f"{customers_scored:,}"
    )

    print(
        "   minimum_activity_probability: "
        f"{minimum_probability:.6f}"
    )

    print(
        "   p10_activity_probability: "
        f"{p10_probability:.6f}"
    )

    print(
        "   median_activity_probability: "
        f"{median_probability:.6f}"
    )

    print(
        "   mean_activity_probability: "
        f"{mean_probability:.6f}"
    )

    print(
        "   p90_activity_probability: "
        f"{p90_probability:.6f}"
    )

    print(
        "   maximum_activity_probability: "
        f"{maximum_probability:.6f}"
    )

    print()

    print(
        "   total_h180_revenue: "
        f"{total_h180_revenue:,.2f}"
    )

    print(
        "   total_stage_b_run_rate: "
        f"{total_stage_b_run_rate:,.2f}"
    )

    print(
        "   total_expected_fcv90: "
        f"{total_expected_fcv90:,.2f}"
    )

    print(
        "   median_expected_fcv90: "
        f"{median_expected_fcv90:,.2f}"
    )

    print(
        "   mean_expected_fcv90: "
        f"{mean_expected_fcv90:,.2f}"
    )

    print(
        "   p90_expected_fcv90: "
        f"{p90_expected_fcv90:,.2f}"
    )

    print(
        "   maximum_expected_fcv90: "
        f"{maximum_expected_fcv90:,.2f}"
    )


# ================================================================
# MAIN VALIDATION
# ================================================================

def main() -> None:
    """
    Execute the full FCV production validation.
    """

    print_header(
        "FUTURE CUSTOMER VALUE END-TO-END PRODUCTION VALIDATION"
    )

    print()

    print(
        "Prediction date: "
        f"{PREDICTION_DATE}"
    )

    print()

    print(
        "Complete FCV formula: "
        f"{FCV_COMPLETE_FORMULA}"
    )

    print()

    # ------------------------------------------------------------
    # 1. LOAD CANONICAL TRANSACTIONS
    # ------------------------------------------------------------

    print(
        "1. Loading canonical transactions..."
    )

    load_result = (
        load_transactions(
            DATA_PATH
        )
    )

    transactions = (
        extract_transaction_dataframe(
            load_result
        )
    )

    if transactions.empty:
        raise RuntimeError(
            "Canonical transaction dataset is empty."
        )

    transaction_rows = len(
        transactions
    )

    transaction_columns = len(
        transactions.columns
    )

    print(
        "   Transaction rows: "
        f"{transaction_rows:,}"
    )

    print(
        "   Transaction columns: "
        f"{transaction_columns:,}"
    )

    print(
        "   Canonical transaction DataFrame: PASS"
    )

    print()

    # ------------------------------------------------------------
    # 2. BUILD EXACT FROZEN FEATURES
    # ------------------------------------------------------------

    print(
        "2. Building frozen FCV features..."
    )

    features = (
        build_fcv_model_features(
            transactions=transactions,
            prediction_date=PREDICTION_DATE,
        )
    )

    if features.empty:
        raise RuntimeError(
            "FCV feature pipeline produced an empty scoring population."
        )

    validate_feature_schema(
        features
    )

    eligible_customers = len(
        features
    )

    frozen_feature_count = len(
        FCV_MODEL_FEATURES
    )

    print(
        "   Eligible customers: "
        f"{eligible_customers:,}"
    )

    print(
        "   Frozen model features: "
        f"{frozen_feature_count}"
    )

    print(
        "   Feature schema: PASS"
    )

    print(
        "   Customer uniqueness: PASS"
    )

    print()

    # ------------------------------------------------------------
    # 3. LOAD REAL RESEARCH ARTIFACT
    # ------------------------------------------------------------

    print(
        "3. Loading frozen FCV research artifact..."
    )

    print(
        "   Artifact: "
        f"{DEFAULT_FCV_ARTIFACT_PATH}"
    )

    print(
        "   Manifest: "
        f"{DEFAULT_FCV_MANIFEST_PATH}"
    )

    scorer = FCVScorer()

    validate_fcv_artifact(
        scorer.model
    )

    validate_fcv_manifest(
        scorer.manifest,
        artifact_path=(
            DEFAULT_FCV_ARTIFACT_PATH
        ),
    )

    artifact_prediction_date = (
        scorer.manifest.get(
            "artifact_prediction_date"
        )
    )

    development_episodes = (
        scorer.manifest.get(
            "artifact_development_episodes"
        )
    )

    latest_matured_outcome = (
        scorer.manifest.get(
            "artifact_latest_matured_outcome_end"
        )
    )

    if development_episodes is None:
        raise RuntimeError(
            "FCV manifest is missing artifact_development_episodes."
        )

    if artifact_prediction_date is None:
        raise RuntimeError(
            "FCV manifest is missing artifact_prediction_date."
        )

    if latest_matured_outcome is None:
        raise RuntimeError(
            "FCV manifest is missing "
            "artifact_latest_matured_outcome_end."
        )

    print(
        "   Artifact integrity: PASS"
    )

    print(
        "   Manifest integrity: PASS"
    )

    print(
        "   Artifact prediction date: "
        f"{artifact_prediction_date}"
    )

    print(
        "   Development episodes: "
        f"{development_episodes:,}"
    )

    print(
        "   Latest matured outcome: "
        f"{latest_matured_outcome}"
    )

    print()

    # ------------------------------------------------------------
    # 4. SCORE
    # ------------------------------------------------------------

    print(
        "4. Generating complete FCV scores..."
    )

    scores = (
        scorer.score(
            features
        )
    )

    validate_scores(
        scores=scores,
        features=features,
    )

    scores_generated = len(
        scores
    )

    print(
        "   Scores generated: "
        f"{scores_generated:,}"
    )

    print(
        "   Stage A probability validation: PASS"
    )

    print(
        "   Stage B run-rate validation: PASS"
    )

    print(
        "   Complete FCV formula validation: PASS"
    )

    print(
        "   Governance metadata: PASS"
    )

    print()

    # ------------------------------------------------------------
    # 5. SCORE DISTRIBUTION
    # ------------------------------------------------------------

    print(
        "5. Score distribution"
    )

    print_score_distribution(
        scores
    )

    print()

    # ------------------------------------------------------------
    # 6. HIGHEST FCV SCORES
    # ------------------------------------------------------------

    print(
        "6. Highest Expected FCV90 scores"
    )

    top_scores = (
        scores[
            [
                "customer",
                "future_active_probability",
                "fcv_history_180d_revenue",
                "future_customer_value_90d",
            ]
        ]
        .sort_values(
            "future_customer_value_90d",
            ascending=False,
        )
        .head(
            10
        )
    )

    print(
        top_scores.to_string(
            index=False
        )
    )

    print()

    # ------------------------------------------------------------
    # 7. LOWEST FCV SCORES
    # ------------------------------------------------------------

    print(
        "7. Lowest Expected FCV90 scores"
    )

    bottom_scores = (
        scores[
            [
                "customer",
                "future_active_probability",
                "fcv_history_180d_revenue",
                "future_customer_value_90d",
            ]
        ]
        .sort_values(
            "future_customer_value_90d",
            ascending=True,
        )
        .head(
            10
        )
    )

    print(
        bottom_scores.to_string(
            index=False
        )
    )

    print()

    # ------------------------------------------------------------
    # 8. SAVE VALIDATION OUTPUT
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = scores.copy()

    output[
        "prediction_date"
    ] = (
        PREDICTION_DATE.normalize()
    )

    output[
        "fcv_stage_b_value"
    ] = (
        output[
            "fcv_history_180d_revenue"
        ]
        *
        FCV_STAGE_B_RUN_RATE_MULTIPLIER
    )

    output[
        "fcv_formula"
    ] = (
        FCV_COMPLETE_FORMULA
    )

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "8. Validation output written:"
    )

    print(
        "   "
        f"{OUTPUT_PATH}"
    )

    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            "FCV validation output was not written."
        )

    saved_output = pd.read_parquet(
        OUTPUT_PATH
    )

    if len(saved_output) != len(scores):
        raise RuntimeError(
            "Persisted FCV validation output row count changed."
        )

    print(
        "   Output persistence: PASS"
    )

    print()

    # ------------------------------------------------------------
    # COMPLETION
    # ------------------------------------------------------------

    print_header(
        "END-TO-END FCV VALIDATION: PASS"
    )

    print()

    print(
        "Validated production path:"
    )

    print(
        "canonical transactions"
    )

    print(
        "    |"
    )

    print(
        "    v"
    )

    print(
        "50 frozen FCV features"
    )

    print(
        "    |"
    )

    print(
        "    v"
    )

    print(
        "fitted research Random Forest Stage A"
    )

    print(
        "    |"
    )

    print(
        "    v"
    )

    print(
        "P(Future Active)"
    )

    print(
        "    |"
    )

    print(
        "    v"
    )

    print(
        "0.50 × h180_revenue"
    )

    print(
        "    |"
    )

    print(
        "    v"
    )

    print(
        "Expected FCV90"
    )


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    main()