"""
End-to-end production validation for the frozen Revenue Leakage model.

Validated path
--------------
canonical transactions
    |
    v
45 frozen Revenue Leakage features
    |
    v
fitted research Gradient Boosting model
    |
    v
governed Revenue Leakage probabilities

This script validates:
- canonical data loading;
- exact feature parity and feature order;
- real artifact integrity;
- real manifest integrity;
- probability generation;
- customer uniqueness;
- probability bounds;
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
# PROJECT IMPORTS
# ================================================================

from src.data.loader import load_transactions

from src.features.leakage_model_features import (
    build_leakage_model_features,
)

from src.models.leakage_scorer import (
    DEFAULT_LEAKAGE_ARTIFACT_PATH,
    DEFAULT_LEAKAGE_MANIFEST_PATH,
    LeakageScorer,
    validate_leakage_artifact,
    validate_leakage_manifest,
)

from src.models.leakage_spec import (
    LEAKAGE_MODEL_FEATURES,
    LEAKAGE_MODEL_METADATA,
)


# ================================================================
# PATHS
# ================================================================

DATA_PATH = (
    PROJECT_ROOT
    /
    "cleaned_online_retail.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "outputs"
    /
    "validation"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    /
    "leakage_end_to_end_validation.parquet"
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

    print(
        "=" * 72
    )

    print(
        title
    )

    print(
        "=" * 72
    )


# ================================================================
# LOADER RESULT NORMALISATION
# ================================================================

def extract_transaction_dataframe(
    load_result,
) -> pd.DataFrame:
    """
    Extract the canonical transaction DataFrame returned by
    load_transactions().

    The canonical loader may return additional validation metadata
    alongside the DataFrame. This validator only needs the canonical
    transaction table.

    Fail closed if a DataFrame cannot be identified.
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

        if len(
            load_result
        ) == 0:
            raise RuntimeError(
                "Canonical transaction loader returned an empty tuple."
            )

        transactions = (
            load_result[
                0
            ]
        )

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

        if len(
            load_result
        ) == 0:
            raise RuntimeError(
                "Canonical transaction loader returned an empty list."
            )

        transactions = (
            load_result[
                0
            ]
        )

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
    Confirm the exact frozen Revenue Leakage feature schema.
    """

    expected_columns = [
        "customer",
        *LEAKAGE_MODEL_FEATURES,
    ]

    actual_columns = list(
        features.columns
    )

    if (
        actual_columns
        !=
        expected_columns
    ):
        raise RuntimeError(
            "Revenue Leakage feature schema does not match "
            "the frozen 45-feature production contract."
        )

    if (
        len(
            LEAKAGE_MODEL_FEATURES
        )
        !=
        45
    ):
        raise RuntimeError(
            "Frozen Revenue Leakage feature count is not 45."
        )

    if features[
        "customer"
    ].isna().any():
        raise RuntimeError(
            "Revenue Leakage feature matrix contains missing "
            "customer identifiers."
        )

    if features[
        "customer"
    ].duplicated().any():
        raise RuntimeError(
            "Revenue Leakage feature matrix contains duplicate "
            "customer identifiers."
        )


# ================================================================
# SCORE VALIDATION
# ================================================================

def validate_scores(
    scores: pd.DataFrame,
    expected_customers: pd.Series,
) -> None:
    """
    Validate generated Revenue Leakage probabilities.
    """

    required_columns = {
        "customer",
        "revenue_leakage_probability",
        "model_name",
        "model_version",
        "model_type",
        "target_name",
        "scored_at_utc",
    }

    missing = (
        required_columns
        -
        set(
            scores.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Revenue Leakage score output is missing columns: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    if (
        len(
            scores
        )
        !=
        len(
            expected_customers
        )
    ):
        raise RuntimeError(
            "Revenue Leakage score count does not match "
            "the feature population."
        )

    if (
        scores[
            "customer"
        ].tolist()
        !=
        expected_customers.tolist()
    ):
        raise RuntimeError(
            "Revenue Leakage scoring changed customer order."
        )

    if scores[
        "customer"
    ].duplicated().any():
        raise RuntimeError(
            "Revenue Leakage score output contains duplicate "
            "customers."
        )

    probabilities = (
        scores[
            "revenue_leakage_probability"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Revenue Leakage probabilities contain non-finite "
            "values."
        )

    if (
        (
            probabilities
            <
            0
        ).any()
        or
        (
            probabilities
            >
            1
        ).any()
    ):
        raise RuntimeError(
            "Revenue Leakage probabilities are outside [0, 1]."
        )

    if not scores[
        "model_name"
    ].eq(
        LEAKAGE_MODEL_METADATA.model_name
    ).all():
        raise RuntimeError(
            "Revenue Leakage model_name metadata mismatch."
        )

    if not scores[
        "model_version"
    ].eq(
        LEAKAGE_MODEL_METADATA.model_version
    ).all():
        raise RuntimeError(
            "Revenue Leakage model_version metadata mismatch."
        )

    if not scores[
        "model_type"
    ].eq(
        LEAKAGE_MODEL_METADATA.model_type
    ).all():
        raise RuntimeError(
            "Revenue Leakage model_type metadata mismatch."
        )

    if not scores[
        "target_name"
    ].eq(
        LEAKAGE_MODEL_METADATA.target_name
    ).all():
        raise RuntimeError(
            "Revenue Leakage target_name metadata mismatch."
        )

    if scores[
        "scored_at_utc"
    ].isna().any():
        raise RuntimeError(
            "Revenue Leakage score metadata contains missing "
            "scored_at_utc values."
        )


# ================================================================
# DISTRIBUTION DISPLAY
# ================================================================

def print_probability_distribution(
    scores: pd.DataFrame,
) -> None:
    """
    Print useful score-distribution diagnostics.
    """

    probabilities = (
        scores[
            "revenue_leakage_probability"
        ]
    )

    print(
        f"   customers_scored: "
        f"{len(probabilities):,}"
    )

    print(
        f"   minimum_probability: "
        f"{probabilities.min():.6f}"
    )

    print(
        f"   p10_probability: "
        f"{probabilities.quantile(0.10):.6f}"
    )

    print(
        f"   p25_probability: "
        f"{probabilities.quantile(0.25):.6f}"
    )

    print(
        f"   median_probability: "
        f"{probabilities.median():.6f}"
    )

    print(
        f"   mean_probability: "
        f"{probabilities.mean():.6f}"
    )

    print(
        f"   p75_probability: "
        f"{probabilities.quantile(0.75):.6f}"
    )

    print(
        f"   p90_probability: "
        f"{probabilities.quantile(0.90):.6f}"
    )

    print(
        f"   maximum_probability: "
        f"{probabilities.max():.6f}"
    )


# ================================================================
# MAIN VALIDATION
# ================================================================

def main() -> None:
    """
    Execute the full Revenue Leakage production validation.
    """

    print_header(
        "REVENUE LEAKAGE END-TO-END PRODUCTION VALIDATION"
    )

    print()

    print(
        f"Prediction date: "
        f"{PREDICTION_DATE}"
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

    print(
        f"   Transaction rows: "
        f"{len(transactions):,}"
    )

    print(
        f"   Transaction columns: "
        f"{len(transactions.columns):,}"
    )

    print(
        "   Canonical transaction DataFrame: PASS"
    )

    print()

    # ------------------------------------------------------------
    # 2. BUILD FROZEN FEATURES
    # ------------------------------------------------------------

    print(
        "2. Building frozen Revenue Leakage features..."
    )

    features = (
        build_leakage_model_features(
            transactions=transactions,
            prediction_date=PREDICTION_DATE,
        )
    )

    if features.empty:
        raise RuntimeError(
            "Revenue Leakage feature pipeline produced "
            "an empty scoring population."
        )

    validate_feature_schema(
        features
    )

    print(
        f"   Eligible customers: "
        f"{len(features):,}"
    )

    print(
        f"   Frozen model features: "
        f"{len(LEAKAGE_MODEL_FEATURES)}"
    )

    print(
        "   Feature schema: PASS"
    )

    print()

    # ------------------------------------------------------------
    # 3. LOAD REAL RESEARCH ARTIFACT
    # ------------------------------------------------------------

    print(
        "3. Loading frozen research artifact..."
    )

    print(
        f"   Artifact: "
        f"{DEFAULT_LEAKAGE_ARTIFACT_PATH}"
    )

    print(
        f"   Manifest: "
        f"{DEFAULT_LEAKAGE_MANIFEST_PATH}"
    )

    scorer = (
        LeakageScorer()
    )

    validate_leakage_artifact(
        scorer.model
    )

    validate_leakage_manifest(
        scorer.manifest,
        artifact_path=(
            DEFAULT_LEAKAGE_ARTIFACT_PATH
        ),
    )

    print(
        "   Artifact integrity: PASS"
    )

    print(
        "   Manifest integrity: PASS"
    )

    print()

    # ------------------------------------------------------------
    # 4. SCORE
    # ------------------------------------------------------------

    print(
        "4. Generating Revenue Leakage probabilities..."
    )

    scores = (
        scorer.score(
            features
        )
    )

    validate_scores(
        scores=scores,
        expected_customers=(
            features[
                "customer"
            ]
        ),
    )

    print(
        f"   Scores generated: "
        f"{len(scores):,}"
    )

    print(
        "   Probability validation: PASS"
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

    print_probability_distribution(
        scores
    )

    print()

    # ------------------------------------------------------------
    # 6. HIGHEST SCORES
    # ------------------------------------------------------------

    print(
        "6. Highest Revenue Leakage probabilities"
    )

    top_scores = (
        scores[
            [
                "customer",
                "revenue_leakage_probability",
            ]
        ]
        .sort_values(
            "revenue_leakage_probability",
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
    # 7. LOWEST SCORES
    # ------------------------------------------------------------

    print(
        "7. Lowest Revenue Leakage probabilities"
    )

    bottom_scores = (
        scores[
            [
                "customer",
                "revenue_leakage_probability",
            ]
        ]
        .sort_values(
            "revenue_leakage_probability",
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
    # 8. WRITE VALIDATION OUTPUT
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "8. Validation output written:"
    )

    print(
        f"   {OUTPUT_PATH}"
    )

    print()

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------

    print_header(
        "END-TO-END REVENUE LEAKAGE VALIDATION: PASS"
    )

    print()

    print(
        "Validated production path:"
    )

    print()

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
        "45 frozen Revenue Leakage features"
    )

    print(
        "    |"
    )

    print(
        "    v"
    )

    print(
        "fitted research Gradient Boosting model"
    )

    print(
        "    |"
    )

    print(
        "    v"
    )

    print(
        "governed Revenue Leakage probabilities"
    )


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    main()