"""
End-to-end production validation for the frozen Reactivation model.

Flow
----
canonical cleaned transactions
    |
    v
frozen Reactivation feature adapter
    |
    v
exact 29-feature model matrix
    |
    v
real fitted research artifact
    |
    v
production ReactivationScorer
    |
    v
governed customer probabilities

This script performs validation only.

It does not:
- retrain the model;
- modify the frozen model specification;
- modify the research notebook;
- overwrite historical research outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# PROJECT ROOT / IMPORT PATH
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

from src.features.reactivation_model_features import (
    build_reactivation_model_features,
)
from src.models.reactivation_scorer import (
    DEFAULT_REACTIVATION_ARTIFACT_PATH,
    DEFAULT_REACTIVATION_MANIFEST_PATH,
    ReactivationScorer,
)
from src.models.reactivation_spec import (
    REACTIVATION_MODEL_FEATURES,
    REACTIVATION_MODEL_METADATA,
)


# ================================================================
# PATHS
# ================================================================

TRANSACTION_PATH = (
    PROJECT_ROOT
    / "cleaned_online_retail.parquet"
)

VALIDATION_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
)

VALIDATION_OUTPUT_PATH = (
    VALIDATION_OUTPUT_DIR
    / "reactivation_end_to_end_validation.parquet"
)


# ================================================================
# VALIDATION DATE
# ================================================================

PREDICTION_DATE = pd.Timestamp(
    "2011-08-31 23:59:59"
)


# ================================================================
# SOURCE DATA
# ================================================================

def load_canonical_transactions() -> pd.DataFrame:
    """
    Load the canonical cleaned transaction dataset.
    """

    if not TRANSACTION_PATH.exists():
        raise FileNotFoundError(
            "Canonical transaction file not found: "
            f"{TRANSACTION_PATH}"
        )

    transactions = pd.read_parquet(
        TRANSACTION_PATH
    )

    if transactions.empty:
        raise ValueError(
            "Canonical transaction dataset is empty."
        )

    return transactions


# ================================================================
# FEATURE VALIDATION
# ================================================================

def validate_feature_table(
    features: pd.DataFrame,
) -> None:
    """
    Validate the frozen production feature table.
    """

    if features.empty:
        raise ValueError(
            "Reactivation feature population is empty."
        )

    expected_columns = [
        "customer",
        *REACTIVATION_MODEL_FEATURES,
    ]

    if list(
        features.columns
    ) != expected_columns:
        raise ValueError(
            "Reactivation feature table does not match "
            "the exact frozen production schema."
        )

    if features[
        "customer"
    ].isna().any():
        raise ValueError(
            "Reactivation feature table contains "
            "missing customer identifiers."
        )

    if features[
        "customer"
    ].duplicated().any():
        raise ValueError(
            "Reactivation feature table contains "
            "duplicate customers."
        )

    if len(
        features.columns
    ) != 30:
        raise ValueError(
            "Expected customer plus 29 frozen model features."
        )


# ================================================================
# SCORE VALIDATION
# ================================================================

def validate_score_table(
    features: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    """
    Validate production Reactivation scores.
    """

    if scores.empty:
        raise ValueError(
            "Reactivation scoring produced no rows."
        )

    if len(
        scores
    ) != len(
        features
    ):
        raise ValueError(
            "Score count does not match feature population."
        )

    if list(
        scores[
            "customer"
        ]
    ) != list(
        features[
            "customer"
        ]
    ):
        raise ValueError(
            "Customer order changed during scoring."
        )

    if scores[
        "customer"
    ].duplicated().any():
        raise ValueError(
            "Score output contains duplicate customers."
        )

    probabilities = pd.to_numeric(
        scores[
            "reactivation_probability"
        ],
        errors="coerce",
    )

    if probabilities.isna().any():
        raise ValueError(
            "Reactivation scores contain missing probabilities."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Reactivation scores contain non-finite probabilities."
        )

    if not probabilities.between(
        0,
        1,
    ).all():
        raise ValueError(
            "Reactivation probabilities fall outside [0, 1]."
        )

    required_metadata = {
        "model_name",
        "model_version",
        "model_type",
        "target_name",
        "scored_at_utc",
    }

    missing_metadata = (
        required_metadata
        -
        set(
            scores.columns
        )
    )

    if missing_metadata:
        raise ValueError(
            "Score output is missing governance metadata: "
            +
            ", ".join(
                sorted(
                    missing_metadata
                )
            )
        )

    if not (
        scores[
            "model_name"
        ]
        ==
        REACTIVATION_MODEL_METADATA.model_name
    ).all():
        raise ValueError(
            "Unexpected model_name in Reactivation output."
        )

    if not (
        scores[
            "model_version"
        ]
        ==
        REACTIVATION_MODEL_METADATA.model_version
    ).all():
        raise ValueError(
            "Unexpected model_version in Reactivation output."
        )


# ================================================================
# DISTRIBUTION SUMMARY
# ================================================================

def build_score_summary(
    scores: pd.DataFrame,
) -> pd.Series:
    """
    Return basic production score diagnostics.
    """

    probabilities = scores[
        "reactivation_probability"
    ]

    return pd.Series(
        {
            "customers_scored":
                len(
                    scores
                ),

            "minimum_probability":
                probabilities.min(),

            "p10_probability":
                probabilities.quantile(
                    0.10
                ),

            "p25_probability":
                probabilities.quantile(
                    0.25
                ),

            "median_probability":
                probabilities.median(),

            "mean_probability":
                probabilities.mean(),

            "p75_probability":
                probabilities.quantile(
                    0.75
                ),

            "p90_probability":
                probabilities.quantile(
                    0.90
                ),

            "maximum_probability":
                probabilities.max(),
        }
    )


# ================================================================
# MAIN END-TO-END VALIDATION
# ================================================================

def run_validation() -> None:
    """
    Execute the full Reactivation production scoring path.
    """

    print(
        "=" * 72
    )

    print(
        "REACTIVATION END-TO-END PRODUCTION VALIDATION"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Prediction date: "
        f"{PREDICTION_DATE}"
    )

    print()

    # ------------------------------------------------------------
    # Step 1 - Load canonical transactions
    # ------------------------------------------------------------

    print(
        "1. Loading canonical transactions..."
    )

    transactions = (
        load_canonical_transactions()
    )

    print(
        f"   Transaction rows: "
        f"{len(transactions):,}"
    )

    print()

    # ------------------------------------------------------------
    # Step 2 - Build exact frozen features
    # ------------------------------------------------------------

    print(
        "2. Building frozen Reactivation features..."
    )

    features = (
        build_reactivation_model_features(
            transactions=transactions,
            prediction_date=PREDICTION_DATE,
        )
    )

    validate_feature_table(
        features
    )

    print(
        f"   Eligible customers: "
        f"{len(features):,}"
    )

    print(
        f"   Frozen model features: "
        f"{len(REACTIVATION_MODEL_FEATURES)}"
    )

    print(
        "   Feature schema: PASS"
    )

    print()

    # ------------------------------------------------------------
    # Step 3 - Load real research artifact
    # ------------------------------------------------------------

    print(
        "3. Loading frozen research artifact..."
    )

    print(
        f"   Artifact: "
        f"{DEFAULT_REACTIVATION_ARTIFACT_PATH}"
    )

    print(
        f"   Manifest: "
        f"{DEFAULT_REACTIVATION_MANIFEST_PATH}"
    )

    scorer = ReactivationScorer()

    print(
        "   Artifact integrity: PASS"
    )

    print(
        "   Manifest integrity: PASS"
    )

    print()

    # ------------------------------------------------------------
    # Step 4 - Generate scores
    # ------------------------------------------------------------

    print(
        "4. Generating Reactivation probabilities..."
    )

    scores = scorer.score(
        features
    )

    validate_score_table(
        features,
        scores,
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
    # Step 5 - Score distribution
    # ------------------------------------------------------------

    print(
        "5. Score distribution"
    )

    summary = build_score_summary(
        scores
    )

    for (
        metric,
        value,
    ) in summary.items():

        if metric == "customers_scored":
            print(
                f"   {metric}: "
                f"{int(value):,}"
            )

        else:
            print(
                f"   {metric}: "
                f"{value:.6f}"
            )

    print()

    # ------------------------------------------------------------
    # Step 6 - Highest scores
    # ------------------------------------------------------------

    print(
        "6. Highest Reactivation probabilities"
    )

    top_scores = (
        scores[
            [
                "customer",
                "reactivation_probability",
            ]
        ]
        .sort_values(
            "reactivation_probability",
            ascending=False,
        )
        .head(10)
    )

    print(
        top_scores.to_string(
            index=False
        )
    )

    print()

    # ------------------------------------------------------------
    # Step 7 - Lowest scores
    # ------------------------------------------------------------

    print(
        "7. Lowest Reactivation probabilities"
    )

    bottom_scores = (
        scores[
            [
                "customer",
                "reactivation_probability",
            ]
        ]
        .sort_values(
            "reactivation_probability",
            ascending=True,
        )
        .head(10)
    )

    print(
        bottom_scores.to_string(
            index=False
        )
    )

    print()

    # ------------------------------------------------------------
    # Step 8 - Save validation output
    # ------------------------------------------------------------

    VALIDATION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        features[
            [
                "customer",
            ]
        ]
        .merge(
            scores,
            on="customer",
            how="left",
            validate="one_to_one",
        )
    )

    output.to_parquet(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print(
        "8. Validation output written:"
    )

    print(
        f"   {VALIDATION_OUTPUT_PATH}"
    )

    print()

    # ------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------

    print(
        "=" * 72
    )

    print(
        "END-TO-END REACTIVATION VALIDATION: PASS"
    )

    print(
        "=" * 72
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
        "29 frozen Reactivation features"
    )

    print(
        "    |"
    )

    print(
        "    v"
    )

    print(
        "fitted research Random Forest"
    )

    print(
        "    |"
    )

    print(
        "    v"
    )

    print(
        "governed Reactivation probabilities"
    )


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    run_validation()