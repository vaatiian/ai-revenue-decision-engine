"""
End-to-end validation for the production Repeat Purchase scorer.

This script validates:
1. canonical input loading;
2. Notebook 02 research-parity historical dataset construction;
3. frozen feature order;
4. exported artifact integrity;
5. manifest integrity;
6. production scorer output;
7. exact parity between direct model probabilities and scorer output;
8. research population metrics against the locked Notebook 02 results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ================================================================
# PROJECT ROOT
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
# PROJECT IMPORTS
# ================================================================

from src.data.loader import (
    load_transactions,
)
from src.features.repeat_purchase_features import (
    build_repeat_purchase_historical_dataset,
    select_repeat_purchase_model_features,
)
from src.models.repeat_purchase_manifest import (
    DEFAULT_REPEAT_PURCHASE_ARTIFACT_PATH,
    DEFAULT_REPEAT_PURCHASE_MANIFEST_PATH,
    load_and_validate_repeat_purchase_manifest,
)
from src.models.repeat_purchase_scorer import (
    score_repeat_purchase_features,
)
from src.models.repeat_purchase_spec import (
    REPEAT_PURCHASE_FINAL_DECISION,
    REPEAT_PURCHASE_MODEL_FEATURES,
    REPEAT_PURCHASE_MODEL_NAME,
    REPEAT_PURCHASE_MODEL_VERSION,
    REPEAT_PURCHASE_REPORTED_TEST_BRIER,
    REPEAT_PURCHASE_REPORTED_TEST_PR_AUC,
    REPEAT_PURCHASE_REPORTED_TEST_ROC_AUC,
)


# ================================================================
# DATA PATH
# ================================================================

DATA_PATH = (
    PROJECT_ROOT
    /
    "cleaned_online_retail.parquet"
)


# ================================================================
# METRICS
# ================================================================

def lift_at_fraction(
    y_true,
    probabilities,
    fraction: float,
) -> float:
    """
    Calculate lift for the highest-scoring fraction of customers.
    """

    scored = pd.DataFrame(
        {
            "actual":
                np.asarray(
                    y_true
                ),

            "probability":
                np.asarray(
                    probabilities
                ),
        }
    )

    scored = (
        scored
        .sort_values(
            "probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    population_rate = (
        scored[
            "actual"
        ].mean()
    )

    if population_rate == 0:
        raise ValueError(
            "Cannot calculate lift when target prevalence is zero."
        )

    top_n = max(
        1,
        int(
            np.ceil(
                len(
                    scored
                )
                *
                fraction
            )
        ),
    )

    top_rate = (
        scored.iloc[
            :top_n
        ][
            "actual"
        ]
        .mean()
    )

    return (
        top_rate
        /
        population_rate
    )


# ================================================================
# VALIDATION
# ================================================================

def main() -> None:
    print(
        "=" * 72
    )

    print(
        "REPEAT PURCHASE END-TO-END VALIDATION"
    )

    print(
        "=" * 72
    )

    print()

    # ------------------------------------------------------------
    # Load canonical data
    # ------------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical parquet not found: {DATA_PATH}"
        )

    data, validation = (
        load_transactions(
            DATA_PATH
        )
    )

    print(
        f"Canonical rows: "
        f"{len(data):,}"
    )

    print(
        "Canonical schema validation: PASS"
    )

    # ------------------------------------------------------------
    # Build research-parity historical dataset
    # ------------------------------------------------------------

    historical = (
        build_repeat_purchase_historical_dataset(
            data
        )
    )

    if historical.empty:
        raise RuntimeError(
            "Repeat Purchase historical dataset is empty."
        )

    print(
        f"Eligible historical customers: "
        f"{len(historical):,}"
    )

    print(
        f"Repeat rate: "
        f"{historical['repeat_within_90d'].mean() * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # Feature contract
    # ------------------------------------------------------------

    X = (
        select_repeat_purchase_model_features(
            historical
        )
    )

    y = (
        historical[
            "repeat_within_90d"
        ]
        .astype(
            int
        )
        .to_numpy()
    )

    if (
        tuple(
            X.columns
        )
        !=
        REPEAT_PURCHASE_MODEL_FEATURES
    ):
        raise RuntimeError(
            "Repeat Purchase feature order mismatch."
        )

    print(
        f"Frozen feature count: "
        f"{X.shape[1]}"
    )

    # ------------------------------------------------------------
    # Manifest + artifact
    # ------------------------------------------------------------

    manifest = (
        load_and_validate_repeat_purchase_manifest(
            manifest_path=(
                DEFAULT_REPEAT_PURCHASE_MANIFEST_PATH
            ),
            artifact_path=(
                DEFAULT_REPEAT_PURCHASE_ARTIFACT_PATH
            ),
        )
    )

    model = joblib.load(
        DEFAULT_REPEAT_PURCHASE_ARTIFACT_PATH
    )

    print(
        "Artifact + manifest validation: PASS"
    )

    # ------------------------------------------------------------
    # Direct model scoring
    # ------------------------------------------------------------

    direct_probabilities = (
        model.predict_proba(
            X
        )[
            :,
            1
        ]
    )

    if (
        len(
            direct_probabilities
        )
        !=
        len(
            historical
        )
    ):
        raise RuntimeError(
            "Direct prediction length mismatch."
        )

    # ------------------------------------------------------------
    # Production scorer parity
    # ------------------------------------------------------------

    scoring_input = (
        historical[
            [
                "customer",
                "first_invoice",
                "first_order_date",
                *REPEAT_PURCHASE_MODEL_FEATURES,
            ]
        ]
        .copy()
    )

    scored = (
        score_repeat_purchase_features(
            features=scoring_input,
            model=model,
        )
    )

    scorer_probabilities = (
        scored[
            "repeat_purchase_score"
        ]
        .to_numpy()
    )

    if not np.allclose(
        direct_probabilities,
        scorer_probabilities,
        rtol=0.0,
        atol=1e-15,
    ):
        max_difference = (
            np.max(
                np.abs(
                    direct_probabilities
                    -
                    scorer_probabilities
                )
            )
        )

        raise RuntimeError(
            "Production scorer probability parity failed. "
            f"Maximum difference: {max_difference}"
        )

    print(
        "Direct model vs scorer probability parity: PASS"
    )

    # ------------------------------------------------------------
    # Research-population metrics
    # ------------------------------------------------------------

    pr_auc = (
        average_precision_score(
            y,
            direct_probabilities,
        )
    )

    roc_auc = (
        roc_auc_score(
            y,
            direct_probabilities,
        )
    )

    brier = (
        brier_score_loss(
            y,
            direct_probabilities,
        )
    )

    lift10 = (
        lift_at_fraction(
            y_true=y,
            probabilities=direct_probabilities,
            fraction=0.10,
        )
    )

    lift20 = (
        lift_at_fraction(
            y_true=y,
            probabilities=direct_probabilities,
            fraction=0.20,
        )
    )

    print()

    print(
        "Research-parity population metrics"
    )

    print(
        "-" * 72
    )

    print(
        f"PR-AUC:   {pr_auc:.6f}"
    )

    print(
        f"ROC-AUC:  {roc_auc:.6f}"
    )

    print(
        f"Brier:    {brier:.6f}"
    )

    print(
        f"Lift@10:  {lift10:.6f}x"
    )

    print(
        f"Lift@20:  {lift20:.6f}x"
    )

    # ------------------------------------------------------------
    # Governance metadata
    # ------------------------------------------------------------

    if (
        manifest[
            "model_name"
        ]
        !=
        REPEAT_PURCHASE_MODEL_NAME
    ):
        raise RuntimeError(
            "Manifest model identity mismatch."
        )

    if (
        manifest[
            "model_version"
        ]
        !=
        REPEAT_PURCHASE_MODEL_VERSION
    ):
        raise RuntimeError(
            "Manifest model version mismatch."
        )

    if (
        manifest[
            "governance_decision"
        ]
        !=
        REPEAT_PURCHASE_FINAL_DECISION
    ):
        raise RuntimeError(
            "Manifest governance decision mismatch."
        )

    if not (
        scored[
            "model_name"
        ]
        .eq(
            REPEAT_PURCHASE_MODEL_NAME
        )
        .all()
    ):
        raise RuntimeError(
            "Scorer model metadata mismatch."
        )

    if not (
        scored[
            "model_version"
        ]
        .eq(
            REPEAT_PURCHASE_MODEL_VERSION
        )
        .all()
    ):
        raise RuntimeError(
            "Scorer model version metadata mismatch."
        )

    if not (
        scored[
            "governance_decision"
        ]
        .eq(
            REPEAT_PURCHASE_FINAL_DECISION
        )
        .all()
    ):
        raise RuntimeError(
            "Scorer governance metadata mismatch."
        )

    print()

    print(
        "Governance metadata parity: PASS"
    )

    # ------------------------------------------------------------
    # Locked final-test context
    # ------------------------------------------------------------

    print()

    print(
        "NOTE:"
    )

    print(
        "The metrics above are calculated over the full eligible "
        "historical research-parity population using the final refit model."
    )

    print(
        "They are therefore NOT expected to equal the locked final-test "
        "metrics exactly."
    )

    print()

    print(
        "Locked Notebook 02 final-test metrics:"
    )

    print(
        f"PR-AUC ~ {REPEAT_PURCHASE_REPORTED_TEST_PR_AUC}"
    )

    print(
        f"ROC-AUC ~ {REPEAT_PURCHASE_REPORTED_TEST_ROC_AUC}"
    )

    print(
        f"Brier ~ {REPEAT_PURCHASE_REPORTED_TEST_BRIER}"
    )

    # ------------------------------------------------------------
    # Final
    # ------------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "REPEAT PURCHASE END-TO-END VALIDATION: PASS"
    )

    print(
        "=" * 72
    )

    print(
        f"Model: "
        f"{REPEAT_PURCHASE_MODEL_NAME}"
    )

    print(
        f"Version: "
        f"{REPEAT_PURCHASE_MODEL_VERSION}"
    )

    print(
        f"Governance decision: "
        f"{REPEAT_PURCHASE_FINAL_DECISION}"
    )

    print(
        f"Scored customers: "
        f"{len(scored):,}"
    )


if __name__ == "__main__":
    main()