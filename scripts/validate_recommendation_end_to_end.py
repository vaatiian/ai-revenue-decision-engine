"""
End-to-end production validation for the frozen Next Best Product
recommendation architecture.

Flow
----
canonical cleaned transactions
    |
    v
qualifying recommendation lines
    |
    v
customer-order sequence
    |
    v
observed next-basket episodes
    |
    v
July-August 2011 consumed holdout
    |
    v
RecommendationScorer
    |
    +--> Customer Repeat Frequency
    |
    +--> Item-to-Item Affinity
    |
    +--> 70% Repeat / 30% Discovery presentation
    |
    v
research-parity metrics
    |
    v
governance manifest + source hash validation
    |
    v
production validation output

This script validates the existing frozen architecture.

It does not:
- train a model;
- tune recommendation parameters;
- alter the research holdout;
- modify the recommendation architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

from src.models.recommendation_engine import (
    RECOMMENDATION_RESEARCH_END_DATE,
    build_customer_orders,
    build_observed_next_basket_population,
    build_recommendation_lines,
)
from src.models.recommendation_manifest import (
    DEFAULT_RECOMMENDATION_MANIFEST_PATH,
    load_recommendation_manifest,
    validate_recommendation_manifest,
)
from src.models.recommendation_scorer import (
    RecommendationScorer,
)
from src.models.recommendation_spec import (
    RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_10,
    RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_REPEAT_ELIGIBLE_HIT_RATE_AT_20,
    RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_10,
    RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_HYBRID_HOLDOUT_HIT_RATE_AT_20,
    RECOMMENDATION_TEST_END,
    RECOMMENDATION_TEST_START,
)


# ================================================================
# PATHS
# ================================================================

TRANSACTION_PATH = (
    PROJECT_ROOT
    /
    "cleaned_online_retail.parquet"
)


VALIDATION_OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "outputs"
    /
    "validation"
)


VALIDATION_OUTPUT_PATH = (
    VALIDATION_OUTPUT_DIR
    /
    "recommendation_end_to_end_validation.parquet"
)


# ================================================================
# PARITY TOLERANCE
# ================================================================

# The frozen notebook metrics are stored to six decimal places.
#
# A small tolerance protects against harmless tie-order effects where
# products have mathematically identical recommendation scores while
# still failing on material model drift.

HOLDOUT_METRIC_TOLERANCE = 0.005


# ================================================================
# DATA LOADING
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

    transactions[
        "date"
    ] = pd.to_datetime(
        transactions[
            "date"
        ]
    )

    return transactions


# ================================================================
# RESEARCH HOLDOUT POPULATION
# ================================================================

def build_holdout_episodes(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct the exact mature next-basket episode population and
    select the frozen July-August 2011 performance holdout.
    """

    recommendation_lines = (
        build_recommendation_lines(
            transactions=transactions,
            observation_end_date=(
                RECOMMENDATION_RESEARCH_END_DATE
            ),
        )
    )

    if recommendation_lines.empty:
        raise ValueError(
            "No qualifying recommendation lines were produced."
        )

    customer_orders = (
        build_customer_orders(
            recommendation_lines=(
                recommendation_lines
            ),
            observation_end_date=(
                RECOMMENDATION_RESEARCH_END_DATE
            ),
        )
    )

    episodes = (
        build_observed_next_basket_population(
            customer_orders
        )
    )

    if episodes.empty:
        raise ValueError(
            "No observed next-basket recommendation episodes "
            "were produced."
        )

    start_period = pd.Period(
        RECOMMENDATION_TEST_START,
        freq="M",
    )

    end_period = pd.Period(
        RECOMMENDATION_TEST_END,
        freq="M",
    )

    holdout = (
        episodes.loc[
            (
                episodes[
                    "prediction_period"
                ]
                >=
                start_period
            )
            &
            (
                episodes[
                    "prediction_period"
                ]
                <=
                end_period
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if holdout.empty:
        raise ValueError(
            "Frozen July-August recommendation holdout is empty."
        )

    return holdout


# ================================================================
# HIT-RATE HELPERS
# ================================================================

def recommendation_hit(
    recommended_products,
    actual_products,
    k: int,
) -> int:
    """
    Return 1 when at least one Top-K recommended product appears in
    the observed target basket, otherwise 0.
    """

    recommended = set(
        list(
            recommended_products
        )[
            :k
        ]
    )

    actual = set(
        actual_products
    )

    return int(
        bool(
            recommended
            &
            actual
        )
    )


def hit_rate(
    recommendation_lists,
    actual_lists,
    k: int,
) -> float:
    """
    Calculate mean episode-level Hit Rate@K.
    """

    hits = [
        recommendation_hit(
            recommended_products=recommended,
            actual_products=actual,
            k=k,
        )
        for recommended, actual
        in zip(
            recommendation_lists,
            actual_lists,
        )
    ]

    if not hits:
        raise ValueError(
            "Cannot calculate hit rate on an empty population."
        )

    return float(
        pd.Series(
            hits,
            dtype=float,
        ).mean()
    )


# ================================================================
# HOLDOUT METRICS
# ================================================================

def calculate_holdout_metrics(
    scored: pd.DataFrame,
) -> dict[str, float]:
    """
    Recalculate the frozen recommendation holdout metrics.

    General repeat / hybrid metrics:
        all observed next-basket holdout episodes.

    Repeat-eligible metric:
        episodes containing at least one repeat product.

    Discovery metric:
        episodes containing at least one genuinely new product.
    """

    metrics = {}

    metrics[
        "repeat_hit_rate_at_10"
    ] = hit_rate(
        recommendation_lists=(
            scored[
                "repeat_recommendations"
            ]
        ),
        actual_lists=(
            scored[
                "next_basket_products"
            ]
        ),
        k=10,
    )

    metrics[
        "repeat_hit_rate_at_20"
    ] = hit_rate(
        recommendation_lists=(
            scored[
                "repeat_recommendations"
            ]
        ),
        actual_lists=(
            scored[
                "next_basket_products"
            ]
        ),
        k=20,
    )

    repeat_eligible = (
        scored.loc[
            scored[
                "repeat_product_count"
            ]
            >
            0
        ]
    )

    metrics[
        "repeat_eligible_hit_rate_at_20"
    ] = hit_rate(
        recommendation_lists=(
            repeat_eligible[
                "repeat_recommendations"
            ]
        ),
        actual_lists=(
            repeat_eligible[
                "repeat_products"
            ]
        ),
        k=20,
    )

    discovery_eligible = (
        scored.loc[
            scored[
                "new_product_count"
            ]
            >
            0
        ]
    )

    metrics[
        "discovery_hit_rate_at_10"
    ] = hit_rate(
        recommendation_lists=(
            discovery_eligible[
                "discovery_recommendations"
            ]
        ),
        actual_lists=(
            discovery_eligible[
                "new_products"
            ]
        ),
        k=10,
    )

    metrics[
        "discovery_hit_rate_at_20"
    ] = hit_rate(
        recommendation_lists=(
            discovery_eligible[
                "discovery_recommendations"
            ]
        ),
        actual_lists=(
            discovery_eligible[
                "new_products"
            ]
        ),
        k=20,
    )

    metrics[
        "hybrid_hit_rate_at_20"
    ] = hit_rate(
        recommendation_lists=(
            scored[
                "hybrid_recommendations"
            ]
        ),
        actual_lists=(
            scored[
                "next_basket_products"
            ]
        ),
        k=20,
    )

    return metrics


# ================================================================
# PARITY VALIDATION
# ================================================================

def validate_metric_parity(
    actual_metrics: dict[str, float],
) -> None:
    """
    Compare production holdout results with the frozen research model card.
    """

    expected_metrics = {
        "repeat_hit_rate_at_10":
            RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_10,

        "repeat_hit_rate_at_20":
            RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_20,

        "repeat_eligible_hit_rate_at_20":
            RECOMMENDATION_REPEAT_ELIGIBLE_HIT_RATE_AT_20,

        "discovery_hit_rate_at_10":
            RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_10,

        "discovery_hit_rate_at_20":
            RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_20,

        "hybrid_hit_rate_at_20":
            RECOMMENDATION_HYBRID_HOLDOUT_HIT_RATE_AT_20,
    }

    failures = []

    for metric_name, expected_value in (
        expected_metrics.items()
    ):

        actual_value = (
            actual_metrics[
                metric_name
            ]
        )

        difference = abs(
            actual_value
            -
            expected_value
        )

        if (
            difference
            >
            HOLDOUT_METRIC_TOLERANCE
        ):
            failures.append(
                (
                    metric_name,
                    expected_value,
                    actual_value,
                    difference,
                )
            )

    if failures:

        failure_lines = [
            (
                f"{name}: expected={expected:.6f}, "
                f"actual={actual:.6f}, "
                f"difference={difference:.6f}"
            )
            for (
                name,
                expected,
                actual,
                difference,
            )
            in failures
        ]

        raise ValueError(
            "Recommendation holdout parity validation failed:\n"
            +
            "\n".join(
                failure_lines
            )
        )


# ================================================================
# GOVERNANCE VALIDATION
# ================================================================

def validate_governance_manifest() -> dict:
    """
    Load the exported manifest and verify the frozen contract plus
    current governed production source hashes.
    """

    manifest = (
        load_recommendation_manifest(
            DEFAULT_RECOMMENDATION_MANIFEST_PATH
        )
    )

    validate_recommendation_manifest(
        manifest=manifest,
        project_root=PROJECT_ROOT,
        verify_source_hashes=True,
    )

    return manifest


# ================================================================
# SCORE HOLDOUT
# ================================================================

def score_holdout(
    transactions: pd.DataFrame,
    holdout: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score the frozen recommendation holdout through the production
    RecommendationScorer.
    """

    scorer = RecommendationScorer(
        top_n=20
    )

    score_episodes = (
        holdout[
            [
                "customer",
                "invoice",
                "order_date",
                "prediction_period",
            ]
        ]
        .copy()
    )

    scores = scorer.score(
        transactions=transactions,
        episodes=score_episodes,
    )

    target_columns = (
        holdout[
            [
                "customer",
                "invoice",
                "next_invoice",
                "next_order_date",
                "days_to_next_order",
                "next_basket_products",
                "historical_products",
                "repeat_product_count",
                "new_product_count",
                "repeat_products",
                "new_products",
            ]
        ]
        .copy()
    )

    validation = (
        scores
        .merge(
            target_columns,
            on=[
                "customer",
                "invoice",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    if (
        validation[
            "next_basket_products"
        ].isna().any()
    ):
        raise RuntimeError(
            "Recommendation validation lost observed target baskets "
            "during score/target merge."
        )

    return validation


# ================================================================
# STRUCTURAL VALIDATION
# ================================================================

def validate_score_structure(
    scores: pd.DataFrame,
    expected_rows: int,
) -> None:
    """
    Fail closed on malformed production score output.
    """

    if len(
        scores
    ) != expected_rows:
        raise RuntimeError(
            "Production recommendation scorer did not return exactly "
            "one output row per holdout episode."
        )

    if scores.duplicated(
        subset=[
            "customer",
            "invoice",
        ]
    ).any():
        raise RuntimeError(
            "Production recommendation scores contain duplicate "
            "customer/invoice keys."
        )

    for column in (
        "repeat_recommendations",
        "discovery_recommendations",
        "hybrid_recommendations",
    ):

        if scores[
            column
        ].isna().any():
            raise RuntimeError(
                f"Recommendation output contains missing {column}."
            )

        if (
            scores[
                column
            ]
            .map(
                len
            )
            >
            20
        ).any():
            raise RuntimeError(
                f"{column} contains more than Top-20 products."
            )


# ================================================================
# OUTPUT
# ================================================================

def save_validation_output(
    scores: pd.DataFrame,
) -> None:
    """
    Persist the governed validation result.
    """

    VALIDATION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = scores.copy()

    # Lists/tuples are normalised to lists for stable parquet output.
    for column in (
        "repeat_recommendations",
        "discovery_recommendations",
        "hybrid_recommendations",
        "next_basket_products",
        "historical_products",
        "repeat_products",
        "new_products",
    ):

        output[
            column
        ] = output[
            column
        ].map(
            lambda value:
                list(
                    value
                )
        )

    output.to_parquet(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )


# ================================================================
# REPORT
# ================================================================

def print_validation_report(
    transactions: pd.DataFrame,
    holdout: pd.DataFrame,
    scores: pd.DataFrame,
    metrics: dict[str, float],
    manifest: dict,
) -> None:
    """
    Print a concise production validation report.
    """

    print()
    print(
        "="
        *
        72
    )

    print(
        "RECOMMENDATION END-TO-END VALIDATION: PASS"
    )

    print(
        "="
        *
        72
    )

    print()

    print(
        f"Canonical transaction rows: {len(transactions):,}"
    )

    print(
        f"Holdout episodes: {len(holdout):,}"
    )

    print(
        "Holdout period: "
        f"{holdout['prediction_period'].min()} "
        "to "
        f"{holdout['prediction_period'].max()}"
    )

    print(
        "Unique holdout customers: "
        f"{holdout['customer'].nunique():,}"
    )

    print()

    print(
        "Production architecture:"
    )

    print(
        "  Repeat: "
        f"{manifest['architecture']['repeat_channel']}"
    )

    print(
        "  Discovery: "
        f"{manifest['architecture']['discovery_channel']}"
    )

    print(
        "  Optional hybrid: "
        f"{manifest['architecture']['optional_hybrid']}"
    )

    print()

    print(
        "Frozen holdout parity:"
    )

    expected = {
        "repeat_hit_rate_at_10":
            RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_10,

        "repeat_hit_rate_at_20":
            RECOMMENDATION_REPEAT_HOLDOUT_HIT_RATE_AT_20,

        "repeat_eligible_hit_rate_at_20":
            RECOMMENDATION_REPEAT_ELIGIBLE_HIT_RATE_AT_20,

        "discovery_hit_rate_at_10":
            RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_10,

        "discovery_hit_rate_at_20":
            RECOMMENDATION_DISCOVERY_HOLDOUT_HIT_RATE_AT_20,

        "hybrid_hit_rate_at_20":
            RECOMMENDATION_HYBRID_HOLDOUT_HIT_RATE_AT_20,
    }

    for metric_name in expected:

        print(
            f"  {metric_name}: "
            f"production={metrics[metric_name]:.6f} | "
            f"research={expected[metric_name]:.6f}"
        )

    print()

    print(
        "Governed source hashes: PASS"
    )

    print(
        "Manifest validation: PASS"
    )

    print(
        "Score structure validation: PASS"
    )

    print(
        "Research holdout parity: PASS"
    )

    print()

    print(
        "Validation output:"
    )

    print(
        f"  {VALIDATION_OUTPUT_PATH}"
    )

    print()

    print(
        "="
        *
        72
    )


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    """
    Execute complete recommendation production validation.
    """

    print(
        "Loading canonical transactions..."
    )

    transactions = (
        load_canonical_transactions()
    )

    print(
        "Validating recommendation governance manifest..."
    )

    manifest = (
        validate_governance_manifest()
    )

    print(
        "Reconstructing frozen recommendation holdout..."
    )

    holdout = (
        build_holdout_episodes(
            transactions
        )
    )

    print(
        "Scoring holdout through production RecommendationScorer..."
    )

    scores = (
        score_holdout(
            transactions=transactions,
            holdout=holdout,
        )
    )

    validate_score_structure(
        scores=scores,
        expected_rows=len(
            holdout
        ),
    )

    print(
        "Calculating holdout recommendation metrics..."
    )

    metrics = (
        calculate_holdout_metrics(
            scores
        )
    )

    print(
        "Validating research parity..."
    )

    validate_metric_parity(
        metrics
    )

    print(
        "Writing validation output..."
    )

    save_validation_output(
        scores
    )

    print_validation_report(
        transactions=transactions,
        holdout=holdout,
        scores=scores,
        metrics=metrics,
        manifest=manifest,
    )


if __name__ == "__main__":
    main()