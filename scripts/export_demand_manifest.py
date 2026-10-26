"""
Export the frozen deterministic weekly demand forecasting manifest.
"""

from __future__ import annotations

import sys

from pathlib import Path


# ================================================================
# PROJECT IMPORT PATH
# ================================================================

PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
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


from src.models.demand_manifest import (  # noqa: E402
    DEMAND_MANIFEST_PATH,
    validate_demand_manifest_file,
    write_demand_manifest,
)


# ================================================================
# EXPORT
# ================================================================

def main() -> None:
    """
    Generate and immediately validate the demand governance manifest.
    """

    path = write_demand_manifest(
        DEMAND_MANIFEST_PATH
    )

    manifest = validate_demand_manifest_file(
        path,
        verify_source_hashes=True,
    )

    print(
        "DEMAND GOVERNANCE MANIFEST EXPORT: PASS"
    )

    print(
        f"Manifest: {path}"
    )

    print(
        f"Model: {manifest['model_name']}"
    )

    print(
        f"Version: {manifest['model_version']}"
    )

    print(
        f"Type: {manifest['model_type']}"
    )

    print(
        "Serialized model artifact required: "
        f"{manifest['serialized_model_required']}"
    )

    print(
        "Horizons: "
        f"{manifest['forecast_horizons_weeks']}"
    )

    print(
        "Seasonal weight: "
        f"{manifest['architecture']['seasonal_weight']}"
    )

    print(
        "Recent weight: "
        f"{manifest['architecture']['recent_weight']}"
    )

    print(
        "Governance decision: "
        f"{manifest['governance']['decision']}"
    )

    print(
        "Research notebook SHA-256: "
        f"{manifest['source_sha256']['research_notebook']}"
    )

    print(
        "Production spec SHA-256: "
        f"{manifest['source_sha256']['production_spec']}"
    )

    print(
        "Weekly feature engine SHA-256: "
        f"{manifest['source_sha256']['weekly_feature_engine']}"
    )

    print(
        "Production scorer SHA-256: "
        f"{manifest['source_sha256']['production_scorer']}"
    )


if __name__ == "__main__":
    main()