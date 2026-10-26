"""
Export the frozen deterministic recommendation governance manifest.

No fitted model artifact is created because the governed recommendation
architecture is deterministic.

This script records:
- frozen architecture;
- ranking formulas;
- support thresholds;
- temporal governance;
- research decisions;
- holdout performance;
- SHA-256 hashes of governed production source files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


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

from src.models.recommendation_manifest import (
    DEFAULT_RECOMMENDATION_MANIFEST_PATH,
    build_recommendation_manifest,
    validate_recommendation_manifest,
)


# ================================================================
# EXPORT
# ================================================================

def main() -> None:
    """
    Build, validate and export the deterministic recommendation
    governance manifest.
    """

    manifest = build_recommendation_manifest(
        project_root=PROJECT_ROOT
    )

    validate_recommendation_manifest(
        manifest=manifest,
        project_root=PROJECT_ROOT,
        verify_source_hashes=True,
    )

    output_path = (
        DEFAULT_RECOMMENDATION_MANIFEST_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print(
        "RECOMMENDATION GOVERNANCE MANIFEST: EXPORTED"
    )

    print(
        f"Path: {output_path}"
    )

    print(
        f"Model: {manifest['model_name']}"
    )

    print(
        f"Version: {manifest['model_version']}"
    )

    print(
        f"Artifact type: {manifest['artifact_type']}"
    )

    print(
        "Serialized model artifact required: "
        f"{manifest['serialized_model_artifact_required']}"
    )

    print(
        "Governed source files: "
        f"{len(manifest['governed_source_sha256'])}"
    )


if __name__ == "__main__":
    main()