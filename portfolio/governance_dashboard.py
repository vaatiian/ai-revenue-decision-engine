"""
Governance Dashboard logic for the public portfolio application.

The dashboard consumes only:
- the curated public six-system model registry; and
- the curated public portfolio summary.

It does not:
- change governance decisions;
- infer new approved uses;
- grant operational permission;
- claim realised commercial impact;
- reconstruct private research evidence.
"""

from __future__ import annotations

from typing import Any

from portfolio.model_explorer import (
    DECISION_ENGINE_SYSTEM_IDS,
    STANDALONE_SYSTEM_IDS,
    approved_use_rows,
    governance_label,
    limitation_rows,
    validate_model_registry,
)


EXPECTED_PRODUCTION_SYSTEM_COUNT = 6


def validate_governance_sources(
    registry: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """
    Fail closed when public governance evidence is incomplete.
    """

    validate_model_registry(
        registry
    )

    if not isinstance(
        summary,
        dict,
    ):
        raise TypeError(
            "Portfolio summary must be a dictionary."
        )

    evidence = summary.get(
        "verified_evidence"
    )

    if not isinstance(
        evidence,
        dict,
    ):
        raise RuntimeError(
            "Portfolio summary is missing verified evidence."
        )

    total = evidence.get(
        "production_scoring_systems_total"
    )

    available = evidence.get(
        "production_scoring_systems_available"
    )

    intelligence_systems = evidence.get(
        "production_intelligence_systems"
    )

    if total != EXPECTED_PRODUCTION_SYSTEM_COUNT:
        raise RuntimeError(
            "Governance Dashboard requires six production scoring systems."
        )

    if available != EXPECTED_PRODUCTION_SYSTEM_COUNT:
        raise RuntimeError(
            "Governance Dashboard requires verified 6/6 runtime availability."
        )

    if intelligence_systems != EXPECTED_PRODUCTION_SYSTEM_COUNT:
        raise RuntimeError(
            "Governance Dashboard requires six intelligence systems."
        )

    claims_note = summary.get(
        "claims_note"
    )

    if not isinstance(
        claims_note,
        str,
    ) or not claims_note.strip():
        raise RuntimeError(
            "Governance Dashboard requires a public claims boundary."
        )


def runtime_availability_label(
    registry: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    """
    Return the verified aggregate runtime state.
    """

    validate_governance_sources(
        registry,
        summary,
    )

    evidence = summary[
        "verified_evidence"
    ]

    available = evidence[
        "production_scoring_systems_available"
    ]

    total = evidence[
        "production_scoring_systems_total"
    ]

    return (
        f"{available}/{total} AVAILABLE"
    )


def _approved_use_text(
    system: dict[str, Any],
) -> str:
    """
    Convert explicit approved use to one display string.
    """

    rows = approved_use_rows(
        system
    )

    if not rows:
        return (
            "Not exposed in public registry"
        )

    return "; ".join(
        rows
    )


def _limitation_text(
    system: dict[str, Any],
) -> str:
    """
    Convert explicit limitations to one display string.
    """

    rows = limitation_rows(
        system
    )

    if not rows:
        return (
            "No explicit limitation text exposed"
        )

    return "; ".join(
        rows
    )


def governance_surface_rows(
    registry: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Flatten system governance into permission surfaces.

    Recommendation governance is channel-specific and therefore produces
    two rows. Every other system produces one row.
    """

    validate_model_registry(
        registry
    )

    rows: list[dict[str, str]] = []

    for system in registry[
        "systems"
    ]:

        governance = system.get(
            "governance"
        )

        approved_use = system.get(
            "approved_use"
        )

        if isinstance(
            governance,
            dict,
        ):

            for channel, decision in governance.items():

                if isinstance(
                    approved_use,
                    dict,
                ):

                    use = approved_use.get(
                        channel
                    )

                else:

                    use = None

                rows.append(
                    {
                        "system_id":
                            system[
                                "system_id"
                            ],

                        "display_name":
                            system[
                                "display_name"
                            ],

                        "governance_surface":
                            (
                                str(
                                    channel
                                )
                                .replace(
                                    "_",
                                    " ",
                                )
                                .title()
                            ),

                        "governance_decision":
                            str(
                                decision
                            ),

                        "approved_use":
                            (
                                str(
                                    use
                                )
                                if use
                                else
                                "Not exposed in public registry"
                            ),
                    }
                )

        else:

            rows.append(
                {
                    "system_id":
                        system[
                            "system_id"
                        ],

                    "display_name":
                        system[
                            "display_name"
                        ],

                    "governance_surface":
                        "System",

                    "governance_decision":
                        str(
                            governance
                        ),

                    "approved_use":
                        _approved_use_text(
                            system
                        ),
                }
            )

    return rows


def system_governance_rows(
    registry: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build the six-system governance comparison matrix.
    """

    validate_governance_sources(
        registry,
        summary,
    )

    rows = []

    for system in registry[
        "systems"
    ]:

        system_id = system[
            "system_id"
        ]

        rows.append(
            {
                "system_id":
                    system_id,

                "system":
                    system[
                        "display_name"
                    ],

                "technical_availability":
                    "AVAILABLE",

                "governance":
                    governance_label(
                        system.get(
                            "governance"
                        )
                    ),

                "decision_engine_role":
                    (
                        "Included"
                        if system_id
                        in DECISION_ENGINE_SYSTEM_IDS
                        else
                        "Standalone"
                    ),

                "validation":
                    (
                        system.get(
                            "validation_method"
                        )
                        or
                        "Not exposed in public registry"
                    ),

                "approved_use":
                    _approved_use_text(
                        system
                    ),

                "limitations":
                    _limitation_text(
                        system
                    ),
            }
        )

    return rows


def governance_summary(
    registry: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, int]:
    """
    Summarise governance coverage without changing policy.
    """

    validate_governance_sources(
        registry,
        summary,
    )

    systems = registry[
        "systems"
    ]

    surfaces = governance_surface_rows(
        registry
    )

    go_surfaces = sum(
        1
        for row in surfaces
        if row[
            "governance_decision"
        ]
        ==
        "GO"
    )

    conditional_surfaces = sum(
        1
        for row in surfaces
        if row[
            "governance_decision"
        ]
        ==
        "CONDITIONAL GO"
    )

    with_approved_use = sum(
        1
        for system in systems
        if bool(
            approved_use_rows(
                system
            )
        )
    )

    with_limitations = sum(
        1
        for system in systems
        if bool(
            limitation_rows(
                system
            )
        )
    )

    with_validation = sum(
        1
        for system in systems
        if bool(
            system.get(
                "validation_method"
            )
        )
    )

    return {
        "production_systems":
            len(
                systems
            ),

        "runtime_available":
            summary[
                "verified_evidence"
            ][
                "production_scoring_systems_available"
            ],

        "decision_engine_systems":
            len(
                DECISION_ENGINE_SYSTEM_IDS
            ),

        "standalone_systems":
            len(
                STANDALONE_SYSTEM_IDS
            ),

        "governance_surfaces":
            len(
                surfaces
            ),

        "go_surfaces":
            go_surfaces,

        "conditional_go_surfaces":
            conditional_surfaces,

        "systems_with_explicit_approved_use":
            with_approved_use,

        "systems_with_explicit_limitations":
            with_limitations,

        "systems_with_validation_method":
            with_validation,
    }


def claims_boundary_rows(
    registry: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Explain the major governance boundaries represented by public evidence.
    """

    validate_governance_sources(
        registry,
        summary,
    )

    governance = governance_summary(
        registry,
        summary,
    )

    return [
        {
            "boundary":
                "Technical availability",

            "evidence":
                (
                    f"{governance['runtime_available']}/"
                    f"{governance['production_systems']} "
                    "production scoring systems verified available"
                ),

            "meaning":
                (
                    "Availability proves the scoring interface can run; "
                    "it does not grant unrestricted business use."
                ),
        },
        {
            "boundary":
                "Governance permission",

            "evidence":
                (
                    f"{governance['governance_surfaces']} governed "
                    "system or channel permission surfaces"
                ),

            "meaning":
                (
                    "GO and CONDITIONAL GO remain separate from "
                    "technical availability."
                ),
        },
        {
            "boundary":
                "Decision Engine inclusion",

            "evidence":
                (
                    f"{governance['decision_engine_systems']} systems "
                    "feed customer action orchestration; "
                    f"{governance['standalone_systems']} remain standalone"
                ),

            "meaning":
                (
                    "A production scoring system does not automatically "
                    "belong inside customer-level action assignment."
                ),
        },
        {
            "boundary":
                "Commercial claims",

            "evidence":
                summary[
                    "claims_note"
                ],

            "meaning":
                (
                    "Verified engineering and model evidence must not be "
                    "presented as realised incremental revenue or causal impact."
                ),
        },
    ]


def system_governance_detail(
    registry: dict[str, Any],
    system_id: str,
) -> dict[str, Any]:
    """
    Return one system's public governance dossier.
    """

    validate_model_registry(
        registry
    )

    for system in registry[
        "systems"
    ]:

        if system[
            "system_id"
        ] == system_id:

            return {
                "system_id":
                    system_id,

                "display_name":
                    system[
                        "display_name"
                    ],

                "governance":
                    governance_label(
                        system.get(
                            "governance"
                        )
                    ),

                "approved_use":
                    approved_use_rows(
                        system
                    ),

                "limitations":
                    limitation_rows(
                        system
                    ),

                "validation_method":
                    system.get(
                        "validation_method"
                    ),

                "decision_engine_role":
                    (
                        "Included"
                        if system_id
                        in DECISION_ENGINE_SYSTEM_IDS
                        else
                        "Standalone"
                    ),
            }

    raise KeyError(
        f"Unknown governance system ID: {system_id}"
    )