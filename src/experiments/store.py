"""
Append-only JSONL experiment store.

Each line in the store is one complete ExperimentRunRecord serialized as
compact JSON.

The store is intentionally simple and dependency-free:
- append only;
- duplicate run IDs fail closed;
- malformed JSON fails closed;
- schema drift fails closed;
- records are reconstructed through ExperimentRunRecord validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.experiments.record import (
    ExperimentRunRecord,
    experiment_run_to_dict,
)
from src.experiments.spec import (
    EXPERIMENT_RUN_FIELDS,
)


# ================================================================
# DEFAULT LOCATION
# ================================================================


DEFAULT_EXPERIMENT_STORE_PATH = (
    Path("artifacts")
    /
    "experiments"
    /
    "experiment_runs.jsonl"
)


# ================================================================
# JSON HELPERS
# ================================================================


def _record_to_json_line(
    record: ExperimentRunRecord,
) -> str:
    """
    Serialize one experiment record as one deterministic JSON line.
    """

    payload = (
        experiment_run_to_dict(
            record
        )
    )

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
        sort_keys=False,
    )


def _payload_to_record(
    payload: Any,
) -> ExperimentRunRecord:
    """
    Convert one parsed JSON object to a validated run record.
    """

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Experiment store line must contain one JSON object."
        )

    if tuple(
        payload.keys()
    ) != EXPERIMENT_RUN_FIELDS:
        raise ValueError(
            "Experiment store record does not match "
            "EXPERIMENT_RUN_FIELDS."
        )

    try:

        return ExperimentRunRecord(
            **payload
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "Experiment store contains an invalid experiment record."
        ) from exc


def _json_line_to_record(
    line: str,
    line_number: int,
) -> ExperimentRunRecord:
    """
    Parse and validate one JSONL line.
    """

    if not isinstance(
        line,
        str,
    ):
        raise TypeError(
            "Experiment store line must be a string."
        )

    if not line.strip():
        raise ValueError(
            f"Experiment store contains an empty line at line "
            f"{line_number}."
        )

    try:

        payload = json.loads(
            line
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Experiment store contains invalid JSON at line "
            f"{line_number}."
        ) from exc

    try:

        return _payload_to_record(
            payload
        )

    except ValueError as exc:

        raise ValueError(
            f"Experiment store contains an invalid record at line "
            f"{line_number}."
        ) from exc


# ================================================================
# READ OPERATIONS
# ================================================================


def list_experiment_runs(
    path: str | Path = (
        DEFAULT_EXPERIMENT_STORE_PATH
    ),
) -> tuple[ExperimentRunRecord, ...]:
    """
    Load all experiment runs from an append-only JSONL store.

    A missing store is treated as an empty store.
    """

    store_path = Path(
        path
    )

    if not store_path.exists():

        return ()

    if not store_path.is_file():

        raise ValueError(
            "Experiment store path must be a file."
        )

    records = []

    seen_run_ids = set()

    with store_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for (
            line_number,
            line,
        ) in enumerate(
            handle,
            start=1,
        ):

            record = (
                _json_line_to_record(
                    line.rstrip(
                        "\n"
                    ),
                    line_number,
                )
            )

            if (
                record.run_id
                in
                seen_run_ids
            ):
                raise ValueError(
                    "Experiment store contains duplicate run_id: "
                    f"{record.run_id}"
                )

            seen_run_ids.add(
                record.run_id
            )

            records.append(
                record
            )

    return tuple(
        records
    )


def get_experiment_run(
    run_id: str,
    path: str | Path = (
        DEFAULT_EXPERIMENT_STORE_PATH
    ),
) -> ExperimentRunRecord:
    """
    Return one experiment run by exact run_id.
    """

    if not isinstance(
        run_id,
        str,
    ):
        raise TypeError(
            "run_id must be a string."
        )

    if not run_id.strip():
        raise ValueError(
            "run_id must not be empty."
        )

    matches = [
        record
        for record
        in list_experiment_runs(
            path
        )
        if record.run_id
        ==
        run_id
    ]

    if not matches:

        raise KeyError(
            f"Unknown experiment run_id: {run_id}"
        )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            "Experiment store contains duplicate run_id records."
        )

    return matches[0]


# ================================================================
# APPEND OPERATION
# ================================================================


def append_experiment_run(
    record: ExperimentRunRecord,
    path: str | Path = (
        DEFAULT_EXPERIMENT_STORE_PATH
    ),
) -> Path:
    """
    Append one validated run record to the JSONL store.

    Duplicate run IDs fail closed.
    """

    if not isinstance(
        record,
        ExperimentRunRecord,
    ):
        raise TypeError(
            "record must be an ExperimentRunRecord."
        )

    store_path = Path(
        path
    )

    if (
        store_path.exists()
        and
        not store_path.is_file()
    ):
        raise ValueError(
            "Experiment store path must be a file."
        )

    existing = (
        list_experiment_runs(
            store_path
        )
    )

    existing_ids = {
        item.run_id
        for item
        in existing
    }

    if (
        record.run_id
        in
        existing_ids
    ):
        raise ValueError(
            "Experiment run_id already exists: "
            f"{record.run_id}"
        )

    store_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    line = (
        _record_to_json_line(
            record
        )
    )

    with store_path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:

        handle.write(
            line
        )

        handle.write(
            "\n"
        )

    return store_path