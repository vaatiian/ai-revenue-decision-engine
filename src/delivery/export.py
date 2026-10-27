"""
CRM/action-delivery export adapters.

This module persists validated DeliveryPayload records to portable delivery
artifacts.

Supported formats
-----------------
- CSV
- JSONL

The export layer never changes Decision Engine or delivery policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.delivery.builder import (
    delivery_batch_to_frame,
)

from src.delivery.payload import (
    DeliveryPayload,
    delivery_payload_to_dict,
)

from src.delivery.spec import (
    DELIVERY_PAYLOAD_COLUMNS,
)


# ================================================================
# VALIDATION HELPERS
# ================================================================


def _normalise_payload_collection(
    payloads,
) -> tuple[
    DeliveryPayload,
    ...,
]:
    """
    Validate and freeze one iterable of DeliveryPayload records.
    """

    if isinstance(
        payloads,
        DeliveryPayload,
    ):
        raise TypeError(
            "payloads must be an iterable of DeliveryPayload records."
        )

    try:

        payload_tuple = tuple(
            payloads
        )

    except TypeError as exc:

        raise TypeError(
            "payloads must be an iterable of DeliveryPayload records."
        ) from exc

    for payload in payload_tuple:

        if not isinstance(
            payload,
            DeliveryPayload,
        ):
            raise TypeError(
                "Every payload must be a DeliveryPayload."
            )

    return payload_tuple


def _prepare_export_path(
    path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """
    Validate one export destination.
    """

    if not isinstance(
        overwrite,
        bool,
    ):
        raise TypeError(
            "overwrite must be a bool."
        )

    export_path = Path(
        path
    )

    if (
        export_path.exists()
        and
        export_path.is_dir()
    ):
        raise ValueError(
            "Delivery export path must be a file."
        )

    if (
        export_path.exists()
        and
        not overwrite
    ):
        raise FileExistsError(
            f"Delivery export already exists: {export_path}"
        )

    export_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return export_path


# ================================================================
# CSV EXPORT
# ================================================================


def export_delivery_csv(
    payloads,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Export one delivery batch to canonical UTF-8 CSV.
    """

    payload_tuple = (
        _normalise_payload_collection(
            payloads
        )
    )

    export_path = (
        _prepare_export_path(
            path,
            overwrite=(
                overwrite
            ),
        )
    )

    frame = (
        delivery_batch_to_frame(
            payload_tuple
        )
    )

    if tuple(
        frame.columns
    ) != DELIVERY_PAYLOAD_COLUMNS:
        raise RuntimeError(
            "Delivery CSV frame does not match "
            "DELIVERY_PAYLOAD_COLUMNS."
        )

    frame.to_csv(
        export_path,
        index=False,
        encoding="utf-8",
    )

    return export_path


# ================================================================
# JSONL EXPORT
# ================================================================


def export_delivery_jsonl(
    payloads,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Export one delivery batch as one JSON object per line.
    """

    payload_tuple = (
        _normalise_payload_collection(
            payloads
        )
    )

    export_path = (
        _prepare_export_path(
            path,
            overwrite=(
                overwrite
            ),
        )
    )

    mode = (
        "w"
        if overwrite
        else "x"
    )

    with export_path.open(
        mode,
        encoding="utf-8",
        newline="\n",
    ) as handle:

        for payload in payload_tuple:

            record = (
                delivery_payload_to_dict(
                    payload
                )
            )

            if tuple(
                record.keys()
            ) != DELIVERY_PAYLOAD_COLUMNS:
                raise RuntimeError(
                    "Delivery JSONL record does not match "
                    "DELIVERY_PAYLOAD_COLUMNS."
                )

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                    sort_keys=False,
                )
            )

            handle.write(
                "\n"
            )

    return export_path


# ================================================================
# READ-BACK HELPERS
# ================================================================


def read_delivery_csv(
    path: str | Path,
) -> pd.DataFrame:
    """
    Read and validate one exported delivery CSV.
    """

    export_path = Path(
        path
    )

    if not export_path.exists():
        raise FileNotFoundError(
            f"Delivery CSV not found: {export_path}"
        )

    if not export_path.is_file():
        raise ValueError(
            "Delivery CSV path must be a file."
        )

    frame = pd.read_csv(
        export_path,
        encoding="utf-8",
    )

    if tuple(
        frame.columns
    ) != DELIVERY_PAYLOAD_COLUMNS:
        raise ValueError(
            "Delivery CSV schema does not match "
            "DELIVERY_PAYLOAD_COLUMNS."
        )

    return frame


def read_delivery_jsonl(
    path: str | Path,
) -> tuple[
    dict,
    ...,
]:
    """
    Read and minimally validate one exported delivery JSONL file.
    """

    export_path = Path(
        path
    )

    if not export_path.exists():
        raise FileNotFoundError(
            f"Delivery JSONL not found: {export_path}"
        )

    if not export_path.is_file():
        raise ValueError(
            "Delivery JSONL path must be a file."
        )

    records = []

    with export_path.open(
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

            if not line.strip():
                raise ValueError(
                    "Delivery JSONL contains an empty line at "
                    f"line {line_number}."
                )

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    "Delivery JSONL contains invalid JSON at "
                    f"line {line_number}."
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    "Delivery JSONL record must be a JSON object."
                )

            if tuple(
                record.keys()
            ) != DELIVERY_PAYLOAD_COLUMNS:
                raise ValueError(
                    "Delivery JSONL schema does not match "
                    "DELIVERY_PAYLOAD_COLUMNS."
                )

            records.append(
                record
            )

    return tuple(
        records
    )