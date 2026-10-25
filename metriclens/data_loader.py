"""Shared loading and cleaning rules for MetricLens analyses.

Every analysis should use this module so revenue, merchandise, customer, and
data-quality definitions cannot silently drift between notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# Canonical fields accept common export names after punctuation is removed.
FIELD_ALIASES = {
    "date": ("date", "invoicedate", "transactiondate", "orderdate", "revenuedate"),
    "revenue": ("revenue", "amount", "sales", "netrevenue", "arr", "mrr"),
    "quantity": ("quantity", "qty", "units"),
    "unit_price": ("price", "unitprice", "rate"),
    "customer": ("customer", "customerid", "account", "accountid"),
    "product": ("product", "productid", "stockcode", "sku"),
    "description": ("description", "productdescription", "itemdescription"),
    "region": ("region", "country", "market", "geography"),
    "channel": ("channel", "source", "saleschannel"),
    "invoice": ("invoice", "invoiceid", "orderid", "transactionid"),
}


# These exact codes are accounting or service lines in the supplied Online Retail data.
# Exact-code logic avoids classifying a normal product such as "COFFEE" as a fee.
NON_MERCHANDISE_CODE_TYPES = {
    "POST": "postage",
    "DOT": "postage",
    "AMAZONFEE": "fee",
    "AMAZON FEE": "fee",
    "BANKCHARGES": "fee",
    "BANK CHARGES": "fee",
    "C2": "fee",
    "M": "adjustment",
    "D": "discount",
    "PADS": "adjustment",
    "ADJUST": "adjustment",
    "CRUK": "adjustment",
    "S": "adjustment",
    "B": "adjustment",
    # Test and gift-accounting codes are not saleable merchandise SKUs.
    "TEST001": "adjustment",
    "TEST002": "adjustment",
    "GIFT_0001_20": "adjustment",
    "GIFT_0001_30": "adjustment",
    "GIFT_0001_40": "adjustment",
}


@dataclass
class DataLoadResult:
    """Clean canonical line items and auditable data-quality diagnostics."""

    data: pd.DataFrame
    warnings: list[str]
    source_rows: int
    retained_rows: int
    quality_report: pd.DataFrame | None = None


def load_revenue_data(path: str | Path) -> DataLoadResult:
    """Read a revenue export and apply the shared MetricLens cleaning rules.

    The returned data keeps line-item detail for EDA and modelling. Missing
    customer IDs remain missing rather than becoming one synthetic customer.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Revenue dataset not found: {source}")
    return clean_revenue_line_items(_read_file(source))


def clean_revenue_line_items(raw: pd.DataFrame) -> DataLoadResult:
    """Standardise a raw revenue DataFrame into canonical MetricLens line items."""
    source_rows = len(raw)
    if raw.empty:
        raise ValueError("The revenue dataset contains no rows.")

    prepared = raw.copy()
    prepared.columns = [str(column).strip() for column in prepared.columns]
    fields = _find_fields(prepared.columns)
    if "date" not in fields:
        raise ValueError("Missing a date column. Accepted names include Date and InvoiceDate.")
    if "revenue" not in fields and not {"quantity", "unit_price"}.issubset(fields):
        raise ValueError("Provide Revenue/Amount, or both Quantity and Price/UnitPrice.")

    data = pd.DataFrame(index=prepared.index)
    data["date"] = pd.to_datetime(prepared[fields["date"]], errors="coerce")
    data["quantity"] = _numeric_or_missing(prepared, fields.get("quantity"))
    data["unit_price"] = _numeric_or_missing(prepared, fields.get("unit_price"))
    data["revenue"] = (
        pd.to_numeric(prepared[fields["revenue"]], errors="coerce")
        if "revenue" in fields
        else data["quantity"] * data["unit_price"]
    )

    # Dedicated canonical dimensions allow every notebook to group the same way.
    for field in ("customer", "product", "description", "region", "channel", "invoice"):
        data[field] = _text_or_missing(prepared, fields.get(field))
    data["source_sheet"] = (
        prepared["source_sheet"].astype("string")
        if "source_sheet" in prepared.columns
        else pd.Series(pd.NA, index=prepared.index, dtype="string")
    )

    # Normalize labels while preserving missing customer identity as missing.
    data["customer"] = data["customer"].str.replace(r"\.0$", "", regex=True)
    data["product"] = data["product"].str.upper()
    data["description"] = data["description"].str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    data["region"] = data["region"].str.title()
    data["invoice"] = data["invoice"].str.upper()

    # Date/revenue failures and negative unit prices cannot support a valid model.
    pre_filter_data = data.copy()
    invalid_required = data["date"].isna() | data["revenue"].isna()
    invalid_price = data["unit_price"].lt(0).fillna(False)
    retained = data.loc[~invalid_required & ~invalid_price].copy()

    # Only exact duplicate lines are removed; multiple items on one invoice remain.
    duplicate_columns = ["invoice", "product", "description", "quantity", "date", "unit_price", "customer", "region"]
    retained["is_exact_duplicate"] = retained.duplicated(duplicate_columns, keep="first")
    duplicates = retained["is_exact_duplicate"]
    data = retained.loc[~duplicates].copy()

    # Transaction semantics take priority over stock-code classifications.
    data["is_cancellation"] = data["invoice"].str.startswith("C", na=False)
    data["is_return"] = data["quantity"].lt(0).fillna(False)
    data["is_zero_price"] = data["unit_price"].eq(0).fillna(False)
    data["non_merchandise_code"] = data["product"].map(NON_MERCHANDISE_CODE_TYPES)
    data["revenue_type"] = _classify_revenue_type(data)

    # A stable code-level description prevents renamed SKU labels from appearing as
    # different products in product-driver and substitution analyses.
    description_counts = (
        data.dropna(subset=["product", "description"])
        .groupby(["product", "description"], as_index=False)
        .agg(occurrences=("revenue", "size"), latest_date=("date", "max"))
        .sort_values(["product", "occurrences", "latest_date"], ascending=[True, False, False])
        .drop_duplicates("product")
    )
    canonical_descriptions = description_counts.set_index("product")["description"]
    data["product_description"] = data["product"].map(canonical_descriptions).fillna(data["description"])

    # The source's first calendar month is left-truncated customer history.
    first_date = data["date"].min()
    customer_first_observed = data.groupby("customer", dropna=True)["date"].transform("min")
    data["customer_first_observed_date"] = customer_first_observed
    data["is_left_truncated_customer"] = (
        customer_first_observed.dt.to_period("M") == first_date.to_period("M")
    ).fillna(False)
    data["customer_is_identified"] = data["customer"].notna()
    data["period_month"] = data["date"].dt.to_period("M").astype("string")

    quality_report = pd.DataFrame(
        {
            "rule": [
                "Source rows", "Invalid or missing date/revenue", "Negative unit price",
                "Exact duplicate line items", "Rows retained", "Stock-code non-merchandise rows retained",
                "Rows with missing customer ID retained", "Left-truncated customer rows retained",
            ],
            "row_count": [
                source_rows, int(invalid_required.sum()), int(invalid_price.sum()), int(duplicates.sum()),
                len(data), int(data["non_merchandise_code"].notna().sum()), int(data["customer"].isna().sum()),
                int(data["is_left_truncated_customer"].sum()),
            ],
            "revenue_impact": [
                float(pre_filter_data["revenue"].sum()), float(pre_filter_data.loc[invalid_required, "revenue"].sum()),
                float(pre_filter_data.loc[invalid_price, "revenue"].sum()), float(retained.loc[duplicates, "revenue"].sum()),
                float(data["revenue"].sum()), float(data.loc[data["non_merchandise_code"].notna(), "revenue"].sum()),
                float(data.loc[data["customer"].isna(), "revenue"].sum()),
                float(data.loc[data["is_left_truncated_customer"], "revenue"].sum()),
            ],
        }
    )

    warnings = [
        f"Excluded {int(invalid_required.sum()):,} rows with invalid dates or revenue.",
        f"Excluded {int(invalid_price.sum()):,} rows with negative unit prices.",
        f"Excluded {int(duplicates.sum()):,} exact duplicate line items.",
        f"Flagged {int(data['is_left_truncated_customer'].sum()):,} left-truncated customer rows.",
    ]
    return DataLoadResult(data.sort_values("date", kind="stable").reset_index(drop=True), warnings, source_rows, len(data), quality_report)


def _read_file(path: Path) -> pd.DataFrame:
    """Read supported files while preserving Excel sheet identity for audit checks."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        return pd.concat([sheet.assign(source_sheet=name) for name, sheet in sheets.items()], ignore_index=True)
    raise ValueError("Unsupported file format. Use CSV, Parquet, XLSX, or XLS.")


def _classify_revenue_type(data: pd.DataFrame) -> pd.Series:
    """Classify rows with explicit accounting-code rules and transaction semantics."""
    # Apply code types first, then let transaction semantics override them. A
    # cancelled POST row is a cancellation, not a postage sale.
    return pd.Series(pd.NA, index=data.index, dtype="string").mask(
        data["non_merchandise_code"].notna(), data["non_merchandise_code"]
    ).mask(data["is_zero_price"], "zero_price").mask(
        data["is_return"], "return"
    ).mask(data["is_cancellation"], "cancellation").fillna("merchandise")


def _numeric_or_missing(data: pd.DataFrame, column: str | None) -> pd.Series:
    """Return a numeric source column or an all-missing series when unavailable."""
    if column is None:
        return pd.Series(pd.NA, index=data.index, dtype="Float64")
    return pd.to_numeric(data[column], errors="coerce")


def _text_or_missing(data: pd.DataFrame, column: str | None) -> pd.Series:
    """Return stripped text or an all-missing series when a source field is absent."""
    if column is None:
        return pd.Series(pd.NA, index=data.index, dtype="string")
    return data[column].astype("string").str.strip().replace("", pd.NA)


def _find_fields(columns: pd.Index) -> dict[str, Any]:
    normalized = {_normalize(name): name for name in columns}
    return {
        field: next((normalized[alias] for alias in aliases if alias in normalized), None)
        for field, aliases in FIELD_ALIASES.items()
        if any(alias in normalized for alias in aliases)
    }


def _normalize(column: str) -> str:
    return "".join(character for character in str(column).lower() if character.isalnum())
