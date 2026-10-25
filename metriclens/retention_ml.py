"""Leakage-safe snapshot modelling utilities for customer retention."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold


SNAPSHOT_NUMERIC_FEATURES = [
    "order_revenue", "order_units", "basket_product_count", "frequency_to_date",
    "revenue_to_date", "average_order_value_to_date", "days_since_prior_order",
    "average_gap_to_date", "customer_age_days", "days_to_christmas",
    "smoothed_product_group_risk", "is_left_truncated_customer",
    "return_cancel_rate_to_date", "category_diversity_to_date",
    "seasonal_revenue_share_to_date",
]
SNAPSHOT_CATEGORICAL_FEATURES = ["country"]
SNAPSHOT_FEATURES = SNAPSHOT_NUMERIC_FEATURES + SNAPSHOT_CATEGORICAL_FEATURES


def build_customer_snapshots(data: pd.DataFrame, target_days: int = 90) -> pd.DataFrame:
    """Create one labelled row per customer purchase day using only as-of features."""
    lines = data.loc[
        data["customer_is_identified"]
        & data["revenue_type"].eq("merchandise")
        & data["quantity"].gt(0)
        & data["revenue"].gt(0)
    ].copy()
    lines["purchase_day"] = lines["date"].dt.normalize()
    # Description-keyword categories carry commercial meaning; stock-code prefixes
    # are arbitrary and can split similar products into unrelated groups.
    lines["product_group"] = _product_category(lines["product_description"].fillna(lines["description"]))
    lines["is_seasonal_product"] = lines["product_description"].fillna(lines["description"]).str.contains(
        r"CHRISTMAS|XMAS|EASTER|VALENTINE|MOTHER.?S DAY|FATHER.?S DAY", regex=True, na=False
    )
    lines["seasonal_revenue"] = np.where(lines["is_seasonal_product"], lines["revenue"], 0.0)

    # Same-day invoices form one purchase occasion, so they cannot create fake repeat orders.
    snapshots = lines.groupby(["customer", "purchase_day"], as_index=False).agg(
        order_revenue=("revenue", "sum"), order_units=("quantity", "sum"),
        basket_product_count=("product", "nunique"), country=("region", "first"),
        is_left_truncated_customer=("is_left_truncated_customer", "max"),
        seasonal_order_revenue=("seasonal_revenue", "sum"),
    ).sort_values(["customer", "purchase_day"])
    primary_group = lines.groupby(["customer", "purchase_day", "product_group"], as_index=False)["revenue"].sum()
    primary_group = primary_group.sort_values("revenue", ascending=False).drop_duplicates(["customer", "purchase_day"])
    snapshots = snapshots.merge(primary_group[["customer", "purchase_day", "product_group"]], on=["customer", "purchase_day"], how="left")

    # Cumulative RFM and basket features are observable at the end of the current purchase day.
    snapshots["frequency_to_date"] = snapshots.groupby("customer").cumcount() + 1
    snapshots["revenue_to_date"] = snapshots.groupby("customer")["order_revenue"].cumsum()
    snapshots["average_order_value_to_date"] = snapshots["revenue_to_date"] / snapshots["frequency_to_date"]
    snapshots["seasonal_revenue_to_date"] = snapshots.groupby("customer")["seasonal_order_revenue"].cumsum()
    snapshots["seasonal_revenue_share_to_date"] = snapshots["seasonal_revenue_to_date"] / snapshots["revenue_to_date"]
    snapshots["days_since_prior_order"] = snapshots.groupby("customer")["purchase_day"].diff().dt.days
    snapshots["average_gap_to_date"] = snapshots.groupby("customer")["days_since_prior_order"].transform(lambda x: x.expanding().mean())
    snapshots["customer_age_days"] = (snapshots["purchase_day"] - snapshots.groupby("customer")["purchase_day"].transform("min")).dt.days
    snapshots["category_diversity_to_date"] = _expanding_category_diversity(snapshots)
    christmas = pd.to_datetime({"year": snapshots["purchase_day"].dt.year, "month": 12, "day": 25})
    next_christmas = christmas + pd.DateOffset(years=1)
    snapshots["days_to_christmas"] = np.where(snapshots["purchase_day"] <= christmas, (christmas - snapshots["purchase_day"]).dt.days, (next_christmas - snapshots["purchase_day"]).dt.days)

    # The label is based on the next purchase day only, leaving recent rows censored.
    end_date = snapshots["purchase_day"].max()
    snapshots["next_purchase_day"] = snapshots.groupby("customer")["purchase_day"].shift(-1)
    snapshots["days_to_next_purchase"] = (snapshots["next_purchase_day"] - snapshots["purchase_day"]).dt.days
    snapshots["eligible_for_90d_target"] = snapshots["purchase_day"] <= end_date - pd.Timedelta(days=target_days)
    snapshots["orders_again_90d"] = snapshots["days_to_next_purchase"].gt(0) & snapshots["days_to_next_purchase"].le(target_days)
    snapshots["lapse_risk_90d"] = (~snapshots["orders_again_90d"]).astype(int)
    # Return/cancellation events are joined as of each purchase day so future
    # service issues cannot influence an earlier snapshot.
    events = data.loc[
        data["customer_is_identified"] & (data["is_return"] | data["is_cancellation"]),
        ["customer", "date"],
    ].copy()
    events["event_day"] = events["date"].dt.normalize()
    events = events.groupby(["customer", "event_day"], as_index=False).size().rename(columns={"size": "events"})
    events["events_to_date"] = events.groupby("customer")["events"].cumsum()
    snapshots = pd.merge_asof(
        snapshots.sort_values(["purchase_day", "customer"]),
        events[["customer", "event_day", "events_to_date"]].sort_values(["event_day", "customer"]),
        left_on="purchase_day", right_on="event_day", by="customer", direction="backward",
    ).drop(columns="event_day")
    snapshots["events_to_date"] = snapshots["events_to_date"].fillna(0)
    snapshots["return_cancel_rate_to_date"] = snapshots["events_to_date"] / snapshots["frequency_to_date"]
    return snapshots


def _product_category(description: pd.Series) -> pd.Series:
    """Map product descriptions into compact, interpretable commercial categories."""
    text = description.astype("string").str.upper().fillna("")
    conditions = [
        text.str.contains(r"BAG|BOX|WRAP|RIBBON|PACKAGING", regex=True, na=False),
        text.str.contains(r"LIGHT|LAMP|LANTERN|CANDLE|TEA LIGHT", regex=True, na=False),
        text.str.contains(r"MUG|CUP|BOWL|PLATE|GLASS|CAKESTAND", regex=True, na=False),
        text.str.contains(r"HEART|FLOWER|DECOR|ORNAMENT|HANGING", regex=True, na=False),
        text.str.contains(r"TOY|DOLL|GAME|PUZZLE|CHILD|KIDS", regex=True, na=False),
        text.str.contains(r"CHRISTMAS|XMAS|EASTER|VALENTINE", regex=True, na=False),
    ]
    # Convert pandas' nullable boolean masks to NumPy booleans for np.select.
    conditions = [condition.to_numpy(dtype=bool, na_value=False) for condition in conditions]
    categories = np.select(
        conditions,
        ["Packaging", "Lighting", "Tableware", "Home decor", "Toys and gifts", "Seasonal"],
        default="Other merchandise",
    )
    return pd.Series(categories, index=description.index, dtype="string")


def _expanding_category_diversity(snapshots: pd.DataFrame) -> pd.Series:
    """Count unique product categories seen by each customer at each purchase day."""
    return snapshots.groupby("customer")["product_group"].transform(
        lambda values: pd.Series(
            [values.iloc[: position + 1].nunique() for position in range(len(values))],
            index=values.index,
        )
    )


def add_train_only_product_risk(train: pd.DataFrame, score: pd.DataFrame, alpha: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use grouped out-of-fold encoding for training and full-train mapping for scoring."""
    base_rate = train["lapse_risk_90d"].mean()
    train, score = train.copy(), score.copy()
    train["smoothed_product_group_risk"] = base_rate

    # Customer-grouped folds prevent one customer's repeated snapshots leaking into
    # another snapshot's encoded training value.
    groups = train["customer"]
    n_splits = min(5, groups.nunique())
    if n_splits >= 2:
        for fit_index, valid_index in GroupKFold(n_splits=n_splits).split(train, groups=groups):
            fit = train.iloc[fit_index]
            mapping_data = fit.groupby("product_group")["lapse_risk_90d"].agg(["sum", "count"])
            mapping = (mapping_data["sum"] + alpha * fit["lapse_risk_90d"].mean()) / (mapping_data["count"] + alpha)
            train.iloc[valid_index, train.columns.get_loc("smoothed_product_group_risk")] = train.iloc[valid_index]["product_group"].map(mapping).fillna(fit["lapse_risk_90d"].mean())

    # The full training mapping is valid only for separately scored future/test rows.
    group = train.groupby("product_group")["lapse_risk_90d"].agg(["sum", "count"])
    mapping = (group["sum"] + alpha * base_rate) / (group["count"] + alpha)
    score["smoothed_product_group_risk"] = score["product_group"].map(mapping).fillna(base_rate)
    return train, score


def make_snapshot_pipeline() -> Pipeline:
    """Build a regularized, interpretable snapshot model."""
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), SNAPSHOT_NUMERIC_FEATURES),
        ("country", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), SNAPSHOT_CATEGORICAL_FEATURES),
    ])
    # Unweighted probabilities preserve prevalence for value-at-risk calculations.
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, random_state=42))])


def rolling_snapshot_evaluation(snapshots: pd.DataFrame, target_days: int = 90, folds: int = 3, test_days: int = 90, top_share: float = 0.2, bootstrap_samples: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate snapshot ranking with rolling-origin folds and purged target windows."""
    labelled = snapshots.loc[snapshots["eligible_for_90d_target"]].copy()
    dates = np.sort(labelled["purchase_day"].drop_duplicates().to_numpy())
    positions = np.linspace(int(len(dates) * .55), int(len(dates) * .85), folds, dtype=int)
    outcomes, predictions = [], []
    for fold, position in enumerate(positions, 1):
        test_start = pd.Timestamp(dates[position])
        train = labelled.loc[labelled["purchase_day"] < test_start - pd.Timedelta(days=target_days)]
        test = labelled.loc[(labelled["purchase_day"] >= test_start) & (labelled["purchase_day"] < test_start + pd.Timedelta(days=test_days))]
        if train["lapse_risk_90d"].nunique() < 2 or test["lapse_risk_90d"].nunique() < 2:
            continue
        train, test = add_train_only_product_risk(train, test)
        model = make_snapshot_pipeline().fit(train[SNAPSHOT_FEATURES], train["lapse_risk_90d"])
        test = test.copy()
        test["predicted_lapse_probability"] = model.predict_proba(test[SNAPSHOT_FEATURES])[:, 1]
        metrics = _rank_metrics(test["lapse_risk_90d"], test["predicted_lapse_probability"], top_share)
        low, mid, high = _bootstrap_lift(test, top_share, bootstrap_samples, fold)
        outcomes.append({"fold": fold, "test_start": test_start, **metrics, "lift_ci_low": low, "lift_ci_median": mid, "lift_ci_high": high, "brier_score": brier_score_loss(test["lapse_risk_90d"], test["predicted_lapse_probability"])})
        test["fold"] = fold
        predictions.append(test)
    if not outcomes:
        raise ValueError("No valid snapshot evaluation folds were available.")
    return pd.DataFrame(outcomes), pd.concat(predictions, ignore_index=True)


def _rank_metrics(y: pd.Series, score: pd.Series, top_share: float) -> dict[str, float]:
    """Return ranking metrics, treating constant scores as a no-skill baseline."""
    frame = pd.DataFrame({"y": y.to_numpy(), "score": score.to_numpy()})
    rate = frame["y"].mean()
    if frame["score"].nunique() == 1:
        return {"pr_auc": rate, "roc_auc": .5, "lift": 1.0, "recall_at_top": top_share}
    frame = frame.sort_values("score", ascending=False)
    top = frame.head(max(1, int(np.ceil(len(frame) * top_share))))
    return {"pr_auc": average_precision_score(frame["y"], frame["score"]), "roc_auc": roc_auc_score(frame["y"], frame["score"]), "lift": top["y"].mean() / rate, "recall_at_top": top["y"].sum() / frame["y"].sum(), "theoretical_lift_ceiling": 1 / rate}


def _bootstrap_lift(frame: pd.DataFrame, top_share: float, samples: int, seed: int) -> tuple[float, float, float]:
    """Quantify lift uncertainty by resampling whole customers, not correlated rows."""
    rng, values = np.random.default_rng(seed), []
    for _ in range(samples):
        customer_ids = frame["customer"].unique()
        sampled_ids = rng.choice(customer_ids, size=len(customer_ids), replace=True)
        sample = pd.concat([frame.loc[frame["customer"].eq(customer)] for customer in sampled_ids], ignore_index=True)
        if sample["lapse_risk_90d"].nunique() == 2:
            values.append(_rank_metrics(sample["lapse_risk_90d"], sample["predicted_lapse_probability"], top_share)["lift"])
    return tuple(np.percentile(values, [2.5, 50, 97.5]))


def build_action_list(snapshots: pd.DataFrame, expected_value: pd.DataFrame | None, capacity: int, holdout_share: float = .1, seed: int = 42, max_days_since_purchase: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank only very recent purchasers to avoid scoring stale snapshots as new ones."""
    labelled = snapshots.loc[snapshots["eligible_for_90d_target"]]
    train, all_rows = add_train_only_product_risk(labelled, snapshots)
    model = make_snapshot_pipeline().fit(train[SNAPSHOT_FEATURES], train["lapse_risk_90d"])
    current = all_rows.sort_values(["customer", "purchase_day"]).groupby("customer", as_index=False).tail(1).copy()
    as_of_date = snapshots["purchase_day"].max()
    current["days_since_last_order_as_of_scoring"] = (as_of_date - current["purchase_day"]).dt.days
    current = current.loc[current["days_since_last_order_as_of_scoring"].le(max_days_since_purchase)].copy()
    current["predicted_lapse_probability"] = model.predict_proba(current[SNAPSHOT_FEATURES])[:, 1]
    if expected_value is not None:
        current = current.merge(expected_value, on="customer", how="left")
    if "expected_revenue_next_90d" not in current:
        current["expected_revenue_next_90d"] = np.nan
    current["expected_revenue_next_90d"] = current["expected_revenue_next_90d"].fillna(current["average_order_value_to_date"])
    current["expected_revenue_at_risk"] = current["predicted_lapse_probability"] * current["expected_revenue_next_90d"]
    current = current.sort_values("expected_revenue_at_risk", ascending=False).head(capacity).copy()
    current["value_at_risk_decile"] = pd.qcut(current["expected_revenue_at_risk"].rank(method="first", ascending=False), q=min(10, len(current)), labels=False) + 1
    rng = np.random.default_rng(seed)
    current["holdout_draw"] = rng.random(len(current))
    current["experiment_group"] = np.where(current.groupby("value_at_risk_decile")["holdout_draw"].transform(lambda x: x.rank(pct=True)) <= holdout_share, "Randomized holdout", "Retention treatment")
    return current, current.groupby("experiment_group", as_index=False).agg(customers=("customer", "nunique"), expected_revenue_at_risk=("expected_revenue_at_risk", "sum"))
