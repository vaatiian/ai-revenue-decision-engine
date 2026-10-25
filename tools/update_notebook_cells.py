"""Apply targeted source-cell repairs to the numbered cleaning notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def cell(source: str) -> dict:
    """Build a cleared code cell to force reproducible re-execution."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def replace_cells() -> None:
    """Replace the specifically reviewed cells without changing the EDA scope."""
    path = ROOT / "01_Data_Cleaning.ipynb"
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    # Clear stale results so the revised dependency chain must run top to bottom.
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            item["execution_count"], item["outputs"] = None, []

    notebook["cells"][24] = cell(
        '''# Apply the shared exact-code classification so COFFEE is not misclassified as a fee.
from metriclens.data_loader import NON_MERCHANDISE_CODE_TYPES

df["StockCode"] = df["StockCode"].astype("string").str.strip().str.upper()
df["Description"] = df["Description"].astype("string").str.upper().str.replace(r"\\s+", " ", regex=True).str.strip()
df["non_merchandise_code"] = df["StockCode"].map(NON_MERCHANDISE_CODE_TYPES)

# Accounting-code rules apply first; transaction semantics override a special code.
df["revenue_type"] = df["non_merchandise_code"].fillna("merchandise")
df.loc[df["is_zero_price"], "revenue_type"] = "zero_price"
df.loc[df["is_return"], "revenue_type"] = "return"
df.loc[df["is_cancellation"], "revenue_type"] = "cancellation"

# Consolidate relabelled descriptions under a stable stock-code product name.
canonical_description = (
    df.dropna(subset=["StockCode", "Description"]).groupby(["StockCode", "Description"], as_index=False)
    .size().sort_values(["StockCode", "size"], ascending=[True, False])
    .drop_duplicates("StockCode").set_index("StockCode")["Description"]
)
df["ProductDescription"] = df["StockCode"].map(canonical_description).fillna(df["Description"])
comparison_df = df.loc[df["period_month"].astype(str).isin([baseline_period, comparison_period])].copy()
merchandise_df = comparison_df.loc[comparison_df["revenue_type"].eq("merchandise")].copy()

revenue_type_summary = comparison_df.assign(period_key=comparison_df["period_month"].astype(str)).groupby(["revenue_type", "period_key"])["net_revenue"].sum().unstack(fill_value=0).reset_index()
for period in [baseline_period, comparison_period]:
    if period not in revenue_type_summary:
        revenue_type_summary[period] = 0.0
revenue_type_summary["revenue_change"] = revenue_type_summary[comparison_period] - revenue_type_summary[baseline_period]
display(revenue_type_summary.sort_values("revenue_change", ascending=False))'''
    )

    notebook["cells"][28] = cell(
        '''# Build a bridge only for identified customers and reconcile anonymous revenue separately.
identified = comparison_df.loc[comparison_df["Customer ID"].notna() & comparison_df["Customer ID"].ne("UNKNOWN_CUSTOMER")].copy()
identified["period_key"] = identified["period_month"].astype(str)
customer_period_revenue = identified.groupby(["Customer ID", "period_key"])["net_revenue"].sum().unstack(fill_value=0).reset_index()
for period in [baseline_period, comparison_period]:
    if period not in customer_period_revenue:
        customer_period_revenue[period] = 0.0
customer_period_revenue = customer_period_revenue.rename(columns={baseline_period: "baseline_revenue", comparison_period: "comparison_revenue"})
customer_period_revenue["revenue_change"] = customer_period_revenue["comparison_revenue"] - customer_period_revenue["baseline_revenue"]
prior_customers = set(df.loc[df["InvoiceDate"] < pd.Period(baseline_period, freq="M").start_time, "Customer ID"].dropna())

def customer_segment(row):
    """Classify monthly movement; a lost month is not proof of permanent churn."""
    if row["baseline_revenue"] == 0 and row["comparison_revenue"] > 0:
        return "Reactivated customers" if row["Customer ID"] in prior_customers else "New customers"
    if row["baseline_revenue"] > 0 and row["comparison_revenue"] == 0:
        return "Lost customers"
    if row["revenue_change"] > 0:
        return "Existing customer growth"
    if row["revenue_change"] < 0:
        return "Existing customer contraction"
    return "Stable or unusual movement"

customer_period_revenue["customer_segment"] = customer_period_revenue.apply(customer_segment, axis=1)
customer_bridge = customer_period_revenue.groupby("customer_segment", as_index=False)["revenue_change"].sum()
anonymous = comparison_df.loc[comparison_df["Customer ID"].isna() | comparison_df["Customer ID"].eq("UNKNOWN_CUSTOMER")].assign(period_key=lambda x: x["period_month"].astype(str)).groupby("period_key")["net_revenue"].sum()
anonymous_change = anonymous.get(comparison_period, 0.0) - anonymous.get(baseline_period, 0.0)
customer_bridge = pd.concat([customer_bridge, pd.DataFrame({"customer_segment": ["Unidentified customers"], "revenue_change": [anonymous_change]})], ignore_index=True)
print("Note: 'lost' means no revenue in one month; it can overstate churn for lumpy buyers.")
display(customer_bridge.sort_values("revenue_change", ascending=False))'''
    )

    notebook["cells"][36] = cell(
        '''# Group products by stock code and canonical description to prevent SKU relabelling from masquerading as product rotation.
product_period_revenue = merchandise_df.assign(period_key=merchandise_df["period_month"].astype(str)).groupby(["StockCode", "ProductDescription", "period_key"])["net_revenue"].sum().unstack(fill_value=0).reset_index()
for period in [baseline_period, comparison_period]:
    if period not in product_period_revenue:
        product_period_revenue[period] = 0.0
product_period_revenue = product_period_revenue.rename(columns={baseline_period: "baseline_revenue", comparison_period: "comparison_revenue"})
product_period_revenue["revenue_change"] = product_period_revenue["comparison_revenue"] - product_period_revenue["baseline_revenue"]
product_period_revenue["product_movement"] = np.select([product_period_revenue["baseline_revenue"].eq(0) & product_period_revenue["comparison_revenue"].gt(0), product_period_revenue["baseline_revenue"].gt(0) & product_period_revenue["comparison_revenue"].eq(0), product_period_revenue["revenue_change"].gt(0), product_period_revenue["revenue_change"].lt(0)], ["New or newly active SKU", "Inactive SKU", "Growing SKU", "Declining SKU"], default="Stable SKU")
display(product_period_revenue.sort_values("revenue_change", key=lambda values: values.abs(), ascending=False).head(20))'''
    )

    notebook["cells"][44] = cell(
        '''# Measure adoption by stable stock code rather than raw description labels.
product_penetration = merchandise_df.assign(period_key=merchandise_df["period_month"].astype(str)).groupby(["StockCode", "ProductDescription", "period_key"], as_index=False).agg(revenue=("net_revenue", "sum"), unique_customers=("Customer ID", "nunique"), orders=("Invoice", "nunique"), units=("Quantity", "sum"))
product_penetration_pivot = product_penetration.pivot_table(index=["StockCode", "ProductDescription"], columns="period_key", values=["revenue", "unique_customers", "orders", "units"], fill_value=0)
product_penetration_pivot.columns = [f"{metric}_{period}" for metric, period in product_penetration_pivot.columns]
product_penetration_pivot = product_penetration_pivot.reset_index()
product_penetration_pivot["revenue_change"] = product_penetration_pivot.get(f"revenue_{comparison_period}", 0) - product_penetration_pivot.get(f"revenue_{baseline_period}", 0)
display(product_penetration_pivot.sort_values("revenue_change", key=lambda values: values.abs(), ascending=False).head(15))'''
    )

    notebook["cells"][56] = cell(
        '''# Exclude anonymous records from customer concentration; report their revenue separately.
def calculate_concentration(data, dimension):
    """Return revenue-share and HHI metrics for a valid analytical dimension."""
    result = data.groupby(dimension)["net_revenue"].sum().loc[lambda values: values > 0].sort_values(ascending=False).reset_index(name="revenue")
    result["revenue_share_pct"] = result["revenue"] / result["revenue"].sum() * 100
    result["cumulative_share_pct"] = result["revenue_share_pct"].cumsum()
    return result, {"dimension": dimension, "top_1_share_pct": result["revenue_share_pct"].head(1).sum(), "top_5_share_pct": result["revenue_share_pct"].head(5).sum(), "top_10_share_pct": result["revenue_share_pct"].head(10).sum(), "hhi": (result["revenue_share_pct"] ** 2).sum()}

comparison_merchandise = merchandise_df.loc[merchandise_df["period_month"].astype(str).eq(comparison_period)].copy()
identified_merchandise = comparison_merchandise.loc[comparison_merchandise["Customer ID"].notna() & comparison_merchandise["Customer ID"].ne("UNKNOWN_CUSTOMER")]
customer_concentration, customer_concentration_metrics = calculate_concentration(identified_merchandise, "Customer ID")
product_concentration, product_concentration_metrics = calculate_concentration(comparison_merchandise, "StockCode")
country_concentration, country_concentration_metrics = calculate_concentration(comparison_merchandise, "Country")
concentration_summary = pd.DataFrame([customer_concentration_metrics, product_concentration_metrics, country_concentration_metrics])
unidentified_share = 1 - identified_merchandise["net_revenue"].sum() / comparison_merchandise["net_revenue"].sum()
print(f"Unidentified-customer revenue share reported separately: {unidentified_share:.1%}")
display(concentration_summary)'''
    )

    notebook["cells"][67] = cell(
        '''# Convert Period values to strings before pivoting, then profile monthly lost and declining buyers.
identified_sales = positive_sales_df.loc[positive_sales_df["Customer ID"].notna() & positive_sales_df["Customer ID"].ne("UNKNOWN_CUSTOMER")].copy()
identified_sales["period_key"] = identified_sales["period_month"].astype(str)
customer_lifecycle = identified_sales.groupby("Customer ID", as_index=False).agg(first_purchase=("InvoiceDate", "min"), last_purchase=("InvoiceDate", "max"), lifetime_revenue=("net_revenue", "sum"), lifetime_orders=("Invoice", "nunique"))
customer_lifecycle["tenure_days"] = (customer_lifecycle["last_purchase"] - customer_lifecycle["first_purchase"]).dt.days
last_product = identified_sales.sort_values("InvoiceDate").drop_duplicates("Customer ID", keep="last")[["Customer ID", "StockCode", "Description"]].rename(columns={"StockCode": "last_product_code", "Description": "last_product_description"})
customer_comparison = identified_sales.loc[identified_sales["period_key"].isin([baseline_period, comparison_period])].groupby(["Customer ID", "period_key"])["net_revenue"].sum().unstack(fill_value=0).reset_index()
for period in [baseline_period, comparison_period]:
    if period not in customer_comparison:
        customer_comparison[period] = 0.0
customer_comparison = customer_comparison.rename(columns={baseline_period: "baseline_revenue", comparison_period: "comparison_revenue"})
customer_comparison["revenue_change"] = customer_comparison["comparison_revenue"] - customer_comparison["baseline_revenue"]
customer_comparison["is_lost_customer"] = customer_comparison["baseline_revenue"].gt(0) & customer_comparison["comparison_revenue"].eq(0)
customer_comparison["is_declining_customer"] = customer_comparison["baseline_revenue"].gt(0) & customer_comparison["comparison_revenue"].gt(0) & customer_comparison["revenue_change"].lt(0)
customer_profile = customer_comparison.merge(customer_lifecycle, on="Customer ID", how="left").merge(last_product, on="Customer ID", how="left")
lost_customers = customer_profile.loc[customer_profile["is_lost_customer"]].sort_values("baseline_revenue", ascending=False)
declining_customers = customer_profile.loc[customer_profile["is_declining_customer"]].sort_values("revenue_change")
print("Note: 'lost' means no revenue in one month; it can overstate churn for lumpy buyers.")
display(lost_customers.head(20)); display(declining_customers.head(20))'''
    )

    notebook["cells"][93] = cell(
        '''# Audit SKU description variants before interpreting apparent product substitution.
description_variants = merchandise_df.groupby("StockCode")["Description"].nunique().rename("description_count").reset_index().query("description_count > 1")
description_variants = description_variants.merge(merchandise_df.groupby("StockCode")["Description"].agg(lambda values: sorted(set(values))).rename("observed_descriptions"), on="StockCode", how="left").sort_values("description_count", ascending=False)
print("Description variants are relabelling candidates, not evidence of product substitution:")
display(description_variants.head(30))'''
    )

    notebook["cells"][129] = cell(
        '''# Recompute monthly leakage from identified customers and disclose anonymous movement separately.
leakage_input = comparison_df.assign(period_key=comparison_df["period_month"].astype(str))
identified_leakage = leakage_input.loc[leakage_input["Customer ID"].notna() & leakage_input["Customer ID"].ne("UNKNOWN_CUSTOMER")]
customer_leakage_detail = identified_leakage.groupby(["Customer ID", "period_key"])["net_revenue"].sum().unstack(fill_value=0).reset_index()
for period in [baseline_period, comparison_period]:
    if period not in customer_leakage_detail:
        customer_leakage_detail[period] = 0.0
customer_leakage_detail = customer_leakage_detail.rename(columns={baseline_period: "baseline_revenue", comparison_period: "comparison_revenue"})
customer_leakage_detail["revenue_change"] = customer_leakage_detail["comparison_revenue"] - customer_leakage_detail["baseline_revenue"]
lost_value = customer_leakage_detail.loc[customer_leakage_detail["baseline_revenue"].gt(0) & customer_leakage_detail["comparison_revenue"].eq(0), "baseline_revenue"].sum()
contraction_value = -customer_leakage_detail.loc[customer_leakage_detail["baseline_revenue"].gt(0) & customer_leakage_detail["comparison_revenue"].gt(0) & customer_leakage_detail["revenue_change"].lt(0), "revenue_change"].sum()
anonymous = leakage_input.loc[leakage_input["Customer ID"].isna() | leakage_input["Customer ID"].eq("UNKNOWN_CUSTOMER")].groupby("period_key")["net_revenue"].sum()
retention_leakage_summary = pd.DataFrame({"metric": ["Monthly revenue absent from identified customers", "Monthly contraction among identified customers", "Unidentified-customer revenue change (separate)"], "revenue_value": [lost_value, contraction_value, anonymous.get(comparison_period, 0.0) - anonymous.get(baseline_period, 0.0)]})
print("Note: 'lost' means no revenue in one month; it can overstate churn for lumpy buyers.")
display(retention_leakage_summary)'''
    )

    notebook["cells"][133] = cell(
        '''# These are sensitivity cases, not empirical forecasts or intervention effects.
# Replace them only after notebook 05 has a randomized-holdout outcome estimate.
illustrative_recovery_rates = [0.05, 0.10, 0.15, 0.20]
monthly_identified_leakage = retention_leakage_summary.iloc[:2]["revenue_value"].sum()
retention_opportunity = pd.DataFrame({"illustrative_recovery_rate": illustrative_recovery_rates})
retention_opportunity["illustrative_revenue_recovered"] = retention_opportunity["illustrative_recovery_rate"] * monthly_identified_leakage
retention_opportunity["scenario_status"] = "Illustrative only — not a causal forecast"
display(retention_opportunity)'''
    )

    with path.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def update_first_order_notebook() -> None:
    """Make the retained first-order notebook an honest, non-operational baseline."""
    path = ROOT / "02_First_Order_Retention_Model.ipynb"
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            item["execution_count"], item["outputs"] = None, []

    notebook["cells"][4] = cell(
        '''# Load the shared cleaning result instead of duplicating cleaning rules in v1.
from metriclens import load_revenue_data

load_result = load_revenue_data(DATA_PATH)
df = load_result.data.copy()

# v1 needs only positive, identified merchandise orders. Anonymous revenue remains
# in the EDA but cannot form a customer-level behavioural target.
model_transactions = df.loc[
    df["customer_is_identified"]
    & df["revenue_type"].eq("merchandise")
    & df["quantity"].gt(0)
    & df["revenue"].gt(0)
].copy()
print(load_result.quality_report)'''
    )
    notebook["cells"][6] = cell(
        '''# Build a compact first-order table. The December 2009 cohort is excluded
# because source history starts then and those first observations are not acquisitions.
order_summary = model_transactions.groupby(["customer", "invoice"], as_index=False).agg(
    order_date=("date", "min"), first_order_value=("revenue", "sum"),
    first_order_units=("quantity", "sum"), first_order_product_count=("product", "nunique"),
    first_order_country=("region", "first"), left_truncated=("is_left_truncated_customer", "max"),
).sort_values(["customer", "order_date", "invoice"])
first_order_features = order_summary.groupby("customer", as_index=False).first()
analysis_end_date = order_summary["order_date"].max()
future_orders = first_order_features[["customer", "order_date"]].merge(order_summary[["customer", "order_date"]], on="customer", how="left", suffixes=("_first", "_later"))
future_orders["days_after_first_order"] = (future_orders["order_date_later"] - future_orders["order_date_first"]).dt.days
repeat_within_window = future_orders.loc[future_orders["days_after_first_order"].gt(0) & future_orders["days_after_first_order"].le(TARGET_WINDOW_DAYS)].groupby("customer").size().gt(0)
first_order_features["eligible_for_target"] = first_order_features["order_date"].le(analysis_end_date - pd.Timedelta(days=TARGET_WINDOW_DAYS))
first_order_features["churn_risk_90d"] = ~first_order_features["customer"].map(repeat_within_window).fillna(False)
model_data = first_order_features.loc[first_order_features["eligible_for_target"] & ~first_order_features["left_truncated"]].copy()
model_data["first_order_month"] = model_data["order_date"].dt.month
print("Dec 2009 left-truncated customers excluded:", int(first_order_features["left_truncated"].sum()))'''
    )
    notebook["cells"][7] = cell("""# Removed: the former stray commented line was not part of the reproducible model.""")
    notebook["cells"][10] = cell(
        '''# Use a 90-day purge before each test period and deliberately avoid sparse SKU flags.
numeric_features = ["first_order_value", "first_order_units", "first_order_product_count", "first_order_month"]
model_data = model_data.sort_values("order_date").reset_index(drop=True)
test_start = model_data.loc[int(len(model_data) * (1 - TEST_SHARE)), "order_date"]
train_data = model_data.loc[model_data["order_date"] < test_start - pd.Timedelta(days=TARGET_WINDOW_DAYS)].copy()
test_data = model_data.loc[model_data["order_date"] >= test_start].copy()
assert train_data["order_date"].max() + pd.Timedelta(days=TARGET_WINDOW_DAYS) < test_data["order_date"].min()
print("Purged train/test boundary:", train_data["order_date"].max(), test_data["order_date"].min())'''
    )
    notebook["cells"][12] = cell(
        '''# Compare no-skill, value-only, and compact first-order signals. A model must
# clear an absolute lift bar with a bootstrap interval excluding 1.0 to be useful.
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

TOP_SHARE, MINIMUM_LIFT, BOOTSTRAPS = 0.20, 1.30, 1000
models = {
    "Dummy prevalence": (DummyClassifier(strategy="prior"), ["first_order_value"]),
    "First-order value only": (Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]), ["first_order_value"]),
    "Compact first-order baseline": (Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]), numeric_features),
}
rows = []
for name, (model, features) in models.items():
    model.fit(train_data[features], train_data["churn_risk_90d"])
    scores = model.predict_proba(test_data[features])[:, 1]
    scored = pd.DataFrame({"actual": test_data["churn_risk_90d"].to_numpy(), "score": scores}).sort_values("score", ascending=False)
    top = scored.head(max(1, int(len(scored) * TOP_SHARE)))
    lift = 1.0 if scored.score.nunique() == 1 else top.actual.mean() / scored.actual.mean()
    rng, lifts = np.random.default_rng(RANDOM_STATE), []
    for _ in range(BOOTSTRAPS):
        sample = scored.iloc[rng.integers(0, len(scored), len(scored))].sort_values("score", ascending=False)
        sample_top = sample.head(max(1, int(len(sample) * TOP_SHARE)))
        lifts.append(1.0 if sample.score.nunique() == 1 else sample_top.actual.mean() / sample.actual.mean())
    rows.append({"model": name, "pr_auc": average_precision_score(scored.actual, scored.score), "roc_auc": roc_auc_score(scored.actual, scored.score) if scored.score.nunique() > 1 else 0.5, "lift_at_top_20pct": lift, "lift_ci_low": np.percentile(lifts, 2.5), "lift_ci_high": np.percentile(lifts, 97.5)})
model_performance = pd.DataFrame(rows)
model_performance["clears_success_bar"] = model_performance.lift_at_top_20pct.ge(MINIMUM_LIFT) & model_performance.lift_ci_low.gt(1.0)
display(model_performance)
Path("outputs").mkdir(exist_ok=True)
model_performance.to_csv("outputs/first_order_model_evaluation.csv", index=False)'''
    )
    notebook["cells"][13] = cell(
        '''# Do not use a default 50% classification threshold at a high base rate.
# v1 is a diagnostic baseline, not an operational scorer. Its honest conclusion is:
# first-order features alone have little predictive power because one purchase
# contains no repeat cadence, retention history, or mature value signal.
print("v1 result: interpret lift, confidence intervals, and calibration before considering any action.")'''
    )
    notebook["cells"][19] = cell(
        '''# Fixed probability bands are not valid until probabilities are calibrated.
# This baseline intentionally produces no action list; notebook 03 is the candidate model.
print("No action list is generated from v1. It exists to measure the limits of first-order information.")'''
    )
    notebook["cells"][20] = cell(
        '''# Rank deciles preserve ordering even when raw probabilities are poorly calibrated.
best_name = model_performance.loc[model_performance.model.ne("Dummy prevalence")].sort_values("pr_auc", ascending=False).iloc[0]["model"]
print("Use rolling-fold calibration and deciles in notebook 03; v1 is retained solely as the documented baseline.")'''
    )
    notebook["cells"][21] = {"cell_type": "markdown", "metadata": {}, "source": ["## Plain-language result\n", "\n", "First-order information alone has limited predictive power. One purchase does not contain enough cadence, retention, or customer-value history to justify intervention ranking. Notebook 03 tests whether order-history snapshots add real out-of-time signal.\n"]}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    replace_cells()
    update_first_order_notebook()
