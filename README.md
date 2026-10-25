# MetricLens: AI Revenue Variance Investigation Agent

## MetricLens: Business Problem Statement
### 1. Summary

Revenue in November 2011 was only about 2.8% (+39.4k) above November 2010. The headline hides how much changed underneath. The business lost far more revenue from existing customers than the growth figure suggests, and acquisition and reactivation covered the gap. Leadership can see that revenue moved but not why, which customers and products drove it, or what to do about it.

MetricLens has two jobs:

- Explain why revenue changed between two periods, with a ranked, evidence-backed list of drivers.
- Turn the biggest controllable driver, customer retention, into a prioritised, costed action list.
  
### 2. Business context

The data is the UCI Online Retail II dataset: two years (Dec 2009 to Dec 2011) of transactions from a UK online gift and homeware retailer. Many of its customers are trade buyers who order in irregular, lumpy patterns. Three features of the setting matter:

Non-contractual relationships. Customers never formally "cancel". Churn can only be inferred from silence.
Strong seasonality. Gifting demand peaks in Q4, so a single month-on-month comparison is easily distorted.
Incomplete customer identity. Roughly 22.6% of November 2011 revenue has no customer ID, so it can't be tied to individuals.

MetricLens is built on a canonical schema so the same agent can later run on SaaS or other revenue data.

### 3. The problem

#### Problem 1: No explanation of change. 
Finance sees a total, not drivers. Decomposing it requires consistent definitions of revenue, returns, cancellations, non-merchandise lines and customer status, which today vary by analyst.

#### Problem 2: Hidden leakage. 
The customer bridge shows:

Driver	Effect
New customers	+283k
Reactivated customers	+199k
Expansion	+218k
Contraction	−253k
Lost customers	−412k

About 665k of revenue leaked through lost and contracting customers. Acquisition masked it.

#### Problem 3: No way to act. 
Even with the diagnosis, no one knows which customers are slipping, which are worth saving, or whether an intervention would work. About 44.5% of new customers place a second order within 90 days, so more than half never do, and nothing currently targets them.

### 4. Questions MetricLens must answer
1. What changed in revenue between period A and period B, and which drivers are largest?
2. Which customers, products, regions and channels explain it?
3. How much of the change is real versus a data or definition artefact (duplicates, returns, unidentified customers, product relabelling)?
4. Which customers are at risk of lapsing, and how much revenue is at risk?
5. Where should limited retention effort go, and is it worth the cost?

### 5. Decision and stakeholders
- Decision-makers: the commercial or revenue lead and the finance lead (explanation); the account or customer-success team (action).
- Decision: where to spend a limited retention budget and team capacity each period.
- Action: outreach to the highest expected-revenue-at-risk customers, with a randomized holdout to measure whether it works.

### 6. Scope

#### In scope:

- Shared data cleaning with an auditable quality report.
- Variance bridge and driver ranking.
- Retention risk modelling: first-order model as a baseline, order-snapshot model, and BG/NBD plus Gamma-Gamma customer value.
- Value-at-risk ranking and action list.
- Executive brief with evidence trail.

#### Out of scope:

- Causal estimates of intervention effect. That needs a live experiment.
- Pricing, inventory and marketing-mix optimisation.
- Modelling unidentified customers.
  
### 7. Success criteria

#### Business:

- Variance bridge reconciles to total revenue change, including an explicit unidentified-customer line.
- Every number in the brief traces back to a query or cleaning rule.
- The action list ranks customers by expected revenue at risk and respects a stated capacity.

#### Model:

Beats dummy and single-feature baselines on rolling-origin, purge-gapped evaluation.
Reports ROC-AUC, PR-AUC, recall at the top 20% and lift, with customer-level bootstrap confidence intervals and calibration checks.
Deployed only if performance clears a bar set in advance.

### 8. Assumptions and limitations
"Lost" is defined as revenue in the baseline month and none in the comparison month. For lumpy buyers this overstates churn, so the leakage figure is an upper bound.
The first month of data (Dec 2009) is left-truncated, so those customers' history is incomplete.
Only two Christmas cycles exist, so seasonality can't be separated cleanly.
The recovery scenarios (for example recovering 5–20% of leakage) are illustrative until grounded in measured rates.
There's no intervention data, so the benefit of any action is untested.

### 9. Current status and risks
The first-order retention model showed little signal (ROC-AUC about 0.56, lift about 1.08 in the top 20%). This is a diagnostic result, and it motivated the move to snapshot and BG/NBD models.
Key risks: data definitions drifting between notebooks, target leakage, overstated churn, and recommending action before the model clears its bar.

## Load the supplied dataset

```python
from metriclens import load_revenue_data

result = load_revenue_data("online_retail_II.xlsx")
revenue_events = result.data

print(revenue_events.head())
print(result.warnings)
```

The loader reads every Excel sheet, recognizes the supplied `InvoiceDate`, `Quantity`, and `Price` columns, and derives `revenue = quantity × price`. It also preserves cancellation rows so a net-revenue investigation does not silently omit credits.

Install dependencies first:

```bash
python -m pip install -r requirements.txt
```
