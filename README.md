# AI Revenue Decision Engine

A production-oriented retail customer decision system that combines multiple machine-learning models into one governed action framework.

The project was developed through a sequence of research notebooks covering customer behaviour, revenue risk, future value, recommendations, demand forecasting, and final decision orchestration.

The current system converts model outputs into a single customer-level decision such as:

- Protect
- Retain
- Reactivate
- Grow
- Monitor
- Unscored — Long Dormant

---

## Project Status

**Research phase:** Complete  
**Decision engine:** Complete  
**Current deployment decision:** CONDITIONAL GO  
**Current release:** `v0.1.0`

The system is approved for:

- decision support
- CRM prioritisation
- controlled retention campaigns
- controlled reactivation campaigns
- leakage-risk prioritisation
- product recommendation pilots
- commercial experimentation
- customer action queue generation

The system is **not yet approved** for fully autonomous customer intervention.

---

## Research Notebooks

The research phase is preserved in the `notebooks/` directory.

### 01 — Data Cleaning

Builds the canonical retail transaction dataset and applies the project cleaning methodology.

Key principles include:

- retaining known accounting adjustments where appropriate
- separating cancellations and returns
- preserving missing customer IDs as missing
- retaining zero-price transactions with flags
- separating merchandise and non-merchandise revenue
- removing only confirmed cross-sheet duplicate instances
- excluding partial December 2011 from complete-period analysis

---

### 02 — Repeat Purchase Model

Predicts whether a customer is likely to make another purchase within 90 days after their first observed order.

**Deployment decision:** CONDITIONAL GO

Primary use:

- early-lifecycle customer segmentation
- retention-risk prioritisation
- controlled experimentation

This model is not treated as a permanent customer score. Its prediction is only valid within its defined early-lifecycle horizon.

---

### 03 — Reactivation Propensity

Estimates the probability that an inactive customer will reactivate within 90 days.

**Deployment decision:** GO for monitored operational use and controlled campaigns

Important limitation:

The score estimates reactivation propensity, not causal uplift from intervention.

---

### 04 — Revenue Leakage Risk

Identifies customers at risk of material future revenue decline.

The model combines:

- leakage probability
- recent customer revenue
- an economic exposure prioritisation measure

**Deployment decision:** GO for monitored risk and exposure prioritisation

The exposure metric is a prioritisation proxy and must not be interpreted as calibrated expected accounting loss.

---

### 05 — Future Customer Value

Estimates expected positive merchandise revenue over the following 90 days.

The final architecture combines:

- probability of future activity
- recent historical customer revenue
- a conservative value multiplier

**Deployment decision:** CONDITIONAL GO

This signal is intended primarily for relative customer-value prioritisation.

It is not:

- lifetime customer value
- guaranteed revenue
- accounting revenue forecast
- contribution margin

---

### 06 — Next Best Product

Ranks products likely to be purchased in the customer's next qualifying order.

Two recommendation channels are supported:

#### Repeat / Replenishment

**Deployment decision:** GO

Used for products the customer has purchased previously.

#### Discovery / Cross-Sell

**Deployment decision:** CONDITIONAL GO

Used for products the customer has not previously purchased.

Discovery recommendations should be deployed through controlled experiments.

---

### 07 — Demand Forecasting

Forecasts demand at two levels.

#### Business-Level Weekly Revenue Forecast

**Deployment decision:** GO

Approved for:

- weekly planning
- budgeting
- capacity planning
- working-capital guidance

#### SKU-Level Forecasting

**Deployment decision:** CONDITIONAL GO

More suitable for:

- high-frequency products
- replenishment guidance
- directional inventory planning

Sparse and intermittent products are not approved for precise standalone point forecasting.

---

### 08 — Customer Decision Engine

Integrates valid model signals into one customer-level action framework.

The historical decision snapshot used during development is:

**31 August 2011**

The final historical customer population contains:

**5,224 identified customers**

The decision engine assigns one primary action per customer based on:

- current customer lifecycle state
- valid model signals
- temporal eligibility
- economic priority
- recommendation availability
- customer value

---

## Final Decision Actions

### Protect

Used primarily for active customers with material revenue leakage exposure.

Goal:

Preserve economically important customer relationships and reduce potential revenue decline.

---

### Retain

Used for customers whose purchase recency has weakened.

Goal:

Prevent further customer disengagement and preserve relationship continuity.

---

### Reactivate

Used for dormant customers with a valid reactivation propensity score.

Goal:

Prioritise customers for controlled win-back campaigns.

---

### Grow

Used for commercially healthy active customers with valid product recommendations and future-value signals.

Goal:

Support cross-sell, discovery, and expansion opportunities.

---

### Monitor

Used when no model-supported intervention currently takes precedence.

---

### Unscored — Long Dormant

Used for customers outside the validated scope of the current customer-action models.

No unsupported model inference is generated for this population.

---

## Historical Decision Snapshot

The final historical decision population is:

| Action | Customers | Share |
|---|---:|---:|
| Protect | 458 | 8.8% |
| Retain | 1,009 | 19.3% |
| Reactivate | 509 | 9.7% |
| Grow | 284 | 5.4% |
| Monitor | 168 | 3.2% |
| Dormant — No Valid Reactivation Score | 340 | 6.5% |
| Unscored — Long Dormant | 2,456 | 47.0% |

The higher-priority Priority 1–2 intervention queue contains:

**680 customers**

This represents approximately:

**13.0% of the identified customer population**

---

## Contact Intensity Framework

Actionable customers are ranked within their own action type.

Priority levels are translated into operational contact intensity:

| Priority Tier | Contact Intensity |
|---|---|
| Priority 1 | High Touch |
| Priority 2 | Campaign |
| Priority 3 | Automated / Low Cost |
| Priority 4 | Hold / Control |

Customers without a model-led action receive:

**No Model-Led Contact**

---

## Temporal Governance

Model scores are not treated as permanently valid.

A signal can only be used when:

1. it was generated on or before the decision date
2. it remains relevant to the customer's current lifecycle state
3. no later customer event has invalidated the signal

Examples:

- Repeat Purchase scores are only valid during the early customer lifecycle
- Reactivation scores are suppressed if the customer has already returned
- Product recommendations are invalidated when a newer qualifying order supersedes them

This prevents stale predictions from entering the decision engine.

---

## Model Governance Principles

### Propensity Is Not Causal Uplift

A high probability of retention, reactivation, or purchase does not prove that an intervention will cause the outcome.

Treatment-control experiments are required before claiming incremental commercial impact.

---

### Action Scores Are Ranking Scores

Protect, Retain, Reactivate, and Grow priority scores are operational ranking indices.

They are not calibrated probabilities.

Their weights currently represent business policy and should eventually be optimised using:

- incremental response
- contribution margin
- contact cost
- campaign cost
- treatment capacity
- customer fatigue
- incremental profit

---

### Revenue Leakage Exposure

The leakage exposure measure is a conservative prioritisation proxy.

It must not be described as expected accounting loss.

---

### Future Customer Value

Future Customer Value represents expected positive merchandise revenue over the next 90 days.

It is not lifetime value.

---

### Product Recommendation Governance

Repeat recommendations have stronger validation support than discovery recommendations.

Discovery remains CONDITIONAL GO.

---

### Inventory Limitation

The current recommendation system does not contain live:

- stock-on-hand
- stock-out
- product availability
- replenishment constraints

Production deployment should add inventory gating before recommendations are delivered to customers.

---

## Repository Structure

```text
ai-revenue-decision-engine/
│
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_ML_Repeat_Purchase_Model.ipynb
│   ├── 03_ML_Reactivation_Propensity.ipynb
│   ├── 04_ML_Revenue_Leakage_Risk.ipynb
│   ├── 05_ML_Future_Customer_Value.ipynb
│   ├── 06_ML_Next_Best_Product.ipynb
│   ├── 07_ML_Demand_Forecasting.ipynb
│   └── 08_Customer_Decision_Engine.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── scoring/
│   ├── decision_engine/
│   └── monitoring/
│
├── tests/
├── config/
├── docs/
├── scripts/
├── metriclens/
├── requirements.txt
├── .gitignore
└── README.md