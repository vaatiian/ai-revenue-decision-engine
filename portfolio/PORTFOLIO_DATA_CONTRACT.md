AI Revenue Decision Engine
Portfolio Data Contract — v1.0
1. Purpose
This document defines the data that may be exposed, transformed, cached, or visualised inside the public AI Revenue Decision Engine portfolio application.

The portfolio layer must remain separate from the production runtime.

Its purpose is to present verified system capability safely and quickly without exposing unnecessary operational, research, or personally identifying detail.
2. Core Principle
The public application must use curated portfolio data rather than depending on the full production dataset.

Portfolio data must be:
- derived from real governed system outputs;
- anonymised where customer or transaction identifiers are present;
- compact enough for fast application startup;
- reproducible;
- safe for public presentation;
- explicitly separated from production state;
- free from future-conditioned research leakage where the public interface implies live production capability.
The portfolio must never fabricate model outputs or business impact metrics.
3. Portfolio Data Zones
The project uses four portfolio data classifications.
PUBLIC
Safe for direct inclusion in the deployed portfolio application.
PUBLIC_AFTER_ANONYMISATION
Safe only after raw identifiers have been replaced with deterministic portfolio identifiers.
TECHNICAL_SUMMARY_ONLY
Underlying data may be used to calculate aggregate metrics or charts but must not be exposed row-by-row.
EXCLUDED
Must not be packaged with the public portfolio application.
4. Customer Decision Data
Source
outputs/phase12_smoke/customer/customer-batch/2011-08-31/customer_decisions.csv
Source Shape
5,220 rows × 33 columns.
Classification
PUBLIC_AFTER_ANONYMISATION
Purpose
This is the primary dataset for:
- Revenue Command Centre;
- Customer Decision Explorer;
- Decision Engine storytelling;
- action distribution;
- priority distribution;
- customer-state distribution;
- risk segmentation;
- FCV segmentation;
- reactivation segmentation;
- contact-policy distribution;
- experimentation distribution.
Identifier Treatment
The source customer identifier must not be exposed directly.

It must be replaced with a deterministic portfolio identifier using a stable mapping such as:

CUST-00001

CUST-00002

CUST-00003

The mapping must be deterministic within the generated portfolio dataset.

The original customer identifier must not be included in the deployed portfolio file.
Allowed Columns
Portfolio output may include:
- portfolio_customer_id
- decision_date
- customer_state
- days_since_last_order
- customer_tenure_days
- lifetime_orders
- lifetime_revenue
- orders_prior_30d
- orders_prior_90d
- revenue_prior_90d
- revenue_prior_180d
- primary_action
- primary_action_score
- action_priority_percentile
- priority_tier
- contact_intensity
- business_action
- decision_reason
- experiment_requirement
- leakage_probability
- leakage_exposure_score
- leakage_exposure_percentile
- economic_priority_band
- future_active_probability
- future_customer_value_90d
- fcv_percentile
- fcv_tier
- reactivation_probability
- reactivation_prediction_date
- recommended_product_channel
- portfolio_recommended_product
- portfolio_repeat_top1_product
- portfolio_discovery_top1_product
Product Identifier Treatment
Product identifiers must not be exposed directly if present.

Any product identifier included in the public portfolio should be deterministically mapped to a portfolio-safe representation such as:

PROD-001

PROD-002

or replaced with a human-readable portfolio label where a safe description is available.
Display Normalisation
Presentation-only labels may be cleaned for public display.

Examples:
- encoding artefacts may be replaced with standard separators;
- internal action labels may be converted into human-readable title case;
- missing values may be displayed as Not Available or an equivalent neutral label.
Production source data must remain unchanged.
5. Repeat Purchase Portfolio Data
Source
outputs/phase12_smoke/repeat/repeat-purchase/2011-08-31/repeat_purchase_scores.csv
Source Shape
4,269 rows × 10 columns.
Classification
PUBLIC_AFTER_ANONYMISATION
Purpose
Supports:
- Repeat Purchase model explorer;
- score distribution;
- propensity segmentation;
- example newly acquired customer profiles;
- governance explanation.
Identifier Treatment
The following source identifiers must not be exposed:
- customer
- first_invoice
They should be replaced with deterministic portfolio-safe identifiers.

Recommended representations:
- CUST-00001
- ORD-000001
Allowed Portfolio Fields
- portfolio_customer_id
- portfolio_first_order_id
- first_order_date
- repeat_purchase_score
- model_name
- model_version
- model_type
- target_name
- governance_decision
scored_at_utc may be retained for technical provenance if needed but should not dominate the public presentation.
6. Demand Forecast Data
Source
outputs/phase12_smoke/demand/demand/2011-11-21/demand_forecasts.csv
Source Shape
3 rows × 15 columns.
Classification
PUBLIC
Purpose
Supports:
- Demand Forecasting page;
- 1-week forecast;
- 2-week forecast;
- 4-week forecast;
- seasonal analogue comparison;
- recent four-week average;
- forecast methodology explanation;
- governance explanation.
Allowed Fields
All current aggregate fields may be used because they do not contain customer-level identifiers.

These include:
- forecast_origin_week
- target_week
- horizon_weeks
- seasonal_analogue_week
- seasonal_analogue_revenue
- recent_four_week_average
- forecast_revenue
- seasonal_weight
- recent_weight
- model_name
- model_version
- model_type
- target_name
- governance_decision
- scored_at_utc
7. Validation Data
FCV Validation
Source:

outputs/validation/fcv_end_to_end_validation.parquet

Classification:

TECHNICAL_SUMMARY_ONLY

May be used for:
- score distributions;
- FCV distributions;
- probability distributions;
- summary statistics;
- technical validation visuals.
Customer identifiers must not be exposed.
Leakage Validation
Source:

outputs/validation/leakage_end_to_end_validation.parquet

Classification:

TECHNICAL_SUMMARY_ONLY

May be used for:
- leakage score distributions;
- model output ranges;
- population counts;
- technical validation visuals.
Customer identifiers must not be exposed.
Reactivation Validation
Source:

outputs/validation/reactivation_end_to_end_validation.parquet

Classification:

TECHNICAL_SUMMARY_ONLY

May be used for:
- reactivation probability distribution;
- population counts;
- technical validation visuals.
Customer identifiers must not be exposed.
Recommendation Validation
Source:

outputs/validation/recommendation_end_to_end_validation.parquet

Classification:

TECHNICAL_SUMMARY_ONLY

This file contains future-conditioned research evaluation information including fields such as:
- next_invoice
- next_order_date
- days_to_next_order
- next_basket_products
- repeat_products
- new_products
These fields must not be exposed in the public application as if they were live production inputs.

The portfolio may use aggregate research metrics derived from governed manifests.

The raw validation rows must not be deployed.
8. Model Manifest Data
Sources
Governed JSON manifests under:

artifacts/models/*/*.json
Classification
PUBLIC
Purpose
These manifests are the canonical source for technical model cards.

Allowed information includes:
- model name;
- model version;
- model type;
- target definition;
- prediction horizon;
- feature count;
- algorithm;
- hyperparameters;
- preprocessing;
- validation methodology;
- governance decision;
- approved uses;
- prohibited uses;
- model limitations;
- performance metrics;
- architecture;
- forecast horizons;
- recommendation architecture;
- production artifact metadata.
Sensitive Technical Metadata
Cryptographic hashes may remain in repository manifests but do not need prominent display inside the portfolio application.

They are implementation provenance rather than hiring-manager-facing information.
9. Locked Portfolio Model Metrics
The portfolio may display only metrics verified by governed model artefacts or explicitly reproduced from the project.
Future Customer Value
Governance:

CONDITIONAL GO

Verified metrics:
- Stage A PR-AUC: 0.822148
- Stage A ROC-AUC: 0.749591
- Complete-model Spearman: 0.550464
- Revenue Capture@10%: 0.573750
- Revenue Capture@20%: 0.686870
- Revenue Capture@30%: 0.768946
- Predicted-to-Actual Revenue Ratio: 0.593481
These metrics must be labelled as research/test performance rather than realised business impact.
Recommendation System
Repeat channel governance:

GO

Discovery channel governance:

CONDITIONAL GO

Verified final holdout metrics:
- Repeat Hit Rate@10: 0.705912
- Repeat Hit Rate@20: 0.792319
- Repeat-Eligible Hit Rate@20: 0.870896
- Discovery Hit Rate@10: 0.216010
- Discovery Hit Rate@20: 0.299238
- Hybrid Hit Rate@20: 0.774634
These are recommendation evaluation metrics, not conversion or revenue uplift metrics.
Demand Forecasting
Governance:

GO

Verified approximate historical WAPE:
- 1-week horizon: 0.1652
- 2-week horizon: 0.1668
- 4-week horizon: 0.1684
- 4-week aggregate WAPE: 0.0916
Verified approximate Forecast / Actual ratios:
- 1-week: 0.974
- 2-week: 0.974
- 4-week: 0.975
- four-week aggregate: 0.977
These are historical evaluation statistics, not guaranteed future performance.
Repeat Purchase
Governance:

CONDITIONAL GO

Verified final test metrics:
- PR-AUC: 0.503
- ROC-AUC: 0.560
- Brier Score: 0.248
- Lift@10%: 1.03
- Lift@20%: 1.11
- Calibration Error: 4.71 percentage points
The portfolio must preserve the conditional governance interpretation rather than presenting the model as unrestricted production intelligence.
10. Leakage and Reactivation Model Metrics
The current production manifests contain architecture, governance, feature, and validation-design metadata but do not expose equivalent locked final performance values in their manifests.

Therefore the public app must not invent headline performance metrics for these models.

They may instead display:
- governance decision;
- algorithm;
- prediction horizon;
- validation method;
- feature count;
- business role;
- production availability.
If later portfolio work retrieves independently governed final metrics from research artefacts, those may be added only after verification.
11. Production Smoke-Test Evidence
The portfolio may display the following verified execution evidence:
- 6 / 6 production scoring systems available;
- 5,220 customer decisions generated in Docker smoke execution;
- 4,269 repeat-purchase scores generated in Docker smoke execution;
- 1-, 2-, and 4-week demand forecasts generated;
- duplicate customer delivery rerun rejected;
- invalid demand schedule boundary rejected;
- non-root Docker runtime verified;
- persistent delivery audit verified;
- 1,596 automated tests passing.
These must be described as:
- test evidence;
- system verification;
- deployment verification.
They must not be described as realised production-business impact.
12. Excluded Public Data
The following must not be bundled into the deployed portfolio application.
Raw canonical transaction data
Reason:

Unnecessary for portfolio presentation and larger than required for a fast public application.
Model binaries
Includes .joblib files.

Reason:

The public portfolio does not need to ship production model artifacts merely to explain the system.
Delivery payload files
Reason:

Operational CRM delivery representations are unnecessary for public demonstration.
Delivery audit ledger
Reason:

Operational state should not be exposed publicly.
Experiment state
Reason:

Operational experiment history is not required for the public application.
Raw recommendation evaluation population
Reason:

Contains future-conditioned information used for research evaluation.
Temporary research files
Includes:
- decision_engine_research_temp.txt
- recommendation_research_temp.py
These remain outside the portfolio layer.
13. Portfolio Directory Contract
The eventual portfolio structure should follow:
portfolio/
├── PORTFOLIO_NARRATIVE.md
├── PORTFOLIO_DATA_CONTRACT.md
├── data/
│   ├── customer_decisions_portfolio.parquet
│   ├── repeat_purchase_portfolio.parquet
│   ├── demand_forecasts_portfolio.parquet
│   ├── model_registry_portfolio.json
│   └── portfolio_summary.json
├── assets/
├── components/
├── pages/
└── theme/
The exact application module structure may evolve during Phase 13, but the separation between production artifacts and portfolio data must remain.
14. Portfolio Dataset Generation
Portfolio datasets should be produced by a deterministic script rather than manually edited.

The future generator should:
1. load governed production outputs;
2. validate expected schemas;
3. anonymise customer identifiers;
4. anonymise order identifiers where required;
5. anonymise product identifiers where required;
6. normalise display-only labels;
7. select only approved columns;
8. derive aggregate portfolio summaries;
9. write compact portfolio-safe files;
10. fail closed if source schemas drift.
The generator must not modify the original production outputs.
15. Data Integrity
Portfolio data generation must fail closed if:
- expected source files are missing;
- source schemas drift unexpectedly;
- duplicate anonymised IDs are created;
- required governed fields are absent;
- unsupported future-conditioned fields enter a public dataset;
- non-approved raw identifiers remain after anonymisation.
Portfolio generation must be covered by automated tests.
16. Claims Integrity
The public portfolio must distinguish among:
Model Performance
Research or test metrics from governed model evaluation.
Production Readiness Evidence
Docker, scheduler, scoring, audit, and regression verification.
System Capability
What the architecture is capable of doing.
Demonstration
Interactive examples shown through portfolio-safe data.
Realised Business Impact
Measured business outcomes such as incremental revenue, conversion uplift, or retention uplift.

No realised business impact should be claimed unless it was actually measured.
17. Performance Requirement
The public portfolio should load primarily from compact curated files.

It must not require:
- loading the complete source transaction dataset;
- retraining models;
- running all production scorers at application startup;
- rebuilding validation populations;
- executing notebooks.
This keeps the application fast, reliable, and inexpensive to host.
18. Interactive Decision Engine Exception
The portfolio may invoke selected lightweight production decision-policy logic interactively where doing so is safe and deterministic.

This is permitted because the Decision Engine policy itself is part of the product being demonstrated.

The portfolio must not create a separate mock policy that disagrees with production logic.
19. Completion Criteria
The Portfolio Data Contract is satisfied when:
- public-safe source classifications are frozen;
- identifiers are anonymised deterministically;
- public datasets contain only approved fields;
- future-conditioned evaluation data is excluded from live portfolio views;
- model metrics come from governed manifests;
- portfolio claims distinguish research, smoke-test, system, and business evidence;
- generation is deterministic;
- schema drift fails closed;
- automated tests cover the portfolio data layer.
20. Governing Rule
The portfolio is a presentation layer over the real system.

It must simplify presentation without weakening technical truth.