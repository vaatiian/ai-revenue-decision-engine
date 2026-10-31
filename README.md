# Revenue OS — AI Revenue Decision Engine

**From retail transactions to intelligent, governed revenue decisions.**

Revenue OS is a production-oriented machine learning and decision-intelligence system that turns historical retail transaction data into customer-risk signals, predictive insights, prioritised commercial decisions, and governed operational outputs.

It connects **six production intelligence systems** to deterministic decision policies, CRM delivery controls, experiment governance, monitoring, and Docker-based batch workflows. The public Streamlit application makes the architecture and verified historical evidence explorable without exposing raw transactions or operational state.

**[Explore the live dashboard](https://revenue-os-ai.streamlit.app/)** · **[View the v1.0.0 tag](https://github.com/vaatiian/ai-revenue-decision-engine/releases/tag/v1.0.0)** · **[Browse source code](https://github.com/vaatiian/ai-revenue-decision-engine)**

![Revenue OS Executive Overview](docs/screenshots/01-executive-overview.png)

*Revenue OS Executive Overview: transactions → intelligence → decisions → governance.*

> **Evidence boundary:** This is a publicly deployed, production-oriented portfolio and historically verified batch system—not evidence of live customer interventions, autonomous campaign execution, or realised incremental revenue.

## At a Glance

| Verified evidence | Result |
|---|---:|
| Production intelligence systems | 6 |
| Production scoring systems available | 6 / 6 |
| Historical customer decisions generated in Docker smoke execution | 5,220 |
| Historical repeat-purchase scores generated | 4,269 |
| Governed demand forecast horizons | 1, 2 and 4 weeks |
| Full automated regression | **1,915 passed** |
| Portfolio-specific regression | **319 passed** |
| Production batch workloads | 3 |
| Interactive application pages | 11 |
| Interactive architecture views | 3 |
| Portfolio release | **v1.0.0 — published** |

### Find what you need

| If you want to... | Start here |
|---|---|
| Understand the business problem | [1. The Business Problem](#1-the-business-problem) |
| Explore models and commercial decisions | [2. The Solution](#2-the-solution) |
| Review the data-to-decision architecture | [3. System Architecture](#3-system-architecture) |
| See application screenshots | [4. Interactive Streamlit Portfolio](#4-interactive-streamlit-portfolio) |
| Audit testing and operational evidence | [5. Verified Engineering Evidence](#5-verified-engineering-evidence) |
| Examine research methods and metrics | [6. Research and Model Validation](#6-research-and-model-validation) |
| Review safeguards and limitations | [7. Governance and Safety Controls](#7-governance-and-safety-controls) |
| Run the app locally | [8. Local Setup](#8-local-setup) |

---

## 1. The Business Problem

Retailers accumulate transaction histories but often struggle to turn them into coordinated customer actions. Traditional reporting describes **what happened**; predictive models estimate **what may happen**. Neither, on its own, determines a governed response.

Revenue OS addresses five operational questions:

1. Which customers require attention?
2. What action should be considered?
3. How urgently should that action be prioritised?
4. What contact and experimentation rules apply?
5. Is the supporting model evidence valid and eligible for the decision?

**The objective is not more predictions; it is a traceable path from predictive signals to actionable, policy-constrained decisions.**

## 2. The Solution

Revenue OS combines six production intelligence systems with a deterministic customer-level Decision Engine. Four systems may contribute to the decision engine under their governance and eligibility contracts; two run as independent lifecycle and forecasting workflows.

### Production Intelligence Systems

| System | Business question | Governance | Operational role |
|---|---|---|---|
| **Reactivation Propensity** | Which inactive customers may return? | GO | Decision Engine input |
| **Revenue Leakage Risk** | Which customers face a risk of declining revenue? | GO | Decision Engine input |
| **Future Customer Value** | What positive merchandise revenue might be generated over the next 90 days? | CONDITIONAL GO | Decision Engine input |
| **Next Best Product** | What product is relevant to the next eligible purchase? | GO for repeat; CONDITIONAL GO for discovery | Optional Decision Engine input |
| **Repeat Purchase Propensity** | Will a first-time customer purchase again within the governed horizon? | CONDITIONAL GO | Standalone scoring workflow |
| **Weekly Demand Forecasting** | What merchandise revenue is forecast over 1, 2 and 4 weeks? | GO | Standalone forecasting workflow |

**Availability is not permission.** A scoring interface being operational does not authorise unrestricted intervention.

### Customer Decision Actions

For eligible customer signals, the Decision Engine applies lifecycle context, deterministic business rules and prioritisation policies to produce one primary governed outcome.

| Outcome | Intended operational purpose |
|---|---|
| **Protect** | Prioritise commercially important relationships exposed to revenue decline |
| **Retain** | Address weakening customer engagement |
| **Reactivate** | Consider controlled win-back activity for eligible dormant customers |
| **Grow** | Identify eligible product expansion opportunities |
| **Monitor** | Maintain visibility without immediate model-led outreach |
| **Non-actionable / Unscored** | Withhold unsupported model-led actions explicitly |

These outcomes are **policy-derived decisions**, not additional independent predictions.

### Priority and Contact Policy

Actionable customers are ranked within relevant action categories. Priority bands **P1–P4** can be mapped to contact-intensity categories such as High Touch, Campaign, Automated / Low Cost, and Hold / Control.

Non-actionable customers receive **No Model-Led Contact**. Priority indices are operational ranking measures, not calibrated causal probabilities or guaranteed returns.

---

## 3. System Architecture

### Customer Decision Pipeline

```text
Historical Retail Transactions
            |
            v
    Canonical Data Contract
            |
            v
   Production Feature Builders
            |
            v
    Governed Model Scorers
            |
            v
    Score Registry / Snapshots
            |
            v
    Signal Eligibility & Alignment
            |
            v
    Deterministic Decision Engine
            |
            v
    Priority & Contact Policy
            |
            v
    Governed Customer Decisions
```

The eligible Decision Engine channels are **Reactivation Propensity**, **Revenue Leakage Risk**, **Future Customer Value**, and **Next Best Product** when a valid recommendation episode is present. **Repeat Purchase Propensity** and **Weekly Demand Forecasting** remain standalone governed workflows.

### Operational Boundaries

```text
Governed Decision Output
            |
            +--> Experiment Governance
            +--> CRM Delivery Preparation
            +--> Delivery Export / Audit
            +--> Monitoring & Health Checks
```

These are separate operational contracts. Running the Decision Engine **does not automatically send communications or run experiments**.

### Production Runtime

Three batch workloads are supported: **month-end customer decisions**, **repeat-purchase scoring**, and **weekly demand forecasting**.

The Docker runtime is based on **Python 3.11**, executes as a **non-root user**, and is intentionally a **one-shot operational container**, not the public web server. An external scheduler supplies job dates and invocation; this repository does not provision that scheduler.

---

## 4. Interactive Streamlit Portfolio

**[Open Revenue OS](https://revenue-os-ai.streamlit.app/)**

The public application is an **11-page interactive portfolio** with executive, model, decision, governance, architecture, and testing views. It reads compact, curated, anonymised historical outputs rather than loading private raw transaction data or scoring production models at application startup.

### Application Pages

| Section | Pages |
|---|---|
| **Command Centre** | Overview · Revenue Command Centre · Customer Decision Explorer |
| **AI Intelligence** | Interactive Decision Demo · Decision Intelligence · ML / Model Explorer |
| **Governance & Engineering** | Governance Dashboard · Production Architecture · Architecture Visuals · Testing & Reliability · Production Evidence |

### Screenshots

**Revenue Command Centre** — filters, customer-decision snapshots and commercial priorities.

![Revenue Command Centre](docs/screenshots/02-revenue-command-centre.png)

**Risk × Value Intelligence** — governed leakage signals and modelled future customer value.

![Risk and Future Customer Value](docs/screenshots/03-risk-value-intelligence.png)

**Customer Decision Explorer** — anonymised decisions, classification and supporting evidence.

![Customer Decision Explorer](docs/screenshots/04-customer-decision-explorer.png)

**Governance Dashboard** — technical availability, permission, inclusion and evidence boundaries.

![Governance Dashboard](docs/screenshots/05-governance-dashboard.png)

**Production Architecture Visuals** — system boundaries and intelligence workflows.

![Production Architecture Visuals](docs/screenshots/06-architecture-visuals.png)

*Screenshots represent curated, anonymised historical evidence, not live customer interventions or achieved revenue uplift.*

**Local launch:** See [8. Local Setup](#8-local-setup). The public cloud entry point is `streamlit_app/app.py`; the local portfolio can also be launched through the root `app.py`.

---

## 5. Verified Engineering Evidence

### Automated Testing

The verified v1.0.0 release regression reported:

```text
1915 passed, 1 warning
```

The portfolio-specific suite reported:

```text
319 passed
```

| Verification | Result |
|---|---:|
| Full regression | 1,915 passed |
| Portfolio subset (included in full regression) | 319 passed |
| Test failures | 0 |
| Known nonfatal warnings | 1 |
| Dependency integrity | No broken requirements |

The warning originates in the Repeat Purchase invalid-date test, where pandas falls back to per-element date parsing. This warning is known and nonfatal.

An **earlier audited inventory**, not an updated enumeration of v1.0.0, recorded **91 test modules**, **1,644 explicitly defined test functions**, and **196 collection-expanded cases**. These counts should not be interpreted as an independent recalculation of the latest 1,915 executed tests.

Run the full suite (with development dependencies, including `pytest`, installed):

```bash
python -m pytest -q
```

Run the portfolio suite:

```bash
python -m pytest tests/test_portfolio_*.py -q
```

Tests cover model and feature contracts, temporal validation, scoring, customer decisioning, governance, operational workflows, application navigation, evidence integrity and fail-closed behaviour.

### Docker Smoke-Test Evidence

| Workload or control | Verified result |
|---|---|
| Customer decision batch | 5,220 output rows |
| Repeat-purchase batch | 4,269 output rows |
| Demand forecasting | 1-, 2- and 4-week horizons |
| Container execution | Linux, non-root |
| Delivery deduplication | Duplicate reruns rejected |
| Audit | Persistent delivery audit verified |
| Scheduling contract | Invalid demand business boundary rejected |

These are **historical execution and engineering verification results**, not proof of realised commercial impact.

---

## 6. Research and Model Validation

The repository preserves research workflows alongside production implementations.

### Research Notebooks

| Notebook | Purpose |
|---|---|
| `01_Data_Cleaning.ipynb` | Canonical transaction preparation |
| `02_ML_Repeat_Purchase_Model.ipynb` | Early-lifecycle repeat purchase |
| `03_ML_Reactivation_Propensity.ipynb` | Reactivation propensity |
| `04_ML_Revenue_Leakage_Risk.ipynb` | Revenue leakage risk |
| `05_ML_Future_Customer_Value.ipynb` | Future customer revenue prioritisation |
| `06_ML_Next_Best_Product.ipynb` | Repeat and discovery recommendations |
| `07_ML_Demand_Forecasting.ipynb` | Weekly demand forecasting |
| `08_Customer_Decision_Engine.ipynb` | Customer-level decision orchestration |

### Selected Governed Research Metrics

| Model | Evaluation metric | Historical value |
|---|---|---:|
| Future Customer Value | Stage A PR-AUC | 0.822148 |
| Future Customer Value | Complete-model Spearman | 0.550464 |
| Future Customer Value | Revenue Capture@20% | 0.686870 |
| Repeat Recommendation | Holdout Hit Rate@20 | 0.792319 |
| Discovery Recommendation | Holdout Hit Rate@20 | 0.299238 |
| Repeat Purchase | Test ROC-AUC | 0.560 |
| Repeat Purchase | Test PR-AUC | 0.503 |
| Demand Forecast | Approx. 1-week historical WAPE | 0.1652 |
| Demand Forecast | Approx. 4-week historical WAPE | 0.1684 |

These values are historical evaluation results—not causal or realised commercial uplift. Repeat Purchase and discovery recommendation systems retain their conditional governance status. No unsupported headline metric is assigned to Reactivation or Revenue Leakage where governed manifests do not expose an equivalent locked final value.

**Population distinction:** The earlier research documentation identified **5,224 customers** in an August 2011 snapshot. The separately verified Docker execution generated **5,220 customer decision rows**. These represent different evidence contexts and must not be conflated.

---

## 7. Governance and Safety Controls

### Temporal Integrity

- Features must not use outcomes unavailable at the scoring boundary.
- Training/evaluation eligibility is separate from operational eligibility.
- Customer lifecycle changes can invalidate prior scores.
- Recommendation episodes cannot be rebuilt using future-conditioned research outcomes.
- Demand forecasts are bounded by complete observed weekly periods.

### Fail-Closed Contracts

Unsupported conditions are rejected rather than silently accepted. Examples include missing required fields, schema drift, invalid forecast origins or horizons, duplicate delivery IDs, malformed audit records, and invalid model or score eligibility.

### Experimentation and Causal Claims

A propensity score does not establish that an intervention caused a sale. Incremental impact requires appropriately controlled treatment-versus-control measurement. Discovery recommendations are subject to **CONDITIONAL GO** governance and controlled validation.

### Financial Interpretation

- **Revenue leakage exposure** is a prioritisation proxy, not calibrated accounting loss.
- **Future Customer Value** estimates positive merchandise revenue over a governed horizon—not lifetime value, margin or guaranteed income.
- Product recommendations do not currently include live inventory availability gating.

### Intervention Policy

Revenue OS supports governed decision support, CRM preparation, prioritisation and experimentation. **The v1.0.0 portfolio is not authorised for unrestricted autonomous customer interventions.**

---

## 8. Local Setup

### Requirements

- Python 3.11 is the production Docker baseline.
- Portfolio dependencies are defined separately from production runtime dependencies.
- `pytest` is required to run development tests.
- Production scoring requires separately provisioned canonical data, governed model artifacts and operational state.

Clone:

```bash
git clone https://github.com/vaatiian/ai-revenue-decision-engine.git
cd ai-revenue-decision-engine
```

Create an environment:

```bash
python -m venv .venv
```

Activate on **Windows PowerShell**:

```powershell
.venv\Scripts\Activate.ps1
```

Or on **macOS/Linux**:

```bash
source .venv/bin/activate
```

Install portfolio dependencies:

```bash
python -m pip install -r requirements-portfolio.txt
```

Start the portfolio:

```bash
python -m streamlit run app.py
```

For the production runtime dependencies, use the separate requirements file:

```bash
python -m pip install -r requirements.txt
```

For development tests, install `pytest` in your chosen compatible environment and run the applicable test commands from [5. Verified Engineering Evidence](#5-verified-engineering-evidence).

> **Reproducibility boundary:** A fresh clone does not automatically include all private datasets, production model binaries, source manifests, audit state or external scheduler configuration. The public portfolio is intentionally separate from those operational dependencies.

---

## 9. Production Command-Line Interface

The governed operational interface is:

```bash
python -m src.cli --help
```

Runtime health (including governed model-source verification by default):

```bash
python -m src.cli health
```

Month-end customer decisions:

```bash
python -m src.cli customer-batch \
  --data /path/to/canonical_transactions.parquet \
  --decision-date 2011-08-31 \
  --output-dir /path/to/outputs
```

Optional inputs include `--recommendation-episodes PATH` and `--delivery-audit PATH`; inputs must satisfy their respective production contracts.

Repeat-purchase scoring:

```bash
python -m src.cli repeat-purchase \
  --data /path/to/canonical_transactions.parquet \
  --modeling-end-date 2011-08-31 \
  --output-dir /path/to/outputs
```

The optional `--include-opening-month` flag changes the historical-parity opening-month exclusion policy.

Weekly demand forecasting:

```bash
python -m src.cli demand \
  --data /path/to/canonical_transactions.parquet \
  --forecast-origin-week 2011-11-21 \
  --output-dir /path/to/outputs \
  --horizons 1 2 4
```

The forecast-origin week must be a valid Monday representing the last complete observed week. Supported horizons are **1, 2 and 4 weeks** only.

*The dates shown are documented historical example boundaries; commands require appropriate governed inputs and artifacts to execute successfully.*

---

## 10. Docker and Scheduled Operations

Build the one-shot production runtime:

```bash
docker build -t ai-revenue-decision-engine .
```

The image uses `python:3.11-slim`, runs as a non-root user, and defaults to the health operation:

```bash
python -m src.cli health
```

Supported external scheduler interface:

```text
run_scheduled_job.sh customer YYYY-MM-DD
run_scheduled_job.sh repeat-purchase YYYY-MM-DD
run_scheduled_job.sh demand YYYY-MM-DD
```

The wrapper accepts environment-provided configuration, including `DATA_PATH`, `OUTPUT_DIR`, `DELIVERY_AUDIT_PATH`, `RECOMMENDATION_EPISODES_PATH`, and `PYTHON_BIN`.

It does **not** derive scheduling dates, provision a scheduler, retrain models, or automatically execute several workflows. Runtime operators must supply storage, inputs, artifacts, permissions and scheduling infrastructure.

---

## 11. Portfolio Data and Privacy

The public presentation layer is separated from raw transactions and operational state. Curated outputs are generated under a restricted schema using deterministic anonymisation and explicit data classifications:

- `PUBLIC`
- `PUBLIC_AFTER_ANONYMISATION`
- `TECHNICAL_SUMMARY_ONLY`
- `EXCLUDED`

The public application excludes raw transaction data, operational audit histories, sensitive research evaluation rows and private production model binaries.

Documentation:

- [Portfolio Data Contract](portfolio/PORTFOLIO_DATA_CONTRACT.md)
- [Portfolio Narrative](portfolio/PORTFOLIO_NARRATIVE.md)

The deterministic generator is:

```bash
python portfolio/generate_portfolio_data.py
```

It requires its documented governed source artifacts. You do not need to rerun it to explore an already prepared portfolio deployment.

---

## 12. Repository Structure

```text
ai-revenue-decision-engine/
├── app.py                       # Main Streamlit portfolio application
├── streamlit_app/               # Streamlit Community Cloud entry point
├── portfolio/                   # UI, curated data, architecture and evidence views
│   ├── data/
│   ├── generate_portfolio_data.py
│   ├── architecture_visuals.py
│   ├── testing_reliability.py
│   ├── PORTFOLIO_DATA_CONTRACT.md
│   └── PORTFOLIO_NARRATIVE.md
├── src/
│   ├── data/                    # Canonical data contracts
│   ├── features/                # Feature engineering
│   ├── models/                  # Model and forecast interfaces
│   ├── scoring/                 # Governed scorers
│   ├── decision_engine/         # Customer decision orchestration
│   ├── deployment/              # Batch execution
│   ├── monitoring/              # Runtime health
│   └── cli.py                   # Operational CLI
├── tests/                       # Automated regression tests
├── notebooks/                   # Research and model evaluation
├── artifacts/                   # Governed artifacts and metadata
├── config/                      # Configuration
├── scripts/                     # Scheduled-job wrapper
├── docs/screenshots/            # Portfolio captures
├── .streamlit/                  # UI configuration
├── Dockerfile                   # One-shot production runtime
├── requirements.txt             # Production dependencies
├── requirements-portfolio.txt   # Public portfolio dependencies
└── README.md
```

The public Streamlit interface demonstrates and inspects governed system behaviour; it does not replace the production scoring and decision implementation.

---

## 13. Project Status and Roadmap

### v1.0.0 — Released (29 October 2026)

The first public portfolio release is tagged as **[`v1.0.0`](https://github.com/vaatiian/ai-revenue-decision-engine/releases/tag/v1.0.0)** at commit **`51cd2ab9dd220ee4c408d197c2bb7ffe2e4b1792`**.

**Implemented and verified:**

- Research workflows and governed model specifications
- Six production intelligence systems
- Customer-level Decision Engine with prioritisation and contact policies
- CRM delivery preparation, audit and duplicate protection
- Experiment governance and runtime health controls
- Three operational batch interfaces and Docker smoke execution
- Curated, anonymised portfolio data
- Public 11-page Streamlit experience
- Architecture explorer, testing and reliability views
- 1,915-test project regression and 319-test portfolio subset
- Public deployment, published screenshots and versioned Git tag

### Post-v1.0.0 Priorities

1. Deployment health monitoring and browser-based visual regression checks.
2. Additional runtime observability and model-drift monitoring.
3. Enterprise integration and CRM/API readiness.
4. Controlled live intervention experiments, subject to permissions and governance.
5. Credible measurement and attribution of incremental commercial impact.

**Release scope:** v1.0.0 is a production-oriented engineering demonstration and public portfolio release. It is not a claim of an unrestricted, production-operating customer campaign service.

---

## 14. Responsible Interpretation

| Evidence category | What it supports |
|---|---|
| **Model evaluation** | Historical research and test-set performance |
| **Production smoke execution** | Verified governed code execution on specified historical inputs |
| **Operational capability** | Available engineering interfaces and controls |
| **Portfolio demonstration** | Curated inspection of verified historical system outputs |
| **Realised business impact** | Requires measured real-world outcomes and an appropriate causal basis |

**No incremental revenue, retention or conversion uplift is claimed without valid experimental evidence.**

---

## 15. Project Principle

> **Predictions are not decisions. Decisions are not interventions. Interventions are not impact.**

Revenue OS is designed to make those distinctions explicit, testable, auditable and commercially useful.

**From retail transactions to governed revenue decisions.**
