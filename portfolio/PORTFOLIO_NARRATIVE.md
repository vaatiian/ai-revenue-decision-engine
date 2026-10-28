# AI Revenue Decision Engine

## Portfolio Narrative Specification — v1.0

---

## 1. Portfolio Objective

The AI Revenue Decision Engine portfolio must demonstrate that this project is not a collection of isolated machine learning notebooks.

It is an end-to-end, production-oriented decision intelligence system that converts historical retail transactions into governed commercial decisions.

The portfolio must be understandable to two audiences:

1. Nontechnical hiring managers and business leaders.
2. Technical hiring managers, data scientists, machine learning engineers, analytics leaders, and software engineers.

The portfolio experience must allow a nontechnical visitor to understand the business value within approximately 60–90 seconds while allowing a technical visitor to progressively inspect the architecture, modelling methodology, governance controls, testing strategy, deployment system, and source code.

---

# 2. Core Positioning

## Primary Project Name

**AI Revenue Decision Engine**

## Supporting Description

An end-to-end machine learning and decision intelligence platform that transforms retail transaction history into customer-level actions, revenue-risk signals, product recommendations, future-value estimates, repeat-purchase propensity, and weekly revenue forecasts.

## One-Line Positioning

**From transactions to governed revenue decisions.**

## Expanded Positioning

The system combines supervised machine learning, recommendation systems, deterministic forecasting, customer analytics, governed decision logic, CRM delivery, experiment tracking, monitoring, automated testing, and scheduled Docker execution.

Its purpose is not merely to generate predictions.

Its purpose is to answer:

- Which customers require attention?
- What commercial action should be taken?
- How urgent is that action?
- Which channel or treatment should be used?
- Which predictions are safe to operationalise?
- How should decisions be audited and monitored?

---

# 3. Portfolio Story

## The Business Problem

Retail organisations can accumulate millions of historical transactions while still struggling to convert that data into coordinated customer actions.

Traditional analytics may identify what happened.

Individual machine learning models may predict what could happen.

Neither automatically answers the operational question:

> What should the business do next?

The project addresses that gap by connecting predictive intelligence to governed business action.

## The Technical Problem

A useful production decision system must solve more than model training.

It must address:

- temporal leakage;
- production feature parity;
- scoring contracts;
- model versioning;
- incomplete populations;
- recommendation eligibility;
- decision precedence;
- governance restrictions;
- contact policy;
- experimentation;
- delivery integrity;
- auditability;
- runtime health;
- scheduled execution;
- repeatability;
- failure handling.

The AI Revenue Decision Engine was designed around those constraints.

---

# 4. End-to-End System Story

The system follows this conceptual flow:

**Retail Transactions**

→ Canonical Data Contract

→ Feature Engineering

→ Production Scoring Systems

→ Score Registry

→ Customer Decision Engine

→ Priority & Contact Policy

→ Experiment Governance

→ CRM Delivery

→ Audit Ledger

→ Monitoring

→ Scheduled Docker Runtime

The final customer-facing decision layer can classify customers into governed actions including:

- Protect
- Retain
- Reactivate
- Grow
- Monitor
- Unscored / non-actionable states

---

# 5. Production Intelligence Systems

The portfolio represents six production intelligence systems.

## 5.1 Reactivation Propensity

### Question

Which inactive customers are likely to return?

### Architecture

Supervised binary classification.

### Production Governance

**GO**

### Role

Supports reactivation decisions for eligible dormant customers.

---

## 5.2 Revenue Leakage Risk

### Question

Which customers are at risk of materially reducing future revenue?

### Architecture

Supervised binary classification using historical customer behaviour and revenue baselines.

### Production Governance

**GO**

### Role

Supports customer retention and revenue protection decisions.

---

## 5.3 Future Customer Value

### Question

How much future customer revenue can reasonably be expected?

### Architecture

Two-stage probability × value framework.

### Production Governance

**CONDITIONAL GO**

### Role

Provides economic context for customer prioritisation and growth decisions.

---

## 5.4 Repeat Purchase Propensity

### Question

Which first-time customers are likely to purchase again within the governed horizon?

### Architecture

Interpretable logistic regression.

### Production Governance

**CONDITIONAL GO**

### Role

Operates as a separate customer-acquisition / repeat-purchase scoring workflow rather than a current Decision Engine input.

---

## 5.5 Next Best Product Recommendation

### Question

What product should an eligible purchasing customer be shown next?

### Architecture

Deterministic recommendation architecture combining:

- repeat-purchase frequency;
- item-item discovery affinity;
- optional hybrid ranking.

### Production Governance

Repeat channel:

**GO**

Discovery channel:

**CONDITIONAL GO**

### Role

Supports growth actions when a valid recommendation episode is externally observed.

The system does not reconstruct future-conditioned research evaluation populations in production.

---

## 5.6 Weekly Demand Forecasting

### Question

What level of positive merchandise revenue should the business expect over the next 1, 2, and 4 weeks?

### Architecture

Deterministic seasonal / recent demand blend.

### Production Governance

**GO**

### Role

Operates as a separate weekly business forecasting workflow.

Demand Forecast is not currently used as a customer-level Decision Engine input.

---

# 6. Decision Engine

The Decision Engine is the portfolio's central differentiator.

The predictive models do not independently determine customer actions.

Their outputs are aligned through governed policy.

Customer actions include:

### ACTION_PROTECT

Used for active customers requiring revenue protection.

### ACTION_RETAIN

Used for at-risk customers where retention intervention is appropriate.

### ACTION_REACTIVATE

Used for eligible dormant customers.

### ACTION_GROW

Used for active customers with governed growth opportunities.

### ACTION_MONITOR

Used when model-led intervention is not currently appropriate.

### Non-actionable governed states

The system also preserves explicit states where a model-led action should not be taken.

This prevents predictions from automatically becoming commercial actions without policy controls.

---

# 7. Priority Framework

Actionable customers receive governed priority bands.

- **P1** — highest priority
- **P2**
- **P3**
- **P4**
- **Not Prioritised** — non-actionable outcomes

Contact strategy is derived from governed priority.

Examples include:

- High Touch
- Campaign
- Automated / Low Cost
- Hold / Control
- No Model-Led Contact

---

# 8. Experiment Governance

The system distinguishes between recommendation and experimentation.

Examples:

- Reactivation actions require experimentation.
- Growth actions require experimentation where the discovery recommendation channel remains CONDITIONAL GO.
- Retention and protection actions can carry recommended experimentation.
- Non-actionable states are marked Not Applicable.

Experiment activity is recorded in a governed append-only experiment store.

---

# 9. Delivery Governance

Customer decisions can be transformed into governed CRM delivery payloads.

The delivery subsystem includes:

- deterministic or caller-controlled delivery IDs;
- duplicate protection;
- CSV export;
- JSONL export;
- append-only audit history;
- batch-level validation;
- fail-closed duplicate handling.

Scheduled customer-batch reruns cannot silently duplicate audited customer deliveries.

---

# 10. Monitoring

The production monitoring system observes:

- scoring runtime availability;
- experiment-store integrity;
- decision outputs;
- delivery audit health.

Health levels include:

- HEALTHY
- DEGRADED
- UNHEALTHY

Monitoring is observational.

It does not independently modify customer decisions.

---

# 11. Scheduled Production Workflows

The production runtime exposes three governed scheduled workloads.

## Month-End Customer Decision Job

Transactions

→ Reactivation scoring

→ Revenue Leakage scoring

→ Future Customer Value scoring

→ Optional Recommendation scoring

→ Customer Decision Engine

→ Delivery payload generation

→ Export

→ Audit

## Repeat Purchase Job

Transactions

→ Production Repeat Purchase Features

→ Repeat Purchase Scoring

→ Governed score export

## Weekly Demand Job

Transactions

→ Locked weekly demand representation

→ 1-week forecast

→ 2-week forecast

→ 4-week forecast

→ Governed forecast export

---

# 12. Production Evidence

The portfolio must prominently communicate real execution evidence.

Current verified production evidence includes:

- **6 / 6 production scoring systems available**
- **5,220 customer decisions generated in Docker smoke execution**
- **4,269 repeat-purchase scores generated in Docker smoke execution**
- **1, 2, and 4 week governed demand forecasts generated**
- duplicate customer-delivery reruns rejected
- invalid demand scheduling boundaries rejected
- Linux non-root Docker runtime verified
- persistent delivery audit verified
- explicit model and data mount boundaries verified
- **1,596 automated tests passing**

The portfolio must never invent metrics that were not generated or verified by the project.

---

# 13. Model Governance Principle

A model being technically available does not automatically mean it is unrestricted for production use.

The project explicitly preserves governance decisions such as:

- GO
- CONDITIONAL GO

This distinction must be visible throughout the portfolio.

Imperfections in model performance should not be hidden.

They should demonstrate mature model-risk thinking.

---

# 14. Failure Philosophy

A defining engineering principle of the project is:

## Fail Closed

Examples include:

- invalid forecast origin → rejected;
- missing required feature → rejected;
- duplicate delivery identifier → rejected;
- malformed audit store → rejected;
- schema drift → rejected;
- unavailable required scoring runtime → surfaced through health monitoring;
- unsupported forecast horizon → rejected.

Failure handling is a product feature, not an afterthought.

---

# 15. Temporal ML Discipline

Temporal correctness is a major portfolio theme.

The system uses chronological reasoning rather than random train/test logic for time-dependent customer behaviour.

Important principles include:

- historical features precede outcomes;
- future outcome windows do not leak into production features;
- evaluation eligibility is separated from live production eligibility;
- incomplete reporting periods are governed explicitly;
- recommendation research populations are not reconstructed from future information during production;
- demand forecasting respects complete observed weekly periods.

The portfolio should include visual explanations of these boundaries.

---

# 16. Application Audiences

The interactive portfolio application should provide two natural exploration paths.

## Executive / Business Path

Focus on:

- business problem;
- customer decisions;
- commercial impact;
- revenue risk;
- customer prioritisation;
- recommendations;
- demand outlook;
- operational readiness.

Technical implementation should remain available but secondary.

## Technical / Data Science Path

Focus on:

- data contracts;
- feature engineering;
- temporal validation;
- model architectures;
- evaluation;
- governance;
- scoring interfaces;
- decision alignment;
- software architecture;
- testing;
- Docker;
- scheduled deployment;
- observability.

---

# 17. Portfolio Application Experience

The Streamlit application should feel like an AI revenue command centre rather than a notebook viewer.

Primary navigation should include:

1. Overview
2. Revenue Command Centre
3. Customer Decision Explorer
4. Decision Engine
5. AI & Models
6. Demand Forecasting
7. Governance
8. Production System
9. Testing & Reliability
10. Architecture
11. Case Study
12. About the Project

---

# 18. Landing Page Narrative

The opening viewport must answer four questions almost immediately.

### What is this?

An AI Revenue Decision Engine.

### What does it do?

Transforms retail transactions into governed customer and revenue decisions.

### Is it only research?

No.

It includes production scorers, governed decision logic, delivery controls, monitoring, testing, Docker execution, and scheduled workloads.

### Why should I care?

It demonstrates the full path from raw data through machine learning to operational business action.

---

# 19. Hero Proof Points

The opening portfolio experience may use verified proof points including:

**6**  
Production Intelligence Systems

**1,596**  
Automated Tests Passing

**5,220**  
Customer Decisions Generated in Production Smoke Execution

**4,269**  
Repeat Purchase Scores Generated

**3**  
Governed Demand Horizons

**6 / 6**  
Production Scoring Systems Available

These numbers must always be labelled accurately so smoke-test evidence is not misrepresented as business impact.

---

# 20. Visual Direction

The portfolio should feel like a serious modern intelligence product.

Desired characteristics:

- visually strong but restrained;
- executive-friendly;
- technically credible;
- high information hierarchy;
- generous whitespace;
- strong typography;
- deliberate use of colour;
- no rainbow dashboard aesthetic;
- minimal decorative elements;
- interactive but not gimmicky.

Suggested visual language:

- graphite / near-black surfaces;
- warm white typography;
- restrained cyan or electric-blue intelligence accent;
- green for GO / healthy;
- amber for CONDITIONAL GO / degraded;
- red only for genuine risk or failed controls;
- large KPI typography;
- compact technical metadata;
- clean analytical charts.

The interface should feel closer to a modern enterprise intelligence platform than a generic Streamlit tutorial.

---

# 21. Portfolio Data Principle

The public portfolio experience should not depend on loading the entire production dataset at startup.

The architecture should separate:

## Production Engine

Real production code, feature pipelines, scorers, policies, delivery, monitoring, and scheduling.

## Portfolio Demonstration Layer

Curated and anonymised demonstration outputs derived from the real system.

This provides:

- fast startup;
- safer public deployment;
- reproducibility;
- low hosting requirements;
- controlled storytelling.

Where appropriate, selected lightweight production logic may also run interactively.

---

# 22. Interactive Customer Explorer

The application should allow visitors to inspect governed customer outcomes.

Potential controls:

- customer state;
- action;
- priority;
- leakage risk band;
- future-value band;
- experiment requirement;
- contact policy.

Visitors should be able to select a customer and view a decision dossier containing:

- state;
- available model scores;
- economic context;
- recommended action;
- priority;
- contact policy;
- experiment requirement;
- recommendation where applicable;
- governance context.

---

# 23. Interactive Decision Demonstration

The portfolio should include a controlled Decision Engine demonstration.

Users may adjust governed example inputs and observe policy outcomes.

The demonstration must use the real Decision Engine policy or an exact portfolio adapter around it.

It must not invent a second set of business rules purely for UI demonstration.

---

# 24. Technical Model Explorer

Each intelligence system should have a dedicated technical view containing, where available:

- business question;
- target definition;
- prediction horizon;
- population;
- feature architecture;
- validation methodology;
- algorithm;
- evaluation metrics;
- governance decision;
- production scorer;
- artifact/version information;
- known limitations;
- production role.

---

# 25. Architecture Story

The portfolio should visually communicate this high-level architecture:

Transactions

→ Canonical Data Layer

→ Feature Pipelines

→ Production Intelligence Systems

→ Score Registry

→ Decision Engine

→ Priority / Contact Policy

→ Experiment Governance

→ CRM Delivery

→ Audit

→ Monitoring

→ Scheduled Docker Runtime

Demand Forecast and Repeat Purchase remain visible as governed adjacent workflows where they are not current customer-action inputs.

---

# 26. Testing & Reliability Story

Testing should be presented as part of the engineering story rather than as a footer.

The portfolio should highlight coverage of areas such as:

- feature contracts;
- schema enforcement;
- temporal boundaries;
- production scorer behaviour;
- governance logic;
- customer decision policy;
- experiment integrity;
- delivery integrity;
- monitoring;
- deployment runtime;
- scheduler behaviour;
- fail-closed conditions.

Verified current regression:

**1,596 tests passing**

One known Repeat Purchase invalid-date parsing warning is intentional and non-fatal.

---

# 27. Repository Story

The GitHub repository should reinforce the application.

The final README should prioritise:

1. project purpose;
2. live demo;
3. architecture;
4. core capabilities;
5. verified results;
6. model register;
7. governance;
8. production engineering;
9. testing;
10. local execution;
11. repository structure;
12. technical documentation.

The README should not duplicate the entire Streamlit application.

It should act as the repository landing page and technical gateway.

---

# 28. Claims Policy

The portfolio must distinguish clearly between:

- model metrics;
- production smoke-test evidence;
- system capability;
- business simulation;
- actual realised business impact.

Do not claim realised revenue uplift unless such uplift has been measured.

Do not describe smoke-test customer counts as production customers served.

Do not describe model forecasts as realised business results.

Technical credibility is more valuable than inflated claims.

---

# 29. Completion Definition

Phase 13 is complete when the project has:

- an interactive portfolio application;
- executive and technical exploration paths;
- curated portfolio-safe data;
- customer decision explorer;
- interactive Decision Engine demonstration;
- model explorer;
- governance dashboard;
- demand forecasting experience;
- production architecture view;
- testing and reliability view;
- architecture diagrams;
- polished GitHub README;
- screenshots / demo assets;
- portfolio application tests;
- local end-to-end validation;
- live deployment;
- final repository cleanup;
- final tagged release.

The target release is:

**v1.0.0 — AI Revenue Decision Engine**

At that point the project is considered complete as a flagship portfolio project.