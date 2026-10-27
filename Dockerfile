# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app


# ================================================================
# DEPENDENCIES
#
# Do not upgrade pip during the image build. The base image already
# includes pip, and avoiding the extra download reduces build
# fragility. Retries are retained for package installation.
# ================================================================

COPY requirements.txt ./requirements.txt

RUN python -m pip install \
    --retries 10 \
    --timeout 120 \
    -r requirements.txt


# ================================================================
# APPLICATION
# ================================================================

COPY src ./src


# ================================================================
# GOVERNED MODEL MANIFESTS
#
# .dockerignore excludes untracked model binaries and executed
# research notebooks. Runtime model binaries are mounted separately.
# ================================================================

COPY artifacts/models ./artifacts/models


# ================================================================
# GOVERNED SOURCE MATERIAL
#
# Demand forecasting verifies the SHA-256 of its research notebook
# as part of the production manifest contract.
# ================================================================

COPY notebooks/07_ML_Demand_Forecasting.ipynb \
     ./notebooks/07_ML_Demand_Forecasting.ipynb


# ================================================================
# NON-ROOT RUNTIME
# ================================================================

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && chown -R app:app /app

USER app


# ================================================================
# DEFAULT OPERATION
#
# This is intentionally a one-shot operational image, not a web
# server. Phase 12 will add scheduled execution.
# ================================================================

CMD ["python", "-m", "src.cli", "health"]