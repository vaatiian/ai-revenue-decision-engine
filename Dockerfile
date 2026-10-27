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

COPY scripts/run_scheduled_job.sh \
     ./scripts/run_scheduled_job.sh


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
    && chmod 0755 /app/scripts/run_scheduled_job.sh \
    && chown -R app:app /app

USER app


# ================================================================
# DEFAULT OPERATION
#
# This remains intentionally a one-shot operational image.
# External schedulers invoke scripts/run_scheduled_job.sh with an
# explicit governed job name and business boundary.
# ================================================================

CMD ["python", "-m", "src.cli", "health"]
