#!/bin/sh

set -eu


# ================================================================
# AI REVENUE DECISION ENGINE — SCHEDULED JOB WRAPPER
#
# Usage
# -----
#   run_scheduled_job.sh customer YYYY-MM-DD
#   run_scheduled_job.sh repeat-purchase YYYY-MM-DD
#   run_scheduled_job.sh demand YYYY-MM-DD
#
# The external scheduler owns WHEN this script runs.
# The caller must always supply the explicit business boundary.
#
# This wrapper does not:
# - calculate dates from wall-clock time;
# - determine month-end or Monday scheduling;
# - retrain models;
# - change governed business logic;
# - run multiple jobs automatically.
# ================================================================


usage() {
    echo "Usage:" >&2
    echo "  $0 customer YYYY-MM-DD" >&2
    echo "  $0 repeat-purchase YYYY-MM-DD" >&2
    echo "  $0 demand YYYY-MM-DD" >&2
}


# ================================================================
# ARGUMENT CONTRACT
# ================================================================


if [ "$#" -ne 2 ]; then
    usage
    exit 2
fi


JOB="$1"
BUSINESS_DATE="$2"


if [ -z "$JOB" ] || [ -z "$BUSINESS_DATE" ]; then
    usage
    exit 2
fi


# ================================================================
# RUNTIME PATH CONTRACT
# ================================================================


PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_PATH="${DATA_PATH:-/app/data/cleaned_online_retail.parquet}"

OUTPUT_DIR="${OUTPUT_DIR:-/app/outputs}"

DELIVERY_AUDIT_PATH="${DELIVERY_AUDIT_PATH:-/app/artifacts/delivery/delivery_audit.jsonl}"


# ================================================================
# GOVERNED JOB ROUTING
# ================================================================


case "$JOB" in

    customer)

        if [ -n "${RECOMMENDATION_EPISODES_PATH:-}" ]; then

            exec "$PYTHON_BIN" \
                -m src.cli \
                customer-batch \
                --data "$DATA_PATH" \
                --decision-date "$BUSINESS_DATE" \
                --output-dir "$OUTPUT_DIR" \
                --delivery-audit "$DELIVERY_AUDIT_PATH" \
                --recommendation-episodes "$RECOMMENDATION_EPISODES_PATH"

        fi

        exec "$PYTHON_BIN" \
            -m src.cli \
            customer-batch \
            --data "$DATA_PATH" \
            --decision-date "$BUSINESS_DATE" \
            --output-dir "$OUTPUT_DIR" \
            --delivery-audit "$DELIVERY_AUDIT_PATH"
        ;;


    repeat-purchase)

        exec "$PYTHON_BIN" \
            -m src.cli \
            repeat-purchase \
            --data "$DATA_PATH" \
            --modeling-end-date "$BUSINESS_DATE" \
            --output-dir "$OUTPUT_DIR"
        ;;


    demand)

        exec "$PYTHON_BIN" \
            -m src.cli \
            demand \
            --data "$DATA_PATH" \
            --forecast-origin-week "$BUSINESS_DATE" \
            --output-dir "$OUTPUT_DIR"
        ;;


    *)

        echo "Unsupported scheduled job: $JOB" >&2
        usage
        exit 2
        ;;

esac