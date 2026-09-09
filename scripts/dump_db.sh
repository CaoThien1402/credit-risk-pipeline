#!/usr/bin/env bash
# Snapshot the running Postgres container (schema + data) into db-seed/01_seed.sql.
# Run this after historical_load.py so a fresh clone can `docker compose down -v &&
# docker compose up -d` and get the full dataset back without re-running the ETL —
# db-seed/ is mounted at /docker-entrypoint-initdb.d, which Postgres auto-loads once
# on an empty data volume.
set -euo pipefail

cd "$(dirname "$0")/.."

docker exec credit-db pg_dump -U postgres -d credit_db --clean --if-exists > db-seed/01_seed.sql

echo "Wrote db-seed/01_seed.sql ($(wc -l < db-seed/01_seed.sql) lines)"
