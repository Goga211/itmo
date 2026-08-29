#!/bin/bash
set -e

if ! psql -U postgres -Atqc "SELECT 1 FROM pg_roles WHERE rolname = 'replicator'" | grep -q 1; then
  psql -v ON_ERROR_STOP=1 --username "postgres" -c "CREATE ROLE replicator WITH REPLICATION PASSWORD '123' LOGIN;"
fi

if ! psql -U postgres -Atqc "SELECT 1 FROM pg_database WHERE datname = 'losiento'" | grep -q 1; then
  psql -v ON_ERROR_STOP=1 --username "postgres" -f "/home/scripts/init-db.sql"
fi

cp /etc/postgresql/postgresql.conf "$PGDATA/postgresql.conf"
cp /etc/postgresql/pg_hba.conf "$PGDATA/pg_hba.conf"

echo "Configuration files copied!"
