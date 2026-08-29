#!/bin/bash
while true; do
    psql -U postgres -d losiento -c "SELECT * FROM employees;"
    psql -U postgres -d losiento -c "SELECT * FROM projects;"
    psql -U postgres -d losiento -c "SELECT * FROM tasks;"
    sleep 2
done
