#!/bin/bash

while true; do
    # 1. Добавляем нового сотрудника
    departments=("IT" "HR" "Sales" "Finance")
    random_department=${departments[$RANDOM % ${#departments[@]}]}
    employee_name="Сотрудник_$(date +%s)"

    psql -U postgres -d losiento -c \
        "INSERT INTO employees (full_name, department) VALUES ('$employee_name', '$random_department');"

    # 2. Добавляем новый проект
    project_title="Проект_$(date +%s)"
    budget=$(( (RANDOM % 90 + 10) * 10000 ))

    psql -U postgres -d losiento -c \
        "INSERT INTO projects (title, budget) VALUES ('$project_title', $budget);"

    # 3. Берём id последнего проекта
    last_project_id=$(psql -U postgres -d losiento -t -A -c \
        "SELECT id FROM projects ORDER BY id DESC LIMIT 1;")

    # 4. Добавляем задачу к последнему проекту
    statuses=("new" "in_progress" "done")
    random_status=${statuses[$RANDOM % ${#statuses[@]}]}
    task_description="Задача_$(date +%s)"

    psql -U postgres -d losiento -c \
        "INSERT INTO tasks (project_id, description, status) VALUES ($last_project_id, '$task_description', '$random_status');"

    sleep 2
done