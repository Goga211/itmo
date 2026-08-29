CREATE DATABASE losiento;

\c losiento;

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    full_name TEXT,
    department TEXT
);

CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    title TEXT,
    budget INTEGER
);

CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    project_id INTEGER,
    description TEXT,
    status TEXT
);


INSERT INTO employees (full_name, department)
SELECT
    'Сотрудник ' || g,
    (ARRAY['IT','HR','Sales','Finance'])[1 + (g % 4)]
FROM generate_series(1, 10) g;

INSERT INTO projects (title, budget)
SELECT
    'Проект ' || g,
    g * 10000
FROM generate_series(1, 10) g;

INSERT INTO tasks (project_id, description, status)
SELECT
    1 + (g % 100),
    'Задача номер ' || g,
    (ARRAY['new','in_progress','done'])[1 + (g % 3)]
FROM generate_series(1, 20) g;