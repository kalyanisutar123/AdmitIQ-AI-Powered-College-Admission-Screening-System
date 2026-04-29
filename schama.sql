-- ============================================
-- AI College Admission System — SQL Schema
-- File: database/schema.sql
-- ============================================

-- Students table: stores every application and its ML prediction
CREATE TABLE IF NOT EXISTS students (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT    NOT NULL,
    email              TEXT,
    age                INTEGER,
    gender             TEXT,
    marks_10           REAL    NOT NULL,
    marks_12           REAL    NOT NULL,
    entrance_score     REAL    NOT NULL,
    preferred_course   TEXT,
    admitted           INTEGER NOT NULL DEFAULT 0,  -- 0=No, 1=Yes
    recommended_course TEXT,
    confidence         REAL,                         -- ML confidence %
    submitted_at       TEXT    DEFAULT (datetime('now'))
);

-- Users table: for login (admin + future student accounts)
CREATE TABLE IF NOT EXISTS users (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,             -- SHA-256 hash
    role     TEXT DEFAULT 'student'     -- 'admin' or 'student'
);

-- ── Sample queries ──

-- Get all admitted students ordered by 12th marks
SELECT name, marks_12, entrance_score, recommended_course
FROM students
WHERE admitted = 1
ORDER BY marks_12 DESC;

-- Acceptance rate
SELECT
    COUNT(*)             AS total,
    SUM(admitted)        AS admitted,
    ROUND(100.0 * SUM(admitted) / COUNT(*), 1) AS accept_rate_pct
FROM students;

-- Average scores by course
SELECT recommended_course,
       ROUND(AVG(marks_10),1) AS avg_10,
       ROUND(AVG(marks_12),1) AS avg_12,
       ROUND(AVG(entrance_score),1) AS avg_ent,
       COUNT(*) AS count
FROM students
WHERE admitted = 1
GROUP BY recommended_course
ORDER BY avg_12 DESC;

-- Course distribution
SELECT recommended_course, COUNT(*) AS applications
FROM students
GROUP BY recommended_course;
