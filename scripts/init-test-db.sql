-- Runs once when the Postgres volume is first created.
-- A separate database keeps integration tests from touching dev data.
CREATE DATABASE mpc_test OWNER mpc;
\connect mpc
CREATE EXTENSION IF NOT EXISTS vector;
\connect mpc_test
CREATE EXTENSION IF NOT EXISTS vector;
