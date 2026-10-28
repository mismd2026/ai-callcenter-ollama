-- Multi-tenant schema for the AI call center (PostgreSQL 16 + pgvector).
--
-- Isolation strategy: shared tables + tenant_id column + Row-Level Security.
-- The application connects as role `app_rw` and sets `app.tenant_id` per request,
-- so even a query that forgets its WHERE clause cannot read another tenant's rows.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE tenants (
    id              TEXT PRIMARY KEY CHECK (id ~ '^[a-z0-9][a-z0-9-]{1,31}$'),
    name            TEXT        NOT NULL,
    plan            TEXT        NOT NULL DEFAULT 'standard',
    max_concurrent  INT         NOT NULL DEFAULT 4,      -- concurrent AI calls allowed
    faq_direct_thr  REAL        NOT NULL DEFAULT 0.86,
    rag_context_thr REAL        NOT NULL DEFAULT 0.70,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE faq_entries (
    tenant_id   TEXT        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    id          TEXT        NOT NULL,
    category    TEXT        NOT NULL DEFAULT 'general',
    question    TEXT        NOT NULL,
    answer      TEXT        NOT NULL,
    embedding   vector(768) NOT NULL,                     -- multilingual-e5-base
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id)
);
CREATE INDEX faq_embedding_hnsw ON faq_entries USING hnsw (embedding vector_cosine_ops);

CREATE TABLE calls (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     TEXT        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    caller_hash   TEXT,                                   -- salted hash, never the raw number
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at      TIMESTAMPTZ,
    final_layer   SMALLINT    CHECK (final_layer BETWEEN 1 AND 3),
    handoff_reason TEXT
);
CREATE INDEX calls_tenant_started ON calls (tenant_id, started_at DESC);

CREATE TABLE call_turns (
    call_id     UUID        NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    tenant_id   TEXT        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    seq         INT         NOT NULL,
    transcript  TEXT        NOT NULL,
    reply       TEXT        NOT NULL,
    layer       SMALLINT    NOT NULL,
    reason      TEXT        NOT NULL,
    sources     TEXT[]      NOT NULL DEFAULT '{}',
    stt_ms      INT, policy_ms INT, tts_ms INT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (call_id, seq)
);

-- ---------------------------------------------------------------- row-level security
ALTER TABLE faq_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE calls       ENABLE ROW LEVEL SECURITY;
ALTER TABLE call_turns  ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON faq_entries
    USING (tenant_id = current_setting('app.tenant_id', true));
CREATE POLICY tenant_isolation ON calls
    USING (tenant_id = current_setting('app.tenant_id', true));
CREATE POLICY tenant_isolation ON call_turns
    USING (tenant_id = current_setting('app.tenant_id', true));

-- Application role: no BYPASSRLS, no DDL.
DO $$ BEGIN
    CREATE ROLE app_rw LOGIN;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
GRANT SELECT, INSERT, UPDATE ON faq_entries, calls, call_turns TO app_rw;
GRANT SELECT ON tenants TO app_rw;

-- Retention: transcripts are personal data; purge after 90 days (run from a CronJob).
-- DELETE FROM calls WHERE started_at < now() - interval '90 days';
