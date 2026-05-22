-- Ethics PDF OCR + RAG schema
-- Run once against the `ethics` database before ingesting.
--   psql -d ethics -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source PDF
CREATE TABLE IF NOT EXISTS ethics_documents (
    id           SERIAL PRIMARY KEY,
    filename     TEXT NOT NULL UNIQUE,
    page_count   INT,
    char_count   INT,
    ocr_model    TEXT,
    full_text    TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per chunk; embedding dim 768 matches nomic-embed-text.
-- If you swap embedding models, change the vector(N) dimension to match.
CREATE TABLE IF NOT EXISTS ethics_chunks (
    id            SERIAL PRIMARY KEY,
    document_id   INT NOT NULL REFERENCES ethics_documents(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    content       TEXT NOT NULL,
    token_count   INT,
    embedding     vector(768),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

-- HNSW index for fast cosine similarity search.
-- (Requires pgvector >= 0.5.0. If your version is older, switch to
--  `USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)`.)
CREATE INDEX IF NOT EXISTS ethics_chunks_embedding_hnsw
    ON ethics_chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS ethics_chunks_document_id_idx
    ON ethics_chunks (document_id);
