-- Hybrid RAG Search System Database Schema
-- Requires PostgreSQL 15+ with pgvector and pg_trgm extensions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL UNIQUE,
    content TEXT NOT NULL,
    format VARCHAR(10) NOT NULL CHECK (format IN ('txt', 'md', 'json')),
    file_size INTEGER NOT NULL CHECK (file_size > 0 AND file_size <= 10485760),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT content_not_empty CHECK (length(trim(content)) > 0)
);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    embedding vector(768) NOT NULL,
    search_vector tsvector NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT unique_chunk_per_document UNIQUE (document_id, chunk_index)
);

-- Indexes for search performance
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_trgm ON chunks
    USING gin (content gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks
    USING gin (search_vector);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);

-- Trigger to update search_vector automatically
CREATE OR REPLACE FUNCTION update_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('simple', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_search_vector_update ON chunks;
CREATE TRIGGER chunks_search_vector_update
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_search_vector();

-- Trigger to update document updated_at
CREATE OR REPLACE FUNCTION update_document_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_updated_at ON documents;
CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_document_timestamp();
