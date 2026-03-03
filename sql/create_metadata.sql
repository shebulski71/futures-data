-- Schema for futures lakehouse metadata (phase 1)

CREATE SCHEMA IF NOT EXISTS md;

-- Each ingest monthly catalog file we processed (for auditing/replay)
CREATE TABLE IF NOT EXISTS md.symbology_catalog_month (
  root            text        NOT NULL,
  month           date        NOT NULL,   -- first day of month
  dataset         text        NOT NULL,
  parent_symbol   text        NOT NULL,
  catalog_path    text        NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (root, month, dataset)
);

-- One row per (root, contract symbol, instrument_id, validity range)
-- We store the raw validity (d0/d1) from Databento.
CREATE TABLE IF NOT EXISTS md.contracts (
  dataset         text        NOT NULL,
  root            text        NOT NULL,
  symbol          text        NOT NULL,   -- e.g., ESH5
  instrument_id   integer     NOT NULL,
  valid_from      date        NOT NULL,
  valid_to        date        NOT NULL,
  is_spread       boolean     NOT NULL DEFAULT false,
  first_seen_month date       NOT NULL,
  last_seen_month  date       NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (dataset, root, symbol, instrument_id, valid_from, valid_to)
);

-- Helpful indexes for querying
CREATE INDEX IF NOT EXISTS contracts_root_symbol_idx
  ON md.contracts (root, symbol);

CREATE INDEX IF NOT EXISTS contracts_instrument_idx
  ON md.contracts (instrument_id);

CREATE INDEX IF NOT EXISTS contracts_validity_idx
  ON md.contracts (root, valid_from, valid_to);

-- Simple trigger to keep updated_at fresh
CREATE OR REPLACE FUNCTION md.set_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_contracts_updated_at ON md.contracts;
CREATE TRIGGER trg_contracts_updated_at
BEFORE UPDATE ON md.contracts
FOR EACH ROW EXECUTE FUNCTION md.set_updated_at();
