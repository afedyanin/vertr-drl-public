-- Table: public.portfolio_snapshots

-- DROP TABLE IF EXISTS public.portfolio_snapshots;

CREATE TABLE IF NOT EXISTS public.portfolio_snapshots
(
    time_utc timestamp with time zone NOT NULL,
    account_id text COLLATE pg_catalog."default" NOT NULL,
    shares numeric,
    bonds numeric,
    futures numeric,
    options numeric,
    currencies numeric,
    portfolio numeric,
    expected_yield numeric,
    CONSTRAINT portfolio_snapshots_pkey PRIMARY KEY (time_utc, account_id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.portfolio_snapshots
    OWNER to postgres;