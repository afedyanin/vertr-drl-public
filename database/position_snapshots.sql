-- Table: public.position_snapshots

-- DROP TABLE IF EXISTS public.position_snapshots;

CREATE TABLE IF NOT EXISTS public.position_snapshots
(
    time_utc timestamp with time zone NOT NULL,
    account_id text COLLATE pg_catalog."default" NOT NULL,
    instrument_uid text COLLATE pg_catalog."default",
    instrument_str text COLLATE pg_catalog."default",
    balance numeric NOT NULL,
    CONSTRAINT position_snapshots_pkey PRIMARY KEY (time_utc, account_id, instrument_uid, instrument_str)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.position_snapshots
    OWNER to postgres;