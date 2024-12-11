-- Table: public.operations

DROP TABLE IF EXISTS public.operations;

CREATE TABLE IF NOT EXISTS public.operations
(
    id uuid NOT NULL,
    account_id text COLLATE pg_catalog."default" NOT NULL,
    parent_operation_id text COLLATE pg_catalog."default",
    currency text COLLATE pg_catalog."default",
    payment numeric,
    price numeric,
    state integer NOT NULL,
    quantity integer,
    quantity_rest integer,
    figi text COLLATE pg_catalog."default",
    instrument_type text COLLATE pg_catalog."default",
    date timestamp with time zone NOT NULL,
    type text COLLATE pg_catalog."default",
    operation_type integer NOT NULL,
    asset_uid text COLLATE pg_catalog."default",
    position_uid text COLLATE pg_catalog."default",
    instrument_uid uuid,
    operation_json json,
    CONSTRAINT operations_pkey PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.operations
    OWNER to postgres;

