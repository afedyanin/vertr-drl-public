-- Table: public.trades

DROP TABLE IF EXISTS public.trades;

CREATE TABLE IF NOT EXISTS public.trades
(
    trade_id uuid NOT NULL,
    operation_id uuid NOT NULL,
    operation_type integer NOT NULL,
    date_time timestamp with time zone NOT NULL,
    quantity integer NOT NULL,
    price numeric NOT NULL,
    trade_json json,
    CONSTRAINT trades_pkey PRIMARY KEY (trade_id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.trades
    OWNER to postgres;