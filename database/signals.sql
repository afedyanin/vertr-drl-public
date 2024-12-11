-- Table: public.signals

DROP TABLE IF EXISTS public.signals;

CREATE TABLE IF NOT EXISTS public.signals
(
    order_request_id uuid NOT NULL,
    time_utc timestamp with time zone NOT NULL,
    quantity integer NOT NULL,
    symbol text COLLATE pg_catalog."default" NOT NULL,
    origin text COLLATE pg_catalog."default" NOT NULL,
    prediction json NOT NULL,
    CONSTRAINT signals_pkey PRIMARY KEY (order_request_id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.signals
    OWNER to postgres;