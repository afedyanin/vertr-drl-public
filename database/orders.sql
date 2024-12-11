-- Table: public.orders

DROP TABLE IF EXISTS public.orders;

CREATE TABLE IF NOT EXISTS public.orders
(
    order_id uuid NOT NULL,
    account_id text COLLATE pg_catalog."default" NOT NULL,
    order_request_id uuid,
    direction integer NOT NULL,
    executed_commission numeric,
    executed_order_price numeric,
    execution_report_status integer NOT NULL,
    initial_commission numeric,
    instrument_uid uuid NOT NULL,
    lots_executed integer,
    lots_requested integer,
    message text COLLATE pg_catalog."default",
    order_type integer NOT NULL,
    server_time timestamp with time zone NOT NULL,
    order_json json NOT NULL,
    CONSTRAINT orders_pkey PRIMARY KEY (order_id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.orders
    OWNER to postgres;