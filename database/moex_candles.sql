-- Table: public.moex_candles

-- DROP TABLE IF EXISTS public.moex_candles;

CREATE TABLE IF NOT EXISTS public.moex_candles
(
    time_utc timestamp with time zone NOT NULL,
    "interval" integer NOT NULL,
    symbol text COLLATE pg_catalog."default" NOT NULL,
    open numeric,
    close numeric,
    high numeric,
    low numeric,
    value numeric,
    volume numeric,
    CONSTRAINT moex_candles_unique UNIQUE (time_utc, "interval", symbol)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.moex_candles
    OWNER to postgres;