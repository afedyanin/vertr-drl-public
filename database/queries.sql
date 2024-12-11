-- get signals
SELECT * FROM public.signals ORDER BY time_utc DESC;

-- get orders
SELECT * FROM public.orders ORDER BY server_time DESC;

-- get operations
SELECT * FROM public.operations
where account_id = '7b65b669-ae2e-4909-9487-412945cda9c5'
ORDER BY date DESC

-- get trades
SELECT
op.date,
tr.*
FROM public.trades tr
JOIN public.operations op ON op.id = tr.operation_id
where op.account_id = '7b65b669-ae2e-4909-9487-412945cda9c5'
ORDER BY op.date DESC


