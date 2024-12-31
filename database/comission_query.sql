SELECT SUM(payment) FROM public.operations
WHERE account_id = 'fc66cf9b-8fb8-4d9e-ba79-a5e8b87c5aa7'
AND operation_type = 19
--ORDER BY date DESC
