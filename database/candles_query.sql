select *
from moex_candles
where interval = 60
order by time_utc DESC
limit 10

