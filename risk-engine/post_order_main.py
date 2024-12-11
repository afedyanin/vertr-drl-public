import sys
import uuid

from orders_sql_adapter import OrdersSqlAdapter

sys.path.append("../airflow/plugins")

from db_connection import DbConnection
from domain_model import Instrument
from tinvest_sandbox_adapter import TinvestSandboxAdapter

if __name__ == '__main__':
    lots = int(sys.argv[1])
    if lots is None:
        print(f"Quantity lots first argument must be specified.")
        exit()

    print(f"Posting new order lots={lots}...")

    db_connection = DbConnection.local_db_connection()
    instrument = Instrument.get_instrument("SBER")
    tinvest_adapter = TinvestSandboxAdapter()
    orders_adapter = OrdersSqlAdapter(db_connection)
    order_request_id = uuid.uuid4()

    print(f'Posting new order to T-invest API. order_request_id={order_request_id} qty={lots}')
    order = tinvest_adapter.post_market_order(order_request_id, instrument, lots)
    orders_adapter.insert_order(tinvest_adapter.account_id, order)
    print(f'New order posted. order_id={order.order_id}')

    position = tinvest_adapter.get_positions()
    print(f'New position={position}')


