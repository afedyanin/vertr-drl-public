from db_connection import DbConnection
from domain_model import Instrument
from orders_sql_adapter import OrdersSqlAdapter
from signals_sql_adapter import SignalsSqlAdapter
from tinvest_sandbox_adapter import TinvestSandboxAdapter


class TinvestRiskEngine:
    def __init__(
            self,
            db_connection: DbConnection,
            instrument: Instrument,
            account_id: str | None = None):
        self.db_connection = db_connection
        self.instrument = instrument
        self._tinvest_adapter = TinvestSandboxAdapter(account_id)
        self.account_id = self._tinvest_adapter.account_id
        self._orders_adapter = OrdersSqlAdapter(self.db_connection)
        self._signals_adapter = SignalsSqlAdapter(self.db_connection)

    def post_order(self, quantity_lots: int = 20):

        last_signal = self._signals_adapter.get_last_signal()

        if last_signal is None:
            print('No new signal found. Skip executing.')
            return

        if last_signal.quantity == 0:
            print('No new orders required. Hold position.')
            return

        order_request_id = last_signal.order_request_id
        order = self._orders_adapter.get_order_by_request_id(order_request_id)

        if order is not None:
            print(f'Existing order found for order_request_id={order_request_id}. Skip executing.')
            return

        order_qty = last_signal.quantity * quantity_lots
        order_quantity = self._adjust_order_quantity(order_qty)

        if order_quantity == 0:
            print('No new orders required. Hold position.')
            return

        print(f'Posting new order to T-invest API. order_request_id={order_request_id}')
        order = self._tinvest_adapter.post_market_order(order_request_id, self.instrument, int(order_quantity))

        self._orders_adapter.insert_order(self.account_id, order)
        print(f'New order posted. order_id={order.order_id}')

    def _adjust_order_quantity(self, signal_quantity: int) -> int:

        position_balance = self._tinvest_adapter.get_position_balance(self.instrument)

        if position_balance is None or position_balance == 0:
            # just open position
            return signal_quantity

        position_balance_lots = position_balance / self.instrument.lot_size

        if signal_quantity > 0:
            if position_balance > 0:
                # same direction - keep position
                return 0
            else:
                # diff direction - swap position
                return position_balance_lots * (-1) + signal_quantity

        if signal_quantity < 0:
            if position_balance > 0:
                # diff direction swap position
                return position_balance_lots * (-1) + signal_quantity
            else:
                # same direction - keep position
                return 0


