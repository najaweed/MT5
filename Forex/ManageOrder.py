import MetaTrader5 as mt5
import pandas as pd
import numpy as np

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()


class ManageOrder:
    def __init__(self,
                 c_symbol,
                 ):
        self.symbol = c_symbol

    def get_opened_positions(self):
        positions = mt5.positions_get(symbol=self.symbol)
        sym_positions = []
        for i_pos, pos in enumerate(positions):
            df = pd.DataFrame(pos._asdict().items(), columns=['index', 'value'])
            # print(df)
            df = df.set_index('index')
            sym_positions.append(df)

        return sym_positions

    def sending_request(self, request, max_try: int = 10):
        request['volume'] = np.round(request['volume'], 3)
        for i in range(max_try):

            order_req = mt5.order_send(request)
            print(order_req)
            if order_req is not None:
                if order_req.retcode == mt5.TRADE_RETCODE_DONE:
                    return True

                else:
                    # TODO based on error try to fix and send again order
                    request['volume'] = 0.01  # np.round(request['volume'], 2)
                    pass
            if i == max_try - 1:
                # print('error  ', self.symbol, order_req.retcode)
                return False

    def close_opened_position(self, ):
        df_positions = self.get_opened_positions()

        for pos in df_positions:

            deal_type = 0.
            deal_price = 0.
            last_deal_type = pos.loc['type'][0]

            last_deal_volume = pos.loc['volume'][0]

            if last_deal_type == mt5.ORDER_TYPE_BUY:
                deal_type = mt5.ORDER_TYPE_SELL
                deal_price = mt5.symbol_info_tick(self.symbol).bid

            elif last_deal_type == mt5.ORDER_TYPE_SELL:
                deal_type = mt5.ORDER_TYPE_BUY
                deal_price = mt5.symbol_info_tick(self.symbol).ask

            position_id = pos.loc['ticket'][0]
            deviation = 20

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": last_deal_volume,
                "type": deal_type,
                "position": position_id,
                "price": deal_price,
                "deviation": deviation,
                "magic": 234000,
                "comment": "python script close",
                "type_time": mt5.ORDER_TIME_SPECIFIED,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            self.sending_request(request)

    def is_opposite_opened_order(self):
        is_opposite = False
        df_positions = self.get_opened_positions()
        type_deals = [pos.loc['type'][0] for pos in df_positions]
        if len(type_deals) > 1:
            if all(element == type_deals[0] for element in type_deals):
                is_opposite = False
            else:
                is_opposite = True
                print('opposite in ', self.symbol)
        return is_opposite

    def prepare_request(self, p_request):
        if p_request['type'] == 'BUY':
            p_request['type'] = mt5.ORDER_TYPE_BUY
        elif p_request['type'] == 'SELL':
            p_request['type'] = mt5.ORDER_TYPE_SELL
        p_request['symbol'] = self.symbol
        return p_request

    def manage(self, p_request):
        # manage request for Only-ONE-deal trader
        p_request = self.prepare_request(p_request)
        positions = self.get_opened_positions()
        if len(positions) == 0:
            # open new position
            self.sending_request(p_request)
            return
        elif len(positions) == 1:
            if p_request['type'] == positions[0].loc['type'][0]:
                print('same side req')
                return
            elif p_request['type'] != positions[0].loc['type'][0]:
                self.close_opened_position()
                self.sending_request(p_request)
                return
        elif len(positions) > 1:
            self.close_opened_position()
            return

# def sample_order(deal_type='BUY'):
#     import pytz
#     from datetime import datetime
#     import MetaTrader5 as mt5
#     timezone = pytz.timezone("Etc/UTC")
#
#     utc_from = datetime(2021, 11, 1, hour=0, tzinfo=timezone)
#
#     # prepare the buy request structure
#     symbol = "AUDNZD.si"
#     symbol_info = mt5.symbol_info(symbol)
#     point = mt5.symbol_info(symbol).point
#
#     deal_t = None
#     price = None
#     sl = None
#     tp = None
#     if deal_type == 'SELL':
#         deal_t = mt5.ORDER_TYPE_SELL
#         price = mt5.symbol_info_tick(symbol).bid
#         sl = price + 100 * point
#         tp = price - 100 * point
#
#     elif deal_type == 'BUY':
#         deal_t = mt5.ORDER_TYPE_BUY
#         price = mt5.symbol_info_tick(symbol).ask
#         sl = price - 100 * point
#         tp = price + 100 * point
#     lot = 0.01
#     deviation = 20
#
#     request = {
#         "action": mt5.TRADE_ACTION_DEAL,
#         "symbol": symbol,
#         "volume": lot,
#         "type": deal_t,
#         "price": price,
#         "sl": sl,
#         "tp": tp,
#         "deviation": deviation,
#         "magic": 234000,
#         "comment": "python script open",
#         "type_time": mt5.ORDER_TIME_GTC,
#         "type_filling": mt5.ORDER_FILLING_IOC,
#     }
#     return request
#
#
# mg = ManageOrder("AUDNZD.si")
# for _ in range(100):
#     t = np.random.choice(['SELL', 'BUY'])
#     mg.manage(sample_order(t))
