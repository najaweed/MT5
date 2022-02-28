import numpy as np
import pandas as pd
from Forex.Agent import SignalSeasonal
import MetaTrader5 as mt5


class AlgoAgent:
    def __init__(self,
                 step_df_rates: pd.DataFrame,
                 slow_fast=(0.0001, 0.1),
                 ):
        self.df = step_df_rates
        self.digit = 1e5  # ,#config['digit']
        self.time_intervals = [30, 20]
        self.ss_config = slow_fast  # config['slow_fast']

        self.ask_modes = self.get_modes('high')
        self.bid_modes = self.get_modes('low')

        self.ask_diff_mode = (self.ask_modes[1] - self.ask_modes[0]) * self.digit
        self.bid_diff_mode = (self.bid_modes[1] - self.bid_modes[0]) * self.digit

    def get_modes(self, price_type: str, ):
        prices = self.df.loc[:, price_type].to_numpy()
        return SignalSeasonal(self.ss_config).trade_modes(prices)

    def _position(self, threshold=(20, 20)):
        if self.ask_diff_mode[-1] > threshold[0]:
            # print('here')
            if self.ask_diff_mode[-1] < self.ask_diff_mode[-2]:
                # print('sell zone')
                return "SELL"

        elif self.bid_diff_mode[-1] < -threshold[1]:
            if self.bid_diff_mode[-1] > self.bid_diff_mode[-2]:
                # print('buy zone')
                return "BUY"
        else:
            return False

    def _price_deal(self):
        alpha_point = 0.0

        if self._position() == 'SELL':
            return self.df.loc[self.df.index[-1], 'close'] + alpha_point
        elif self._position() == 'BUY':
            return self.df.loc[self.df.index[-1], 'close'] - alpha_point
        else:
            return False

    def _stop_loss_estimate(self, st_point=50.):
        # calculate stop loss price alpha_point
        if self._position() == 'SELL':
            st_point = abs(self.ask_modes[1][-1] - self.ask_modes[0][-1])
            st_point = np.clip(st_point,a_min=0.00015,a_max=0.00080)

            return self._price_deal() + 1 * st_point
        elif self._position() == 'BUY':
            st_point = abs(self.bid_modes[1][-1] - self.bid_modes[0][-1])
            st_point = np.clip(st_point,a_min=0.00015,a_max=0.00080)

            return self._price_deal() - 1 * st_point
        else:
            return None

    def _take_profit_estimate(self, tp_point=50):

        # calculate take profit alpha_point
        if self._position() == 'SELL':
            tp_point = abs(self.bid_modes[1][-1] - self.bid_modes[0][-1])
            tp_point = np.clip(tp_point,a_min=0.00015,a_max=0.00080)

            return self._price_deal() - tp_point

        elif self._position() == 'BUY':
            tp_point = abs(self.ask_modes[1][-1] - self.ask_modes[0][-1])
            tp_point = np.clip(tp_point, a_min=0.00015, a_max=0.00080)
            return self._price_deal() + tp_point

        else:
            return None

    def _price_deviation(self):
        # calculate price deviation
        return 5

    def _time_expire_order(self, time_delay=2):

        return  self.time_intervals[-1]

    def _volume_lot(self):

        return 0.01

    def take_action(self, params=(20.03, 20.03, 30, 30, 2)):

        if not self._position((params[0], params[1])):
            # return {"type":"CLOSE"}
            return {}
        elif self._price_deal():

            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                # "symbol": self.symbol,
                "volume": self._volume_lot(),
                "type": self._position((params[0], params[1])),
                "price": self._price_deal(),
                "sl": self._stop_loss_estimate(params[2]),
                "tp": self._take_profit_estimate(params[3]),
                "deviation": self._price_deviation(),
                "magic": 234000,
                "comment": "python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
                # "expiration": self._time_expire_order(params[4]),
            }

            return req
        else:
            return {}




# import pandas as pd
# import matplotlib.pyplot as plt
# from Forex.Market import LiveTicks
# WINDOW = (1 * 6 * 60)
#
# live = LiveTicks({'symbol':'EURUSD.si' , 'window': 3 *WINDOW})
# df = live.get_rates()
#
# import time
# s_time = time.time()
# for _ in range(1000):
#     AlgoAgent(df,(0.001,0.01,0.1)).take_action()
# print((time.time()-s_time)/1000)

# al = AlgoAgent(df,(0.001,0.01,0.1))
#
#
# def numpy_ewma( data, alpha):
#     # alpha = 2 / (window + 1.0)
#     # scale = 1 / (1 - alpha)
#     n = data.shape[0]
#     scale_arr = (1 - alpha) ** (-1 * np.arange(n))
#     weights = (1 - alpha) ** np.arange(n)
#     pw0 = (1 - alpha) ** (n - 1)
#     mult = data * pw0 * scale_arr
#     cumsums = mult.cumsum()
#     out = cumsums * scale_arr[::-1] / weights.cumsum()
#
#     return out
#
# plt.plot(df.index, df['close'])
# modes = al.ask_modes
# f_modes = (0.001,0.01,0.1)
# for i in range(len(modes)):
#     plt.plot(df.index, modes[i])
#     plt.plot(df.index, numpy_ewma( df['high'].to_numpy(),f_modes[i]), c='r')
#
# plt.show()
