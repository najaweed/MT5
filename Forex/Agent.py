import numpy as np
import pandas as pd
import MetaTrader5 as mt5


class Agent:
    def __init__(self,
                 strategy,
                 ):
        self.strategy = strategy
        self.df = None
        self.signal = None

    def _position(self):

        if self.signal['position'] == 'SELL':
            return "SELL"

        elif self.signal['position'] == 'BUY':
            return "BUY"

        else:
            return False

    def _price_deal(self):

        if self._position() == 'SELL':
            return self.df.loc[self.df.index[-1], 'close']

        elif self._position() == 'BUY':
            return self.df.loc[self.df.index[-1], 'close']

        else:
            return False

    def _stop_loss(self, st_point=50.):
        if self._position() == 'SELL':
            if self.signal['st']:
                return self._price_deal() + self.signal['st']
            else:
                return self._price_deal() + 1 * st_point

        elif self._position() == 'BUY':
            if self.signal['st']:
                return self._price_deal() - self.signal['st']
            else:
                return self._price_deal() - 1 * st_point
        else:
            return None

    def _take_profit(self, tp_point=50):
        if self._position() == 'SELL':
            if self.signal['tp']:
                return self._price_deal() - self.signal['tp']
            else:
                return self._price_deal() - tp_point

        elif self._position() == 'BUY':
            if self.signal['tp']:
                return self._price_deal() + self.signal['tp']
            else:
                return self._price_deal() + tp_point
        else:
            return None

    def _price_deviation(self):
        return 5

    def _time_expire_order(self, time_delay=2):
        return False

    def _volume_lot(self):

        if self.signal['volume']:
            return self.signal['volume']
        else:
            return 0.01

    def take_action(self, step_data_frame):
        self.df = step_data_frame
        self.signal = self.strategy(step_data_frame).signal

        if not self._position():
            return {}
        elif self._price_deal():

            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                # "symbol": self.symbol,
                "volume": self._volume_lot(),
                "type": self._position(),
                "price": self._price_deal(),
                "sl": self._stop_loss(),
                "tp": self._take_profit(),
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
