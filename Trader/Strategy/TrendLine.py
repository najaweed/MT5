import pandas as pd
from scipy.signal import argrelextrema
import numpy as np

import datetime
from dateutil.relativedelta import *

from dateutil.relativedelta import relativedelta


class TrendLine:
    def __init__(self,
                 df,
                 time_frame,
                 # windows,
                 ):
        self.df = df
        self.tf = time_frame
        # self.windows = windows
        self.trends_index = {}
        self.step_predict = 3
        self._low_trend()
        self._high_trend()

    def _low_trend(self):
        for order in reversed(range(10, 11)):
            # for win in self.windows:
            low = self.df.low

            low_index = argrelextrema(-low.to_numpy(), np.greater, order=order)[0]

            # print(order,low_index)
            if len(low_index) == 2:
                alpha = (low[low_index[1]] - low[low_index[0]]) / (low_index[1] - low_index[0])
                support = low[low_index]
                support[low.index[-1]] = low[low_index[1]] + alpha * (low.shape[0] - low_index[1])
                self.trends_index[f'low'] = support
                break

    def _high_trend(self):
        for order in reversed(range(10, 11)):
            # for win in self.windows:
            high = self.df.high
            h = high.nlargest(2)
            print((h.index[1] - h.index[0]).days)
            print(high.nlargest(2))

            high_index = argrelextrema(high.to_numpy(), np.greater, order=order)[0]
            # print(order,high_index)
            if len(high_index) == 2:
                alpha = (high[high_index[1]] - high[high_index[0]]) / (high_index[1] - high_index[0])
                resistance = high[high_index]
                resistance[high.index[-1]] = high[high_index[1]] + alpha * (high.shape[0] - high_index[1])
                self.trends_index[f'high'] = resistance
            break


from Forex.Market import LiveTicks
import time
import MetaTrader5 as mt5
import matplotlib.pyplot as plt

window = 90000
trader_config = {
    "symbol": 'XAUUSD',
    "window": window,
    "digit": 1e5,
    "window_imb": 256,
    "volume_threshold": 50,
    # "agent": Agent(SimpleStrategy),

}
time_frames = {
    '01min': mt5.TIMEFRAME_M1,
    '02min': mt5.TIMEFRAME_M2,
    '03min': mt5.TIMEFRAME_M3,
    '04min': mt5.TIMEFRAME_M4,
    '05min': mt5.TIMEFRAME_M5,
    '06min': mt5.TIMEFRAME_M6,
    '10min': mt5.TIMEFRAME_M10,
    '12min': mt5.TIMEFRAME_M12,
    '15min': mt5.TIMEFRAME_M15,
    '20min': mt5.TIMEFRAME_M20,
    '30min': mt5.TIMEFRAME_M30,
    '01H': mt5.TIMEFRAME_H1,
    '02H': mt5.TIMEFRAME_H2,
    '03H': mt5.TIMEFRAME_H3,
    '04H': mt5.TIMEFRAME_H4,
    '06H': mt5.TIMEFRAME_H6,
    '08H': mt5.TIMEFRAME_H8,
    '12H': mt5.TIMEFRAME_H12,
    '1D': mt5.TIMEFRAME_D1,
    '1W': mt5.TIMEFRAME_W1,

}

live_trader = LiveTicks(trader_config)
rates = live_trader.get_rates(time_frame=mt5.TIMEFRAME_D1)
print(rates.high)
breakpoint()

#
# for window in [120, 60, 30]:
#     trader_config = {
#         "symbol": 'XAUUSD',
#         "window": window,
#         "digit": 1e5,
#         "window_imb": 256,
#         "volume_threshold": 50,
#         # "agent": Agent(SimpleStrategy),
#
#     }
#     time_frames = {
#         '01min': mt5.TIMEFRAME_M1,
#         '02min': mt5.TIMEFRAME_M2,
#         '03min': mt5.TIMEFRAME_M3,
#         '04min': mt5.TIMEFRAME_M4,
#         '05min': mt5.TIMEFRAME_M5,
#         '06min': mt5.TIMEFRAME_M6,
#         '10min': mt5.TIMEFRAME_M10,
#         '12min': mt5.TIMEFRAME_M12,
#         '15min': mt5.TIMEFRAME_M15,
#         '20min': mt5.TIMEFRAME_M20,
#         '30min': mt5.TIMEFRAME_M30,
#         '01H': mt5.TIMEFRAME_H1,
#         '02H': mt5.TIMEFRAME_H2,
#         '03H': mt5.TIMEFRAME_H3,
#         '04H': mt5.TIMEFRAME_H4,
#         '06H': mt5.TIMEFRAME_H6,
#         '08H': mt5.TIMEFRAME_H8,
#         '12H': mt5.TIMEFRAME_H12,
#         '1D': mt5.TIMEFRAME_D1,
#         '1W': mt5.TIMEFRAME_W1,
#
#     }
#
#     live_trader = LiveTicks(trader_config)
#     rates = live_trader.get_rates(time_frame=mt5.TIMEFRAME_W1)
#     trend_line = TrendLine(df=rates, time_frame='W', step_predict=20)
#
#     plt.plot(rates.close)
#     # plt.plot(rates.high)
#     # plt.plot(rates.low)
#
#     lows = trend_line._low_trend()
#     for low in lows:
#         plt.plot(low, '--', c='b')
#     plt.plot(trend_line._mid_trend())
#
#     highs = trend_line._high_trend()
#     for high in highs:
#         plt.plot(high, '--', c='r')
#
# for window in [10 * 24, 120, 60]:
#     trader_config = {
#         "symbol": 'XAUUSD',
#         "window": window,
#         "digit": 1e5,
#         "window_imb": 256,
#         "volume_threshold": 50,
#         # "agent": Agent(SimpleStrategy),
#
#     }
#     time_frames = {
#         '01min': mt5.TIMEFRAME_M1,
#         '02min': mt5.TIMEFRAME_M2,
#         '03min': mt5.TIMEFRAME_M3,
#         '04min': mt5.TIMEFRAME_M4,
#         '05min': mt5.TIMEFRAME_M5,
#         '06min': mt5.TIMEFRAME_M6,
#         '10min': mt5.TIMEFRAME_M10,
#         '12min': mt5.TIMEFRAME_M12,
#         '15min': mt5.TIMEFRAME_M15,
#         '20min': mt5.TIMEFRAME_M20,
#         '30min': mt5.TIMEFRAME_M30,
#         '01H': mt5.TIMEFRAME_H1,
#         '02H': mt5.TIMEFRAME_H2,
#         '03H': mt5.TIMEFRAME_H3,
#         '04H': mt5.TIMEFRAME_H4,
#         '06H': mt5.TIMEFRAME_H6,
#         '08H': mt5.TIMEFRAME_H8,
#         '12H': mt5.TIMEFRAME_H12,
#         '1D': mt5.TIMEFRAME_D1,
#         '1W': mt5.TIMEFRAME_W1,
#
#     }
#
#     live_trader = LiveTicks(trader_config)
#     rates = live_trader.get_rates(time_frame=mt5.TIMEFRAME_MN1)
#     trend_line = TrendLine(df=rates, time_frame='M', step_predict=20)
#
#     plt.plot(rates.close)
#     plt.plot(rates.high)
#     plt.plot(rates.low)
#
#     lows = trend_line._low_trend()
#     for low in lows:
#         plt.plot(low, '.-', c='b')
#     plt.plot(trend_line._mid_trend())
#
#     highs = trend_line._high_trend()
#     for high in highs:
#         plt.plot(high, '.-', c='r')


def _gold_plot(df_rates):
    # fplt.candle_bull_body_color = '#26a69a'
    # fplt.background = '#000'
    # setup dark mode, and red/green candles
    import finplot as fplt

    w = fplt.foreground = '#eef'
    b = fplt.background = fplt.odd_plot_background = '#242320'
    fplt.candle_bull_color = fplt.volume_bull_color = fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#352'
    fplt.candle_bear_color = fplt.volume_bear_color = '#810'

    fplt.cross_hair_color = w + 'a'

    # plot renko + renko-transformed volume

    # ax1, ax2 ,ax3 = fplt.create_plot('US Brent Oil Renko [dark mode]', rows=3, maximize=False)
    # fplt.candlestick_ochl( df_rates[['open', 'close', 'high', 'low']],)
    fplt.candlestick_ochl(df_rates[['open', 'close', 'high', 'low']], )
    trend_line = TrendLine(df=rates, time_frame='min', )
    print(trend_line.trends_index)
    # for win in [2000, ]:
    high_trends = trend_line.trends_index[f'high']
    low_trends = trend_line.trends_index[f'low']

    print(high_trends)
    #
    # print(support)
    print(low_trends)
    # print(resistance)
    # fplt.set_y_range(1850,2000)
    # fplt.set_y_scale('linear')
    fplt.add_line((high_trends.index[0], high_trends[0]), (high_trends.index[1], high_trends[1]),
                  color='#9900ff',
                  interactive=True)
    fplt.add_line((high_trends.index[1], high_trends[1]), (high_trends.index[2], high_trends[2]),
                  color='#9900ff',
                  interactive=True)

    fplt.add_line((low_trends.index[0], low_trends[0]), (low_trends.index[1], low_trends[1]),
                  color='#9900ff',
                  interactive=True)
    fplt.add_line((low_trends.index[1], low_trends[1]), (low_trends.index[2], low_trends[2]),
                  color='#9900ff',
                  interactive=True)
    fplt.show()


_gold_plot(rates)
