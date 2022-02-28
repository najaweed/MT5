from Forex.Market import LiveTicks
import time
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import finplot as fplt
import pandas as pd
from scipy.signal import argrelextrema
import numpy as np

window = 99999
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
print(rates)

# w = fplt.foreground = '#eef'
# b = fplt.background = fplt.odd_plot_background = '#242320'
# fplt.candle_bull_color = fplt.volume_bull_color = fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#352'
# fplt.candle_bear_color = fplt.volume_bear_color = '#810'
# fplt.cross_hair_color = w + 'a'
from finplot import ColorMap


def _plot_gold_history():
    window = 99999
    trader_config = {
        "symbol": 'XAUUSD',
        "window": window,
        "digit": 1e5,
        "window_imb": 256,
        "volume_threshold": 50,
        # "agent": Agent(SimpleStrategy),

    }
    live_trader = LiveTicks(trader_config)
    rates = live_trader.get_rates(time_frame=mt5.TIMEFRAME_D1)

    tf = f'{1}W'
    dfdd = rates.open.resample(tf).first().to_frame()
    dfdd['close'] = rates.close.resample(tf).last()
    dfdd['high'] = rates.high.resample(tf).max()
    dfdd['low'] = rates.low.resample(tf).min()
    print(dfdd)
    fplt.candlestick_ochl(dfdd[['open', 'close', 'high', 'low']].dropna())
    for i in reversed(range(1, 24)):
        for t in ['D', 'W', 'M']:
            tf = f'{i}{t}'
            dfd = rates.open.resample(tf).first().to_frame()
            dfd['close'] = rates.close.resample(tf).last()
            dfd['high'] = rates.high.resample(tf).max()
            dfd['low'] = rates.low.resample(tf).min()
            # print(dfd)
            # fplt.candlestick_ochl(dfd[['open', 'close', 'high', 'low']].dropna())

            extr_high = argrelextrema(dfd.high.to_numpy(), np.greater, order=2)[0]
            extr_low = argrelextrema(-dfd.low.to_numpy(), np.greater, order=2)[0]

            # print(extr_high)
            # print(extr_low)
            if len(extr_high) >= 2:
                fplt.add_line((dfd.high.index[extr_high[-2]], dfd.high[extr_high[-2]]),
                              (dfd.high.index[extr_high[-1]], dfd.high[extr_high[-1]]),
                              color='#1f77b4',
                              interactive=False)

                alpha = (dfd.high[extr_high[-1]] - dfd.high[extr_high[-2]]) / abs(
                    (dfd.high.index[extr_high[-1]] - dfd.high.index[extr_high[-2]]).days)
                t_end = abs((dfd.high.index[extr_high[-1]] - dfdd.high.index[-1]).days)
                pr = dfd.high[extr_high[-1]] + alpha * t_end
                fplt.add_line((dfd.high.index[extr_high[-1]], dfd.high[extr_high[-1]]),
                              (dfdd.high.index[-1], pr),
                              color='#1f77b4',
                              interactive=False)

            if len(extr_low) >= 2:
                fplt.add_line((dfd.low.index[extr_low[-2]], dfd.low[extr_low[-2]]),
                              (dfd.low.index[extr_low[-1]], dfd.low[extr_low[-1]]),
                              color='#9467bd',
                              interactive=False)

                alpha = (dfd.low[extr_low[-1]] - dfd.low[extr_low[-2]]) / abs(
                    (dfd.low.index[extr_low[-1]] - dfd.low.index[extr_low[-2]]).days)
                t_end = abs((dfd.low.index[extr_low[-1]] - dfdd.low.index[-1]).days)
                pr = dfd.low[extr_low[-1]] + alpha * t_end
                fplt.add_line((dfd.low.index[extr_low[-1]], dfd.low[extr_low[-1]]),
                              (dfdd.low.index[-1], pr),
                              color='#9467bd',
                              interactive=False)
    fplt.show()


def _plot_gold_intraday():
    window = 99999
    trader_config = {
        "symbol": 'XAUUSD',
        "window": window,
        "digit": 1e5,
        "window_imb": 256,
        "volume_threshold": 50,
        # "agent": Agent(SimpleStrategy),

    }
    live_trader = LiveTicks(trader_config)
    rates = live_trader.get_rates(time_frame=mt5.TIMEFRAME_M1)
    fplt.foreground = '#eef'
    fplt.background = '#242320'
    fplt.odd_plot_background = '#242320'

    ax = fplt.create_plot('Things move', rows=1, init_zoom_periods=1000, maximize=False)

    tf = f'{1}min'
    dfdd = rates.open.resample(tf).first().to_frame()
    dfdd['close'] = rates.close.resample(tf).last()
    dfdd['high'] = rates.high.resample(tf).max()
    dfdd['low'] = rates.low.resample(tf).min()
    fplt.candlestick_ochl(dfdd[['open', 'close', 'high', 'low']].dropna(), ax=ax)

    for i in reversed(range(1, 240)):
        for t in ['H', 'min', ]:
            tf = f'{i}{t}'
            dfd = rates.open.resample(tf).first().to_frame()
            dfd['close'] = rates.close.resample(tf).last()
            dfd['high'] = rates.high.resample(tf).max()
            dfd['low'] = rates.low.resample(tf).min()

            extr_high = argrelextrema(dfd.high.to_numpy(), np.greater, order=2)[0]
            extr_low = argrelextrema(-dfd.low.to_numpy(), np.greater, order=2)[0]

            if len(extr_high) >= 2:

                inx2 = dfd.high.index[extr_high[-2]]
                s2 = dfdd.high[inx2:(inx2 + pd.Timedelta(i, unit=t))]
                index_2 = s2[s2 == dfd.high[extr_high[-2]]].index[0]

                inx1 = dfd.high.index[extr_high[-1]]
                s1 = dfdd.high[inx1:(inx1 + pd.Timedelta(i, unit=t))]
                index_1 = s1[s1 == dfd.high[extr_high[-1]]].index[0]

                fplt.add_line((index_2, dfd.high[extr_high[-2]]),
                              (index_1, dfd.high[extr_high[-1]]),
                              color='#eca5a6',
                              interactive=False,
                              width=1.2 if t == 'H' else 1)

                alpha = (dfd.high[extr_high[-1]] - dfd.high[extr_high[-2]]) / abs(
                    (index_1 - index_2).total_seconds() / 60)
                t_end = abs((dfdd.high.index[-1] - index_1).total_seconds() / 60)
                pr = dfd.high[extr_high[-1]] + alpha * t_end

                fplt.add_line((index_1, dfd.high[extr_high[-1]]),
                              (dfdd.high.index[-1], pr),
                              color='#df6668',
                              interactive=False,
                              width=6.2 if t == 'H' else 3)

            if len(extr_low) >= 2:

                inx2 = dfd.low.index[extr_low[-2]]
                s2 = dfdd.low[inx2:(inx2 + pd.Timedelta(i, unit=t))]
                index_2 = s2[s2 == dfd.low[extr_low[-2]]].index[0]

                inx1 = dfd.low.index[extr_low[-1]]
                s1 = dfdd.low[inx1:(inx1 + pd.Timedelta(i, unit=t))]
                index_1 = s1[s1 == dfd.low[extr_low[-1]]].index[0]

                fplt.add_line((index_2, dfd.low[extr_low[-2]]),
                              (index_1, dfd.low[extr_low[-1]]),
                              color='#a5caec',
                              interactive=False,
                              width=1.2 if t == 'H' else 1)

                alpha = (dfd.low[extr_low[-1]] - dfd.low[extr_low[-2]]) / abs((index_1 - index_2).total_seconds() / 60)
                t_end = abs((dfdd.low.index[-1] - index_1).total_seconds() / 60)
                pr = dfd.low[extr_low[-1]] + alpha * t_end

                fplt.add_line((index_1, dfd.low[extr_low[-1]]),
                              (dfdd.low.index[-1], pr),
                              color='#66a5df',
                              interactive=False,
                              width=6.2 if t == 'H' else 3)

    fplt.show()


# _plot_gold_history()
_plot_gold_intraday()
