import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


class TrendChannels:
    def __init__(self,
                 df_rates,
                 ):
        self.df = df_rates
        self.df = self._gen_tick_bar('1min')

    def get_trends(self):
        trend_points = {}

        for i in reversed(range(1, 60)):
            for tf in ['min', 'H']:
                df = self._gen_tick_bar(f'{i}{tf}')
                arg_high = argrelextrema(df.high.to_numpy(), np.greater, order=2)[0]
                arg_low = argrelextrema(-df.low.to_numpy(), np.greater, order=2)[0]
                if len(arg_high) >= 2:
                    points = self._get_resistance(df.high, arg_high, i, tf)
                    trend_points[f'{i},{tf},resistance'] = points
                if len(arg_low) >= 2:
                    points = self._get_supports(df.low, arg_low, i, tf)
                    trend_points[f'{i},{tf},support'] = points

        return trend_points

    def _gen_tick_bar(self, time_frame: str):
        df = self.df.open.resample(time_frame).first().to_frame()
        df['close'] = self.df.close.resample(time_frame).last()
        df['high'] = self.df.high.resample(time_frame).max()
        df['low'] = self.df.low.resample(time_frame).min()
        return df

    def _get_resistance(self, df, arg_df, i, tf):

        temp_index_1 = df.index[arg_df[-1]]
        range_price_1 = self.df.high[temp_index_1:(temp_index_1 + pd.Timedelta(i, unit=tf))]
        price_1 = df[arg_df[-1]]
        index_1 = range_price_1[range_price_1 == price_1].index[0]
        temp_index_2 = df.index[arg_df[-2]]
        range_price_2 = self.df.high[temp_index_2:(temp_index_2 + pd.Timedelta(i, unit=tf))]
        price_2 = df[arg_df[-2]]
        index_2 = range_price_2[range_price_2 == price_2].index[0]

        alpha = (df[arg_df[-1]] - df[arg_df[-2]]) / abs(
            (index_1 - index_2).total_seconds() / 60)
        t_end = abs((self.df.close.index[-1] - index_1).total_seconds() / 60)
        predict_price = df[arg_df[-1]] + alpha * t_end
        index_last = self.df.close.index[-1]
        points = [[(index_2, price_2), (index_1, price_1)],
                  [(index_1, price_1), (index_last, predict_price)],
                  [alpha, t_end]
                  ]
        return points

    def _get_supports(self, df, arg_df, i, tf):

        temp_index_1 = df.index[arg_df[-1]]
        range_price_1 = self.df.low[temp_index_1:(temp_index_1 + pd.Timedelta(i, unit=tf))]
        price_1 = df[arg_df[-1]]
        index_1 = range_price_1[range_price_1 == price_1].index[0]
        temp_index_2 = df.index[arg_df[-2]]
        range_price_2 = self.df.low[temp_index_2:(temp_index_2 + pd.Timedelta(i, unit=tf))]
        price_2 = df[arg_df[-2]]
        index_2 = range_price_2[range_price_2 == price_2].index[0]

        alpha = (df[arg_df[-1]] - df[arg_df[-2]]) / abs(
            (index_1 - index_2).total_seconds() / 60)
        t_end = abs((self.df.close.index[-1] - index_1).total_seconds() / 60)
        predict_price = df[arg_df[-1]] + alpha * t_end
        index_last = self.df.close.index[-1]
        points = [[(index_2, price_2), (index_1, price_1)],
                  [(index_1, price_1), (index_last, predict_price)],
                  [alpha, t_end]
                  ]
        return points


from Forex.Market import LiveTicks
import MetaTrader5 as mt5
import finplot as fplt

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
rates = live_trader.get_rates(time_frame=mt5.TIMEFRAME_M1)
print(rates)

trend_channels = TrendChannels(rates)
trs = trend_channels.get_trends()
r1m = trend_channels._gen_tick_bar('1min')
fplt.foreground = '#eef'
fplt.background = '#0a081b'
fplt.odd_plot_background = '#0a081b'
ax = fplt.create_plot('Things move', rows=1, init_zoom_periods=4 * 60, maximize=True)
fplt.candlestick_ochl(r1m[['open', 'close', 'high', 'low']].dropna(), ax=ax)
for k, v in trs.items():
    width = 4
    time_ = int(v[-1][-1])

    if time_ > 60*24:
        width = 25
    elif 60*24 > time_ > 60*6:
        width = 15
    elif 60*6 > time_ > 60:
        width = 8
    else:
        pass

    if k.split(',')[-1] == 'resistance':
        fplt.add_line(v[0][0], v[0][1], color='#360000', width=0.2, ax=ax)
        fplt.add_line(v[1][0], v[1][1], color='#4e0000', width=width, ax=ax)
    elif k.split(',')[-1] == 'support':
        fplt.add_line(v[0][0], v[0][1], color='#132a43', width=0.2, ax=ax)
        fplt.add_line(v[1][0], v[1][1], color='#162640', width=width, ax=ax)
fplt.show()
