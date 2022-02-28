from Forex.Market import LiveTicks
from Trader.Strategy.abcStrategy import Strategy


class MicroChannel(Strategy):
    def __init__(self,
                 step_df):
        self.df = step_df

    def _buy_zone(self):
        return False

    def _sell_zone(self):
        return False

    def _estimate_st(self):
        return 1

    def _estimate_tp(self):
        return 1

    def _estimate_volume(self):
        return 0.01

    def _micro_channel(self, max_len_time_tick: int = 12, max_high_low_spread: float = 0.50, time_frame: str = '5min'):

        for win in reversed(range(3, max_len_time_tick)):
            # print(win)
            low = self.df.low[-win:]
            high = self.df.high[-win:]
            # print(low, high)
            low_up_trend = np.all(np.diff(low.to_numpy()) >= 0)
            high_down_trend = np.all(np.diff(high.to_numpy()) <= 0)
            max_high_trend = np.all(abs(np.diff(high.to_numpy())) <= max_high_low_spread)
            high_low_spread = np.all((high - low).to_numpy() <= max_high_low_spread)
            if low_up_trend and high_low_spread and max_high_trend:
                diff_trend = (low[-1] - low[0])
                if time_frame[-3:] == 'min':
                    prediction = np.round(diff_trend / int(float(time_frame[:2])) * win + self.df.close[-1], 2)
                    print(
                        f'{time_frame} - Up Trend {win} ticks - {int(float(time_frame[:2]) * win)} min  | {np.round(diff_trend / int(float(time_frame[:2])), 3)} /min | {prediction} in {time_frame} ')
                elif time_frame[-1] == 'H':
                    prediction = np.round(diff_trend / int(float(time_frame[:2]) * 60) * win * 60 + self.df.close[-1],
                                          2)

                    print(
                        f'{time_frame} - Up Trend {win} ticks - {int(float(time_frame[:2]) * 60 * win)} min  | {np.round(diff_trend / (int(float(time_frame[:2]) * 60 * win)), 3)} /min |{prediction}in {time_frame} ')
                else:
                    print(f'{time_frame} - Up Trend {win} ticks   ')
                return
            elif high_down_trend and high_low_spread and max_high_trend:
                diff_trend = (high[-1] - high[0])
                if time_frame[-3:] == 'min':
                    prediction = np.round(diff_trend / int(float(time_frame[:2])) * win + self.df.close[-1], 2)

                    print(
                        f'{time_frame} - Down Trend {win} ticks - {int(float(time_frame[:2]) * win)} min  | {np.round(diff_trend / int(float(time_frame[:2])), 3)} /min  |{prediction} in {time_frame} ')
                elif time_frame[-1] == 'H':
                    prediction = np.round(diff_trend / int(float(time_frame[:2]) * 60) * win * 60 + self.df.close[-1],
                                          2)

                    print(
                        f'{time_frame} - Down Trend {win} ticks - {int(float(time_frame[:2]) * 60 * win)} min  | {np.round(diff_trend / (int(float(time_frame[:2]) * 60 * win)), 3)} /min |{prediction} in {time_frame} ')

                else:
                    print(f'{time_frame} - Down Trend {win} ticks   ')
                return
        print(f'{time_frame} - No Trend')


from Forex.Agent import Agent
from Trader.Strategy.SimpleStrategy import SimpleStrategy
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import mplfinance as mpf
import datetime
import pandas as pd
from scipy.signal import argrelextrema
import numpy as np
import finplot as fplt

# Find peaks in the window

trader_config = {
    "symbol": 'XAUUSD',
    "window": (1 * 1 * 24 * 24),
    "digit": 1e5,
    "window_imb": 256,
    "volume_threshold": 50,
    "agent": Agent(SimpleStrategy),

}
import time

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


# while True:
#     live_trader = LiveTicks(trader_config)
#     for time_frame, mt5_time_frame in time_frames.items():
#         rates = live_trader.get_rates(time_frame=mt5_time_frame)
#         MicroChannel(rates)._micro_channel(max_len_time_tick=20, max_high_low_spread=2000, time_frame=time_frame)
#
#     time.sleep(2)

def _gold_plot(df_rates):
    rates = df_rates
    fig, ax = plt.subplots()
    # rates['ewma_slow_1'] = (rates.loc[:,'close']).ewm(alpha=0.01).mean()
    # rates['ewma_slow_2'] = (rates.loc[:,'close']).ewm(alpha=0.05).mean()
    # rates['ewma_slow_3'] = (rates.loc[:,'close']).ewm(alpha=0.15).mean()
    # ax.plot(rates.index, rates['ewma_slow_1'],)
    # ax.plot(rates.index, rates['ewma_slow_2'],)
    # ax.plot(rates.index, rates['ewma_slow_3'],)
    # for i in range(2,20,5):
    #
    #     low_index = argrelextrema(rates['low'].values, np.less_equal, order=i)[0]
    #     plt.scatter(rates.index[low_index], rates.loc[rates.index[low_index], 'low'],s=i*40)
    #
    #     high_index = argrelextrema(rates['high'].values, np.greater_equal, order=i)[0]
    #     plt.scatter(rates.index[high_index], rates.loc[rates.index[high_index], 'high'],s=i*40)

    #
    # rates['min'] = rates.iloc[argrelextrema(rates.close.values, np.less_equal, order=n)[0]]['low']
    # rates['max'] = rates.iloc[argrelextrema(rates.close.values, np.greater_equal, order=n)[0]]['high']
    # plt.plot(rates['max'],'r')
    # plt.plot(rates['min'],'b')
    # plt.plot(rates.index, rates['close'], '.-')
    # plt.plot(rates.index, rates['high'], '.-')
    # plt.plot(rates.index, rates['low'], '.-')

    fplt.candle_bull_body_color = '#26a69a'
    # fplt.background = '#000'
    #
    # fplt.candlestick_ochl(rates[['open', 'close', 'high', 'low']], )
    # fplt.candlestick_ochl(rates[['open', 'close', 'high', 'low']], )

    # fplt.plot(rates.close.ewm(alpha=0.15).mean())
    print()
    print(rates.close.ewm(alpha=0.1, adjust=True).mean())
    emw = (rates.loc[:, 'close']).ewm(alpha=0.05).mean()
    fplt.candlestick_ochl(rates.index, emw)

    # fplt.plot(rates.index[-100:], rates.close[-100:].values)
    fplt.show()

    # end_days = pd.bdate_range(rates.index[0], rates.index[-1], freq='C', )

    # print(end_days)

    # for end_day in end_days:
    #     ax.axvline(end_day)
    #
    # ax.grid(True, alpha=0.5)
    #
    # plt.show()


def _line(self, df, index_1, index_2):
        alpha = (df[index_2] - df[index_1]) / (index_2 - index_1)
        print(self.df.shape[0])
        step_time = self.df.shape[0] + self.step_predict - index_1
        index_time = pd.date_range(df.index[index_1], periods=step_time, freq=self.tf)
        pindex_time = pd.date_range(df.index[-1], periods=step_time, freq=self.tf)

        iindex_time = pd.date_range(start=df.index[index_1], end=df.index[-1], freq=self.tf)

        t = np.linspace(0, len(iindex_time), len(iindex_time))

        p_line = pd.DataFrame(index=iindex_time)
        p_line['pre'] = df[index_1] + alpha * t

        return p_line

trader_config = {
    "symbol": 'XAUUSD',
    "window": (1 * 1 * 24 * 24),
    "digit": 1e5,
    "window_imb": 256,
    "volume_threshold": 50,
    "agent": Agent(SimpleStrategy),

}
import time

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


# print(rates)


def trend_line(df, step_predict=10):
    # print(df.shape[0])
    alpha = (df.close[-1] - df.open[0]) / df.shape[0]
    time = np.linspace(0, df.shape[0], df.shape[0])
    line = df.open[0] + alpha * time

    predict_time_steps = np.linspace(0, step_predict, step_predict)
    p_line = pd.DataFrame(index=pd.date_range(rates.index[-1], periods=step_predict, freq="M"))
    p_line['pre'] = df.close[-1] + alpha * predict_time_steps
    return line, p_line


from scipy.signal import argrelextrema


def _anti_trend(df, step_predict=110, ):
    for order in reversed(range(3, int(df.shape[0] / 2))):
        high_index = argrelextrema(-df.to_numpy(), np.greater, order=order)
        if len(high_index[0]) == 10:
            print(df.index[high_index])
            low_last = 0
            print(df.iloc[high_index])
            plt.figure(1)
            plt.scatter(df.index[high_index], df.iloc[high_index])
            alpha = (df.iloc[high_index][-1] - df.iloc[np.argmin(df)]) / (high_index[0][-1] - df.iloc[np.argmin(df)])
            print(alpha)
            print(df.index[high_index][0])
            p_line = pd.DataFrame(
                index=pd.date_range(df.iloc[np.argmin(df)], periods=df.shape[0] - df.iloc[np.argmin(df)] + step_predict,
                                    freq="D"))

            time = np.linspace(0, df.shape[0] - df.iloc[np.argmin(df)] + step_predict,
                               df.shape[0] - df.iloc[np.argmin(df)] + step_predict)
            p_line['pre'] = df.iloc[np.argmin(df)] + alpha * time
            plt.figure(1)

            plt.plot(p_line)
            print(order, alpha)
            break


def _aanti_trend(df, step_predict=110, ):
    for order in reversed(range(3, int(df.shape[0] / 2))):
        high_index = argrelextrema(df.to_numpy(), np.greater, order=order)
        if len(high_index[0]) == 10:
            print(df.index[high_index])
            low_last = 0
            print(df.iloc[high_index])
            plt.figure(1)
            plt.scatter(df.index[high_index], df.iloc[high_index])

            alpha = (df.iloc[high_index][-1] - df.iloc[high_index]) / (high_index[0][-1] - high_index[0][0])
            p_line = pd.DataFrame(
                index=pd.date_range(df.iloc[np.argmax(df)], periods=df.shape[0] - df.iloc[np.argmax(df)] + step_predict,
                                    freq="D"))

            time = np.linspace(0, df.shape[0] - df.iloc[np.argmax(df)] + step_predict,
                               df.shape[0] - df.iloc[np.argmax(df)] + step_predict)
            p_line['pre'] = df.iloc[np.argmax(df)] + alpha * time
            plt.figure(1)

            plt.plot(p_line)
            print(order, alpha)
            break


# print('alpha low ', alpha)


rates['trend'], p_line = trend_line(rates, step_predict=40)
de_trend = rates.high  # - rates.trend
_anti_trend(rates.low, 40, )
_aanti_trend(rates.high, 40, )

plt.figure(1)
plt.plot(rates.trend)
plt.plot(rates.high)
plt.plot(rates.low)

# plt.plot(p_line)
# plt.figure(2)
# plt.plot(rates.high - rates.trend)

plt.show()


# _gold_plot(rates)


# window = 5
# for t in range(rates.shape[0] - window):
#
#     low = rates.low[t:(t + window)]
#     high = rates.high[t:(t + window)]
#
#     low_up_trend = np.all(np.diff(low.to_numpy()) >= 0)
#     high_up_trend = np.all(np.diff(high.to_numpy()) <= 0)
#     high_low_spread = np.all((high - low).to_numpy() <= 0.50)
#
#     if low_up_trend and high_low_spread:
#         # print(rates.iloc[t:(t+window),...])
#         print(low)
#         # breakpoint()
#
#         plt.plot(low)
#         plt.plot(high)
#
#         plt.show()

#     elif high_up_trend and high_low_spread:
#         # print(rates.iloc[t:(t+window),...])
#         print(high)
#         plt.plot(low)
#         plt.plot(high)
#
#         plt.show()
#     else:
#         pass


class TrendLine:
    def __init__(self,
                 df,
                 time_frame,
                 step_predict,
                 ):
        self.df = df
        self.tf = time_frame
        self.step_predict = step_predict

    def _line(self, index_1, index_2):
        pass

    def _low_trend(self):
        pass

    def _high_trend(self):
        pass

    def _mid_trend(self):
        pass
