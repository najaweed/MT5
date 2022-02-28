import MetaTrader5 as mt5
import pandas as pd
import numpy as np


class LiveTicks:
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    def __init__(self,
                 config,
                 ):
        self.symbol = config['symbol']
        self.window = config['window']
        self.all_rates = None

    def get_ticks(self, start_datetime, end_datetime):
        all_ticks = pd.DataFrame(mt5.copy_ticks_range(self.symbol, start_datetime, end_datetime, mt5.COPY_TICKS_ALL))
        return all_ticks

    def get_rates(self, time_frame=mt5.TIMEFRAME_M1):
        all_prices = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, time_frame, 0, self.window))
        #print(all_prices)
        all_prices['time'] = pd.to_datetime(all_prices['time'], unit='s')
        all_prices = all_prices.set_index('time')
        # all_prices = all_prices[~all_prices.index.duplicated(keep='first')]
        # all_prices = all_prices.reindex(pd.date_range(all_prices.index[0], all_prices.index[-1], freq='min'))
        # all_prices = all_prices.fillna(method='bfill')
        # all_prices = all_prices.fillna(method='ffill')
        self.all_rates = all_prices
        return all_prices

    def percentage_return(self, price_type='close'):
        all_prices = self.all_rates[f'{price_type}']
        pct_prices = all_prices.pct_change().fillna(method='bfill')
        return pct_prices.to_numpy(dtype=float)


class ImbalanceTick:

    def __init__(self,
                 df_rates,
                 config,
                 ):
        self.df = df_rates
        self.window_imb = config['window_imb']
        self.volume_threshold = config['volume_threshold']

        self.imb_index = self._get_imb_index(self.volume_threshold)

    def _get_imb_index(self, volume_threshold):
        imb_index = []
        imb_volume = 0.
        for i, tick_index in enumerate(self.df.index):
            if i == 0:
                imb_index.append(tick_index)
            tick_volume = self.df.loc[tick_index, 'tick_volume']
            imb_volume += tick_volume
            if imb_volume >= volume_threshold:
                # print(tick_index)
                imb_index.append(tick_index)
                imb_volume = 0.

        return imb_index

    def gen_imb_df(self):

        imb_df = pd.DataFrame(columns=['low', 'open', 'close', 'high'])
        for i in range(1, len(self.imb_index)):
            df_volume = self.df.loc[self.imb_index[i - 1]:self.imb_index[i], 'tick_volume']

            imb_df.loc[self.imb_index[i], 'low'] = np.average(
                self.df.loc[self.imb_index[i - 1]:self.imb_index[i], 'low'],
                weights=df_volume)
            imb_df.loc[self.imb_index[i], 'close'] = np.average(
                self.df.loc[self.imb_index[i - 1]:self.imb_index[i], 'close'],
                weights=df_volume)
            imb_df.loc[self.imb_index[i], 'open'] = np.average(
                self.df.loc[self.imb_index[i - 1]:self.imb_index[i], 'open'],
                weights=df_volume)

            imb_df.loc[self.imb_index[i], 'high'] = np.average(
                self.df.loc[self.imb_index[i - 1]:self.imb_index[i], 'high'],
                weights=df_volume)

            imb_df.loc[self.imb_index[i], 'volume'] = self.volume_threshold
        return imb_df.tail(self.window_imb)

# live = LiveTicks('EURUSD.si')
# df = live.get_rates(window_size=300)
# print(df)
# from datetime import datetime
# import pytz
#
# timezone = pytz.timezone("Etc/UTC")
# utc_from = datetime(2020, 10, 10, hour=13, tzinfo=timezone)
# utc_to = datetime(2021, 12, 11, hour=13, tzinfo=timezone)
# all_prices = pd.DataFrame(mt5.copy_rates_range("EURUSD.si", mt5.TIMEFRAME_D1, utc_from, utc_to))
# all_prices['time'] = pd.to_datetime(all_prices['time'], unit='s')
# all_prices = all_prices.set_index('time')
# all_prices = all_prices[~all_prices.index.duplicated(keep='first')]
# all_prices = all_prices.reindex(pd.date_range(all_prices.index[0], all_prices.index[-1], freq='d'))
# all_prices = all_prices.fillna(method='bfill')
# all_prices = all_prices.fillna(method='ffill')
#
# print(all_prices)
