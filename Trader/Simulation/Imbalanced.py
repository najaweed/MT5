import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Forex.Market import LiveTicks


class ImbalanceTick:

    def __init__(self,
                 df_rates,
                 config,
                 volume_threshold=1000,
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

        imb_df = pd.DataFrame(columns=['low', 'open', 'close', 'high' , 'volume'])
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


WINDOW = (3 * 24 * 60)

live = LiveTicks({'symbol': 'USDCAD.si', 'window': WINDOW})
df = live.get_rates()

imb = ImbalanceTick(df, {'window_imb': 256, 'volume_threshold': 1000})
imb_index = imb.imb_index
imb_df = imb.gen_imb_df()

plt.figure(12)
plt.errorbar(df.index, df['close'], yerr=df['high'] - df['low'])
#
# for idx in imb_index:
#     plt.axvline(idx)
# plt.scatter(imb_df.index, imb_df['low'], c='b')
plt.scatter(imb_df.index, imb_df['close'], c='r')
# plt.scatter(imb_df.index, imb_df['high'], c='r')

plt.figure(121)

imb_index = imb.imb_index
imb_df = imb.gen_imb_df()
plt.scatter(imb_df.index, imb_df['low'], c='b')
plt.scatter(imb_df.index, imb_df['close'], c='g')
plt.scatter(imb_df.index, imb_df['high'], c='r')

plt.figure(1121)

plt.scatter(range(len(imb_df.index)), imb_df['low'].to_numpy(), c='b')
plt.scatter(range(len(imb_df.index)), imb_df['close'].to_numpy(), c='g')
plt.scatter(range(len(imb_df.index)), imb_df['high'].to_numpy(), c='r')

import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from ta.momentum import rsi
from ta.trend import macd

plt.figure(11121)

plt.scatter(imb_df.index, rsi(imb_df['close'], window=24), c='b')
plt.figure(111121)

plt.scatter(imb_df.index, macd(imb_df['close'], window_slow=24, window_fast=12, ), c='g')
plt.scatter(imb_df.index, macd(imb_df['close'], window_slow=24, window_fast=12, ), c='b')

plt.show()
