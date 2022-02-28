import pandas as pd
import numpy as np


class ImbalancedTick:

    def __init__(self,
                 df_rates,
                 config,
                 ):
        self.symbols = list(set([col[:9] for col in df_rates.columns]))
        self.market_time_limit = ('01:00', '23:00')  # config['market_time_limit']
        self.df = self._get_df(df_rates).between_time(*self.market_time_limit)
        self.volume_min = 1000# config['volume_min']
        # self.diff_min = config['diff_min']
        # self.total_volume_min = config['total_volume_min']
        # self.market_time_limit = config['market_time_limit']
        self.df['sum_volume'] = self.df[[f'{sym},tick_volume' for sym in self.symbols]].sum(axis=1)

    def generate_df_imbalanced(self):

        imb_df = pd.DataFrame(columns=self.df.columns)
        t_vol = 0
        idx_last = self.df.index[0]
        for idx in self.df.index:
            t_vol += self.df.loc[idx, 'sum_volume']
            if t_vol >= self.volume_min:
                # imb_df.loc[idx,'EURUSD.si,close'] = 11
                for sym in self.symbols:
                    print(sym,idx)
                    imb_df.loc[idx, f'{sym},close'] = np.average(self.df.loc[idx_last:idx, f'{sym},close'],
                                     weights=self.df.loc[idx_last:idx, f'{sym},tick_volume'])
                    imb_df.loc[idx, f'{sym},tick_volume'] =self.df.loc[idx_last:idx, f'{sym},tick_volume'].sum()
                imb_df.loc[idx,'sum_voume'] = t_vol
                t_vol = 0
                idx_last = idx

        return imb_df
    def _get_df(self, df_rates):
        df_volumes = df_rates[[f'{sym},tick_volume' for sym in self.symbols]].fillna(1)
        df_prices = df_rates[[f'{sym},close' for sym in self.symbols]].fillna(method='bfill')
        return pd.concat([df_volumes, df_prices], axis=1)



#
# # # generate Tick with min volume
# # from ImbalancedTick import ImbalancedTick
# # df = pd.read_csv('F:\\all_rates_m1.csv', index_col='time')
# # from datetime import datetime
# # df.index = pd.to_datetime(df.index)
# # imb = ImbalancedTick(df.iloc[:,:],[])
# #
# # imb_df =imb.generate_df_imbalanced()
# # imb_df.to_csv('F:\\imb_m1_v1000.csv')
#
