from Trader.Strategy.abcStrategy import Strategy
from ta import momentum, volume, trend, volatility


class SimpleStrategy(Strategy):
    def __init__(self,
                 step_df,
                 ):
        self.df = step_df
        self.df['rsi'] = momentum.rsi(self.df['close'], window=8).bfill()
        self.df['macd'] = trend.macd(self.df['close'], window_slow=32, window_fast=8, ).bfill()
        self.df['macd2'] = trend.macd(self.df['macd'], window_slow=32, window_fast=8, ).bfill()

        self.df['very_fast_ema'] = trend.ema_indicator(self.df['close'], window=8).bfill()
        self.df['fast_ema'] = trend.ema_indicator(self.df['close'], window=16).bfill()
        self.df['slow_ema'] = trend.ema_indicator(self.df['close'], window=32).bfill()

        fast_bb = volatility.BollingerBands(self.df['close'], window=16, window_dev=1)
        self.df['fast_high_channel'] = fast_bb.bollinger_hband().bfill()
        self.df['fast_mid_channel'] = fast_bb.bollinger_mavg().bfill()
        self.df['fast_low_channel'] = fast_bb.bollinger_lband().bfill()

        slow_bb = volatility.BollingerBands(self.df['close'], window=32, window_dev=2)
        self.df['slow_high_channel'] = slow_bb.bollinger_hband().bfill()
        self.df['slow_mid_channel'] = slow_bb.bollinger_mavg().bfill()
        self.df['slow_low_channel'] = slow_bb.bollinger_lband().bfill()

    def _buy_zone(self):
        if self.macd_signal_buy() and self.bb_signal_buy():
            return True
        else:
            return False

    def _sell_zone(self):
        if self.macd_signal_sell() and self.bb_signal_sell():
            return True
        else:
            return False

    def _estimate_st(self):
        st_point = abs(self.df['slow_high_channel'][-1] - self.df['slow_mid_channel'][-1])
        return st_point

    def _estimate_tp(self):
        tp_point = abs(self.df['slow_high_channel'][-1] - self.df['slow_mid_channel'][-1])
        return tp_point

    def _estimate_volume(self):
        return 0.01

    # BB strategy
    def bb_signal_buy(self):
        out_of_band = self.df['close'][-2] < self.df['fast_low_channel'][-2]
        cross_over = self.df['close'][-1] > self.df['fast_low_channel'][-1]
        if  cross_over:
            return True
        else:
            return False

    def bb_signal_sell(self):
        out_of_band = self.df['close'][-2] > self.df['fast_low_channel'][-2]
        cross_over = self.df['close'][-1] < self.df['fast_low_channel'][-1]

        if  cross_over:
            return True
        else:
            return False

    # MACD
    def macd_signal_buy(self):
        if self.df['macd2'][-1] > 0:
            if self.df['macd2'][-1] > self.df['macd'][-2]:
                return True
        else:
            return False

    def macd_signal_sell(self):
        if self.df['macd2'][-1] < 0:
            if self.df['macd2'][-1] > self.df['macd2'][-2]:
                return True
        else:
            return False


# from Forex.Market import LiveTicks, ImbalanceTick
# import matplotlib.pyplot as plt
# 
# WINDOW = (3 * 24 * 60)
# 
# live = LiveTicks({'symbol': 'EURUSD.si', 'window': WINDOW})
# df = live.get_rates()
# 
# imb = ImbalanceTick(df, {'window_imb': 256, 'volume_threshold': 750})
# imb_index = imb.imb_index
# imb_df = imb.gen_imb_df()
# 
# sim = SimpleStrategy(imb_df)
# print(sim.signal)
# fig , axs = plt.subplots(2)
# 
# axs[0].plot(sim.df.index, sim.df['fast_high_channel'], '--', c='r')
# 
# # plt.plot(sim.df.index, sim.df['fast_mid_channel'], '--', c='g')
# axs[0].plot(sim.df.index, sim.df['fast_low_channel'], '--', c='b')
# 
# axs[0].plot(sim.df.index, sim.df['slow_high_channel'], ':', c='r')
# axs[0].plot(sim.df.index, sim.df['slow_mid_channel'], ':', c='g')
# axs[0].plot(sim.df.index, sim.df['slow_low_channel'], ':', c='b')
# axs[0].plot(sim.df.index, sim.df['fast_ema'], '.-', c='r')
# axs[0].plot(sim.df.index, sim.df['slow_ema'], '.-', c='b')
# 
# axs[0].scatter(sim.df.index, sim.df['close'], c='g')
# 
# #plt.figure(11)
# axs[1].plot(sim.df.index, sim.df['macd'], '.-', c='r')
# axs[1].plot(sim.df.index, sim.df['macd2'], '.-', c='g')
# axs[1].plot(sim.df.index, sim.df['fast_ema']- sim.df['slow_ema'], '.-', c='b')
# axs[1].axhline(0.0)
# #plt.plot(sim.df.index, sim.df['close']-sim.df['fast_ema'] , '.-', c='b')
# 
# plt.show()
