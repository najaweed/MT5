from Forex.Market import LiveTicks, ImbalanceTick
from Forex.ManageOrder import ManageOrder
import time


class LiveTrader:
    def __init__(self,
                 config,
                 ):
        self.manager = ManageOrder(config['symbol'])
        self.live = LiveTicks(config)
        self.agent = config['agent']

        self.config_imb = {'window_imb': config['window_imb'],
                           'volume_threshold': config['volume_threshold']}

        self.last_tick = self.live.get_rates().index[-1]
        self.last_imb_tick = self.last_tick

    def _check_new_tick(self):
        tick = self.live.get_rates().index[-1]
        if tick > self.last_tick:
            self.last_tick = tick
            return True
        else:
            return False

    def _check_new_imbalanced_tick(self, data_frame):
        tick_imb = ImbalanceTick(data_frame, self.config_imb).imb_index[-1]
        if tick_imb > self.last_imb_tick:
            self.last_imb_tick = tick_imb
            return True
        else:
            return False

    def main(self):
        while True:

            if self._check_new_tick():
                print(self.last_tick)
                df = self.live.get_rates()

                if self._check_new_imbalanced_tick(df):
                    imb_df = ImbalanceTick(df, self.config_imb).gen_imb_df()
                    print(imb_df)
                    print(self.last_imb_tick)

                    request = self.agent.take_action(imb_df)
                    if request != {}:
                        self.manager.manage(request)
                    else:
                        pass
            else:
                pass

            time.sleep(5)

