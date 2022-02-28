import gym
import numpy as np
from Forex.Agent import AlgoAgent


class TradingEnv(gym.Env):
    def __init__(self,
                 config_env,
                 ):
        # self.state_observation = config_env["state_observations"]
        self.state_reward = config_env["state_rewards"]
        self.window = config_env['window']
        self.df = config_env['df_rates']
        self.config = config_env

        # gym config
        # self.action_space = gym.spaces.Box(low=np.float32([0.005, 0.1, 20, 20, 20, 20, 5]),
        #                                    high=np.float32([0.05, 0.6, 50, 50, 50, 50, 50]),)
        #
        # self.action_space = gym.spaces.MultiDiscrete([3 for _ in range(self.num_symbols)])
        # self.action_space = gym.spaces.Discrete(3)
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
        #                                         shape=self.state_observation.shape[1:],
        #                                         dtype=np.float32
        #                                         )
        # self.observation_space = gym.spaces.Box(low=np.float32([0.000001, 0.001]),
        #                                         high=np.float32([0.001, 1]),
        #                                         )

        # env step parameters
        self.index_step: int = self.window
        self.index_last_step: int = self.state_reward.shape[0] - 1
        # render
        self.balance = self.config['balance']
        self.opened_position = {'type': "CLOSE"}
        self.step_position = {}
        self.history_positions = {'step_position': [], 'opened_position': []}

    def reset(self):
        self.index_step = self.window
        self.balance = self.config['balance']
        self.opened_position = {'type': "CLOSE"}
        self.step_position = {}
        self.history_positions = {'step_position': [], 'opened_position': []}

        return self.observation_space.sample()  # self.state_observation[self.index_step, ...]

    def step(self, step_action):
        # observation for Algo
        df_rates = self.df.iloc[(self.index_step - self.window):self.index_step, :]
        self.step_position = AlgoAgent(df_rates, step_action).take_action()
        self._hold_close_filter_position()
        self.history_positions['step_position'].append(self.step_position)
        self.history_positions['opened_position'].append(self.opened_position)
        # Next step
        self.index_step += 1
        # RL observation provide
        step_observation = np.zeros_like(step_action)  # self.state_observation[self.index_step, ...]
        # RL reward calculate from Algo step position
        step_reward = self._calculate_reward()
        # done condition
        done = True if self.index_step == self.index_last_step else False
        print(self.balance) if done else None
        self.balance += step_reward
        # print(step_action ,self.balance )
        return step_observation, step_reward, done, {}

    def render(self, mode="human"):
        pass

    def _hold_close_filter_position(self):
        # translate {} from Algo to 'type'  HOLD/CLOSE

        if self.opened_position['type'] != 'CLOSE':
            if self.step_position == {}:
                self.step_position['type'] = 'HOLD'
                self.step_position['index'] = self.index_step
                return
        elif self.step_position == {}:
            self.step_position['type'] = 'CLOSE'
            self.step_position['index'] = self.index_step

    def _calculate_reward(self):
        step_reward = 0.
        # print(self.df.index[self.index_step], self.df.iloc[self.index_step, 0])

        if self.step_position['type'] == 'HOLD':
            sl_tp = self._check_sl_tp(self.df.iloc[self.index_step, :])
            if sl_tp:
                self.opened_position = {'type': "CLOSE"}
                step_reward += sl_tp
                return step_reward

            time_expire = self._check_time_expire()
            if time_expire:
                self.opened_position = {'type': "CLOSE"}
                step_reward = time_expire
                return step_reward
        else:

            step_reward = self._calculate_profit()
        return step_reward

    def _calculate_profit(self):
        step_reward = 0.
        step_df = self.df.iloc[self.index_step, :]
        # open new position
        if self.step_position['type'] != 'CLOSE' and self.step_position['type'] != 'HOLD':
            if self.opened_position['type'] == 'CLOSE':
                # place deal
                if step_df['low'] <= self.step_position['price'] <= step_df['high']:

                    # print(self.df.index[self.index_step], self.step_position['type'], ' Placed')
                    # step_reward -= 0. * self.balance # commission
                    self.opened_position = self.step_position
                    self.opened_position['index'] = self.index_step
                    return step_reward
                # if not placed
                else:
                    # print(self.df.index[self.index_step], self.step_position['type'], ' Not Placed')
                    self.step_position = {}
                    return step_reward
            elif self.opened_position['type'] != 'CLOSE':
                # reverse
                if self.step_position['type'] != self.opened_position['type']:
                    step_reward = self.df.loc[self.df.index[self.index_step], 'open'] - self.opened_position[
                        'price']
                    step_reward *= self.balance
                    #  print(self.df.index[self.index_step], 'Reversed Spot', step_reward)
                    self.opened_position['type'] = 'CLOSE'
                    # place reverse order
                    if step_df['low'] <= self.step_position['price'] <= step_df['high']:
                        # step_reward -= 0. * self.balance # commission
                        #print(self.df.index[self.index_step], self.step_position['type'], 'Open Reverse ', )

                        self.opened_position = self.step_position
                        self.opened_position['index'] = self.index_step

                        return step_reward
                    # if not placed
                    else:
                        # print(self.df.index[self.index_step], self.step_position['type'],
                        #       'Reversed Not Placed',
                        #       )

                        return step_reward
                # same direction
                elif self.step_position['type'] == self.opened_position['type']:
                    # print(self.df.index[self.index_step], 'SAME DIRECTION')
                    return step_reward
            else:
                return step_reward

        return step_reward

    def _check_sl_tp(self, step_df):
        step_reward = 0.
        if self.opened_position['type'] != 'CLOSE' and self.step_position['type'] == 'HOLD':

            if step_df['low'] <= self.opened_position['sl'] <= step_df['high']:
                step_reward = - self.balance * abs(self.opened_position['sl'] - self.opened_position['price'])
                # print(self.df.index[self.index_step], 'StopLoss', step_reward)
                return step_reward

            elif self.opened_position['type'] == 'SELL' and self.opened_position['sl'] < step_df['low']:
                step_reward = - self.balance * abs(self.opened_position['sl'] - self.opened_position['price'])
                # print(self.df.index[self.index_step], 'StopLoss', step_reward)
                return step_reward

            elif self.opened_position['type'] == 'BUY' and self.opened_position['sl'] > step_df['high']:
                step_reward = - self.balance * abs(self.opened_position['sl'] - self.opened_position['price'])
                # print(self.df.index[self.index_step], 'StopLoss', step_reward)
                return step_reward

            # check take profit

            elif step_df['low'] <= self.opened_position['tp'] <= step_df['high']:
                step_reward = self.balance * abs(self.opened_position['tp'] - self.opened_position['price'])
                # print(self.df.index[self.index_step], 'TakeProfit', step_reward)
                return step_reward

            elif self.opened_position['type'] == 'SELL' and self.opened_position['tp'] > step_df['high']:
                step_reward = self.balance * abs(self.opened_position['tp'] - self.opened_position['price'])
                # print(self.df.index[self.index_step], 'TakeProfit', step_reward)
                return step_reward

            elif self.opened_position['type'] == 'BUY' and self.opened_position['tp'] < step_df['low']:
                step_reward = self.balance * abs(self.opened_position['tp'] - self.opened_position['price'])
                # print(self.df.index[self.index_step], 'TakeProfit', step_reward)
                return step_reward

            else:
                return False

    def _check_time_expire(self):
        step_reward = 0.

        if self.step_position['type'] == 'HOLD':
            if self.index_step - self.opened_position['index'] >= self.opened_position['expiration']:

                step_reward = self.df.loc[self.df.index[self.index_step], 'close'] - self.opened_position['price']
                step_reward = -1 * step_reward if self.opened_position['type'] == "SELL" else step_reward
                step_reward *= self.balance
                # print(self.df.index[self.index_step], 'Time Expired ', step_reward)
                return step_reward
            else:
                return False


