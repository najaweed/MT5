import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Forex.Market import LiveTicks

from Trader.Simulation.TradeEnv import TradingEnv
import time
import itertools

WINDOW = (1 * 6 * 60)

live = LiveTicks('EURGBP.si')
df = live.get_rates(window_size=2 * WINDOW)
print(df['tick_volume'])
breakpoint()
# df -> imb_df
# percentage_return = imb_return
# state_observations = imb_obs

env_config = {
    "df_rates": df,
    "state_observations": live.percentage_return(),
    "state_rewards": live.percentage_return(),
    "balance": 100000,
    "window": WINDOW,
}

trade_env = TradingEnv(env_config)

log = {'params': [], 'balance': []}
s_time = time.time()

p1 = np.linspace(0.000001, 0.0005, 30)
p2 = np.linspace(0.05, 0.5, 10)
grid_params = [p1, p2]

all_state_params = []
for pieces in itertools.product(*grid_params):
    all_state_params.append(pieces)
x_params = []
for st in all_state_params:
    x_params.append(st )

for p in x_params:
    params = p

    log['params'].append(params)

    trade_env.reset()
    for _ in range(int(1e6)):
        _, _, done, _ = trade_env.step(params)
        if done:
            log['balance'].append(trade_env.balance)
            break

print(time.time() - s_time)

index_profit_log = [i for i, v in enumerate(log['balance']) if v > 100050]
profit_params = []
param_max = log['params'][np.argmax(log['balance'])]
print(param_max)

np.save('data.npy', param_max)  # save
