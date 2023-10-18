'''
    metrics run Q-learning model
'''

from Util import RandomWalkEnv
from discreteQ import discreteQAgent
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# LOAD Q table
with open("Qtable.pkl", "rb") as pkl_handle:
	Q = pickle.load(pkl_handle)


seeds = [19, 20, 21, 22, 23, 24, 25, 26, 28, 29]
results = {'seed':[], 'RW_p_l':[], 'Sharpie_ratio':[], 'accuracy':[], 'p_l':[]}

df = pd.DataFrame()
for seed in  seeds:
    print(seed)
    results['seed'].append(seed)
    env = RandomWalkEnv(size=10000, random_seed=seed, equity=1)
    env.trim_df()
    Q_agent = discreteQAgent(env, bins_n=24)

    Q_agent.play_game(Q) # play game

    df['equity_'+str(seed)] = env.equity

df['total'] = df.sum(axis=1)/len(seeds)

plt.figure(figsize=(10,4))
df['total'].plot()
plt.xlabel('Time')
plt.ylabel('Equity')
plt.grid(True)
plt.title('Portfolio of the all seeds, Q-learning algorithm')
plt.show()

df.to_csv('./xls//Portfolio_Q.csv')

