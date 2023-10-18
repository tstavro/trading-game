
from Util import RandomWalkEnv
from discreteQ import discreteQAgent
import pickle
import pandas as pd
import numpy as np

# LOAD Q table
with open("Qtable.pkl", "rb") as pkl_handle:
	Q = pickle.load(pkl_handle)


seeds = [18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]
# seeds = [18, 19]
results = {'seed':[], 'RW_p_l':[], 'Sharpie_ratio':[], 'accuracy':[], 'p_l':[]}


for seed in  seeds:
    results['seed'].append(seed)
    env = RandomWalkEnv(size=10000, random_seed=seed, equity=1)
    env.trim_df()
    Q_agent = discreteQAgent(env, bins_n=24)

    Q_agent.play_game(Q) # play game
    env.disp_dataset(save_plot='Q_learning')
    env.disp_equity(
            extra_str='Q-learning accuracy: '+str(round(env.accuracy, 4)), 
            save_plot='Q_learning'
            )
    # print('accuracy:', str(round(env.accuracy, 3)))
    # print('p/l:', str(round(env.p_l, 3)))

    RW_pl = env.df['process'].iloc[-1] - env.df['process'].iloc[0]

    df = pd.DataFrame()
    df['equity'] = env.equity
    df['returns'] = df['equity'].pct_change(1)
    df.dropna(inplace=True)
    sharpe_ratio = np.sqrt(252) * df['returns'].mean() / df['returns'].std()

    results['RW_p_l'].append(round(RW_pl, 4))
    results['Sharpie_ratio'].append(round(sharpe_ratio, 3))
    results['accuracy'].append(round(env.accuracy, 3))
    results['p_l'].append(round(env.p_l, 3))


results_df = pd.DataFrame.from_dict(results)
results_df.to_csv('xls//Q_learn.csv')
