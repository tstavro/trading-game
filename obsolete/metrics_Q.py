
import RandomWalk
from  RL_models import discreteQlearning
import pickle
import numpy as np
import pandas as pd

###########
# Q_Predict
###########

# LOAD Q table
with open("Qtable.pkl", "rb") as pkl_handle:
	Q = pickle.load(pkl_handle)


for i in range(19, 30):
    env = RandomWalk.make(size=10000, random_seed=i, equity=1)
    env.trim_df()
    qlearn = discreteQlearning(env, bins_n=24)

    qlearn.play_game(Q) # play game
    # env.disp_dataset()
    # env.disp_equity('accuracy: '+str(round(env.accuracy, 4)))


    RW_pl = env.df['process'].iloc[-1]/env.df['process'].iloc[0]-1
    print(i,'P/L RW:', round(RW_pl, 4))
    print(i, 'P/L equity:', round(env.equity[-1]-1, 4))
    df = pd.DataFrame()
    df['equity'] = env.equity
    df['returns'] = df['equity'].pct_change(1)
    df.dropna(inplace=True)
    sharpe_ratio = np.sqrt(252) * df['returns'].mean() / df['returns'].std()
    print(i, 'Sharp ratio:', round(sharpe_ratio, 3))


