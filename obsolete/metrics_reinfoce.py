
from reinforce_torch import PolicyGradientAgent
import matplotlib.pyplot as plt
import RandomWalk
import numpy as np
import pandas as pd


agent = PolicyGradientAgent(ALPHA=0.001, # 0.001
                            input_dims=[5], 
                            GAMMA=0.99, # 0.99
                            n_actions=2, 
                            layer1_size=128, 
                            layer2_size=128
                            )
agent.load_model()


for i in range(19, 30):

    env = RandomWalk.make(size=10000, random_seed=i, equity=1)
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.greedy_action(observation)
        observation, reward, done = env.step(action)
        score += reward

    # env.disp_dataset('Reinforce')
    # env.disp_equity('Reinforce accuracy:'+ str(round(env.accuracy, 4)))
    # print('accuracy:', env.accuracy)

    RW_pl = env.df['process'].iloc[-1]/env.df['process'].iloc[0]-1
    print(i,'P/L RW:', round(RW_pl, 4))
    print(i, 'P/L equity:', round(env.equity[-1]-1, 4))
    df = pd.DataFrame()
    df['equity'] = env.equity
    df['returns'] = df['equity'].pct_change(1)
    df.dropna(inplace=True)
    sharpe_ratio = np.sqrt(252) * df['returns'].mean() / df['returns'].std()
    print(i, 'Sharp ratio:', round(sharpe_ratio, 3))




