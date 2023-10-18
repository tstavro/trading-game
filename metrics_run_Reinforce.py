'''
    metrics for Reinforce model
'''

import numpy as np
import torch
from Util import RandomWalkEnv
import pandas as pd


model = torch.load('./models/reinforce.torch')

seeds = [18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]
results = {'seed':[], 'RW_p_l':[], 'Sharpie_ratio':[], 'accuracy':[], 'p_l':[]}

for seed in  seeds:
    results['seed'].append(seed)
    env = RandomWalkEnv(size=10000, random_seed=seed, equity=1)
    observation = env.reset()
    done = False
    tot_score = 0
    while not done:
        act_prob = model(torch.from_numpy(observation).float())
        # act_prob = torch.where(torch.isnan(act_prob), torch.tensor(0.5), act_prob)
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
        observation, reward, done, _ = env.step(action)
        tot_score += reward

    env.disp_dataset(save_plot='Reinforce')
    env.disp_equity(
        extra_str='Reinforce accuracy:'+ str(round(env.accuracy, 4)),
        save_plot='Reinforce'
        )

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
results_df.to_csv('xls//Reinforce.csv')
