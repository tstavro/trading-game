'''
    metrics for Supervised model
'''

from Util import RandomWalkEnv
import tensorflow as tf
import pandas as pd
import numpy as np


tf.compat.v1.disable_eager_execution()

model = tf.keras.models.load_model('./models/supervised.keras')

seeds = [18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]
results = {'seed':[], 'RW_p_l':[], 'Sharpie_ratio':[], 'accuracy':[], 'p_l':[]}

def play_game(model_path=''):
    # model.load()
    observation = env.reset()
    done = False
    cnt = 0  # number of moves in an episode
    while not done:
        cnt += 1
        action = model.predict(observation.reshape(1,-1))
        if action > 0.5:
            action = 0
        else:
            action = 1
        observation, reward, done, _ = env.step(action)
    return cnt

for seed in seeds:
    results['seed'].append(seed)
    env = RandomWalkEnv(size=10000, random_seed=seed, equity=1)
    play_game('')
    env.disp_dataset(save_plot='Supervised')

    # env.disp_equity('Supervised - accuracy:'+ str(round(env.accuracy, 4)))
    env.disp_equity(
        extra_str='Supervised accuracy:'+ str(round(env.accuracy, 3)),
        save_plot='Supervised'
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
results_df.to_csv('xls//Supervised.csv')

