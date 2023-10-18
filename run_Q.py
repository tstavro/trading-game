'''
	Q_agent
'''

from Util import RandomWalkEnv
from discreteQ import discreteQAgent
import pickle

###########
# Q_Predict
###########

# LOAD Q table
with open("Qtable.pkl", "rb") as pkl_handle:
	Q_table = pickle.load(pkl_handle)

env = RandomWalkEnv(size=10000, random_seed=23, equity=1)
env.trim_df()
Q_agent = discreteQAgent(env, bins_n=24)

Q_agent.play_game(Q_table) # play game
Q_agent.env.disp_dataset(save_plot='Q_learning')
Q_agent.env.disp_equity(
		extra_str='accuracy: '+str(round(env.accuracy, 4)), 
		save_plot='Q_learning'
		)
print('accuracy:', str(round(env.accuracy, 4)))


