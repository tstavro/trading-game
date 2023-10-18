
 
import os
import sys
sys.path.append(os.pardir)

from Util import RandomWalkEnv
import torch
import numpy as np

model = torch.load('..\\models\\reinforce.torch')

env = RandomWalkEnv(size=10000, random_seed=20, equity=1)
env.observation_space_n
env.action_space_n

env.action_space.sample()

observation = env.reset()
done = False
tot_score = 0
while not done:
    act_prob = model(torch.from_numpy(observation).float())
    # act_prob = torch.where(torch.isnan(act_prob), torch.tensor(0.5), act_prob)
    action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
    observation, reward, done, _ = env.step(action)
    tot_score += reward

env.disp_dataset('Reinforce')
env.disp_equity('Reinforce accuracy:'+ str(round(env.accuracy, 4)))
print('accuracy:', env.accuracy)



