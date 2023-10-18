'''
Reinforce test
'''

import numpy as np
import torch
from Util import RandomWalkEnv


model = torch.load('./models/reinforce.torch')

Stochastic_action = True
env = RandomWalkEnv(size=10000, random_seed=18, equity=1)
observation = env.reset()
done = False
tot_score = 0
cum_prob = 0
i = 0
while not done:
    i += 1
    act_prob = model(torch.from_numpy(observation).float())
    act_prob = torch.where(torch.isnan(act_prob), torch.tensor(0.5), act_prob)
    prob = torch.max(act_prob).item()
    cum_prob += prob
    if Stochastic_action:
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
    else:
        action = torch.argmax(act_prob).item()  # act greedily
    observation, reward, done, _ = env.step(action)
    tot_score += reward

cum_prob = cum_prob / i
env.disp_dataset('Reinforce')
env.disp_equity('Reinforce accuracy:'+ str(round(env.accuracy, 4)))
print('accuracy:', env.accuracy)
print('mean greedy prob: ', round(cum_prob, 3))

