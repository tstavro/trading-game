'''
	Autocorrelation
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 


# plt.rcParams["figure.figsize"] = (10, 8)
spacing = np.linspace(-9 * np.pi, 9 * np.pi, num=1000)
s = pd.Series(0.7 * np.random.rand(1000) + 0.3 * np.sin(spacing))
plt.plot(s)
plt.grid(True)
plt.show()
pd.plotting.autocorrelation_plot(s)


random.seed(1)
random_walk = list()
random_walk.append(-1 if random.random() < 0.5 else 1)
for i in range(1, 1000):
	movement = -1 if random.random() < 0.5 else 1
	value = random_walk[i-1] + movement
	random_walk.append(value)
plt.plot(random_walk)
plt.grid(True)
plt.show()
pd.plotting.autocorrelation_plot(random_walk)


from statsmodels.tsa.stattools import adfuller
# statistical test
result = adfuller(random_walk)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# take difference
diff = list()
for i in range(1, len(random_walk)):
	value = random_walk[i] - random_walk[i - 1]
	diff.append(value)

pd.plotting.autocorrelation_plot(diff)
# no significant relationship between the lagged observations

