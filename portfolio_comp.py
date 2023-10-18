
'''
    portfolio comparison of Q-learning and Reiforce
'''

import pandas as pd
import matplotlib.pyplot as plt

df_Q = pd.read_csv('./xls//Portfolio_Q.csv')
df_R = pd.read_csv('./xls//Portfolio_Reinforce.csv')

df = pd.DataFrame()
df['Q'] = df_Q['total']
df['R'] = df_R['total']

plt.figure(figsize=(10,4))
df['Q'].plot(label='Q-learning', legend=True)
df['R'].plot(label='Reinforce', legend=True)
plt.xlabel('Time')
plt.ylabel('Equity')
plt.grid(True)
plt.title('Portfolio comparison Reinforce vs Q-learning')
plt.legend()
plt.show()


