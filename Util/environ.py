

import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import enum
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

FIG_SIZE = (10, 4)


class Actions(enum.Enum):
    Buy = 0
    Sell = 1


class RandomWalkEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("RandomWalkEnv-v0")

    def __init__(self, 
                    size=10000, 
                    random_seed=18, 
                    equity=100,
                    enable_metric=False,
                    zero_start=True) -> None:

        self.process_size = size
        self.equity = []
        self.orig_equity = equity 
        self.accEquity = equity
        self.enable_metric = enable_metric
        self.random_seed = random_seed
        self.idx = 0 # index of process
        self.profit_trades = 0
        self.zero_start = zero_start
        self.__generate_data(random_seed)

        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(5,), dtype=np.float32)

    def __generate_data(self, seed=18):
        np.random.seed(seed)
        returns = np.random.normal(loc=0, scale=1, size=self.process_size+101)/100
        process = np.cumsum(returns)+1 
        self.df = pd.DataFrame()
        self.df['process'] = process
        min = self.df['process'].min()
        if min < 0:
            self.df['process'] = self.df['process']-min+0.2
        self.df['pct_1'] = self.df['process'].pct_change(1)
        self.df['pct_2'] = self.df['process'].pct_change(2)
        self.df['pct_3'] = self.df['process'].pct_change(3)
        self.df['pct_4'] = self.df['process'].pct_change(4)
        self.df['pct_5'] = self.df['process'].pct_change(5)
        self.df['pct_6'] = self.df['process'].pct_change(6)
        self.df['pct_7'] = self.df['process'].pct_change(7)
        self.df['pct_8'] = self.df['process'].pct_change(8)
        self.df['pct_10'] = self.df['process'].pct_change(10)
        self.df['pct_20'] = self.df['process'].pct_change(20)
        self.df['pct_30'] = self.df['process'].pct_change(30)
        self.df['pct_50'] = self.df['process'].pct_change(50)
        self.df['pct_100'] = self.df['process'].pct_change(100)
        self.df['returns'] = self.df['pct_1'].shift(-1)
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True)

    def reset(self):
        self.equity = []
        self.best_returns = []
        self.agent_returns = []
        self.accEquity = self.orig_equity

        if self.zero_start: # start of process
            self.idx = 0
        else:
            self.idx = np.random.randint(low=0, high=self.process_size-100)

        self.profit_trades = 0
        self.metric = []
        return np.array([
            self.df['pct_10'].iloc[self.idx],
            self.df['pct_20'].iloc[self.idx],
            self.df['pct_30'].iloc[self.idx],
            self.df['pct_50'].iloc[self.idx],
            self.df['pct_100'].iloc[self.idx] 
            ]
        )

    def step(self, action_idx):
        """
        action: (0:buy, 1:sell)
        """
        action = Actions(action_idx)

        if action == Actions.Buy:
            return_ = self.df['returns'].iloc[self.idx]
        elif action == Actions.Sell:
            return_ = -self.df['returns'].iloc[self.idx]
        
        if self.enable_metric:
            self.best_returns.append(np.abs(self.df['returns'].iloc[self.idx]))
            self.agent_returns.append(return_)

            best_yield = self.best_returns[-100:]
            agent_yield = self.agent_returns[-100:]
            metric = np.sum(agent_yield) / np.sum(best_yield)
            self.metric.append(metric) # logged just for plotting

        if return_ > 0:
            self.profit_trades += 1
        
        self.accEquity += return_
        self.equity.append(self.accEquity)

        self.idx += 1 # increases the index

        # calculate metric
        # win_len =100
        # max_value = 0
        # if len(self.equity) - win_len > 0:
        #     window = self.equity[-win_len:]
        #     max_value = np.max(window)
        #     metric = max_value - self.equity[-1]
        # else:
        #     metric = 0
        # self.metric.append(metric) # logged just for plotting            


        if self.idx == len(self.df)-1:
            done = True
        else:
            if self.enable_metric:
                if metric < 0.08: # 0.08
                # if metric > 0.03: # 0.08
                    done = True
                else:
                    done = False
            else:
                done = False

        obs = np.array([
            self.df['pct_10'].iloc[self.idx],
            self.df['pct_20'].iloc[self.idx],
            self.df['pct_30'].iloc[self.idx],
            self.df['pct_50'].iloc[self.idx],
            self.df['pct_100'].iloc[self.idx] 
            ]
        )
        info = {}        
        return obs, return_, done, info
        
    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def disp_dataset(self, extra_str='', save_plot=None):
        plt.figure(figsize=FIG_SIZE)
        plt.plot(self.df['process'])
        X = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df['process'].values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        plt.plot(X, reg.predict(X), color='gray', linestyle='--')
        plt.grid(True)
        plt.title('Process - random seed:' + str(self.random_seed) + ' ' + extra_str)
        plt.tight_layout()
        if save_plot:
            plt.savefig('figures//'+'Process - random seed_' + str(self.random_seed) + '.jpg') 
        else: 
            plt.show()  # Display the plot

    def disp_equity(self, extra_str='', save_plot=None):
        plt.figure(figsize=FIG_SIZE)
        plt.plot(self.equity)
        X = np.arange(len(self.equity)).reshape(-1, 1)
        y = np.array(self.equity).reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        plt.plot(X, reg.predict(X), color='gray', linestyle='--')
        plt.grid(True)
        plt.title('Equity - random seed:'+str(self.random_seed)+' '+extra_str)
        plt.tight_layout()
        if save_plot:
            plt.savefig('figures//'+ 'Equity '+save_plot+'- random seed_'+str(self.random_seed) + '.jpg') 
        else: 
            plt.show()  # Display the plot

    def _disp_metric_(self):
        plt.figure(figsize=FIG_SIZE)
        plt.plot(self.metric)
        plt.grid(True)
        plt.title('Metric - random seed:'+str(self.random_seed))
        plt.tight_layout()
        plt.show

    def limits(self):
        return {
        # 'pct_1': (self.df['pct_1'].min(), self.df['pct_1'].max()),
        # 'pct_2': (self.df['pct_2'].min(), self.df['pct_2'].max()),
        # 'pct_3': (self.df['pct_3'].min(), self.df['pct_3'].max()),
        # 'pct_4': (self.df['pct_4'].min(), self.df['pct_4'].max()),
        # 'pct_5': (self.df['pct_5'].min(), self.df['pct_5'].max()),
        # 'pct_6': (self.df['pct_6'].min(), self.df['pct_6'].max()),
        # 'pct_7': (self.df['pct_7'].min(), self.df['pct_7'].max()),
        # 'pct_8': (self.df['pct_8'].min(), self.df['pct_8'].max()),
        'pct_10': (self.df['pct_10'].min(), self.df['pct_10'].max()),
        'pct_20': (self.df['pct_20'].min(), self.df['pct_20'].max()),
        'pct_30': (self.df['pct_30'].min(), self.df['pct_30'].max()),
        'pct_50': (self.df['pct_50'].min(), self.df['pct_50'].max()),
        'pct_100': (self.df['pct_100'].min(), self.df['pct_100'].max())
        }

    def save_limits(self):
        a_file = open("limits.pkl", "wb")
        pickle.dump(self.limits(), a_file)
        a_file.close()

    def load_limits(self):
        a_file = open("limits.pkl", "rb")
        output = pickle.load(a_file)
        a_file.close()        
        return output

    def trim_df(self):
        a1 = self.load_limits()
        
        for col in ('pct_10', 'pct_20', 'pct_30', 'pct_50', 'pct_100'):
            min = a1[col][0]
            max = a1[col][1]
            self.df[col] = np.where(self.df[col] < min, min, self.df[col])
            self.df[col] = np.where(self.df[col] > max, max, self.df[col])

    @property
    def steps(self):
        return self.process_size
    
    @property
    def accuracy(self):
        return round(self.profit_trades / self.idx, 3)

    @property
    def p_l(self):
        return self.equity[-1] - self.equity[0]

    @property
    def observation_space_n(self):
        return 5

    @property
    def action_space_n(self):
        return 2    






