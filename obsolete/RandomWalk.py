
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=10)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)
import pickle


class make:
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
        self.acc = 0
        self.zero_start = zero_start
        self.generate_data(random_seed)

    def generate_data(self, seed=18):
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
        # start of process
        # if self.zero_start:
        #     self.idx = 0
        # else:
        #     if self.enable_metric:
        #         self.idx = np.random.randint(low=0, high=self.process_size-100)
        #     else:
        #         self.idx = 0 
        if self.zero_start:
            self.idx = 0
        else:
            self.idx = np.random.randint(low=0, high=self.process_size-100)


        self.acc = 0
        self.metric = []
        return np.array([
            # self.df['pct_1'].iloc[self.idx],
            # self.df['pct_2'].iloc[self.idx],
            # self.df['pct_3'].iloc[self.idx],
            # self.df['pct_4'].iloc[self.idx],
            # self.df['pct_5'].iloc[self.idx]
            # self.df['pct_6'].iloc[self.idx],
            # self.df['pct_7'].iloc[self.idx],
            # self.df['pct_8'].iloc[self.idx],
            self.df['pct_10'].iloc[self.idx],
            self.df['pct_20'].iloc[self.idx],
            self.df['pct_30'].iloc[self.idx],
            self.df['pct_50'].iloc[self.idx],
            self.df['pct_100'].iloc[self.idx] 
            ]
        )

    def step(self, action):
        """
        action: (0:buy, 1:sell)
        """

        if action == 0:
            return_ = self.df['returns'].iloc[self.idx]
        elif action == 1:
            return_ = -self.df['returns'].iloc[self.idx]
        
        if self.enable_metric:
            self.best_returns.append(np.abs(self.df['returns'].iloc[self.idx]))
            self.agent_returns.append(return_)

            best_yield = self.best_returns[-100:]
            agent_yield = self.agent_returns[-100:]
            metric = np.sum(agent_yield) / np.sum(best_yield)
            self.metric.append(metric)

        if return_ > 0:
            self.acc += 1
        
        self.accEquity += return_
        self.equity.append(self.accEquity)

        self.idx += 1

        if self.idx == len(self.df)-1:
            done = True
        else:
            if self.enable_metric:
                if metric < 0.1: # 0.08
                    done = True
                else:
                    done = False
            else:
                done = False
        obs = np.array([
            # self.df['pct_1'].iloc[self.idx],
            # self.df['pct_2'].iloc[self.idx],
            # self.df['pct_3'].iloc[self.idx],
            # self.df['pct_4'].iloc[self.idx],
            # self.df['pct_5'].iloc[self.idx],
            # self.df['pct_6'].iloc[self.idx],
            # self.df['pct_7'].iloc[self.idx],
            # self.df['pct_8'].iloc[self.idx],
            self.df['pct_10'].iloc[self.idx],
            self.df['pct_20'].iloc[self.idx],
            self.df['pct_30'].iloc[self.idx],
            self.df['pct_50'].iloc[self.idx],
            self.df['pct_100'].iloc[self.idx] 
            ]
        )

        return obs, return_, done

    def disp_dataset(self, extra_str=''):
        plt.figure()
        # plt.plot(self.df['process'][:int(self.process_size/2)])
        plt.plot(self.df['process'])
        plt.grid(True)
        plt.title('Process - random seed:'+str(self.random_seed)+' '+extra_str)
        plt.show

    def disp_equity(self, extra_str=''):
        plt.figure()
        plt.plot(self.equity)
        plt.grid(True)
        plt.title('Equity - random seed:'+str(self.random_seed)+' '+extra_str)
        plt.show

    def disp_metric(self):
        plt.figure()
        plt.plot(self.metric)
        plt.grid(True)
        plt.title('Metric - random seed:'+str(self.random_seed))
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


    def sample_action(self):
        return np.random.randint(0,2)


    @property
    def observation_space(self):
        return 5

    @property
    def action_space_n(self):
        return 2    
     
    @property
    def steps(self):
        return self.process_size
    
    @property
    def accuracy(self):
        return round(self.acc / self.idx, 3)



        

