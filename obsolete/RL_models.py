
import numpy as np
import matplotlib.pyplot as plt
import time

class discreteQlearning():

    def __init__(self, env, bins_n=10, alpha=0.01, gamma=0.9):
        """
            env: environment
            bins_n: number of bins
            alpha: alpha parameter in Q learing
            gamma: gamma parameter in Q learing

        """
        self.bins_n = bins_n
        self.set_env(env)
        self.alpha = alpha
        self.gamma = gamma
        self.bins = self.create_bins()


    def set_env(self, env):
        self.env = env


    def max_dict(self, d):
        # d = {0: 0, 1: 0}
        # for key, val in d.items():
        # 	print('key',key,'val',val)
        max_v = float('-inf')
        for key, val in d.items():
            if val > max_v:
                max_v = val
                max_key = key
        return max_key, max_v


    def create_bins(self):
        '''
        create bins array [observation_space.shape[0], bins]
        '''
        bins = []
        a1 = 0.0001
        # bins.append(np.linspace(self.env.df['pct_1'].min()-a1, self.env.df['pct_1'].max()+a1, self.bins_n))
        # bins.append(np.linspace(self.env.df['pct_2'].min()-a1, self.env.df['pct_2'].max()+a1, self.bins_n))
        # bins.append(np.linspace(self.env.df['pct_3'].min()-a1, self.env.df['pct_3'].max()+a1, self.bins_n))
        # bins.append(np.linspace(self.env.df['pct_4'].min()-a1, self.env.df['pct_4'].max()+a1, self.bins_n))
        # bins.append(np.linspace(self.env.df['pct_5'].min()-a1, self.env.df['pct_5'].max()+a1, self.bins_n))
        bins.append(np.linspace(self.env.df['pct_10'].min()-a1, self.env.df['pct_10'].max()+a1, self.bins_n))
        bins.append(np.linspace(self.env.df['pct_20'].min()-a1, self.env.df['pct_20'].max()+a1, self.bins_n))
        bins.append(np.linspace(self.env.df['pct_30'].min()-a1, self.env.df['pct_30'].max()+a1, self.bins_n))
        bins.append(np.linspace(self.env.df['pct_50'].min()-a1, self.env.df['pct_50'].max()+a1, self.bins_n))
        bins.append(np.linspace(self.env.df['pct_100'].min()-a1, self.env.df['pct_100'].max()+a1, self.bins_n))

        self.state_len = len(bins)
        self.max_states = self.bins_n ** self.state_len 
        self.digit_pad = len(str(self.max_states - 1))
        return np.asarray(bins)        


    def assign_bins(self, observation):
        '''
        returns the state that the observation value belongs to
        '''
        state = np.zeros(self.state_len)
        for i in range(self.state_len):
            # returns the index of the bins observation belongs to
            state[i] = np.digitize(observation[i], self.bins[i])  
        return state


    def get_state_as_string(self, state):
        a2 = 0
        for i, el in enumerate(state[::-1]):
            a2 += el * self.bins_n ** i
        return str(int(a2)).zfill(self.digit_pad)


    def get_all_states_as_string(self):
        states = []
        for i in range(self.max_states):
            states.append(str(i).zfill(self.digit_pad)) 
        return states


    def initialize_Q(self):
        Q = {}
        all_states = self.get_all_states_as_string()
        for state in all_states:
            Q[state] = {}
            for action in range(self.env.action_space_n):
                Q[state][action] = 0
        return Q


    def play_game(self, Q, eps=0.02):
        observation = self.env.reset()
        done = False
        cnt = 0  # number of moves in an episode
        state_numeric = self.assign_bins(observation)
        state = self.get_state_as_string(state_numeric)
        total_reward = 0
        while not done:
            cnt += 1
            if np.random.uniform() < eps:
                act = self.env.action_space.sample()
            else:
                act = self.max_dict(Q[state])[0]  
            observation, reward, done, _ = self.env.step(act)
            total_reward += reward

            state_numeric = self.assign_bins(observation)
            state_new = self.get_state_as_string(state_numeric)
            state = state_new
        return total_reward, cnt


    def Q_learn(self, Q, eps=0.5):
        '''
            1. receives an observation
            2. Defines an action based on epsilon. Random or Qvalue(best action)
            3. Executes the action and Receives the new state: state_new
            4. cumsums the reward
            5. Takes the best Qvalue and the responding action for the new state
            6. Updates the Qvalue for the current state action Q[state][act]

        '''
        observation = self.env.reset()
        done = False
        cnt = 0  # number of moves in an episode
        state_numeric = self.assign_bins(observation)
        state = self.get_state_as_string(state_numeric)
        total_reward = 0

        while not done:
            cnt += 1
            # np.random.randn() seems to yield a random action 50% of the time ?
            if np.random.uniform() < eps:
                act = self.env.action_space.sample()
            else:
                act = self.max_dict(Q[state])[0] 
            observation, reward, done, _ = self.env.step(act)
            total_reward += reward
            state_numeric = self.assign_bins(observation)
            state_new = self.get_state_as_string(state_numeric)
            max_q_s1a1 = self.max_dict(Q[state_new])[1]  
            Q[state][act] += self.alpha * (reward + self.gamma * max_q_s1a1 - Q[state][act])
            state = state_new
        return total_reward, cnt


    def plot_eps(self, N=4000):
        eps = []
        eps1 = []
        eps2 = []
        for n in range(N):
            eps.append(1.0 / np.sqrt(n + 1))
            eps1.append(1 / (1 + n * 10e-3))
            eps2.append(0.5 / (1 + n * 10e-3))
        plt.figure()
        plt.plot(eps, label='eps:3 = 1.0 / np.sqrt(n + 1)')
        plt.plot(eps1, label='eps:1 = 1.0 / (1+n*10e-3)')
        plt.plot(eps2, label=' eps:2 = 0.5 / (1+n*10e-3)')
        plt.legend()
        plt.show()


    def train(self, episodes=4000, eps_strategy=3):
        """
            eps_strategy = 1: eps = 1.0 / (1+n*10e-3)
            eps_strategy = 2: eps = 0.5 / (1+n*10e-3)
            eps_strategy = 3: eps = 1.0 / np.sqrt(n + 1)

        """        
        Q = self.initialize_Q()

        length = []
        rewards = []
        start = time.time()                            
        for n in range(episodes):
            if eps_strategy == 1:
                eps = 1.0 / (1+n*10e-3)
            elif eps_strategy == 2:
                eps = 0.5 / (1+n*10e-3)
            elif eps_strategy == 3:
                eps = 1.0 / np.sqrt(n + 1)
            episode_reward, episode_length = self.Q_learn(Q, eps)
            if n % 10 == 0:
                print(n, 'epsilon %.4f' % eps, 'reward:', round(episode_reward,4))
            length.append(episode_length)
            rewards.append(episode_reward)
        end = time.time()
        print('time elapsed:', time.strftime('%H:%M:%S', time.gmtime(end - start)))
        return length, rewards, Q


    def plot_running_avg(self, totalrewards):
        N = len(totalrewards)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(totalrewards[max(0, t - 100):(t + 1)])
        plt.figure()
        plt.plot(running_avg)
        plt.title("Running Average")
        plt.show()


    def plot_totalrewards(self, totalrewards):
        plt.figure()
        plt.plot(totalrewards)
        plt.title("Total Rewards")
        plt.show()
    

