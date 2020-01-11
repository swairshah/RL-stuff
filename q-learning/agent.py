import gym
import numpy as np

class Agent(object):
    def __init__(self, lr, gamma, eps_max, eps_min, eps_dec, n_actions, n_states):
        self.lr = lr
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = lr
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.eps = self.eps_max
        #self.Q = np.random.randn(self.n_states, self.n_actions)
        self.Q = None

    def pick_action(self):
        pass

    def decrease_eps(self):
        self.eps = self.eps*self.eps_dec if self.eps > self.eps_min\
                else self.eps_min

    def learn(self, s, a, r, s_):
        pass

class TabularQAgent(Agent):
    def __init__(self, lr, gamma, eps_max, eps_min, eps_dec, n_actions, n_states):
        super(TabularQAgent, self).__init__(lr, gamma, eps_max, eps_min, eps_dec, n_actions, n_states)
        self.Q = np.zeros((self.n_states, self.n_actions))

    def pick_action(self, obs):
        if np.random.rand() <= self.eps: # explore
            a = np.random.choice(self.n_actions)
        else:
            a = np.argmax(self.Q[obs, :])

        return a

    def learn(self, s, a, r, s_):
        self.Q[s, a] += self.lr*(r + self.gamma*np.max(self.Q[s_,:]) - self.Q[s, a])
        self.decrease_eps()
