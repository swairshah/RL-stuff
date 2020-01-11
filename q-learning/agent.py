import gym
import numpy as np
import torch, torch.nn as nn
from network import FCNet

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

class DeepQAgent(Agent):
    def __init__(self, lr, gamma, eps_max, eps_min, eps_dec, n_actions, n_states, input_dims):
        super(DeepQAgent, self).__init__(lr, gamma, eps_max, eps_min, eps_dec, n_actions, n_states)

        self.input_dims = input_dims
        self.Q = FCNet(self.lr, self.input_dims, self.n_actions)

    def pick_action(self, obs):
        if np.random.rand() <= self.eps: # explore
            a = np.random.choice(self.n_actions)
        else:
            s = torch.tensor(obs, dtype=torch.float).to(self.Q.device)
            a = self.Q.forward(s)
            a = torch.argmax(a).item()
        return a

    def learn(self, s, a, r, s_):
        self.Q.optimizer.zero_grad()
        s = torch.tensor(s, dtype=torch.float).to(self.Q.device)
        s_ = torch.tensor(s_, dtype=torch.float).to(self.Q.device)
        a = torch.tensor(a).to(self.Q.device)
        r = torch.tensor(r).to(self.Q.device)

        q_pred = self.Q.forward(s)[a]
        q_next = self.Q.forward(s_).max()

        q_target = r + self.gamma*q_next
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrease_eps()
