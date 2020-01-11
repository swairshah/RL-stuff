from agent import TabularQAgent, DeepQAgent
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
n_actions = env.action_space.n
n_states = env.observation_space.shape

A = DeepQAgent(lr=0.001, gamma=0.9, 
          eps_max = 1.0, eps_min = 0.01, eps_dec = 0.9999995, 
          n_actions = n_actions, n_states = n_states, input_dims = n_states)

n_episodes = 5000000
win_pct_list = []
scores = []

for i in range(n_episodes):
    done = False
    score = 0
    s = env.reset()
    done = False

    while not done:
        a = A.pick_action(s)
        s_, r, done, info = env.step(a)
        A.learn(s, a, r, s_)
        score += r
        s = s_
        
    scores.append(score)
    if i % 100 == 0:
        win_pct = np.mean(scores[-100:])
        win_pct_list.append(win_pct)
        if i % 1000 == 0:
            print('episode', i, 'win pct %.2f' % win_pct, 
                                'eps %2.f' % A.eps)

plt.plot(win_pct_list)
plt.show()
