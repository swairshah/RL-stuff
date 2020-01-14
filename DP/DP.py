import gym
from envs.gridworld import GridworldEnv
import numpy as np
from collections import defaultdict

env = GridworldEnv()

def policy_evaluation(env, policy, theta=0.0001, discount=0.99):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, p_a in enumerate(policy[s]):
                for p, s_, r, done in env.P[s][a]:
                    v += p_a * p * (r + discount * V[s_])

            delta = max(delta, np.abs(v - V[s]))
            V[s] = v

        if delta < theta:
            break
    return V
