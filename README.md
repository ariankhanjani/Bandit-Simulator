# Bandit-Simulator

Multi-Armed Bandit Simulator with implementation of core algorithms.

---

## 📋 Overview

This repository implements a framework for simulating and comparing **multi-armed bandit** algorithms. Currently, the ε-Greedy policy is implemented, with a bandit environment that uses (for now) a Gaussian (normal) reward distribution.  

Over time the aim is to add more algorithms (UCB, Thompson Sampling, etc.) and more bandit types.

---
## ⚡Quick Start

import numpy as np
from bandit_env import Bandit
from epsilon_greedy import EGreedy

num_arms= 5
epsilon=0.1
steps=5000

agent = EGreedy(num_arms=num_arms, epsilon_decay=False)
env = Bandit(num_arms=num_arms, stationary=True, seed=44)

rewards = np.zeros(steps)
regrets = np.zeros(steps)


for t in range(steps):
    action, reward = agent.step(env)                # agent chooses and updates
    optimal_action = env.optimal_action()           # oracle best action
    optimal_reward = env.q_true[optimal_action]     # true best reward
    regrets[t] = optimal_reward - reward
    rewards[t] = reward

print(f"Average reward: {rewards.mean():.3f}")
print(f"Total regret: {regrets.sum():.3f}")

