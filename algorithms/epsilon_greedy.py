import numpy as np

class EGreedy:
    def __init__(self, num_arms, epsilon=0.1, epsilon_decay=False, seed=None):
        """
        Epsilon-Greedy Agent
        Args:
            num_arms (int): number of arms in the bandit
            epsilon (float): probability of exploration
            seed (int or None): RNG seed
        """
        self.num_arms = num_arms                    # Number of arms, same as Bandit Env num_arms
        self.epsilon = epsilon                      # Epsilon 
        self.epsilon_decay = epsilon_decay          # If True, epsilon decays over time
        self.rng = np.random.default_rng(seed)
        self.q_estimates = np.zeros(num_arms)       # Q value estimated by agent
        self.action_count = np.zeros(num_arms)      # Number of each arm pulled 
        self.step_count = 0
    
    
    def step(self, env):
        
        # Decaying epsilon if enabled
        if self.epsilon_decay:
            eps = self.epsilon / (1 + self.step_count / self.num_arms)
        
        else:
            eps = self.epsilon
        
        # Exploration and Exploitation
        if np.random.rand() < eps:
            action = self.rng.integers(0, self.num_arms)        # Explore
            
        else:
            action = int(np.argmax(self.q_estimates))           # Exploit

          
        reward = env.step(action)       # Get reward from environment

        self.action_count[action] += 1
        n = self.action_count[action]
        self.q_estimates[action] += self.q_estimates[action] + (reward - self.q_estimates[action]) / n
        self.step_count += 1

        return action, reward