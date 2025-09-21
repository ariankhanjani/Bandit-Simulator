import numpy as np

class Bandit:
    """
    Args:
        num_arms(int): Number of arms
        stationary(bool): If False, means reward distribution will drift over time
        seed(int or None): Random seed for reproducibility
    """
    def __init__(self, num_arms, stationary=True, seed=None):    
        
        self.num_arms = num_arms
        self.stationary = stationary  
        self.rng = np.random.default_rng(seed)                  # Random number generator object from numpy
        self.q_true = self.rng.normal(0, 1.0, size=num_arms)    # True reward distributions initialized randomly
        self.q_estimates = np.zeros(num_arms)                   # Estimated optimal value for each arm
        self.action_count = np.zeros(num_arms)                  # Number of each arm pulled
        self.step_count = 0                                     # Step counter
        
    def step(self, action): 
        """Take an action (arm index), return reward """
        
        if not (0 <= action < self.num_arms):
            raise ValueError(f"Action range must be in [0, {self.num_arms-1}]")     # Invalid action out of range
        
        reward = self.rng.normal(self.q_true[action], 1.0)
        self.action_count[action] += 1
        self.step_count += 1
        
        if not self.stationary:
            self.q_true += self.rng.normal(0, 0.01, size=self.num_arms)

        return reward       # Reward, Num_Steps, Num_Actions

    def reset(self):
        
        self.q_true = self.rng.normal(0, 1.0, size=self.num_arms) 
        self.q_estimates = np.zeros(self.num_arms)
        self.action_count = np.zeros(self.num_arms)  
        self.step_count = 0 
          
    def optimal_action(self):
        
        return np.argmax(self.q_true)       # Returns index of the current optimal action, lets compute regret later
