# Problem Statement
# There are 2 car lots with a maximum of 10 cars that
# can be stored at each location. At the end of each day,
# the total number of cars rented and returned are
# tallied. These numbers are drawn i.i.d. from a Poisson
# distribution. Location 1 has a expected rental and return rate of 3 cards.
# Location 2 has an expected rental rate of 4 and return rate of 2
# Obviously, a location cannot rent out more cars than they have on location.
# At the end of each day, you can move up to
# 5 cars from one location to another. This transfer has no
# intrinsic cost. Each rental nets you a constant monetary
# value. What policy should be followed to maximize the
# money made in total from both locations?

import numpy as np
from numpy.random import poisson
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

class Environment:
    def __init__(self, lot1_max = 10, lot2_max = 10, lot1_return_rate = 3, lot1_rental_rate = 3, lot2_return_rate = 2, lot2_rental_rate = 4):
        self.info = {
            'lot1': {
                'max': lot1_max,
                'cars': lot1_max, 
                'return_rate': lot1_return_rate, 
                'rental_rate': lot1_rental_rate
            }, 
            'lot2': {
                'max': lot2_max,
                'cars': lot2_max, 
                'return_rate': lot2_return_rate, 
                'rental_rate': lot2_rental_rate
            }
        }
        
        self._rental_reward = 100
        
    def state(self):
        return (self.info['lot1']['cars'], self.info['lot2']['cars'])
        
    def actions(self, state):
        max_action = min(state[1], self.info['lot1']['max'] - state[0])
        min_action = -min(state[0], self.info['lot2']['max'] - state[1])
        return range(min_action, max_action + 1)
        
    def sample(self, action):
        
        self.info['lot1']['cars'] += action
        self.info['lot2']['cars'] -= action
        
        sample_day = {
            'lot1': {
                'rentals': min(poisson(self.info['lot1']['rental_rate']), self.info['lot1']['cars']),
                'returns': min(poisson(self.info['lot1']['return_rate']), self.info['lot1']['max'] - self.info['lot1']['cars'])
            },
            'lot2': {
                'rentals': min(poisson(self.info['lot2']['rental_rate']), self.info['lot2']['cars']),
                'returns': min(poisson(self.info['lot2']['return_rate']), self.info['lot2']['max'] - self.info['lot2']['cars'])
            }
        }
        
        
        if sample_day['lot1']['rentals'] < 0:
            print(self.info['lot1']['cars'])
        
        self.info['lot1']['cars'] += sample_day['lot1']['returns'] - sample_day['lot1']['rentals']
        self.info['lot2']['cars'] += sample_day['lot2']['returns'] - sample_day['lot2']['rentals']
        
        reward = self._rental_reward * (sample_day['lot1']['rentals'] + sample_day['lot2']['rentals'])
        
        return reward
        
class Agent:
    def __init__(self, discount = 0.95, td_lambda = 0.8, learning_rate = 0.1, epsilon = 0.1, trace_threshold = 0.001):
        self.discount = discount
        self.td_lambda = td_lambda
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.Q = {}
        self.history = deque(maxlen = 1 + int(-np.log(1 / trace_threshold) / np.log(self.discount * self.td_lambda + 0.00001)))
        
    def set_environment(self, env):
        self.env = env
        self.history.clear()
        self.time_step = 1
        self.state = self.env.state()
    
    def policy(self, state):
        best_action = None
        best_action_value = None
        
        for action in self.env.actions(state):
            state_action = (state, action)
            if not state_action in self.Q:
                self.Q[state_action] = 0
            if best_action is None:
                best_action = action
                best_action_value = self.Q[state_action]
            action_value = self.Q[state_action]
            if action_value > best_action_value:
                best_action = action
                best_action_value = action_value
                
        return best_action
    
    def epsilon_policy(self, state):
        action = None
        if np.random.rand() < self.epsilon: #p(epsilon) random action
            action = np.random.choice(self.env.actions(state))
        else:   #p(1 - epsilon) greedy w.r.t. Q
            action = self.policy(state)
            
        return action
    
    def act(self):
        action = self.epsilon_policy(self.env.state())
        state_action = (self.env.state(), action)
        if not state_action in self.Q:
            self.Q[state_action] = 0
        self.history.append(state_action)
        
        reward = self.env.sample(action)
        next_action = self.policy(self.env.state())
        next_state_action = (self.env.state(), next_action)
        
        eligibility = 1
        for sa in self.history:
            delta = reward + self.discount * self.Q[next_state_action] - self.Q[sa]
            self.Q[sa] += self.learning_rate * delta * eligibility
            eligibility *= self.discount * self.td_lambda
            
        self.time_step += 1
        
    def value(self, state):
        action = self.policy(state)
        return self.Q[(state, action)]
    
def main():
    np.random.seed(0)
    max1 = 10
    max2 = 10
    agent = Agent(discount = 0.95, td_lambda = 0.9)
    for episode in range(50):
        env = Environment(lot1_max = max1, lot2_max = max2)
        agent.set_environment(env)
        agent.epsilon = 0.5
        agent.learning_rate = 1 / (2 * episode + 1)
        for step in range(200):
            agent.act()
        
    x = range(max1 - 1)
    y = range(max2 - 1)
    
    value = np.zeros((max1 - 1, max2 - 1))
    policy = np.zeros((max1 - 1, max2 - 1))
    for i in x:
        for j in y:
            value[i][j] = agent.value((i, j))
            policy[i][j] = agent.policy((i, j))
            
    x, y = np.meshgrid(x, y)
            
    fig = plt.figure(figsize = plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
    ax2 = fig.add_subplot(1, 2, 2, projection = '3d')
    
    ax1.plot_wireframe(x, y, value)
    ax2.plot_wireframe(x, y, policy)
    
    plt.show()
    
if __name__ == '__main__':
    main()