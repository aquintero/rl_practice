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

class Environment:
    def __init__(self, lot1_max = 10, lot2_max = 10, lot1_return_rate = 3, lot1_rental_rate = 3, lot2_return_rate = 2, lot2_rental_rate = 4):
        self._internal_state = {
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
        
        def sample(action):
            
            self._internal_state['lot1']['cars'] += action
            self._internal_state['lot2']['cars'] -= action
            
            sample_day = {
                'lot1': {
                    'rentals': min(poisson(self._internal_state['lot1']['rental_rate']), self._internal_state['lot1']['cars']),
                    'returns': min(poisson(self._internal_state['lot1']['return_rate']), self._internal_state['lot1']['max'] - self._internal_state['lot1']['cars'])
                },
                'lot2': {
                    'rentals': min(poisson(self._internal_state['lot2']['rental_rate']), self._internal_state['lot2']['cars']),
                    'returns': min(poisson(self._internal_state['lot2']['return_rate']), self._internal_state['lot2']['max'] - self._internal_state['lot2']['cars'])
                }
            }
            
            
            if sample_day['lot1']['rentals'] < 0:
                print(self._internal_state['lot1']['cars'])
            
            self._internal_state['lot1']['cars'] += sample_day['lot1']['returns'] - sample_day['lot1']['rentals']
            self._internal_state['lot2']['cars'] += sample_day['lot2']['returns'] - sample_day['lot2']['rentals']
            
            reward = self._rental_reward * (sample_day['lot1']['rentals'] + sample_day['lot2']['rentals'])
            next_state = (self._internal_state['lot1']['cars'], self._internal_state['lot2']['cars'])
            
            return reward, next_state
        
        start_state = (self._internal_state['lot1']['max'], self._internal_state['lot2']['max'])
        self.agent = Agent(self._internal_state, start_state, 0.9, 0.9, sample)
        
class Agent:
    def __init__(self, env_info, start_state, discount, td_lambda, sample):
        self.env_info = env_info
        self.state = start_state
        self.discount = discount
        self.td_lambda = td_lambda
        self.sample = sample
        
        self.time_step = 1
        self.Q = {}
        self.elibility = {}
    
    def _learning_rate(self):
        return 1 / np.sqrt(self.time_step)
    
    def _epsilon(self):
        return 1 / np.sqrt(self.time_step)
    
    def policy(self, state):
        best_action = None
        best_action_value = None
        max_action = min(state[1], self.env_info['lot1']['max'] - state[0])
        min_action = -min(state[0], self.env_info['lot2']['max'] - state[1])
        
        for action in range(min_action, max_action + 1):
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
    
    def act(self):
        action = None
        if np.random.rand() < self._epsilon(): #p(epsilon) random action
            max_action = min(self.state[1], self.env_info['lot1']['max'] - self.state[0])
            min_action = -min(self.state[0], self.env_info['lot2']['max'] - self.state[1])
            action = np.random.randint(min_action, max_action + 1)
        else:   #p(1 - epsilon) greedy w.r.t. Q
            action = self.policy(self.state)
        state_action = (self.state, action)
        if not state_action in self.Q:
            self.Q[state_action] = 0
        if not state_action in self.elibility:
            self.elibility[state_action] = 0
        
        reward, next_state = self.sample(action)
        next_action = self.policy(next_state)
        next_state_action = (next_state, next_action)
        
        delta = reward + self.discount * self.Q[next_state_action] - self.Q[state_action]
        if reward < 0:
            print(state_action, reward)
        self.elibility[state_action] += 1
        
        for sa in self.elibility:
            self.Q[sa] += self._learning_rate() * delta * self.elibility[sa]
            self.elibility[sa] *= self.discount * self.td_lambda
        #print(state_action, reward, next_state_action)
        self.state = next_state
        self.time_step += 1
        
    def value(self, state):
        action = self.policy(state)
        return self.Q[(state, action)]
    
def main():
    np.random.seed(0)
    max1 = 10
    max2 = 10
    env = Environment(lot1_max = max1, lot2_max = max2)
    for i in range(100000):
        env.agent.act()
        
    x = range(max1 + 1)
    y = range(max2 + 1)
    
    value = np.zeros((max1 + 1, max2 + 1))
    policy = np.zeros((max1 + 1, max2 + 1))
    for i in x:
        for j in y:
            value[i][j] = env.agent.value((i, j))
            policy[i][j] = env.agent.policy((i, j))
            
    x, y = np.meshgrid(x, y)
            
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
    ax2 = fig.add_subplot(1, 2, 2, projection = '3d')
    
    ax1.plot_wireframe(x, y, value)
    ax2.plot_wireframe(x, y, policy)
    
    plt.show()
    
if __name__ == '__main__':
    main()