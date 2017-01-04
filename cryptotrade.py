import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from scipy.optimize import basinhopping

N_EPISODES = 10
TRAINING_SIZE = 0.5
VALIDATION_SIZE = 0.1
LEARNING_RATE = 0.01
#TD_LAMBDA = 0.9
DISCOUNT = 0.95

class Environment:
    def __init__(self, data, initial_btc = 1.0):
        self.data = data
        self.btc = initial_btc
        self.eth = 0
        self.time_step = 0
        
    def done(self):
        return self.time_step == len(self.data) - 1
        
    def agent_state(self):
        state = self.data.iloc[[self.time_step]].squeeze()
        state['btc'] = self.btc
        state['eth'] = self.eth
        return state
        
    def score(self):
        state = self.agent_state()
        return state['btc'] + state['eth'] * state['close']
        
    def step(self, action):
        self.time_step += 1
        state = self.agent_state()
        reward = 0
                
        next_btc = self.btc
        next_eth = self.eth
        #print(state)
        #print(action)
                
        if action['btc_amt'] > 0 and action['btc_amt'] <= state['btc'] and action['btc_price'] >= state['low']:   
            amt = action['btc_amt']
            price = action['btc_price']
            next_btc -= amt
            next_eth += amt / price
            reward -= amt
            
        if action['eth_amt'] > 0 and action['eth_amt'] <= state['eth'] and action['eth_price'] <= state['high'] and action['eth_price'] >= 0:   
            amt = action['eth_amt']
            price = action['eth_price']
            next_btc += amt * price
            next_eth -= amt
            reward += amt * price
            
        self.btc = next_btc
        self.eth = next_eth
        next_state = self.agent_state()
        
        #print(next_state)
        #print(reward)
            
        return next_state, reward

state_vector = ['btc', 'eth', 'high', 'low', 'open', 'close', 'volume', 'weightedAverage']
action_vector = ['btc_amt', 'btc_price', 'eth_amt', 'eth_price']

class History:
    def __init__(self, max_size):
        self.max_size = max_size
        self.idx = 0
        self.queue = np.zeros((max_size, 2), dtype = object) 
        
    def size(self):
        return min(self.idx, self.max_size)
        
    def sample(self, size):
        size = min(size, self.size())
        sample_indices = np.random.choice(range(self.size()), replace = True)
        x = np.atleast_2d(self.queue[sample_indices, 0])
        y = np.atleast_1d(self.queue[sample_indices, 1])
        
        return x, y
        
    def append(self, sample, target):
        idx = self.idx % self.max_size
        self.queue[idx, 0] = sample
        self.queue[idx, 1] = target
        self.idx += 1
        
class QEstimator:
    def __init__(self, n_hidden_layers = 2, batch_size = 32):
        self.train_step = 0
        self.batch_size = batch_size
        self.n_hidden_layers = n_hidden_layers
        self.history = History(self.batch_size * self.batch_size)
        self.model = self._build_model()
        self.frozen_model = self._build_model()
        self._update_model()
        
    def _update_model(self):
        for i in range(len(self.model.layers)):
            weights = self.model.layers[i].get_weights()
            self.frozen_model.layers[i].set_weights(weights)
        
    def _build_model(self):
        model = Sequential()
        dim = len(state_vector) + len(action_vector)
        model.add(Dense(dim, input_dim = len(state_vector) + len(action_vector),  init = 'he_normal'))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        for i in range(1, self.n_hidden_layers):
            model.add(Dense(model.output_shape[1], init = 'he_normal'))
            if i == self.n_hidden_layers - 1:
                model.add(Dropout(0.5))
            model.add(Activation('relu'))
            
        model.add(Dense(1, init = 'he_normal'))
        model.add(Activation('linear'))
        
        model.compile(loss = 'mean_squared_error', optimizer = SGD(lr = LEARNING_RATE, momentum = 0.9, nesterov = True))
        
        return model
        
    def _get_input_vector(self, state, action):
        x = []
        for s in state_vector:
            x.append(state[s])
        for a in action_vector:
            x.append(action[a])
            
        return np.array(x)
        
    def train(self, state, action, target):
        sample = self._get_input_vector(state, action)
        self.history.append(sample, target)
        
        x, y = self.history.sample(self.batch_size)        
        self.model.train_on_batch(x, y)
        
        self.train_step += 1
        if self.train_step % (self.batch_size * self.batch_size) == 0:
            self._update_model()
        
    def predict(self, state, action):
        x = self._get_input_vector(state, action)
        return self.frozen_model.predict(np.atleast_2d(x), batch_size = 1)
        
    def best_action(self, state):
        def objective(action_values):
            action = {}
            for i, a in enumerate(action_vector):
                action[a] = action_values[i]
            value = np.asscalar(self.predict(state, action))
            return value
            
        start_action_values = np.random.rand(len(action_vector))
        res = basinhopping(objective, start_action_values, niter = 25, niter_success = 5)
        
        action = {}
        for i, a in enumerate(action_vector):
            action[a] = res.x[i]
            
        value = -res.fun
        return action, value
            
        
    def random_action(self, state):
        lo_price = max(0, 2 * state['low'] - state['weightedAverage'])
        hi_price = 2 * state['high'] - state['weightedAverage']
        action = {
            'btc_amt': np.random.rand() * state['btc'],
            'btc_price': np.random.rand() * (hi_price - lo_price) + lo_price,
            'eth_amt': np.random.rand() * state['eth'],
            'eth_price': np.random.rand() * (hi_price - lo_price) + lo_price
        }
        
        return action, self.predict(state, action)
        
class Agent:
    def __init__(self, q_estimator, discount = 0.95, epsilon = 0.5):
        self.Q = q_estimator
        self.discount = discount
        self.epsilon = epsilon
        
    def policy(self, state):
        return self.Q.best_action(state)
        
    def epsilon_policy(self, state):
        if np.random.rand() < self.epsilon:
            return self.Q.random_action(state)
        return self.policy(state)
        
    def train(self, env):
        state = env.agent_state()
        action, value = self.epsilon_policy(state)
        next_state, reward = env.step(action)
        next_action, next_value = self.policy(next_state)
        target = reward + self.discount * next_value
        self.Q.train(state, action, target)
        
    def act(self, env):
        state = env.agent_state()
        action, value = self.policy(state)
        env.step(action)
        
def main():
    np.random.seed(0)
    data = []
    with open('data/btc_eth.json') as data_file:
        data = json.load(data_file)
        
    n_samples = len(data)
    n_train_samples = int(TRAINING_SIZE * n_samples)
    n_validation_samples = int(VALIDATION_SIZE * n_samples)
    n_test_samples = n_samples - n_train_samples - n_validation_samples
    
    df = pd.DataFrame(data)
    df.drop(['date', 'quoteVolume'], inplace = True, axis = 1)
    train = df.values[:n_train_samples]
    validation = df.values[n_train_samples: n_train_samples + n_test_samples]
    test = df.values[n_train_samples + n_test_samples:]
    
    scaler = StandardScaler()
    scaler.fit(df.values[:n_train_samples])
    #df[:] = scaler.transform(df[:])
    
    train = df[:n_train_samples]
    validation = df[n_train_samples: n_train_samples + n_validation_samples]
    test = df[n_train_samples + n_validation_samples:]
    
    estimator = QEstimator()
    agent = Agent(estimator, discount = DISCOUNT)
    
    train_scores = []
    validation_scores = []
    
    for i in range(N_EPISODES):
        train_env = Environment(train)
        while not train_env.done():
            agent.train(train_env)
            if train_env.time_step % 50 == 0:
                print(train_env.time_step)
                print(train_env.agent_state())
                print(agent.Q.best_action(train_env.agent_state()))

        train_scores.append(train_env.score())
        print('t-score: ', train_scores[-1])
        
        validation_env = Environment(validation)
        while not validation_env.done():
            agent.act(validation_env)
            if validation_env.time_step % 50 == 0:
                print(validation_env.time_step)
                print(agent.Q.best_action(validation_env.agent_state()))
        validation_scores.append(validation_env.score())
        print('v-score: ', validation_scores[-1])
        
    test_env = Environment(test)
    while not test_env.done():
        agent.act(test_env)
        if test_env.time_step % 50 == 0:
                print(test_env.time_step)
    test_score = test_env.score()
        
    print(train_scores)
    print(validation_scores)
    print(test_score)
    
if __name__ == '__main__':
    main()