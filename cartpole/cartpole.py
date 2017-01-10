import gym
from gym import wrappers
import tensorflow as tf
import numpy as np

class Estimator:
    def __init__(self, learning_rate = 1):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            self.global_step = tf.Variable(1, trainable = False)
            self.learning_rate = learning_rate / self.global_step
            
            self.x = tf.placeholder(tf.float32, [None, 4], name = 'x')
            self.y = tf.placeholder(tf.float32, [None, 2], name = 'y')
            self.W = tf.Variable(tf.random_normal([4, 2]))
            self.b = tf.Variable(tf.zeros([2]))
            self.predicted_y = tf.matmul(self.x, self.W) + self.b
            self.loss =  tf.reduce_mean(tf.squared_difference(self.predicted_y, self.y))
            self.optimize = tf.train.AdadeltaOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            self.train_step = (
                tf.assign(self.global_step, self.global_step + 1),
                self.optimize
            )
            init = tf.global_variables_initializer()
            self.sess.run(init)
        
    def train(self, state, target):
        state = np.atleast_2d(state)
        target = np.atleast_2d(target)
        feed_dict = {
            self.x: state,
            self.y: target
        }
        self.sess.run(self.train_step, feed_dict = feed_dict)
        #print(self.sess.run([self.W, self.b]))
            
    def predict(self, state):
        state = np.atleast_2d(state)
        feed_dict = {
            self.x: state
        }
        action_values = self.sess.run(self.predicted_y, feed_dict = feed_dict)
        
        return action_values.squeeze()
            
            
class Agent:
    def __init__(self, estimator, env, discount = 0.95):
        self.estimator = estimator
        self.env = env
        self.discount = discount
                 
    def train(self, epsilon = 0):
        state = self.env.reset()
        done = False
        trace = {}
        while not done:
            self.env.render()
            action_values = self.estimator.predict(state)
            action = self.env.action_space.sample()
            if np.random.rand() > epsilon:
                action = np.argmax(action_values)
            next_state, reward, done, info = self.env.step(action)
            next_action_values = self.estimator.predict(next_state)
            next_action = np.argmax(next_action_values)
            
            target = np.array(action_values)
            target[action] = reward + self.discount * next_action_values[next_action]
            
            self.estimator.train(state, target)
            
            state = next_state
        
    def run(self):
        state = env.reset()
        done = False
        rewards = 0
        while not done:
            action_values = self.estimator.predict(state)
            action = np.argmax(action_values)
            state, reward, done, info = self.env.step(action)
            rewards += reward
            
        return rewards

def main():
    np.random.seed(0)
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, './record', force = True)
    estimator = Estimator()
    agent = Agent(estimator, env)
    for episode in range(200):
        agent.train(epsilon = 1 / (episode + 1))
    
if __name__ == '__main__':
    main()