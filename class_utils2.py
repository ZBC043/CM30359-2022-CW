import numpy as np
import gym
import random
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam

obs_space = 8
action_space_size = 4
l1_neurons = 128
l2_neurons = 128
model_name = 'ddqn_model.h5'
learning_rate = 0.001 #by default = 0.001 change by passing in Adam(lr=learning_rate)

class replay_memory():
    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size 
        self.buffer = np.empty(shape=(max_buffer_size), dtype=object)
        self.buffer_idx = 0
        self.size = 0
        
    def print_buffer(self):
        print(self.buffer)
        print(self.buffer.shape)
    
    def append_buffer(self, state, reward, action, new_state, done):
        self.buffer[self.buffer_idx] = [state, reward, action, new_state, done]
        self.size = min(self.size+1, self.max_buffer_size)
        self.buffer_idx = (self.buffer_idx + 1) % self.max_buffer_size
        
    def sample_buffer(self, batch_size):
        batch_found = False
        if self.size >= batch_size:
            batch_found = True
            batch = np.random.choice(self.buffer[0:self.size], batch_size)
            batch = np.vstack(batch)
            return batch, batch_found

        return 0, batch_found


def build_dqn():
    model = Sequential([
                Dense(l1_neurons, input_shape=(obs_space,)),
                Activation('relu'),
                Dense(l2_neurons),
                Activation('relu'),
                Dense(action_space_size)])

    model.compile(optimizer='Adam', loss='huber_loss')

    return model

class ddqn_agent(object):
    def __init__(self, gamma, epsilon, buffer_size, batch_size):
        #Spare vals: , epsilon_dec=0.996, epsilon_end=0.01, replace_target=100
        #self.epsilon_dec = epsilon_dec
        #self.epsilon_min = epsilon_end
        #self.replace_target = replace_target
        self.action_space = [0, 1, 2, 3]
        self.actions = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.buffer = replay_memory(buffer_size)
        self.q_eval = build_dqn()
        self.q_target = build_dqn()

    def store(self, state, action, reward, new_state, done):
        self.buffer.append_buffer(state, reward, action, new_state, done)

    def choose_action(self, state):
        state = np.array([state])   #neural network needs 2d array or 2d tensor 
        actions = []
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
        
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action, actions
    
    def print_buffer(self):
        self.buffer.print_buffer()
    
    def calculate(self):
        batch, flag = self.buffer.sample_buffer(self.batch_size)
        if(flag == False): #Memory Smaller than Batch Size
            
            #Batch cannot be considered as no batch was returned
            print("0") #Filler

        else: #Memory >= Batch Size
            print("Batch")
            action_list = batch[:,2] #Perform Subscripting
            return action_list, batch
        
        


    def update_network_parameters(self):
        self.q_target.model.set_weights(self.q_eval.model.get_weights())

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()