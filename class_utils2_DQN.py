import numpy as np
import gym
import random
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import tensorflow as tf

obs_space = 8
action_space_size = 4
l1_neurons = 128
l2_neurons = 128
model_name = 'dqn_model.h5'
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

class dqn_agent(object):
    def __init__(self, gamma, epsilon, buffer_size, batch_size, epsilon_dec, epsilon_end):
        #Spare vals: , replace_target=100
        #self.epsilon_min = epsilon_end
        #self.replace_target = replace_target
        self.action_space = [0, 1, 2, 3]
        self.actions = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.buffer = replay_memory(buffer_size)
        self.q_eval = build_dqn()
        self.q_target = build_dqn()
        self.target_weight_copy = 100

    def save_vals(self):
        self.q_eval.save('C:\\Users\\alexa\\OneDrive\\Documents\\GitHub\\CM30359-2022-CW\\q_eval_DQN_bl.h5')
        self.q_target.save('C:\\Users\\alexa\\OneDrive\\Documents\\GitHub\\CM30359-2022-CW\\q_target_DQN_bl.h5')

    def store(self, state, action, reward, new_state, done):
        self.buffer.append_buffer(state, reward, action, new_state, done)

    def choose_action(self, state):
        state = np.array([state])   #neural network needs 2d array or 2d tensor 
        actions = []
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
        
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)
        return action, actions
    
    def print_buffer(self):
        self.buffer.print_buffer()
    
    def calculate(self):
        self.target_weight_copy += 1
        batch, flag = self.buffer.sample_buffer(self.batch_size)
  
        if(flag == False): #Memory Smaller than Batch Size
            #Batch cannot be considered as no batch was returned
            print("0") #Filler

        else: #Memory >= Batch Size
            state = batch[:,0]
            reward = batch[:,1]
            action = batch[:,2] #Perform Subscripting
            new_state = batch[:,3]
            done = batch[:,4]
            terminal = 1 - done.astype(int) #converting to integers, false is 0 but as we want the value function to produce 0 if it is a terminal state, we want flip the values. 
            new_state = np.array([np.array(x) for x in new_state])
            state = np.array([np.array(y) for y in state])
           
          
            q_target_evaluate = self.q_target.predict(new_state, verbose=0) #this is using the target value to produce the pred values in a 2d array.
            q_max_action = np.argmax(q_target_evaluate, axis=1) #e.g. [0.5, 0.2, 0.3, 0.7] then takes index of the best action, so in this      case it will return 3. as 0.7 is at index 3. 
            batch_index = np.arange(self.batch_size, dtype=np.int32) 
            target_evaluate = q_target_evaluate[batch_index, q_max_action] #indexing the max action value according to the eval. Using the q_max_action array, it uses the eval networks choices to select from action values that the target network produced. 
            
            target_value = reward+self.gamma*target_evaluate*terminal 
            
            target_matrix = self.q_eval.predict(state, verbose=0)
            target_matrix[batch_index, action.astype(int)] = target_value
            self.q_eval.fit(state, target_matrix, verbose=0)
            
            if self.target_weight_copy == 25:
               
                self.q_target.set_weights(self.q_eval.get_weights())
 
                self.target_weight_copy = 0
            if(self.epsilon >= self.epsilon_end):
                self.epsilon = self.epsilon * self.epsilon_dec
            else:
                self.epsilon = self.epsilon_end
           
           
        
         

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()