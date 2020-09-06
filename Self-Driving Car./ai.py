# AI programming for Self Driving Car

# Importing the libraries 

import numpy as np 
import random 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd 
from torch.autograd import Variable

# Creating the architecture

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action): # input_size is the number of neurons in the input layer. nb_action in the number of possible actions.
        super(Network, self).__init__()  # Basically the input layer is input_size and the output layer is nb_actions
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) # fc1 is the full connection between the input and the hidden layer, we've taken 30 neurons in the hidden layer
        self.fc2 = nn.Linear(30 , nb_action) # fc2 is the full connection between the hidden and the output layer.
        
    def forward(self, state):
        x = F.relu(self.fc1(state)) # We activate hidden neurons , x , using fc1. We use the rectifier function i.e. relu.
        q_values = self.fc2(x) # We generate the output using fc2
        return q_values
    
# Implement Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity): # capacity is the number of previous instances it'll take into consideration as experience.
        self.capacity = capacity
        self.memory = [] # the memory is basically an empty list which would not exceed the capacity
        
    def push(self , event): # event is the tuple of 4 elements which are , last_state , new_state , last_action and new_action
        self.memory.append(event) # We store the event elements in the memory.
        if len(self.memory) > self.capacity:
            del self.memory[0]  # If the memory exceeds the capacity , the first memory is deleted.
            
    def sample(self, batch_size): # batch_size is the number of samples we take at a time
        # if list = ( (1,2,3) , (4,5,6)),  then zip(*list) = ((1,4),(2,3),(5,6)) , basically zip reshapes a list
        samples = zip(*random.sample(self.memory, batch_size)) # We reshape the lists and variables to a torch tensor
        return map(lambda x: Variable(torch.cat(x,0)) , samples) # We return a pytorch variable which contains a tensor and a variable
        # lambda is the function which is going to convert our samples into a pytorch variable which corresponds to the states. 
        # This is done to achieve an alignment where the state , the action and the reward corresponds to the same time t.
    
# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self , input_size, nb_action, gamma): # gamma is the delay coefficient
        self.gamma = gamma
        self.reward_window = [] # The mean of the last 100 rewards
        self.model = Network(input_size , nb_action) # We create the neural network for the Deep Q Model
        self.memory = ReplayMemory(100000) # Memory size is 100000
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) # We choose Adam as our optimizer ,  lr is the learning rate
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # We make a tensor which contains inputs , where the first dimension is last_state
        self.last_action = 0 # Since we're initializing 
        self.last_reward = 0
       
    def select_action(self , state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*0) # T = 0 , temperature parameter. 
        #softmax([1,2,3]) = [0.04, 0.11 , 0.85] => softmax ([1,2,3]*3) = [0,0.02, 0.98] , the higher the temperature value the more chances of high probablity actions.
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def learn(self , batch_state , batch_next_state, batch_reward , batch_action):
        outputs = self.model(batch_state).gather(1 , batch_action.unsqueeze(1)).squeeze(1) # We gather the output of the action only , and we crate a fake dimension for that , then we remove the fake dimension.
        next_outputs = self.model(batch_next_state).detach().max(1)[0] # We gather the max output of the next batch
        target = self.gamma*next_outputs + batch_reward # We calculate the target
        td_loss = F.smooth_l1_loss(outputs, target) # Calculate the temporal difference loss
        self.optimizer.zero_grad()  # Re initializes the optimizer for every iteration of the loop
        td_loss.backward(retain_variables = True) # Back propagates the error into neural networks
        self.optimizer.step() #Updates the weights
        
    def update(self , reward , new_signal): # Updates all the elements in transition and select new action
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) # Create a torch sensor with float values and add a fake dimension 
        self.memory.push((self.last_state , new_state , torch.LongTensor([int(self.last_action)]) , torch.Tensor([self.last_reward]))) # Basically pushes the new information in the memory in form of tensor.
        action = self.select_action(new_state) # We play the new action in the new state.
        if len(self.memory.memory) > 100: # Memory is limited to 100 samples if exceeds 100
           batch_state , batch_next_state, batch_reward , batch_action = self.memory.sample(100)
           self.learn(batch_state , batch_next_state, batch_reward , batch_action)
        self.last_action = action 
        self.last_state = new_state # Last state becomes new state.
        self.last_reward = reward # Last reward becomes new reward or reward.
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000: # Deletes first reward if rewards exceed 1000
            del self.reward_window[0]
        return action 
    
    def score(self): # Compute the scores i.e mean of the rewards
        return sum(self.reward_window)/(len(self.reward_window)+1) # We add +1 in order for the denominator to never be zero
    
    def save(self):
        torch.save({
            'state_dict' : self.model.state_dict() , 
            'optimizer' : self.optimizer.state_dict() ,
                    }, 'last_brain.pth') # Saves as dictionary on a .pth file
        
    def load(self): # To load the saved file
        if os.path.isfile('last_brain.pth'):
            print('= > Loading the model ...')
            checkpoint = torch.load('last_brain.pth') # Save the file as a variable
            self.model.load_state_dict(checkpoint['state_dict']) # Load the model from state_dict
            self.optimizer.load_state_dict(checkpoint['optimizer']) # Load the optimizer fromm optimizer
            print('... Done!')
        else : 
            print('no checkpoint found ...')
    
            
        
        
        
        
        
        
        
        
    
    
    
        
