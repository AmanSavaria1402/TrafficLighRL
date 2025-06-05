# import libraries
import torch
import pandas as pd
import numpy as np
import os
import cityflow
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import copy
import math
from collections import deque
import random

# class for environment
class PressureEnv:
    '''
        This class is the environment implemented in cityflow for a single intersection.
    '''
    def __init__(self, maxSteps, configPath=os.path.join('generated', 'config.json'), numThreads=1):
        # initializing the cityflow engine
        self.engine = cityflow.Engine(configPath, thread_num=numThreads)
        self.numSteps = 0 # to track how many steps have been taken
        self.maxSteps = maxSteps # the maximum number of steps allowed
        self.directions = [('road_0_1_0_0', 'road_1_1_1_0'), # left
              ('road_1_0_1_0', 'road_1_1_2_0'), # left
              ('road_2_1_2_0', 'road_1_1_3_0'), # left
              ('road_1_2_3_0', 'road_1_1_0_0'), # left
              ('road_0_1_0_1', 'road_1_1_0_1'), # straight
              ('road_1_0_1_1', 'road_1_1_1_1'), # straight
              ('road_2_1_2_1', 'road_1_1_2_1'), # straight
              ('road_1_2_3_1', 'road_1_1_3_1'), # straight
              ('road_0_1_0_2', 'road_1_1_3_2'), # right
              ('road_1_0_1_2', 'road_1_1_0_2'), # right
              ('road_2_1_2_2', 'road_1_1_1_2'), # right
              ('road_1_2_3_2', 'road_1_1_2_2') # right
              ]
        self.incoming = [t[0] for t in self.directions]
        self.capacity = 40 # capacity of the lanes
    
    def _getState(self, currTLPhase):
        '''
            This function returns the state the environment is in right now
        '''
        # get lanecounts
        laneCounts = self.engine.get_lane_vehicle_count()
        # add to a dictionary and return
        stArray = []
        cumLaneLenghts = {'road_0_1_0':0, 'road_2_1_2':0, 'road_1_2_3':0, 'road_1_0_1':0}
        for k,v in laneCounts.items():
            if k in self.incoming:
                stArray.append(v)
        # appending the current phase
        stArray.append(currTLPhase)
        
        return stArray
    
    def _getReward(self):
        '''
            This function returns the reward after taking the current state
        '''
        # NOTE: reward will be generated after the action is done, so we need to implement the do_action and simulate traffic for the next 10 seconds
        # after that, calculate the reward
        # get the lanelengths
        r = 0
        vicCounts = self.engine.get_lane_waiting_vehicle_count()
        for d in self.directions:
            # calculate the number of incoming and outgoing vehicles
            nIn = vicCounts[d[0]]
            nOut = vicCounts[d[1]]
            r_i = -1 * nIn * (1 - (nOut/self.capacity))
            r += r_i
        return r
    
    def _peformAction(self):
        '''
            This function will take action, which is setting the traffic light to a specific phase.
        '''
        pass
        # set trafficlight phase
        # simulate for the next 10 seconds
        self._step(10)

    def _step(self, t=10):
        '''
            This function steps the environment for the next t seconds.
        '''
        # NOTE TO SELF: rn, the interval is hardcoded to 1 second, same as the config definition, REMEMBER to make this dynamic
        finished = False
        for i in range(t):
            self.numSteps+=1
            if self.numSteps==self.maxSteps:
                finished = True
                break
            self.engine.next_step()
        return finished

    def take_action(self, action, t=10, intersection_id='intersection_1_1'):
        '''
            This is the main callable function for taking a step in the environment. It does the following:
                1. takes the action.
                2. simulates for the next t seconds.
                3. gets the reward
                4. get next state
            Action will be the index of the tl phase for the intersection defined as defined in the roadnet file for that intersection
        '''
        # take action, set the tl phase to the provided index
        self.engine.set_tl_phase(intersection_id, action)
        # run the engine
        finished = self._step(t)
        # get the state
        next_state = self._getState(action)
        # get the reward
        r = self._getReward()

        return next_state, r, finished
    
    def reset(self,currTLPhase):
        '''
            This function resets the environment to the original state.
        '''
        self.engine.reset()
        self.numSteps = 0
        # clearing the replay and the roadnetlog files
        open(os.path.join('generated', 'GeneratedRoadNetLogExpt.json'), 'w').close()
        open(os.path.join('generated', 'GeneratedReplayLogExpt.txt'), 'w').close()
        return self._getState(currTLPhase)
    

# declaring the environment
env = PressureEnv(maxSteps=3600)

# defining parameters
# Hyperparameters
PARAM_learning_rate = 0.001
PARAM_gamma = 0.99
PARAM_epsilon = 1.0
PARAM_epsilon_min = 0.01
PARAM_epsilon_decay = 35000
PARAM_batch_size = 64
PARAM_target_update_freq = 10
PARAM_memory_size = 10000
PARAM_episodes = 500
PARAM_TAU = 0.005
PARAM_device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# other helper functions
class DQN(nn.Module):
    '''
        This class defines the neural network used for the model.
    '''
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # creating the layers
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
    
class ReplayMemory:
    '''
        This class defines the replay memory
    '''
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.memory = deque(maxlen=self.bufferSize)

    def insert(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))
    
    def sample(self, batch_size):
        sampleBatch = random.sample(self.memory, batch_size)
        states, actions, rewards, nextStates, done = zip(*sampleBatch)
        return np.array(states), np.array(actions, dtype=int), np.array(rewards), np.array(nextStates), np.array(done)
    
    def size(self):
        return len(self.memory)
    
def selectAction(state, epsilon):
    '''
        This function selects and returns an action using the epsilon greedy method.
    '''
    # select a random number between 0 and 1
    rNum = random.random()
    # print('Rnum: ', rNum)
    if rNum<epsilon:
        # explore, select a random action
        return random.choice([0,1,2,3])
    else: # exploit, get the best action
        state = torch.tensor(state, device=PARAM_device, dtype=torch.float32).unsqueeze(0)
        qVals = policyNet(state)
        return torch.argmax(qVals).item() # the action that has the highest Q value
    
# defining the input and output dimensions
input_dim, output_dim = 13, 4
# Declaring the networks
policyNet = DQN(input_dim, output_dim)
targetNet = DQN(input_dim, output_dim)
# loading the state dict from policy to target so that they have the same starting weights
targetNet.load_state_dict(policyNet.state_dict())
# optimizer
optimizer = optim.Adam(policyNet.parameters(), lr=PARAM_learning_rate)
# moving the models to gpu
policyNet.to(PARAM_device)
targetNet.to(PARAM_device)

# copying the epsilon value to use
INUSE_epsilon = copy.deepcopy(PARAM_epsilon)
# initializing the memory
memory = ReplayMemory(PARAM_memory_size)

# Training the model
episodeRewards = [] # to store the episode rewards for logging/plotting purposes
stepsDone = 0
# creating a loop for training
for ep in range(PARAM_episodes):
    print('###-'* 30)
    print("Episode: ", ep)
    print("Epsilon Used for the episode: ", INUSE_epsilon)
    # reset the environment
    state = env.reset(0)
    epReward = []
    done = False

    # training in the episode
    while not done:
        # if stepsDone%250==0:
        #     print("250 Steps done!, ", stepsDone)
        # recording the previous action
        # recording previous action the previous TL phase
        # print("stepsDone: ", stepsDone)
        if stepsDone==0:
            prevAction = 0
        else:
            prevAction = action
        # print("Prev action: ", prevAction)
        # select an action and perform it
        action = selectAction(state, INUSE_epsilon)
        stepsDone+=1
        # print("ACTION SELECTED: ", action)
        nextSt, r, done = env.take_action(action = action)
        # print("nextSt: ", nextSt)
        # debug
        # if env.numSteps == 10800:
        #     print("\n!!!10800 steps done!!!\n")

        # store current state, action, reward, next state, done in replay memory
        memory.insert(state, action, r, nextSt, done)

        # update current state to next state
        state = nextSt
        epReward.append(r)

        # update model parameters
        # if memory does not have enough tuples to sample (batch_size), we cant sample so we continue the loop
        # print('Memory size: ', memory.size())
        if memory.size() < PARAM_batch_size:
            continue
        # else, we train the networks
        # take a sample
        stBatch, aBatch, rBatch, nstBatch, doneBatch = memory.sample(PARAM_batch_size)
        stBatch = torch.tensor(stBatch, device=PARAM_device, dtype=torch.float32)
        aBatch = torch.tensor(aBatch, device=PARAM_device).unsqueeze(1)
        rBatch = torch.tensor(rBatch, device=PARAM_device, dtype=torch.float32)
        nstBatch = torch.tensor(nstBatch, device=PARAM_device, dtype=torch.float32)
        # getting Q values for current states
        QVals = policyNet(stBatch).gather(1, aBatch)

        # Updating q values using target network
        with torch.no_grad():
            # next state q values using target network
            nextQVals = targetNet(nstBatch).max(1)[0]
            # updating the q values for policy network
            targetQVals = rBatch + PARAM_gamma * nextQVals

        # calculating the loss
        lossFunction = nn.SmoothL1Loss()
        loss = lossFunction(QVals, targetQVals.unsqueeze(1))

        # updating model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # updaing the target network periodically
        if ep%PARAM_target_update_freq==0:
            # targetNetStateDict = targetNet.state_dict()
            # policyNetStateDict = policyNet.state_dict()
            # for key in policyNetStateDict:
            #     targetNetStateDict[key] = policyNetStateDict*PARAM_tau
            targetNet.load_state_dict(policyNet.state_dict())
        

    # decay epsilon after episode is complete
    INUSE_epsilon = PARAM_epsilon_min + (PARAM_epsilon - PARAM_epsilon_min) * math.exp(-1. * stepsDone/PARAM_epsilon_decay)
    episodeRewards.append(epReward)

    # printing training stats
    print("EPISODE COMPLETE")
    print(f"min_reward: {min(epReward)} || max_reward: {max(epReward)} || total_reward: {sum(epReward)} || average_reward: {np.mean(epReward)}")


# saving the models
# need to save everything
import pickle
# saving the models
torch.save(targetNet, "models/TargetNetv5.pth")
torch.save(policyNet, "models/PolicyNetv5.pth")

# saving the lists
with open('models/rewardListv5.pkl', 'wb') as f:
    pickle.dump(episodeRewards, f)