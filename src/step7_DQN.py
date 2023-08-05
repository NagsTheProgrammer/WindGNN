import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from step2_graph_builder import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_time_steps, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_time_steps)
        self.layer2 = nn.Linear(n_time_steps, n_time_steps)
        self.layer3 = nn.Linear(n_time_steps, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
episode_durations = []

def createActions():
    actions = []
    for i in range(34):
        for j in range(34):
            if i != j:
                action = [i,j,0.01]
                actions.append(action)
                action = [i,j,-0.01]
                actions.append(action)
    return actions

class DQNTrainer(object):
    def __init__(self, model, graph, wind_max, wind_min, actions):
        self.model = model
        self.graph = graph
        self.wind_max = wind_max
        self.wind_min = wind_min
        self.actions = actions

        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4

        self.n_actions = len(actions)

        self.steps_done = 0

    def step(self, modelAdj, attr, action, truth):
        action = action.detach().cpu().numpy()
        graph = self.graph
        graph[action[0][0][0][0]][action[0][0][0][1]] = graph[action[0][0][0][0]][action[0][0][0][1]] + action[0][0][0][2]

        terminated = bool( np.max(self.graph) > 1 or np.min(self.graph) < 0 )

        lab = truth.detach().cpu().numpy() * (self.wind_max - self.wind_min) + self.wind_min

        modelOutput = self.model(modelAdj,attr)
        modelOut = modelOutput.detach().cpu().numpy() * (self.wind_max - self.wind_min) + self.wind_min

        previousAdj = torch.tensor(build_A_star(self.graph)).float().to(device)
        previousOutput = self.model(previousAdj, attr)
        previousOut = previousOutput.detach().cpu().numpy() * (self.wind_max - self.wind_min) + self.wind_min

        currentAdj = torch.tensor(build_A_star(graph)).float().to(device)
        currentOutput = self.model(currentAdj, attr)
        currentOut = currentOutput.detach().cpu().numpy() * (self.wind_max - self.wind_min) + self.wind_min
        
        modelDiff = abs(lab[0][-1] - modelOut[-1])
        previousDiff = abs(lab[0][-1] - previousOut[-1])
        currentDiff = abs(lab[0][-1] - currentOut[-1])

        reward = 1.0

        if np.mean(currentDiff) > np.mean(modelDiff):
            reward = -1.0

        if np.mean(currentDiff) < np.mean(modelDiff):
            reward = -1.0

        if np.mean(currentDiff) > np.mean(previousDiff):
            reward = -1.0

        if np.max(currentDiff) > np.max(currentDiff):
            reward = -2.0
        
        if np.mean(currentDiff) < np.mean(previousDiff):
            reward = 1.0

        if np.max(currentDiff) < np.max(previousDiff):
            reward = 2.0
            if np.max(currentDiff) < 15:
                reward = 10.0

        return graph, reward, terminated, {}

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                x = self.policy_net(state)
                b0 = x.max(3)[0].max(2)[0].max(1)[0].max(0)[1]
                b1 = x.max(3)[0].max(2)[0].max(1)[1][b0]
                b2 = x.max(3)[0].max(2)[1][b0][b1]
                sample = x.max(3)[1][b0][b1][b2]
                return torch.tensor([[[self.actions[sample]]]], device=device, dtype=torch.long)
        else:
            return torch.tensor([[[random.choice(self.actions)]]], device=device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(3)[0].max(2)[0].max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def train(self, adj, attr, truth):
        # TRAINING
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer
        
        # Get the number of state observations
        primeState = torch.matmul(adj, attr)
        graph = self.graph
        n_observations = primeState.size(3)
        n_time_steps = primeState.size(1)

        self.policy_net = DQN(n_observations, n_time_steps, self.n_actions).to(device)
        self.target_net = DQN(n_observations, n_time_steps, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        for i in range(2):
            graph = self.graph
            state = primeState

            for j in range(150):
                print(j)
                action = self.select_action(state)
                
                graph, reward, terminated, _ = self.step(adj, attr, action, truth)
                reward = torch.tensor([reward], device=device)
                
                if terminated:
                    next_state = None
                else:
                    tempAdj = torch.tensor(build_A_star(graph)).float().to(device)
                    next_state = torch.matmul(tempAdj,attr)

                self.memory.push(state, action, next_state, reward)
                state = next_state

                self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if terminated:
                    break
            
            output = self.model(tempAdj,attr)
            modelOut = output.detach().cpu().numpy() * (self.wind_max - self.wind_min) + self.wind_min
            lab = truth.detach().cpu().numpy() * (self.wind_max - self.wind_min) + self.wind_min
            modelDiff = abs(lab[0][-1] - modelOut[-1])
            avgDiff = np.mean(modelDiff)
            print("generation: %d, diff: %1.5f" % (i, avgDiff))
            self.steps_done = 0
        
        return torch.tensor(build_A_star(graph)).float().to(device)
            