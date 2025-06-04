import os
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        #the NN takes the state as input and outputs the q values for each action
        # n outputs = number of actions
        # n inputs = number of features in the state
        #(cart position, cart velocity, pole angle, pole angular velocity)
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_inputs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class DQNAgent:
    def __init__(self,
                 state_space,
                 action_space,
                 episodes = 500):
        self.action_space = action_space
        
        self.memory = deque(maxlen=10000) #experience replay buffer, can change length (i.e. number of past experiences to store)
        #stores past experiences as tuples (sars+done)
        self.gamma = 0.9 # discount factor
        #exploration - exploitation
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay**(1./float(episodes))
        
        #DQN weights
        self.weights_file = "dqn_weights_cartpole.pth"
        n_inputs = state_space.shape[0]
        n_outputs = action_space.n

        self.q_model = QNetwork(n_inputs, n_outputs)
        self.target_model = QNetwork(n_inputs, n_outputs)

        self.replay_counter = 0

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_model.parameters(), lr = 0.001)

    def save_weights(self):
            torch.save(self.q_model.state_dict(), self.weights_file)
            print(f"Saved weights to {self.weights_file}")

    def load_weights(self):
            if os.path.exists(self.weights_file):
                self.q_model.load_state_dict(torch.load(self.weights_file))
                self.target_model.load_state_dict(torch.load(self.weights_file))
                print(f"Loaded weights from {self.weights_file}")
            else:
                print(f"File {self.weights_file} does not exist")
        
    def update_weights(self):
            #update the weights of the target model
            self.target_model.load_state_dict(self.q_model.state_dict())

    def act(self, state):
            if np.random.rand() < self.epsilon:
                return self.action_space.sample()
           
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.q_model(state_tensor)
            action = torch.argmax(q_values).item()
            return action
            
    def remember(self, state, action, reward, next_state, done):
            item = (state, action, reward, next_state, done)
            self.memory.append(item) #add new experience to the replay buffer (self.memory)

    def get_target_q_value(self, next_state, reward):
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.target_model(next_state_tensor)
                max_q = torch.max(q_vals, dim = 1)[0].item()
            target_q = reward + self.gamma * max_q
            return target_q
        
    def replay(self, batch_size):
            #experience replay
            #allows the agent to learn from past experiences
            #by sampling a batch of experiences from the replay buffer (self.memory)
            #sars = state, action, reward, next state
            if len(self.memory) < batch_size:
                return

            sars_batch = random.sample(self.memory, batch_size) #sample a random batch of experiences from the replay buffer
            #split the sampled experiences into individual batches
            state_batch, next_state_batch = [], []
            action_batch, reward_batch = [], []
            done_batch = []
            #correct format
            for state, action, reward, next_state, done in sars_batch:
                state_batch.append(np.array(state).squeeze())
                next_state_batch.append(np.array(next_state).squeeze())
                action_batch.append(action)
                reward_batch.append(reward)
                done_batch.append(done)
            state_batch = torch.FloatTensor(np.array(state_batch))
            next_state_batch = torch.FloatTensor(np.array(next_state_batch))
            action_batch = torch.LongTensor(np.array(action_batch))
            reward_batch = torch.FloatTensor(np.array(reward_batch))
            done_batch = torch.BoolTensor(np.array(done_batch))

            #get the q values for the current state
            q_values = self.q_model(state_batch) #pass the current state to the q network

            with torch.no_grad():
                next_q_values = self.target_model(next_state_batch) #target model (next state)
            #get the max q value for the next state
                max_next_q = torch.max(next_q_values, dim = 1)[0]
            #compute the target q value
            target_q_vals = reward_batch + (self.gamma * max_next_q * (~done_batch))
            #if done, set target q value to reward

            #exract the current q value for the action taken
            current_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
            #q_values has shape (batch_size, n_actions) --> pred q values
            #gather means that we are getting the q value for dimension 1 (action dimension)
            #unsqueeze gives the index of the action taken
            #what we get is a tensor of shape (batch_size, 1) with the q value for the action taken
            #squeeze removes the dimension of size 1 to only get a tensor of shape (batch_size)

            loss = self.loss_function(current_q, target_q_vals) #MSE loss between current q value and target q value
            self.optimizer.zero_grad() #avoid gradient accumulation
            loss.backward() #compute the gradients of the loss with respect to the model parameters
            self.optimizer.step() #update the model parameters using the optimizer

            #update the epsilon value for exploration-exploitation tradeoff
            self.update_epsilon() 

            if self.replay_counter % 10 == 0:
                #update the weights of the target model 
                #the target model is a delayed version of the q model
                #makes the training process more stable
                self.update_weights()
            self.replay_counter += 1
        
    def update_epsilon(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)

    # the number of trials required without failing
    win_trials = 100
    win_reward = 195

    #scores = deque(maxlen=win_trials)
    scores = []
    env = gym.make('CartPole-v1')
 
    agent = DQNAgent(env.observation_space, env.action_space)
    episodes = 1000
    state_size = env.observation_space.shape[0]
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.reshape(state, [1, state_size])
        done = False
        total_score = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_score += reward
        
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)
        scores.append(total_score)

        mean_score = np.mean(scores[-100:])
        if mean_score >= win_reward and episode >= win_trials:
            print(f"Solved in {episode} episodes, Mean Score: {mean_score:.2f} in {win_trials} episodes")
            agent.save_weights()
            break
        if (episode + 1) % 100 == 0:
            #agent.save_weights()
            print(f"Episode: {episode + 1}, Mean Score: {mean_score:.2f}")
        
    env.close()

    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Reward Over Time')
    plt.grid()
    plt.savefig('dqn_cartpole_rewards.png')
    plt.show()

            
