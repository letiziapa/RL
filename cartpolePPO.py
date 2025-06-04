import os
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict
from collections import deque
import random
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

np.random.seed(42)
#simple PPO implementation on classic cartpole environment
#environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0] #4
action_size = env.action_space.n #2
#hyperparameters
gamma = 0.99 #discount factor
lr_actor = 0.001 #learning rate for actor network
lr_critic = 0.001 #learning rate for critic network
clip_ratio = 0.2 #the 'epsilon' in the clip function (PPO clip ratio)
epochs = 10 #number of epochs to train on each batch
batch_size = 64 # batch size for optimisation

#start by defining the agent
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.dense_layer = nn.Linear(state_size, 32)
        self.policy_layer = nn.Linear(32, action_size)
        self.value_layer = nn.Linear(32, 1)
    def forward(self, state):
        x = F.relu(self.dense_layer(state))
        policy_logits = self.policy_layer(x)
        value = self.value_layer(x)
        return policy_logits, value

def ppo_loss(model, optimizer, old_logits, old_values, advantages, states, actions, returns, clip_ratio, epochs, action_size):
    def compute_loss(logits, values, actions, returns, advantages, old_logits):
        #policy loss
        policy = F.softmax(logits, dim=-1)
        old_policy = F.softmax(old_logits, dim=-1)

        actions_one_hot = F.one_hot(actions, num_classes=action_size).float()
        
        action_probs = torch.sum(policy * actions_one_hot, dim=1)
        old_action_probs = torch.sum(old_policy * actions_one_hot, dim=1)

        ratio = torch.exp(torch.log(action_probs+1e-10) - torch.log(old_action_probs+1e-10))
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))

        #value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns)

        #entropy bonus
        entropy = torch.sum(policy * torch.log(policy + 1e-10), dim=1)
        entropy_bonus = torch.mean(entropy)

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        return total_loss
    
    def get_advantages(returns, values):
        #compute advantages 
        advantages = returns - values.squeeze(-1)
        return (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    
    advantages = get_advantages(returns, old_values)

    for _ in range(epochs):
        logits, values = model(states)
        loss = compute_loss(logits, values, actions, returns, advantages, old_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss

model = ActorCritic(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=lr_actor)

max_episodes = 1000
max_steps_per_episode = 1000

all_losses = []
episode_rewards = []
for episode in range(max_episodes):
    states, actions, rewards, values = [], [], [], []
    state, _ = env.reset()
    

    for step in range(max_steps_per_episode):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(state_tensor)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample().item()
        next_state, reward, done, _ , _ = env.step(action)
        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        state = next_state
        if done:
            #compute discounted returns
            returns = []
            discounted_sum = 0
            for r in reversed(rewards):
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            
            states_tensor = torch.cat(states, dim = 0)
            actions_tensor = torch.tensor(actions, dtype=torch.long)
            returns_tensor = torch.tensor(returns, dtype=torch.float32)
            values_tensor = torch.cat(values, dim=0).detach()

            with torch.no_grad():
                old_logits, _ = model(states_tensor)
            
            loss = ppo_loss(
                model = model,
                optimizer = optimizer,
                old_logits = old_logits.detach(), 
                old_values = values_tensor,
                advantages = returns_tensor - values_tensor.squeeze(-1),
                states = states_tensor, 
                actions = actions_tensor, 
                returns = returns_tensor, 
                clip_ratio = 0.2, 
                epochs = 10, 
                action_size = action_size)
            
            all_losses.append(loss.item())
            episode_rewards.append(sum(rewards))
            
            
            print(f"Episode {episode+1}, Loss: {loss.item():.4f}, Reward: {sum(rewards)}")
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Average Reward over last 10 episodes: {avg_reward:.2f}")
            break
plt.figure(figsize=(10, 5))
plt.plot(all_losses, color = 'orange')
plt.title('PPO Training Loss over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.grid()

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.title('Episode Rewards over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.grid()
plt.show()

# Evaluation function and call
def evaluate(model, env, episodes=5):
    model.eval()
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
            state, reward, terminated , truncated , _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        print(f"Evaluation Episode {ep+1}: Total Reward = {total_reward}")

print("\nEvaluating trained model...")
evaluate(model, env)

  
