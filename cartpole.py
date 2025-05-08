import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#env = gym.make('CartPole-v1', render_mode="human")
env = gym.make('CartPole-v1')

#environment characteristics
observation_space = env.observation_space
action_space = env.action_space
print(f"""Observation space:
0- cart position
1- cart velocity
2- pole angle (radians)
3- pole angular velocity
low bounds: {observation_space.low}, 
high bounds: {observation_space.high}, 
shape: {observation_space.shape}, 
type: {observation_space.dtype}""")

print(f"""\nAction space: {action_space.n}
0- push cart to the left
1- push cart to the right""")

#define the agents
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        #randomly select an action
        return self.action_space.sample()
    
class LearningAgent:
    def __init__(self, env, alpha = 0.1, epsilon = 0.9, gamma = 0.99, episodes = 1000, is_random = False, total_episodes_trained = 0, render = False):
        
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        #self.bins = bins
        self.episodes = episodes
        self.epsilon_decay = epsilon / episodes
        self.q_table = self._create_q_table()
        self.render = render
        self.is_random = is_random
        self.total_episodes_trained = total_episodes_trained

    def digitize_state(self, state):
        position_bins = np.linspace(-2.4, 2.4, 10)
        velocity_bins = np.linspace(-4, 4, 10)
        angle_bins = np.linspace(-0.2095, 0.2095, 10)
        angular_velocity_bins = np.linspace(-4, 4, 10)
        # Discretize the state using the bins

        new_position = np.digitize(state[0], position_bins)
        new_velocity = np.digitize(state[1], velocity_bins)
        new_angle = np.digitize(state[2], angle_bins)
        new_angular_velocity = np.digitize(state[3], angular_velocity_bins)
        # Return the discretized state as a tuple
        new_state_digitized = [new_position, new_velocity, new_angle, new_angular_velocity]
        return new_state_digitized
    
    def _create_q_table(self):
        position_bins = np.linspace(-2.4, 2.4, 10)
        velocity_bins = np.linspace(-4, 4, 10)
        angle_bins = np.linspace(-0.2095, 0.2095, 10)
        angular_velocity_bins = np.linspace(-4, 4, 10)
        return np.zeros(
            (
                len(position_bins) + 1,
                len(velocity_bins) + 1,
                len(angle_bins) + 1,
                len(angular_velocity_bins) + 1,
                self.env.action_space.n,
            )
        )
    def act(self, state):
        """Selects an action based on the epsilon-greedy policy."""
        discrete_state = self.digitize_state(state)
 
        if self.is_random == True or np.random.uniform(0,1) < self.epsilon:
            #action = self.env.action_space.sample()
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(self.q_table[state[0], state[1], state[2], state[3], :])
       # print(f"Action: {action}")
        return action
    
    def train(self):
        self.cumulative_reward = []
        for episode in range(self.episodes):
            ep_reward = 0 #initialize episode reward counter
            state, _ = self.env.reset()
            #state = self.digitize_state(self.env.reset()[0])
            state = self.digitize_state(state)
            done = False
            while not done:
                #if self.render:
                #self.env.render()
                action = self.act(state)
                next_state, reward, done, _ , _ = self.env.step(action)
                next_state = self.digitize_state(next_state)
                ep_reward += reward
                #find the max q-value for the next state
                #max next q-value is found from the maximum of all possible actions
                #this is the agents estimate of the best possible future reward
                #for the next state
                max_next_q = np.max(self.q_table[
                    next_state[0],
                    next_state[1],
                    next_state[2],
                    next_state[3],
                    :,
                ])
                #update the q-table using the q-learning update rule
                #q(s,a) = q(s,a) + alpha * (r + gamma * max(q(s',a')) - q(s,a))
                self.q_table[
                    state[0],
                    state[1],
                    state[2],
                    state[3],
                    action,
                ] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[
                    state[0],
                    state[1],
                    state[2],
                    state[3],
                    action,
                ])
                ep_reward += reward
                state = next_state

            self.epsilon -= self.epsilon_decay
            self.cumulative_reward.append(ep_reward)
            mean_reward = np.mean(self.cumulative_reward[-100:])

            if episode % 100 == 0:
                print(
                    f"Episode: {episode + self.total_episodes_trained} Epsilon: {self.epsilon:0.2f}  Mean Rewards {mean_reward:0.1f}"
                )

            if mean_reward >= 195:
                print(f"Mean rewards: {mean_reward} - no need to train model longer")
                break
        self.env.close()
        
    def test(self):
        """Tests the agent in the environment."""
        #store the rewards for each episode
        self.cumulative_reward = []
        for episode in range(self.episodes):
            ep_reward = 0
            state, _ = self.env.reset()
           # state = self.digitize_state(self.env.reset()[0])
            state = self.digitize_state(state)
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ , _ = self.env.step(action)
                next_state = self.digitize_state(next_state)
                ep_reward += reward
                state = next_state
            
            self.cumulative_reward.append(ep_reward)
            mean_reward = np.mean(self.cumulative_reward[-100:])
            if episode % 100 == 0:
                print(
                    f"Episode: {episode + self.total_episodes_trained} Mean Reward: {mean_reward:0.1f}"
                )

       
# agent = LearningAgent(env, alpha=0.05, epsilon=0.99, gamma=0.99, episodes=args.episodes, render = True)
#agent.train()

if __name__ == "__main__":
    #choose whether to train or test the agent
    parser = argparse.ArgumentParser()
#     parser.add_argument(
#     "--episodes",
#     type=int,
#     default=1000,
#     help="Number of episodes to train or test the agent. Default is 1000.",
# )
    choice = input("Do you want to train or test the agent? (1: train/ 2: test): ")
    #args = parser.parse_args()
    episodes = input("Enter the number of episodes: ")
    try:
        episodes = int(episodes)
    except ValueError:
        print("Invalid input. Defaulting to 1000 episodes.")
        episodes = 1000

    agent = LearningAgent(env, alpha=0.05, epsilon=0.99, gamma=0.99, episodes=episodes, render = True)
    #agent = LearningAgent(env, alpha=0.05, epsilon=0.99, gamma=0.99, episodes=args.episodes, render = True)
    if choice == "1":
        print("Training the agent...")
        agent.train()
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, "cartpole_model.npy")
        np.save(model_path, agent.q_table)
        print(f"Model saved to {model_path}")
    elif choice == "2":
        # Load the trained model
        model_path = "models/cartpole_model.npy"
        if os.path.exists(model_path):
            agent.q_table = np.load(model_path)
            print(f"Model loaded from {model_path}")
            agent.test()
        else:
            print("Trained model not found. Please train the agent first.")
            agent.test()
    else:
        print("No action specified. Defaulting to training the agent.")
        agent.train()
    
    #plot cumulative rewards
    plt.plot(agent.cumulative_reward)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward vs Episode")
    plt.savefig("cumulative_reward.png")
    plt.show()

    # # Save the trained model
    # model_dir = "models"
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # model_path = os.path.join(model_dir, "cartpole_model.npy")
    # np.save(model_path, agent.q_table)
    # print(f"Model saved to {model_path}")

    # # Load the trained model
    # loaded_model_path = os.path.join(model_dir, "cartpole_model.npy")
    # if os.path.exists(loaded_model_path):
    #     agent.q_table = np.load(loaded_model_path)
    #     print(f"Model loaded from {loaded_model_path}")
    # else:
    #     print(f"Model not found at {loaded_model_path}")
    
    
  