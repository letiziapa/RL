from environment import PendulumVerticalEnv
import numpy as np
import h5py
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor

f = h5py.File("/Users/letizia/Desktop/INFN/new model/SaSR_test.hdf5", "r")
seism = f['SR/V1:ENV_CEB_SEIS_V_dec'][:] #seismic data

seism = seism[:12500]*1e-6

env = PendulumVerticalEnv(seism, T = 200, dt = 1e-3, episode_length = 300000) 
env = Monitor(env)
#env = gym.make('CartPole-v1', render_mode = 'human')  
#Validate custom environment

#env.reset()

#print(f"Initial observation shape: {env.observation_space.shape}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")


model = PPO(
    "MlpPolicy", 
    env, 
    gamma = 0.99, 
    batch_size = 32, 
    learning_rate= 1e-4, 
    verbose=1, 
    tensorboard_log="./ppo_tensorboard/"
    )

model.learn(total_timesteps=800000, progress_bar=True)

obs, info = env.reset()
for i in range(300000):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    # if i % 10 == 0:
    #     print(f"\nStep: {i}, "
                
    #             f"\nAction: {action}, "
    #             f"\nReward: {rewards:.3f}")
    # print(f"Displacement: {info['displacement']:.3e}")
    # print(f"Seismic input: {info['seismic_input']:.3e}")
    # print(f"Action taken: {action}, Reward: {rewards}")
    
    if terminated or truncated:
        obs, _ = env.reset()

x6       = np.array(env.history["x6"])       # vertical displacements
rewards  = np.array(env.history["reward"])   # step-by-step rewards
actions  = np.array(env.history["force"])    # control forces you applied
steps    = np.array(env.history["step"])  


plt.figure()
plt.subplot(3,1,1)
plt.plot(x6)
plt.title("Displacement x6")
plt.subplot(3,1,2)
plt.plot(actions)
plt.title("Control force (N)")
plt.subplot(3,1,3)
plt.plot(rewards)
plt.title("Reward")
plt.tight_layout()
plt.show()
# # # Create a new instance of the environment for evaluation
# # eval_env = PendulumVerticalEnv(render_mode='human')

# # # Wrap it with Monitor
# # eval_env = Monitor(eval_env)
# # evaluation_results = evaluate_policy(model, eval_env, n_eval_episodes=100, return_episode_rewards=True)
# analysis = env.get_transfer_function_analysis()
# env.plot_transfer_function_comparison(analysis)