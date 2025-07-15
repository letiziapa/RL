from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from noControl import matrix, AR_model

    
class PendulumVerticalEnv(gym.Env):
    """
    Custom Gym environment to control a vertical pendulum system based on seismic input.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30}

    def __init__(self, seism, interp_displacement, T, dt=1e-3, episode_length=350000, num_envs=1):
        super(PendulumVerticalEnv, self).__init__()
        self.num_envs = num_envs
        self.interp_displacement = interp_displacement
        self.seism = seism
        self.dt = dt
        self.T = T 
        self.episode_length = episode_length
        self.current_step = 0
        self.history = {"x6": [], "v6": [], "step": [], "force": [], "reward": []}
        self.max_force_mag = 10.0 
        
        # Physical parameters
        self.M = [160, 125, 82]
        self.K = [700, 1500, 564]
        self.gamma = [5, 5]
        self.physical_params = [*self.M, *self.K, *self.gamma, self.dt]

        self.A, self.B = matrix(*self.M, *self.K, *self.gamma, self.dt)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        obs_low = np.array([-10.] * 6 , dtype=np.float32)
        obs_high = np.array([10.] * 6 , dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)


    def reset(self, seed: Optional[int] = 42):
        self.done = False
        self.current_step = 0
        self.state = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)
        return np.array(self.state, dtype=np.float32), {}


    def step(self, action):  
        if self.done:
            raise RuntimeError("Episode has finished. Call reset().")

        self.control_force = float(action * self.max_force_mag) 
        prev_displacement = self.state[5]
        seismic_input = self.interp_displacement[self.current_step]
        force = self.K[0] * np.real(seismic_input)
        total_force = force + self.control_force
        
        self.state = AR_model(self.state, self.A, self.B, total_force)

        self.history["x6"].append(self.state[5])  #vertical displacement
        self.history["v6"].append(self.state[2])  #vertical velocity
        self.history["step"].append(self.current_step)
        self.history["force"].append(self.control_force)  #control force from the agent

        output = self.state[5]  # vertical displacement

        reward = self._compute_reward(prev_displacement, output)
        self.history["reward"].append(reward)

        self.current_step += 1
        terminated = bool(abs(self.state[5]) > 10)
        truncated = bool(self.current_step > self.episode_length )
        self.done = terminated or truncated

        obs = np.array(self.state, dtype=np.float32)

        return obs, reward, terminated, truncated, {"displacement":output}
    
    def _compute_reward(self, prev_disp, current_disp):
        if self.current_step == 0:
            return 0.0  # No reward at the first step
        disp = abs(current_disp)
        improvement = abs(prev_disp) - disp
        beta = 1000.0 
        progress_bonus = beta * improvement #ignore this for now

        if current_disp <= 1e-6 and current_disp >= -1e-6:
            return 10.0 
        # elif disp <= 1e-3 and disp > 1e-6:
        #      return 1.0 
        # elif disp <= 1e-3 and disp > 1e-4:
        #     return 0.3 
        # elif disp <= 1e-2 and disp > 1e-3:
        #     return 0.1 
        # elif disp <= 9e-1 and disp > 1e-2:
        #     return -1.0
        else:
            return -10.0
        # alpha = 10.0
        # main_reward = np.exp(-alpha * current_disp**2)
        # progress_bonus = 0.0
        # if len(self.history["x6"]) >= 2:
        #     prev_disp = self.history["x6"][-2]
        #     improvement = abs(prev_disp) - abs(current_disp)
        #     if improvement > 1e-6:
        #     # beta scales the importance of the progress bonus
        #         beta = 1.0 
        #         progress_bonus = beta * improvement

        # return main_reward 
  


    def render(self, mode='human'):        
        print(f"""
        Step: {self.current_step}
        x6: {self.state[5]:.6e} m
        v6: {self.state[2]:.6e} m/s
        Control Force: {self.control_force:.3f} N
        Differential displacement: {self.state[5] - (self.history["x6"][-2] if len(self.history["x6"]) >= 2 else 0):.6e} m
        Seismic Input: {self.interp_displacement[self.current_step]:.6e} m
        Reward: {self.history["reward"][-1] if self.history["reward"] else 0:.4f}
        """)
