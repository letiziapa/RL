from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from noControl import matrix, AR_model
from collections import deque

    
class PendulumVerticalEnv(gym.Env):
    """
    Custom Gym environment to control a vertical pendulum system based on seismic input.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30}

    def __init__(self, interp_displacement, T = 1800, dt=1e-3, episode_length=350000, history_length=10000, seed: Optional[int] = 42):
        super(PendulumVerticalEnv, self).__init__()
        self.interp_displacement = interp_displacement
        self.dt = dt
        self.T = T 
        self.episode_length = episode_length
        self.current_step = 0
        self.history_length = history_length
        self.history = {"x6": [], "v6": [], "step": [], "force": [], "reward": []}
        self.max_force_mag = 5.0 
        
        # Physical parameters
        self.M = [160, 125, 82]
        self.K = [700, 1500, 564]
        self.gamma = [5, 5]

        self.A, self.B = matrix(*self.M, *self.K, *self.gamma, self.dt)
        self.AR_model = AR_model

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        obs_low = np.array([-10.] * 6 , dtype=np.float32)
        obs_high = np.array([10.] * 6 , dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.displacement_history = deque(maxlen = history_length)
        self.input_history = deque(maxlen = history_length)

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.zeros(6, dtype=np.float32)
        
        # Clear histories
        self.displacement_history.clear()
        self.input_history.clear()
        
        # Initialize histories with zeros
        for _ in range(self.history_length):
            self.displacement_history.append(0.0)
            self.input_history.append(0.0)
            
        return self._get_obs(), {}
        # self.done = False
        # self.current_step = 0
        # self.state = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)
        # return np.array(self.state, dtype=np.float32), {}
    def _get_obs(self):
        obs = self.state.copy()
        return obs
        # if len(self.displacement_history) > 0:
        #     disp_array = np.array(self.displacement_history)
        #     input_array = np.array(self.input_history)
        #     #rms of output displacement
        #     rms_out = np.sqrt(np.mean(disp_array**2))

        #     #peak-to-peak of recent output
        #     peak_out = np.max(np.abs(disp_array))

        #     #rms input
        #     rms_in = np.sqrt(np.mean(input_array**2))

        #     ratio = rms_out / (rms_in + 1e-12)

        #     features = np.array([rms_out, peak_out, rms_in, ratio], dtype = np.float32)
        # else:
        #     features = np.zeros(4, dtype = np.float32)
        # return np.concatenate([obs, features])



    def step(self, action):  
        # if self.done:
        #     raise RuntimeError("Episode has finished. Call reset().")
        if self.current_step >= self.episode_length:
            return self._get_obs(), 0.0, False, True, {}
        
        self.control_force = float(action[0]*self.max_force_mag)
        seismic_input = self.interp_displacement[self.current_step]
        self.input_history.append(seismic_input)

        force = self.K[0] * np.real(seismic_input) + self.control_force

        #self.control_force = float(action * self.max_force_mag) 
        # prev_displacement = self.state[5]
        # seismic_input = self.interp_displacement[self.current_step]
        # input_force = self.K[0] * np.real(seismic_input)
        # force = input_force + self.control_force
        
        self.state = self.AR_model(self.state, self.A, self.B, force)

        # self.history["x6"].append(self.state[5])  #vertical displacement
        # self.history["v6"].append(self.state[2])  #vertical velocity
        # self.history["step"].append(self.current_step)
        # self.history["force"].append(self.control_force)  #control force from the agent

        self.output_disp = self.state[5]  # vertical displacement
        self.displacement_history.append(self.output_disp)

        reward = self._compute_reward()
        #self.history["reward"].append(reward)
        self.current_step += 1
        terminated = bool(abs(self.output_disp) > 100.0) #changed from 100
        truncated = bool(self.current_step >= self.episode_length)
        #done = terminated or truncated

        #obs = np.array(self.state, dtype=np.float32)
        obs = self._get_obs()
        info = {
            'displacement': self.output_disp,
            'force': self.control_force,
            'seismic_input': seismic_input,
            'rewards': reward
        }
        return obs, reward, terminated, truncated, info
    # def _compute_reward(self, prev_disp, current_disp):
    #     if self.current_step == 0:
    #         return 0.0  # No reward at the first step
    #     disp = abs(current_disp)
    #     improvement = abs(prev_disp) - disp
    #     beta = 100.0 
    #     progress_bonus = beta * improvement #ignore this for now

    #     if current_disp <= 1e-6 and current_disp >= -1e-6:
    #         return 10.0 

    #     else:
    #         return -10.0  
    def _compute_reward(self):
        #evaluate over multiple timesteps
        if self.current_step < 100:
            return 0.0
        
        recent_out = np.array(list(self.displacement_history)[-1000:])
        recent_in = np.array(list(self.input_history)[-1000:])
        #recent_force = np.array(self.history["force"][-100:]) if self.history["force"] else np.zeros(100)


        rms_out = np.sqrt(np.mean(recent_out**2))
        rms_in = np.sqrt(np.mean(recent_in**2))
        #rms_force = np.sqrt(np.mean(recent_force**2))

    
        # if rms_in < 1e-12:
        #     return 0.0
        # transfer_ratio = rms_out / rms_in

        suppression_reward = -np.log10(rms_out / (rms_in + 1e-12) + 1e-10)
        #displacement_penalty = -np.log10(abs(self.output_disp) + 1e-12)
            # stronger penalty for absolute displacement (quadratic is sharper around 0)
        center_reward = - (self.output_disp**2)
        #alpha = 0.5 #changed from 0.5
        alpha, beta = 1.0, 0.1
        reward = alpha * suppression_reward + beta * center_reward
        #force_penalty = -0.01 * rms_force    # discourage large forces
        #stability_penalty = -10.0 if np.isnan(rms_out) else 0.0  # harsh penalty if unstable

        #reward = suppression_reward + alpha * displacement_penalty 

        return reward



    def render(self, mode='human'):        
        print(f"""
        Step: {self.current_step}
        x6: {self.state[5]:.6e} m
        v6: {self.state[2]:.6e} m/s
        Control Force: {self.history['force'][-1] if self.history['force'] else 0:.3f} N
        Differential displacement: {self.state[5] - (self.history["x6"][-2] if len(self.history["x6"]) >= 2 else 0):.6e} m
        Seismic Input: {self.interp_displacement[self.current_step]:.6e} m
        Reward: {self.history["reward"][-1] if self.history["reward"] else 0:.4f}
        """)
