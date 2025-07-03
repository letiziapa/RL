import math
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import h5py
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
#from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from noControl import matrix, AR_model, evolution 
from scipy.interpolate import interp1d
    
class PendulumVerticalEnv(gym.Env):
    """
    Custom Gym environment to control a vertical pendulum system based on seismic input.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30}

    def __init__(self, seism, T, dt=1e-3, episode_length=350000, num_envs=1):
        super(PendulumVerticalEnv, self).__init__()
        self.num_envs = num_envs
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

        self.history = {
            "x6": [],
            "v6": [],
            "step": [],
            "force": [],
            "reward": []
        }

        # Actions: discrete set of forces [0, ..., +10]
        # self.force_levels = np.linspace(0, 10, 11)
        # self.action_space = spaces.Discrete(len(self.force_levels))

        # #observation space: [v1, v2, v6, x1, x2, x6, seismic_input]
        # #TODO: check if the upper and lower bounds for seismic input make sense or if it's better to just have inf bounds
        # #if min / max bounds make sense maybe apply to x and v as well
        # obs_low = np.array([-np.inf] * 6 + [min(self.zt)], dtype=np.float32)
        # obs_high = np.array([np.inf] * 6 + [max(self.zt)], dtype=np.float32)
        # self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        #---------prepare the data-----------#
        #velocity in frequency domain
        vf = np.fft.fft(seism)
        
        #frequencies
        self.frequencies = np.fft.fftfreq(len(seism), d=1/62.5)
        
        #displacement in frequency domain
        zf = np.zeros_like(vf, dtype=complex)
        nonzero = self.frequencies != 0
        #avoid division by zero
        zf[nonzero] = vf[nonzero] / (1j * 2 * np.pi * self.frequencies[nonzero])
        #displacement in time domain
        self.zt = np.fft.ifft(zf).real
        
        #acceleration in frequency domain
        acc = vf * (self.frequencies * 2 * np.pi * 1j)
        #acceleration in time domain
        self.At = np.fft.ifft(acc).real
        
        # Time vector for original data
 
        self.tt_data = np.arange(0, self.T, 1/62.5)

        #simulation time vector
        self.tmax = self.T*1e3 #1800000 s
        self.tt_sim = np.arange(0, self.tmax, self.dt)
        
        #interpolate the data to match the simulation time vector
        self.interp_displacement = interp1d(self.tt_data, self.zt, kind='linear',
                                            bounds_error=False, fill_value=0.0)(self.tt_sim)
        self.interp_acceleration = interp1d(self.tt_data, self.At, kind='linear',
                                                bounds_error=False, fill_value=0.0)(self.tt_sim)
        self.interp_velocity = interp1d(self.tt_data, seism, kind='linear',
                                                bounds_error=False, fill_value=0.0)(self.tt_sim)
        
        self.v0 = self.seism[3000]  # initial velocity
        self.x0 = self.zt[3000]  # initial displacement
        #self.force_levels = np.linspace(0, 10, 11)
        #force_values = np.linspace(-10, 10, 20)
        #self.action_space = spaces.Discrete(20)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        #observation space: [v1, v2, v6, x1, x2, x6, seismic_input]
        #TODO: check if the upper and lower bounds for seismic input make sense or if it's better to just have inf bounds
        #if min / max bounds make sense maybe apply to x and v as well
        obs_low = np.array([-100.] * 6 , dtype=np.float32)
        obs_high = np.array([100.] * 6 , dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)


    def reset(self, seed: Optional[int] = 42):
        self.done = False
        self.current_step = 0
        self.state = np.array([self.v0, 0., 0., self.x0, 0., 0.], dtype=np.float32)
        for key in self.history:
            self.history[key] = []
        return np.array(self.state, dtype=np.float32), {}


    def step(self, action):  
        if self.done:
            raise RuntimeError("Episode has finished. Call reset().")
        
        def force_function(t, k, displacement):
            F = k * displacement
            return F
        
        #force_values = np.linspace(-10., 10., 20)
        #self.control_force = force_values[action]
        self.control_force = float(action * self.max_force_mag) 
        prev_displacement = self.state[5]
        seismic_input = self.interp_displacement[self.current_step]
        force = self.K[0] * seismic_input
        total_force = force + self.control_force
        
        self.state = AR_model(self.state, self.A, self.B, total_force)

        self.history["x6"].append(self.state[5])  # vertical displacement
        self.history["v6"].append(self.state[2])  # vertical velocity
        self.history["step"].append(self.current_step)
        self.history["force"].append(self.control_force)  # control force

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
        if current_disp == 0:
            return 0.0  # No reward at the first step
        disp = abs(current_disp)
        improvement = abs(prev_disp) - disp
        beta = 1000.0 
        progress_bonus = beta * improvement
        #print(f'Progress bonus = {beta} * {improvement} = {progress_bonus}')

        if current_disp <= 1e-5 and current_disp >= -1e-9:
            #print(f'Current displacement: {current_disp}')
            return 100.0 
        # elif disp <= 1e-3 and disp > 1e-6:
        #      return 1.0 
        # elif disp <= 1e-3 and disp > 1e-4:
        #     return 0.3 
        # elif disp <= 1e-2 and disp > 1e-3:
        #     return 0.1 
        # elif disp <= 9e-1 and disp > 1e-2:
        #     return -1.0
        else:
            return 0.0
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
  
        
    def _compute_reward_smooth_displacement(self, prev_disp, current_displacement):
        """Smooth exponential reward"""
        abs_disp = abs(current_displacement)
        # Exponential decay with scaling
        k = 10000  # Adjust this based on your displacement scale
        reward = np.exp(-k * abs_disp)
        improvement = abs(prev_disp) - abs(current_displacement)
        beta = 500.
        final_reward = reward + (beta * improvement)
        return final_reward 



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
