from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from noControl import matrix, AR_model
from collections import deque
from scipy import signal
from scipy.interpolate import interp1d
import h5py
    
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

    def _get_obs(self):
        obs = self.state.copy()
        return obs


    def step(self, action):  
        if self.current_step >= self.episode_length:
            return self._get_obs(), 0.0, False, True, {}
        
        self.control_force = float(action[0]*self.max_force_mag)
        seismic_input = self.interp_displacement[self.current_step]
        self.input_history.append(seismic_input)

        force = self.K[0] * np.real(seismic_input) + self.control_force
        
        self.state = self.AR_model(self.state, self.A, self.B, force)

        self.output_disp = self.state[5]  # vertical displacement
        self.displacement_history.append(self.output_disp)

        reward = self._compute_reward()
        self.current_step += 1
        terminated = bool(abs(self.output_disp) >= 10.0) #changed from 100
        truncated = bool(self.current_step > self.episode_length)

        obs = self._get_obs()
        info = {
            'displacement': self.output_disp,
            'force': self.control_force,
            'seismic_input': seismic_input,
            'rewards': reward
        }
        return obs, reward, terminated, truncated, info
 
    def _compute_reward(self):
        #evaluate over multiple timesteps
        if self.current_step < 100:
            return 0.0
        
        recent_out = np.array(list(self.displacement_history))
        recent_in = np.array(list(self.input_history))

        #window = np.hanning(len(recent_in))
        out_freq = np.fft.fft(recent_out)
        in_freq = np.fft.fft(recent_in)
        
        trfn = (out_freq / in_freq + 1e-10) * self.dt
        
        freqs, Pxx = signal.welch(recent_in, fs=1/self.dt, window='hann', nperseg=min(112499, len(recent_in)))
        
        #frequencies = np.fft.fftfreq(len(recent_in), d=self.dt)
        #print(f'freq {len(frequencies)}')
        half = len(freqs) // 2
        df = np.diff(freqs[1:half])
        
        control = trfn[1:half] * np.sqrt(Pxx)[1:half]
        
        var = np.cumsum(np.flip(df * (control[:-1])**2))
       
        rms = np.flip(np.sqrt(var))
        mean_rms = np.mean(rms)  
        rms_reward = -np.log1p(mean_rms.real + 1e-12)  #avoid log(0)

        center_reward = -np.log1p(abs(self.output_disp))
        alpha, beta = 1.0, 1.0
        reward = alpha * rms_reward + beta * center_reward
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
if __name__ == '__main__':
    T = 1000
    dt = 1e-3
    Nt_step = T * 1e3
    tmax = Nt_step * dt  
    end = int(T * 62.5)
    nperseg = 2**16
    episode_length = T*1e3

    f = h5py.File("/Users/letizia/Desktop/INFN/new model/SaSR_test.hdf5", "r")
    seism = f['SR/V1:ENV_CEB_SEIS_V_dec'][:] #seismic data
    seism = seism[:end]*1e-6
    f.close()    

    tt_data = np.linspace(0, T, len(seism))
    tt_sim = np.arange(0, tmax, dt)


    vf = np.fft.fft(seism)
    frequencies = np.fft.fftfreq(len(seism), d = 1/62.5)
    half = len(frequencies) // 2  # half of the frequencies array (positive frequencies only)

    X_f = np.zeros_like(vf, dtype=complex)  # create an array of zeros with the same shape as V
    nonzero = frequencies != 0  # boolean mask: true if freq is not zero
    X_f[nonzero] = vf[nonzero] / (1j * 2 * np.pi * frequencies[nonzero])

    # choose one of the two for the displacement
    displacement = np.fft.ifft(X_f).real
    interp_displacement = interp1d(tt_data, displacement.real, kind='linear', bounds_error=False, fill_value=0.0)(tt_sim)

    # --- Test environment without training ---
    test_env = PendulumVerticalEnv(
        interp_displacement,
        T=T,
        dt=dt,
        episode_length=episode_length,
        history_length=1000,
        seed=42,
    )

    obs, info = test_env.reset()
    print("Initial observation:", obs)

    for step in range(10000):  # run for 10 steps
        action = test_env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = test_env.step(action)
        print(f"Step {step}: obs={obs}, reward={reward:.3f}, done={terminated or truncated}")
        
        if terminated or truncated:
            print("Episode finished, resetting...")
            obs, info = test_env.reset()

    test_env.close()