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
    """PendulumVerticalEnv is a custom OpenAI Gym environment for simulating and controlling a vertical pendulum system
    subjected to a force derived from the measured seismic noise.
    This environment models a multi-mass pendulum system, where the agent's goal is to control the system's response 
    to seismic disturbances by applying a force. 
    The environment provides observations of the system's state, 
    accepts continuous control actions, 
    and computes rewards based on the effectiveness of the control in minimising the displacement at the bottom mass.
    Attributes:
        interp_displacement (array-like): seismic displacement input, interpolated to match the simulation timestep.
        T (int): Total simulation time in seconds.
        dt (float): Simulation timestep in seconds.
        episode_length (int): Maximum number of steps per episode.
        history_length (int): Number of timesteps to keep in the displacement and input history.
        options (str): Reward computation mode ('log' or 'inverse').
        seed (Optional[int]): Random seed for reproducibility.
        max_force_mag (float): Maximum magnitude of the control force applied by the agent.
        M (list): Mass values for the pendulum system.
        K (list): Stiffness values for the pendulum system.
        gamma (list): Damping coefficients for the pendulum system.
        A, B: System matrices for state-space representation.
        AR_model (callable): Function to update the state of the system.
        action_space (gym.spaces.Box): Continuous action space for control force.
        observation_space (gym.spaces.Box): Continuous observation space for system state.
        displacement_history (collections.deque): History of output displacements.
        input_history (collections.deque): History of input displacements.
    Methods:
        reset(seed=None, options=None):
            Resets the environment to its initial state (zero velocity and zero displacement).
        step(action):
            Advances the simulation by one timestep using the action chosen by the agent, 
            updates the system state, computes the reward, 
            and returns the new observation, reward, termination/truncation flags, and any additional info.
        _get_obs():
            Returns the current observation of the system state.
        _compute_reward(options='log'):
            Computes the reward based on recent system response, using either logarithmic or inverse reward modes.
            (logarithmic is recommended)
        render(mode='human'):
            Prints the current state of the environment for visualisation.
    Observation:
        A 6-dimensional vector representing the current state of the pendulum system, 
        i.e. the displacements and velocities of the three masses.
    Action:
        A 1-dimensional continuous value in [-1, 1], scaled to the maximum force magnitude (which defaults to 5 N).
    Reward:
        Computed based on the reduction of displacement and system response, with options for logarithmic or inverse scaling.
    Termination:
        The episode terminates if the output displacement exceeds a threshold (10 m) or the maximum episode length is reached.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30}

    def __init__(self, interp_displacement, T = 1800, dt=1e-3, episode_length=350000, history_length=10000, options = 'log', seed: Optional[int] = 42):
        super(PendulumVerticalEnv, self).__init__()
        self.interp_displacement = interp_displacement
        self.dt = dt
        self.T = T 
        self.episode_length = episode_length
        self.current_step = 0
        self.history_length = history_length
        self.options = options
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
        #initialise the state to zero
        self.state = np.zeros(6, dtype=np.float32)
        
        #clear histories
        self.displacement_history.clear()
        self.input_history.clear()
        
        #initialise histories with zeros
        for _ in range(self.history_length):
            self.displacement_history.append(0.0)
            self.input_history.append(0.0)
            
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.state.copy()
        return obs


    def step(self, action):  
        """
        Advances the environment by one time step using the provided action.
        The evolution of the system is computed using a state-space representation,
        where the state is updated based on the current state, system matrices, and the applied force.
        Parameters:
            action (array-like): The action to be taken by the agent, i.e. the force applied.
        Returns:
            obs (np.ndarray): The next observation of the environment state.
            reward (float): The reward obtained after taking the action.
            terminated (bool): Whether the episode has ended due to a terminal condition (e.g. displacement > 10 m).
            truncated (bool): Whether the episode has ended due to reaching the maximum episode length.
            info (dict): Additional information:
                - 'displacement': The current output displacement.
                - 'force': The applied control force.
                - 'seismic_input': The current input displacement.
                - 'rewards': The computed reward for this step.
        """
        if self.current_step >= self.episode_length:
            return self._get_obs(), 0.0, False, True, {}
        
        self.control_force = float(action[0]*self.max_force_mag)
        seismic_input = self.interp_displacement[self.current_step]

        force = self.K[0] * np.real(seismic_input) + self.control_force
        
        self.state = self.AR_model(self.state, self.A, self.B, force)

        self.input_disp = self.state[0]  # vertical displacement at the top mass
        self.output_disp = self.state[5] # vertical displacement at the bottom mass
        
        self.displacement_history.append(self.output_disp)
        self.input_history.append(self.input_disp)

        reward = self._compute_reward(self.options)
        self.current_step += 1

        terminated = bool(abs(self.output_disp) >= 10.0) 
        truncated = bool(self.current_step > self.episode_length)

        obs = self._get_obs()
        info = {
            'displacement': self.output_disp,
            'force': self.control_force,
            'seismic_input': self.input_disp,
            'rewards': reward
        }
        return obs, reward, terminated, truncated, info
 
    def _compute_reward(self, options = 'log'):
        """
        Compute the reward for the current environment state.
        The reward is based on the recent input and output signal histories and on the immediate system response (output displacement).
        This method evaluates the reward over multiple timesteps using frequency domain analysis 
        and power spectral density estimation.
        It supports two reward calculation options: 'log' and 'inverse' (log is recommended).
        Args:
            options (str, optional): The reward calculation method. 
                - 'log': Uses the negative logarithm of the absolute output displacement and RMS value.
                - 'inverse': Uses the inverse of the squared output displacement and RMS value.
                Default is 'log'.
        Returns:
            reward (float): The computed reward value for the current step. Returns 0.0 if the current step is less than 100.
        Notes:
            - The reward combines two components: one based on the output displacement 
            and one based on the RMS of a frequency-weighted control signal.
            - Uses Welch's method for power spectral density estimation and FFT for frequency analysis.
            - The total reward is a weighted sum of the two components, controlled by the tunable parameters alpha and beta.
        """
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
        half = len(freqs) // 2
        df = np.diff(freqs[1:half])
        
        control = trfn[1:half] * np.sqrt(Pxx)[1:half]
        
        var = np.cumsum(np.flip(df * (control[:-1])**2))
       
        rms = np.flip(np.sqrt(var))
        mean_rms = np.mean(rms)  
        if options == 'log':
            center_reward = -np.log1p(abs(self.output_disp)+1e-12)
            rms_reward = -np.log1p(mean_rms.real + 1e-12)

        elif options == 'inverse':
            center_reward = 1.0 / (1.0 + self.output_disp**2)
            rms_reward = 1.0 / (1.0 + mean_rms.real) 
      
        alpha, beta = 0.8, 0.5
        reward = alpha * rms_reward + beta * center_reward 
        return reward


    def render(self, mode='human'):        
        """
        Renders the current state of the environment by printing key information to the console.

        Args:
            mode (str, optional): The mode in which to render the environment. Defaults to 'human'.

        Prints:
            - The current simulation step.
            - The input displacement for the current step.
            - The input displacement as controlled by the agent.
            - The output displacement (state variable).
            - The differential displacement between the current and previous output.
        """

        print(f"""
        Step: {self.current_step}
        Input: {self.interp_displacement[self.current_step]:.6e} m
        Input (controlled): {self.input_disp:.6e} m
        Output: {self.state[5]:.6e} m
        Differential displacement: {self.state[5] - (self.history["x6"][-2] if len(self.history["x6"]) >= 2 else 0):.6e} m
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
        print(f"Step {step}: obs={obs[5]}, reward={reward:.3f}, done={terminated or truncated}")
        test_env.render()
        if terminated or truncated:
            print("Episode finished, resetting...")
            obs, info = test_env.reset()

    test_env.close()