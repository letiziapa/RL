import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from environment import PendulumVerticalEnv
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from noControl import matrix, TransferFunc
import h5py
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

np.random.seed(42)

class LoggingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        
    def _on_step(self) -> bool:
        """
        Callback function called at each training step.
        Logs the current number of timesteps every 1000 steps. If the model has a replay buffer,
        it also logs the current buffer size and prints it to the console. 
        Dumps the logger at each logging interval.

        Returns:
            bool: Always returns True to indicate the callback should continue.
        """
        if self.num_timesteps % 1000 == 0:
            self.logger.record('train/timesteps', self.num_timesteps)
            
            
            # Log buffer size if available
            if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer.size() > 0:
                buffer_size = self.model.replay_buffer.size()
                self.logger.record('train/buffer_size', buffer_size)
                print(f"Replay buffer size: {buffer_size}")
            
            self.logger.dump(step=self.num_timesteps)
                
        return True

#callback during training
callback = LoggingCallback()

def evaluate_control_performance(model, env, max_steps=50_000):  
    """
    Evaluates the performance of a control model in a given environment over a single episode.
    This function runs the provided model in the environment for up to `max_steps` steps or until the first episode ends.
    It collects and returns the displacements, seismic inputs, forces, and rewards at each step for all parallel environments.
    
    Parameters:
        model: The control model with a `predict` method, used to select actions based on observations.
        env: The environment in which to evaluate the model. Must support vectorized operations and provide info dicts with
        'displacement', 'seismic_input', and 'force' keys.
        max_steps (int, optional): Maximum number of steps to run the evaluation. Defaults to 500,000.
    
    Returns:
        all_displacements (np.ndarray): Array of displacements with shape (steps, n_envs).
        all_inputs (np.ndarray): Array of seismic inputs with shape (steps, n_envs).
        all_forces (np.ndarray): Array of forces with shape (steps, n_envs).
        all_rewards (np.ndarray): Array of rewards with shape (steps, n_envs).

    Notes:
        - The function disables training and reward normalization in the environment if applicable.
        - Only the first episode is evaluated, even if the environment supports multiple parallel environments.
        - Prints progress and sample statistics during evaluation.
    """

    env.training = False
    if hasattr(env, 'norm_reward'):
        env.norm_reward = False
    
    #initialise empty lists to collect data
    all_displacements = []
    all_inputs = []
    all_forces = []
    all_rewards = []

    #reset environment
    obs = env.reset()
    steps = 0
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        #collect data from all environments
        displacements_step = [info_i['displacement'] for info_i in info]
        inputs_step = [info_i['seismic_input'] for info_i in info]
        forces_step = [info_i['force'] for info_i in info]
        rewards_step = rewards if np.ndim(rewards) else np.array([float(rewards)]*len(info))
        
        all_displacements.append(displacements_step)
        all_inputs.append(inputs_step)
        all_forces.append(forces_step)
        all_rewards.append(rewards_step)
        
        #check progress
        if steps % 1000 == 0:
            print(f'Step {steps}')

        steps += 1
        if bool(np.asarray(dones)[0]):     # stop at the end of the first episode
            break
    print(f"Collected {len(all_displacements)} steps from envs (one episode).")

    #Convert lists to numpy arrays with shape (steps, n_envs)
    all_displacements = np.array(all_displacements)
    all_inputs = np.array(all_inputs)
    all_forces = np.array(all_forces)
    all_rewards = np.array(all_rewards)
    
    #Average across environments
    mean_displacements = np.mean(all_displacements, axis=1)

    print(f"Sample controlled displacements: {all_displacements[600:620, 0]}")
    print(f"\nSample input displacements: {all_inputs[600:620, 0]}")
    print(f"\nMean displacement: {mean_displacements.mean()}")
    print(f"\nSample rewards: {all_rewards[600:620, 0]}")
    
    return all_displacements, all_inputs, all_forces, all_rewards  

def calculate_tf_rms(output, input_sig, dt):
    """
    Calculates the transfer function magnitude between an output and input signal using the Fast Fourier Transform (FFT).
    
    Parameters:
        output (array): The output signal in the time domain.
        input_sig (array): The input signal in the time domain.
        dt (float): The time step between samples.
    
    Returns:
        freqs (numpy.ndarray): Array of frequency bins corresponding to the FFT.
        abs_trfn_windowed (numpy.ndarray): Magnitude of the transfer function at each frequency bin.
    
    Notes:
        - The transfer function is computed as the ratio of the FFT of the output to the FFT of the input, scaled by the time step.
        - The function returns the absolute value (magnitude) of the transfer function.
    """

    freqs = np.fft.fftfreq(len(input_sig), d=dt)
    output_windowed = np.fft.fft(output )
    input_windowed = np.fft.fft(input_sig )

    trfn_windowed = (output_windowed / input_windowed) *dt
    
    return freqs, abs(trfn_windowed)


def plot_control_results(controlled_disp, input_disp, forces, rewards, dt):
    """
    Plot results generated from the RL control system.
    This function generates a 2x2 grid of subplots to visualize various aspects of a control system's performance:
        1. Time domain comparison of controlled and input displacements.
        2. Forces applied over time.
        3. Transfer function magnitude (requires 'freqs' and 'abs_tf').
        4. Rewards over time.
    
    Parameters:
        controlled_disp (array): The displacement values resulting from the control system.
        input_disp (array): The input displacement values (reference or desired).
        forces (array): The force values applied by the controller.
        rewards (array): The reward values at each time step (e.g., for reinforcement learning).
        dt (float): Time step between samples, used to generate the time axis.

    Returns:
        fig (matplotlib.figure): The matplotlib Figure object containing the subplots.
    Notes:
        - The function assumes that `freqs` and `abs_tf` are available in the global or calling scope for the transfer function subplot.
        - All input arrays must have the same length.
    """

    t = np.arange(len(controlled_disp)) * dt    
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    #Comparison of input and output displacements
    ax = axes[0, 0]
    ax.plot(t, controlled_disp, 'b-', alpha=0.7, label='Controlled')
    ax.plot(t, input_disp, 'r-', alpha=0.7, label='Input')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Displacement [m]')
    ax.set_title('Time Domain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    #Force applied over time
    ax = axes[0, 1]
    ax.plot(t, forces, 'b-', label='force')
    ax.set_xlabel('Time')
    ax.set_ylabel('Force [N]')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    #Transfer function magnitude
    ax = axes[1, 0]
    ax.loglog(abs(freqs), abs_tf, 'g-', linewidth=2)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Unity')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|H(f)|')
    ax.set_title('Transfer Function Magnitude')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    #Rewards over time
    ax = axes[1, 1]
    ax.plot(t, rewards, 'cyan', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Rewards')
    ax.set_title('Rewards')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_sac_model(env, log_path="./sac_control"):    
    """
    Creates and configures a Soft Actor-Critic (SAC) model for reinforcement learning.
    Args:
        env (gym.Env): The environment to train the agent on.
        log_path (str, optional): Path for logging and tensorboard summaries. Defaults to "./sac_control".
    Returns:
        model: An instance of the SAC model configured with the specified parameters.
    Notes:
        - The variables 'lr', 'buffer_size', 'learning_starts', 'batch_size', and 'tau' need to be defined.
        - The model uses an MLP policy with three hidden layers of 256 units each, but can be changed.
        - Logging is set up for stdout, CSV, and TensorBoard.
        - The random seed is set to 42 for reproducibility.
    """
    model = SAC(
        "MlpPolicy",
        env,
        gamma=0.999,  # Slightly less forward-looking
        learning_rate=lr,  # Slower learning for stability
        buffer_size=buffer_size,  # changed from 500000
        learning_starts=learning_starts,   #changed from 10000
        batch_size=batch_size,  # changedfrom 256
        tau=tau,  # changed from 0.005
        ent_coef='auto_0.1',  
        target_update_interval=1,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=[256, 256, 256]  
        ),
        verbose=1,
        tensorboard_log=log_path,
        seed = 42
    )
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    return model

def make_env():
    """
    Creates an environment initializer function for the PendulumVerticalEnv.

    Returns:
    function: A function that, when called, returns an instance of PendulumVerticalEnv 
    initialized with the specified parameters:
        - interp_displacement: displacement input (interpolated).
        - T: Total time.
        - dt: Time step.
        - episode_length: Length of each episode.
        - history_length: Length of the observation history.
        - options: Choice of reward function. Set to 'log' (see 'environment.py' for details).
        - seed: Set to 42 for reproducibility.

    Note:
        The parameters interp_displacement, T, dt, episode_length, and history_length
        must be defined by the user.
    """
    def _init():
        return PendulumVerticalEnv(
            interp_displacement, 
            T = T, 
            dt = dt, 
            episode_length = episode_length,
            history_length = history_length,
            options = 'log',
            seed = 42,
            )
    return _init

T = 1800
dt = 1e-3
Nt_step = T * 1e3
tmax = Nt_step * dt  
end = int(T * 62.5)
nperseg = 2**16
episode_length = T*1e3
history_length = 1000
timesteps = 500_000   

tau = 0.001
batch_size = 256
lr = 0.0003
buffer_size = 500_000
learning_starts = 10000

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

displacement = np.fft.ifft(X_f).real
interp_displacement = interp1d(tt_data, displacement.real, kind='linear', bounds_error=False, fill_value=0.0)(tt_sim)
#interp_displacement = interp_displacement[:half]
if __name__ == '__main__':
    n_envs = 32
    #Create vectorized environment with 32 parallel instances
    base_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs = 100.0)
    model = create_sac_model(env, log_path="./sac_test")
    print('Model hyperparameters:')
    print(f'Learning rate: {lr} \nBatch size: {batch_size} \nTau: {tau} \nBuffer size: {buffer_size} \nLearning starts: {learning_starts}')

    model.learn(
        total_timesteps=timesteps,
        log_interval = 1,
        callback=callback,
        progress_bar=True)
    
    model.save("110_model2_newreward")
    #model.load("3009_model")
    print('Starting evaluation...')
    filename = "110_file_newreward.txt"
    outputs, inputs, forces, rewards = evaluate_control_performance(model, env)
    mean_outputs = outputs.mean(axis=1)
    mean_inputs = inputs.mean(axis=1)
    mean_forces = forces.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    freqs, abs_tf = calculate_tf_rms(mean_outputs, mean_inputs, dt)
    np.savetxt(filename, np.column_stack((freqs, abs_tf, mean_outputs[:len(freqs)], mean_inputs[:len(freqs)])),
               header='Frequency(Hz) |H(f)| mean_output mean_input')
    print(f"Transfer function data saved to {filename}")
    print(f"Evaluation complete.")
    print(f"Output displacement: {outputs.shape}")
    print(f"Forces: {forces.shape}")
    print(f"Rewards: {rewards.shape}")  
    print("Controlled disp stats:", np.min(mean_outputs), np.max(mean_outputs))
    t = np.arange(len(mean_outputs)) * dt     
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    #(controlled) input and output displacements
    ax = axes[0, 0]
    ax.plot(t, mean_outputs*1e-6, 'b-', alpha=0.7, label='Mean Output')
    ax.plot(t, mean_inputs*1e-6, 'r-', alpha=0.7, label='Mean Input')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Displacement [m]')
    ax.set_title('Time Domain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    #forces applied
    ax = axes[0, 1]
    ax.plot(t, mean_forces, 'b-', label='force')
    ax.set_xlabel('Time')
    ax.set_ylabel('Force [N]')

    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    #Transfer function magnitude
    ax = axes[1, 0]
    ax.loglog(abs(freqs), abs_tf, 'g-', linewidth=2)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Unity')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|H(f)|')
    ax.set_title('Transfer Function Magnitude')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    #rewards
    ax = axes[1, 1]
    ax.plot(t, mean_rewards, 'cyan', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Rewards')
    ax.set_title('Rewards')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    base_env.close()