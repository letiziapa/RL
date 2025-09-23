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
        if self.num_timesteps % 1000 == 0:
            self.logger.record('train/timesteps', self.num_timesteps)
            
            
            # Log buffer size if available
            if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer.size() > 0:
                buffer_size = self.model.replay_buffer.size()
                self.logger.record('train/buffer_size', buffer_size)
                print(f"Replay buffer size: {buffer_size}")
            
            self.logger.dump(step=self.num_timesteps)
                
        return True

# Use it in training:
callback = LoggingCallback()

def evaluate_control_performance(model, env, max_steps=50_000):  
    env.training = False
    if hasattr(env, 'norm_reward'):
        env.norm_reward = False
    
    all_displacements = []
    all_inputs = []
    all_forces = []
    all_rewards = []

    obs = env.reset()
    steps = 0
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
 
         # Collect from all environments
        displacements_step = [info_i['displacement'] for info_i in info]
        inputs_step = [info_i['seismic_input'] for info_i in info]
        forces_step = [info_i['force'] for info_i in info]
        rewards_step = rewards if np.ndim(rewards) else np.array([float(rewards)]*len(info))
        
        all_displacements.append(displacements_step)
        all_inputs.append(inputs_step)
        all_forces.append(forces_step)
        all_rewards.append(rewards_step)
        
        if steps % 1000 == 0:
            print(f'Step {steps}')

        steps += 1
        if bool(np.asarray(dones)[0]):     # stop at the end of the FIRST episode
            break
    print(f"Collected {len(all_displacements)} steps from envs (one episode).")

    # Average across episodes
        # Convert lists to numpy arrays with shape (steps, n_envs)
    all_displacements = np.array(all_displacements)
    all_inputs = np.array(all_inputs)
    all_forces = np.array(all_forces)
    all_rewards = np.array(all_rewards)
    
    # Average across environments
    mean_displacements = np.mean(all_displacements, axis=1)

    print(f"Sample controlled displacements: {all_displacements[600:620, 0]}")
    print(f"\nSample input displacements: {all_inputs[600:620, 0]}")
    print(f"\nMean displacement: {mean_displacements.mean()}")
    print(f"\nSample rewards: {all_rewards[600:620, 0]}")
    
    return all_displacements, all_inputs, all_forces, all_rewards  # Return arrays with shape (steps, n_envs)

def calculate_tf_rms(output, input_sig, dt):
    """Calculate transfer function using FFT"""

    freqs = np.fft.fftfreq(len(input_sig), d=dt)
    output_windowed = np.fft.fft(output )
    input_windowed = np.fft.fft(input_sig )

    trfn_windowed = (output_windowed / input_windowed) *dt
    
    return freqs, abs(trfn_windowed)


def plot_control_results(controlled_disp, input_disp, forces, rewards, dt):
    """Comprehensive plotting of control results"""
    t = np.arange(len(controlled_disp)) * dt    
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    # 1. Time domain comparison
    ax = axes[0, 0]
    ax.plot(t, controlled_disp, 'b-', alpha=0.7, label='Controlled')
    ax.plot(t, input_disp, 'r-', alpha=0.7, label='Input')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Displacement [m]')
    ax.set_title('Time Domain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # # 2. forces
    ax = axes[0, 1]
    ax.plot(t, forces, 'b-', label='force')
    ax.set_xlabel('Time')
    ax.set_ylabel('Force [N]')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # 3. Transfer function magnitude
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
    ax.plot(t, rewards, 'cyan', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Rewards')
    ax.set_title('Rewards')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_sac_model(env, log_path="./sac_control"):    
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
            net_arch=[256, 256, 256]  # Deeper network
        ),
        verbose=1,
        tensorboard_log=log_path,
        seed = 42
    )
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    return model

def make_env():
    def _init():
        return PendulumVerticalEnv(
            interp_displacement, 
            T = T, 
            dt = dt, 
            episode_length = episode_length,
            history_length = history_length,
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
    base_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs = 100.0)
    model = create_sac_model(env, log_path="./sac_test")
    print('Model hyperparameters:')
    print(f'Learning rate: {lr} \nBatch size: {batch_size} \nTau: {tau} \nBuffer size: {buffer_size} \nLearning starts: {learning_starts}')

    model.learn(
        total_timesteps=500_000,
        log_interval = 1,
        callback=callback,
        progress_bar=True)
    #model.save("testreward")
    #model.load("sac_pendulum_vertical_increased_timesteps")
    print('Starting evaluation...')
    filename = "2309.txt"
    outputs, inputs, forces, rewards = evaluate_control_performance(model, env)
    #freqs, abs_tf = calculate_tf_rms(outputs, inputs, half, dt)
    #np.savetxt(filename, np.column_stack((freqs, abs_tf, outputs, inputs)), header='Frequency(Hz) |H(f)| output input')
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
    # print("Controlled disp stats:", np.min(outputs), np.max(outputs))
    # t = np.arange(len(outputs)) * dt  
    print("Controlled disp stats:", np.min(mean_outputs), np.max(mean_outputs))
    t = np.arange(len(mean_outputs)) * dt      
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    # 1. Time domain comparison
    ax = axes[0, 0]
    #ax.plot(t, outputs, 'b-', alpha=0.7, label='Output')
    #ax.plot(t, inputs, 'r-', alpha=0.7, label='Controlled Input')
    ax.plot(t, mean_outputs, 'b-', alpha=0.7, label='Mean Output')
    ax.plot(t, mean_inputs, 'r-', alpha=0.7, label='Mean Input')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Displacement [m]')
    ax.set_title('Time Domain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # # 2. forces
    ax = axes[0, 1]
    ax.plot(t, mean_forces, 'b-', label='force')
    ax.set_xlabel('Time')
    ax.set_ylabel('Force [N]')

    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # 3. Transfer function magnitude
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