import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from environment import PendulumVerticalEnv
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from noControl import matrix
import stable_baselines3
import h5py
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
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

def evaluate_control_performance(model, env, max_steps=1800000):  
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
        
        # Collect from first environment
        all_displacements.append(info[0]['displacement'])
        all_inputs.append(info[0]['seismic_input'])
        all_forces.append(info[0]['force'])
        all_rewards.append(rewards[0] if np.ndim(rewards) else float(rewards))

        steps += 1
        if bool(np.asarray(dones)[0]):     # stop at the end of the FIRST episode
            break
    print(f"Collected {len(all_displacements)} steps from env0 (one episode).")

    # Average across episodes
    mean_displacements = np.mean(all_displacements)
    mean_inputs = np.mean(all_inputs)
    mean_rewards = np.mean(all_rewards)
    #(all_rewards[0]) has shape (eval_steps)

    # Add this to your evaluation to see raw values:
    print(f"Sample controlled displacements: {all_displacements[110:120]}")
    print(f"\nSample input displacements: {all_inputs[110:120]}")
    print(f"\nMean displacement: {mean_displacements}")
    #print(f"\nFirst actions from model: {[model.predict(env.reset())[0] for _ in range(10)]}")
    print(f"\nSample rewards: {all_rewards[110:120]}")
    
    return np.array(all_displacements), np.array(all_inputs), np.array(all_forces), np.array(all_rewards)  # Return one force trajectory

def calculate_tf_rms(output, input_sig, nperseg, dt):
    """Calculate transfer function"""
    fs = 1.0 / dt
    f, Pxy = signal.csd(output, input_sig, fs=fs, nperseg=nperseg)
    f, Pxx = signal.welch(input_sig, fs=fs, nperseg=nperseg)
    H = (Pxy / (Pxx + 1e-20)) *dt

    return f, H


def plot_control_results(controlled_disp, input_disp, forces, rewards, dt, T):
    """Comprehensive plotting of control results"""
    freqs, tf = calculate_tf_rms(controlled_disp, input_disp, 2**16, dt)

    # Time axis
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
    
    # # 2. ASDs
    ax = axes[0, 1]
    ax.plot(t, forces, 'b-', label='force')
    # ax.loglog(f, asd_in, 'r-', label='Input ASD')
    #ax.set_xlabel('Frequency [Hz]')
    ax.set_xlabel('Time')
    ax.set_ylabel('Force [N]')
    # ax.set_ylabel('ASD [m/√Hz]')
    # ax.set_title('Amplitude Spectral Densities')
    #ax.set_xlim([0.01, 100])
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # 3. Transfer function magnitude
    ax = axes[1, 0]
    ax.loglog(freqs, np.abs(tf), 'g-', linewidth=2)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Unity')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|H(f)|')
    ax.set_title('Transfer Function Magnitude')
    #ax.set_xlim([0.01, 100])
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    
    # Suppression ratio
    ax = axes[1, 1]
    # suppression = asd_in / (asd_out + 1e-12)
    ax.plot(t, rewards, 'cyan', linewidth=2)
    # ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Rewards')
    ax.set_title('Rewards')
    # ax.set_xlim([0.01, 100])
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_sac_model(env, log_path="./sac_control"):    
    model = SAC(
        "MlpPolicy",
        env,
        gamma=0.999,  # Slightly less forward-looking
        learning_rate=3e-4,  # Slower learning for stability
        buffer_size=500000,  # changed from 500000
        learning_starts=10000,   #changed from 10000
        batch_size=256,
        tau=0.005,  # Softer target updates
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
            history_length = 1000,
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



if __name__ == '__main__':
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


    n_envs = 32
    base_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs = 100.0)
    model = create_sac_model(env, log_path="./sac_test")
    # Test manual logging

# # Use it before training:
#     if test_logging_system(model):
#         print("Logging system is working, proceeding with training...")
#     else:
#         print("Logging system has issues, need to debug further")

#     model.logger.record('test/manual_log', 42.0)
#     model.logger.dump(step=0)
    #print("Manual log written - check if this appears in TensorBoard")
    model.learn(
        total_timesteps=500_000,
        log_interval = 1,
        callback=callback,
        progress_bar=True)
    #model.save("testreward")
    #model.load("sac_pendulum_vertical_increased_timesteps")
    print('Starting evaluation...')
    displacements, inputs, forces, rewards = evaluate_control_performance(model, env)
    print(f"Evaluation complete.")
    print(f"Output displacement: {displacements.shape}")
    print(f"Forces: {forces.shape}")
    print(f"Rewards: {rewards.shape}")
    print("Controlled disp stats:", np.min(displacements), np.max(displacements))
    fig = plot_control_results(displacements, inputs, forces, rewards, dt, T)
    plt.show()
    base_env.close()