from environment import PendulumVerticalEnv
import numpy as np
import h5py
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from stable_baselines3.common.monitor import Monitor

np.random.seed(42)

T = 370
dt = 1e-3
Nt_step = T * 1e3
tmax = Nt_step * dt  
end = int(T * 62.5)

episode_length = 370000  # Total number of steps in the episode
f = h5py.File("/Users/letizia/Desktop/INFN/new model/SaSR_test.hdf5", "r")
seism = f['SR/V1:ENV_CEB_SEIS_V_dec'][:] #seismic data
seism = seism[:end]*1e-6

tt_data = np.linspace(0, T, len(seism))
tt_sim = np.arange(0, tmax, dt)

env = PendulumVerticalEnv(seism, T = T, dt = dt, episode_length = episode_length) 

obs, info = env.reset()

#model = SAC.load("sac_pendulum_vertical_control_5", env=env)
model = SAC(
    "MlpPolicy", 
    env, 
    gamma = 0.9, 
    batch_size = 256, 
    learning_rate= 1e-4, 
    verbose=1, 
    tensorboard_log="./sac_control/"
    )

model.learn(total_timesteps=370000, progress_bar=True)
print("Training completed.")
model.save("sac_pendulum_vertical_control_6")

#obs, info = env.reset()

controlled_displacements = []
for i in range(100000):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    controlled_displacements.append(info['displacement'])
    env.render()

    
    if terminated or truncated:
        obs, info = env.reset()
        #break


def calculate_transfer_function(input_signal, output_signal, dt):
    """Helper function to compute the FFT-based transfer function.
    Input data in time domain"""
    input_signal = np.array(input_signal)
    output_signal = np.array(output_signal)
    window = np.hanning(len(input_signal))

    freqs = np.fft.fftfreq(len(input_signal), d=1/62.5)
    
    fft_input = np.fft.fft(input_signal )
    fft_output = np.fft.fft(output_signal )
    #freqs = np.fft.fftfreq(n, d=dt)
    
    transfer_function = np.abs(fft_output/ fft_input)
    print(len(input_signal), len(output_signal))
    return freqs, transfer_function

x6       = np.array(env.history["x6"])       # vertical displacements
rewards  = np.array(env.history["reward"])   # step-by-step rewards
actions  = np.array(env.history["force"])    # control forces you applied
steps    = np.array(env.history["step"])  

# Ensure all lists have the same length for FFT
#min_len = min(len(seismic_inputs), len(uncontrolled_displacements), len(controlled_displacements))
#seismic_inputs = seismic_inputs[:min_len]
#uncontrolled_displacements = uncontrolled_displacements[:min_len]
#controlled_displacements = controlled_displacements[:min_len]
vf = np.fft.fft(seism)


frequencies = np.fft.fftfreq(len(seism), d = 1/62.5)
half = len(frequencies) // 2 #half of the frequencies array (positive frequencies only)

xf = vf[1:] / (1j * 2 * np.pi * frequencies[1:]) # Fourier Transform of the seismic data
X_f = np.zeros_like(vf, dtype=complex) #create an array of zeros with the same shape as V
nonzero = frequencies != 0 #boolean mask: true if freq is not zero

#for all non-zero frequencies, divide the FT by 2 pi f the take the IFT to get the displacement
X_f[nonzero] = vf[nonzero] / (1j * 2 * np.pi * frequencies[nonzero])

zt = np.fft.ifft(X_f).real
xt = np.fft.ifft(xf) 
#freqs, tf_uncontrolled = calculate_transfer_function(seismic_inputs, uncontrolled_displacements, dt)
print(len(x6))

#interpolate the controlled displacements to match the seismic data length
disp_interp = interp1d(np.arange(len(x6)), x6,
                       kind='linear', bounds_error=False, fill_value=0.0)
x6_interp = disp_interp(tt_data)
input_interp = interp1d(np.arange(len(zt)), zt,
                        kind='linear', bounds_error=False, fill_value=0.0)(np.linspace(0, len(zt)-1, len(x6)))
freqs, tf_controlled = calculate_transfer_function(zt, x6_interp, dt)

half = len(freqs) // 2
plt.figure(figsize=(14, 8))
    
# Plot 1: Transfer Functions
plt.subplot(2, 1, 1)
#positive_freqs_mask = (freqs > 0) & (freqs < 10) # Focus on relevant frequencies
#plt.plot(freqs[:half], tf_uncontrolled[:half], label='Without Control')
plt.plot(abs(frequencies), tf_controlled, label='PPO Control', linestyle='--')
plt.plot(abs(frequencies), np.abs(X_f), label='Seismic Input (frequency domain)')
plt.plot(abs(frequencies), np.abs(np.fft.fft(x6_interp)), label='output (freq domain)', linestyle=':')
plt.title('System Transfer Function Comparison')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Transfer function ($\tilde{x}_6 / \tilde{x}_{in}$)')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid(True)

# Plot 2: Displacement over time
plt.subplot(2, 1, 2)
#time_axis = np.arange(min_len) * dt
#plt.plot(time_axis, uncontrolled_displacements, label='Without Control', alpha=0.8)
plt.plot(x6_interp, label='PPO Control', alpha=0.8)
plt.title('Bottom Mass Displacement Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

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
env.close()
