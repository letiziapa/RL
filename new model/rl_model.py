from environment import PendulumVerticalEnv
import numpy as np
import h5py
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal

T = 900
dt = 1e-3
Nt_step = T * 1e3
tmax = Nt_step * dt  
end = int(T * 62.5)
nperseg = 2**14

episode_length = 900000  
f = h5py.File("/Users/letizia/Desktop/INFN/new model/SaSR_test.hdf5", "r")
seism = f['SR/V1:ENV_CEB_SEIS_V_dec'][:] #seismic data
seism = seism[:end]*1e-6

tt_data = np.linspace(0, T, len(seism))
tt_sim = np.arange(0, tmax, dt)

vf = np.fft.fft(seism)
frequencies = np.fft.fftfreq(len(seism), d = 1/62.5)
half = len(frequencies) // 2 #half of the frequencies array (positive frequencies only)

xf = vf[1:] / (1j * 2 * np.pi * frequencies[1:]) # Fourier Transform of the seismic data
X_f = np.zeros_like(vf, dtype=complex) #create an array of zeros with the same shape as V
nonzero = frequencies != 0 #boolean mask: true if freq is not zero
X_f[nonzero] = vf[nonzero] / (1j * 2 * np.pi * frequencies[nonzero])

#choose one of the two for the displacement
zt = np.fft.ifft(X_f).real
xt = np.fft.ifft(xf).real 

fVel, psdVel = signal.welch(seism.real, fs = 62.5, window='hann', nperseg=nperseg)
fZ, psdZ = signal.welch(xt.real, fs = 62.5, window='hann', nperseg=nperseg)

input_interp = interp1d(tt_data, zt.real,
                        kind='linear', bounds_error=False, fill_value=0.0)(tt_sim)

def calculate_transfer_function(input_signal, output_signal, dt):
    freqs = np.fft.fftfreq(len(input_signal), d=1/62.5)
    
    fft_input = np.fft.fft(input_signal )
    fft_output = np.fft.fft(output_signal )
    
    transfer_function = np.abs(fft_output/ fft_input)*dt
    print(len(input_signal))
    return freqs, transfer_function

def make_env():
    def _init():
        return PendulumVerticalEnv(seism, input_interp, T=T, dt=dt, episode_length=episode_length)
    return _init

#env = PendulumVerticalEnv(seism, input_interp, T = T, dt = dt, episode_length = episode_length) 
if __name__ == "__main__":
    n_envs = 32
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    obs = env.reset()

    model = SAC.load("sac_1", env=env)
    # model = SAC(
    #     "MlpPolicy", 
    #     env, 
    #     gamma = 0.7, 
    #     batch_size = 256, 
    #     learning_rate= 1e-5, 
    #     verbose=1, 
    #     tensorboard_log="./sac_control/"
    #     )

    # model.learn(total_timesteps=900000, progress_bar=True)
    # model.save("sac_1")

#obs, info = env.reset()

    controlled_displacements = []
    rewards_n = []
    actions = []
    obs = env.reset()
    for _ in range(56250):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        actions.append(action)
        controlled_displacements.append(info[0]['displacement'])
        rewards_n.append(rewards)   

        env.render()
    
    actions = np.array(actions)
    print(len(controlled_displacements), len(rewards_n), len(actions))
    disp_interp = interp1d(np.arange(len(controlled_displacements)), controlled_displacements,
                        kind='linear', bounds_error=False, fill_value=0.0)
    x6_interp = disp_interp(tt_data)

    # zt_interp = interp1d(tt_data, zt.real,
    #                      kind='linear', bounds_error=False, fill_value=0.0)(tt_sim)
    print(len(x6_interp), len(zt))
    print(len(actions), len(rewards_n))
    x6_f = np.fft.fft(x6_interp.real)
    zt_f = np.fft.fft(zt.real)
    trfn = zt_f[:half] / x6_f[:half]
    #freqs, tf_controlled = calculate_transfer_function(zt, x6_interp, dt)

    #half = len(freqs) // 2

    fVel, psdVel = signal.welch(seism.real, fs = 62.5, window='hann', nperseg=nperseg)
    fZ, psdZ = signal.welch(xt.real, fs = 62.5, window='hann', nperseg=nperseg)
    fOut, psdOut = signal.welch(controlled_displacements, fs = 1, window='hann', nperseg=nperseg)
    plt.figure()
    plt.loglog(fOut, np.sqrt(psdOut), label = 'output')
    plt.loglog(fZ, np.sqrt(psdZ), label = 'input')
    plt.legend()
    plt.title('Displacement Comparison (freq domain)')

    plt.figure(figsize=(14, 8))

    #transfer functions
    plt.subplot(2, 1, 1)
    plt.plot(abs(frequencies[:half]), abs(trfn), label='PPO Control', linestyle='--')
    plt.plot(abs(frequencies[:half]), np.abs(zt_f[:half]), label='Seismic Input (frequency domain)')
    plt.plot(abs(frequencies[:half]), np.abs(x6_f[:half]), label='output (freq domain)', linestyle=':')
    plt.title('System Transfer Function Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Transfer function ($\tilde{x}_6 / \tilde{x}_{in}$)')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)

    #displacement over time
    plt.subplot(2, 1, 2)
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
    plt.plot(controlled_displacements)
    plt.title("Displacement x6")
    plt.subplot(3,1,2)
    plt.plot(actions[:][30].flatten())
    plt.title("Control force (N)")
    plt.subplot(3,1,3)
    plt.plot(rewards[:][30].flatten())
    plt.title("Reward")
    plt.tight_layout()
    plt.show()


    env.close()
