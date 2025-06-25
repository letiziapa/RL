import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py
from scipy import signal
from noControl import matrix, TransferFunc, AR_model
from scipy.interpolate import interp1d
from gymnasium import spaces
import time
np.random.seed(42)


#structure of the hdf5 object 
def print_hdf5_structure(obj, indent=0):
    """Stampa ricorsivamente la struttura di un file HDF5"""
    for key in obj:
        print(" " * indent + f"📂 {key}")
        if isinstance(obj[key], h5py.Group):
            print_hdf5_structure(obj[key], indent + 2)
        elif isinstance(obj[key], h5py.Dataset):
            print(" " * (indent + 2) + f"🔢 Dataset: {obj[key].shape}, {obj[key].dtype}")

def print_items(dset):
# with h5py.File('SaSR_test.hdf5', 'r') as f:
#     dset = f['SR/V1:ENV_CEB_SEIS_V_dec']
    for key, value in dset.attrs.items():
        print(f"{key}: {value}")

#--------------------------Channels---------------------------#
tower = 'SR'
channels = ['V1:Sa_' + tower + '_F0_LVDT_V_500Hz',
            'V1:Sa_' + tower + '_F1_LVDT_V_500Hz', 
            'V1:Sa_' + tower + '_F2_LVDT_V_500Hz',
            'V1:Sa_' + tower + '_F3_LVDT_V_500Hz', 
            'V1:Sa_' + tower + '_F4_LVDT_V_500Hz']                
            #not considering F7

#required data
#f = h5py.File("SaSR_test.hdf5", "r")
#with open("SaSR_test.hdf5", "r") as f:
f = h5py.File("/Users/letizia/Desktop/INFN/new model/SaSR_test.hdf5", "r")
dset = f['SR/V1:ENV_CEB_SEIS_V_dec']
seism = f['SR/V1:ENV_CEB_SEIS_V_dec'][:] #seismic data
#seism = seism[2000:] #remove the first 2000 samples 
seism = seism[:14000] *1e-6
#print(seism[2000:2050])

#constants
nperseg = 2 ** 16 #samples per segment (useful for the PSD)

T = 224 #since we have removed the first 2000 samples, the signal duration is reduced
t = np.linspace(0, T, len(seism)) #time vector

#parameter to be used in the time evolution
dt = 1e-3 #time step


#take the fourier transform of the data
vf = np.fft.fft(seism)


frequencies = np.fft.fftfreq(len(seism), d = 1/62.5)
half = len(frequencies) // 2 #half of the frequencies array (positive frequencies only)

xf = vf[1:] / (1j * 2 * np.pi * frequencies[1:]) # Fourier Transform of the seismic data
X_f = np.zeros_like(vf, dtype=complex) #create an array of zeros with the same shape as V
nonzero = frequencies != 0 #boolean mask: true if freq is not zero

#for all non-zero frequencies, divide the FT by 2 pi f the take the IFT to get the displacement
X_f[nonzero] = vf[nonzero] / (1j * 2 * np.pi * frequencies[nonzero])

zt = np.fft.ifft(X_f).real
xt = np.fft.ifft(xf)  # inverse Fourier Transform to get the displacement
#print(xt)
#print(zt)
#multiply the FT by 2 pi f then take the IFT to get the acceleration
acc = vf * (frequencies * 2 * np.pi * 1j)
At = np.fft.ifft(acc)


#calculate the PSDs to plot the velocity and acceleration spectra
fAcc, psdAcc = signal.welch(At.real, fs = 62.5, window='hann', nperseg=nperseg)
fVel, psdVel = signal.welch(seism.real, fs = 62.5, window='hann', nperseg=nperseg)
fZ, psdZ = signal.welch(xt.real, fs = 62.5, window='hann', nperseg=nperseg)


def force_function(t, k, displacement):
    return k * np.real(displacement)

def evolution(evol_method, Nt_step, dt, physical_params, signal_params,
              F, file_name = None):
    """
    Simulates the temporal evolution of the system under the influence of an
    external force.

    Parameters
    ----------
    evol_method : function
        The function used to evolve the system (e.g. Euler or ARMA methods).
    Nt_step : int
        The number of temporal steps to simulate.
    dt : float
        The time step size.
    physical_params : list
        The list of physical parameters for the system.
    signal_params : list
        The list of parameters for the external force signal.
    F : function
        The function modeling the external force.
    file_name : str, optional
        The name of the file to save simulation data. Default is None.

    Returns
    -------
    tuple
        A tuple containing the time grid and the arrays of velocities
        and positions for each mass.
    """
    # Initialize the problem
    tmax = Nt_step * dt  # maximum time
    tt = np.arange(0, tmax, dt)  # time grid
    y0 = np.array(
        (0, 0, 0, 0., 0., 0.))  # initial condition
    y_t = np.copy(y0)  # create a copy to evolve it in time
    #run the simulation on the finer time grid
    F_signal = F(tt, *signal_params)  # external force applied over time (cambia)


    # Initialize lists for velocities and positions
    v1, v2, v6 = [[], [], []]
    x1, x2, x6 = [[], [], []]

    # compute the system matrices
    A, B = matrix(*physical_params)

    # time evolution when the ext force is applied
    i = 0
    for t in tt:
        Fi = F_signal[i]  # evaluate the force at time t
        i = i + 1
        y_t = evol_method(y_t, A, B, Fi)  # evolve to step n+1
        v1.append(y_t[0])
        v2.append(y_t[1])
        v6.append(y_t[2])
        x1.append(y_t[3])
        x2.append(y_t[4])
        x6.append(y_t[5])

    # save simulation's data (if a file name is provided)
    if file_name is not None:
        data = np.column_stack((tt, v1, v2, v6,
                                x1, x2, x6))
       # np.savetxt(os.path.join(data_dir, file_name), data,
        #           header='time, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6')

    return (tt, np.array(v1), np.array(v2), np.array(v6), np.array(x1), np.array(x2), np.array(x6))

# temporal steps (for simulation)
Nt_step = T * 1e3
#physical parameters of the system
gamma = [5, 5]  # viscous friction coeff [kg/m*s]
M = [160, 125, 82]  # filter mass [Kg]
K = [700, 1500, 564]  # spring constant [N/m]

F = force_function

wn = 2*np.pi*frequencies[:half]

physical_params = [*M, *K, *gamma, dt]

# Interpolate the acceleration onto the simulation time grid
# Time vector for seismic data (real data)
tt_data = np.arange(0, T, 1 / 62.5)  

# Time vector for simulation
tmax = Nt_step * dt  
tt_sim = np.arange(0, tmax, dt)

interp_acc = interp1d(tt_data, At, kind='linear', bounds_error=False, fill_value=0.0)
At_interp = interp_acc(tt_sim)

interp_displacement = interp1d(tt_data, zt, kind='linear', bounds_error=False, fill_value=0.0)
zt_interp = interp_displacement(tt_sim)

simulation_params = [AR_model, Nt_step, dt] 
#signal_params = [M[0], At] 
signal_params = [K[0], zt_interp] 

tt, v1, v2, v6, x1, x2, x6 = (
                        evolution(*simulation_params, physical_params, signal_params,
                        F, file_name = None))

Tf, poles = TransferFunc(wn, *M, *K, *gamma)
# Compute the magnitude of the transfer function (from simulation)
H = (np.real(Tf) ** 2 + np.imag(Tf) ** 2) ** (1 / 2)

#apply a Hanning window to the data to remove spectral leakage
window = np.hanning(len(seism))

x6_interp = interp1d(tt_sim, x6, kind='linear', bounds_error=False, fill_value=0.0)(tt_data)
v6_interp = interp1d(tt_sim, v6, kind='linear', bounds_error=False, fill_value=0.0)(tt_data)

A, B = matrix(*M, *K, *gamma, dt)  # system matrices
print(A.shape, B.shape) 
action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
observation_space = spaces.Discrete(10)
vout, xout = [], []  # lists to store the output signals


start = time.time()


i = 0
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
force = force_function(tt_sim, K[0], zt_interp)  # external force applied over time
for t in tt_sim:
    Fi = force[i] 
    i = i + 1
    state = AR_model(state, A, B, Fi)
    #state = np.array([v1, v2, v6, x1, x2, x6], dtype=np.float32)  # update the state
    vout.append(state[2])
    xout.append(state[5])
vout_interp = interp1d(tt_sim, vout, kind='linear', bounds_error=False, fill_value=0.0)(tt_data)
xout_interp = interp1d(tt_sim, xout, kind='linear', bounds_error=False, fill_value=0.0)(tt_data)

end = time.time()

print(f"Simulation time: {end - start:.3f} seconds")

    #print(f'\nAction {i}: {action}, State: {state}')
# print(action_space)
# action1 = (action_space.sample()) + (K[0] * xt[0].real)   # sample an action
# action2 = (action_space.sample()) + (K[0] * xt[1].real) # sample another action
# action3 = (action_space.sample()) + (K[0] * xt[2].real)  # sample another action
# print(f'Action 1: {action1}, Action 2: {action2}, Action 3: {action3}')
# state_evol = AR_model(state, A, B, action1)  # evolve the state for 1 step
# state2 = AR_model(state_evol, A, B, action2)
# state3 = AR_model(state2, A, B, action3)  # evolve the state for 2 steps
# print(state_evol)
# print(state_evol.shape)
# print(state2)
# print(state3)
# #output in frequency domain (resampled to match the data time vector)
#xf_in = np.fft.fft(X_f)  # input signal (zt)
#xf_out = np.fft.fft(x6_interp)

#force = F(tt_data, M[0], At)  # external force applied over time

#only keep positive frequencies
#the frequencies array is symmetric, so we only need the first half
xf_in = np.fft.fft(zt)

xf_out = np.fft.fft(x6_interp)
xf_out = xf_out[:half]

xf_out_windowed = np.fft.fft(x6_interp*window)
xf_in_windowed = np.fft.fft(zt * window) # apply the window to the input signal

vf_in = np.fft.fft(seism)  # input signal (seismic data)
vf_out = np.fft.fft(v6_interp) # output signal (v6)

vf_in_windowed = np.fft.fft(seism*window) # apply the window to the input signal
vf_out_windowed = np.fft.fft(v6_interp*window)  # apply the window to the output signal
#trfn = vf_out[:half] / vf_in[:half]
# # Compute transfer function and its magnitude
trfn = (xf_out / xf[:half])*dt # experimental transfer function
trfn_windowed = (xf_out_windowed[:half] / xf_in_windowed[:half])*dt  # experimental transfer function with windowing

trfn_vel = (vf_out[:half] / vf_in[:half])*dt  # experimental transfer function for velocity
trfn_vel_windowed = (vf_out_windowed[:half] / vf_in_windowed[:half])*dt  # experimental transfer function for velocity with windowing
# trfn = (vf_out)/vf_in

freq = frequencies[1:half]
Pxx = psdVel[1:half]
omega = 2 * np.pi * freq # angular frequencies
Pxx = Pxx/omega**2
ASDvelocity = np.sqrt(Pxx)  # Amplitude Spectral Density of the velocity (from data)
OUT = (H[2][1:] * ASDvelocity ) # output without control for velocity (ASD)
#OUT = (H[2] * np.fft.fft(x6_interp[:half]))  # output without control for velocity (ASD)
df = np.diff(frequencies[1:half])  # frequency resolution
varxx = np.cumsum(np.flip(df * OUT[:-1]) ** 2) # cumulative variance
rms_nc = np.flip(np.sqrt(varxx))  # RMS value of the output
#-----------queste cose qua sotto non sono plottate-----------#
Hfn = (np.real(trfn) ** 2 + np.imag(trfn) ** 2) ** (1 / 2)
Hfn_mag = np.abs(Hfn)

# _, psdZ = signal.welch(zt.real, fs = 62.5, window='hann', nperseg=nperseg)  # PSD of the displacement 
# psdZ = interp1d(fZ, psdZ, kind='linear', bounds_error=False, fill_value=0.0)(frequencies[:half])  # PSD of the displacement
# w = 2 * np.pi * frequencies[:half]  # angular frequencies
# psdVel = interp1d(fVel, psdVel, kind='linear', bounds_error=False, fill_value=0.0)(frequencies[:half])  # PSD of the velocity
#psd_disp = psdVel / omega**2 #convert to displacement

#ASDdisplacement = np.sqrt(psd_disp)  # Amplitude Spectral Density of the displacement (from data)
# ASDvelocity = np.sqrt(psdVel)  # Amplitude Spectral Density of the velocity (from data)

# out_asd_velocity = H[2] * ASDvelocity  # output without control for velocity (ASD)
# out_asd_displacement = H[2] * ASDdisplacement  # output without control for displacement (ASD)
# fA_interp, psdAinterp = signal.welch(At_interp.real, fs = 1e3, window='hann', nperseg=nperseg)  # PSD of the acceleration
# out_nocontrol = (abs(trfn)* (abs(xf[:half]))) # output without control
# out_nocontrol_velocity =(abs(trfn_vel) * (abs(vf_in[:half])))  # output without control for velocity

# df = np.diff(frequencies[:half]) # frequency resolution
# var_nocontrol = np.cumsum(np.flip(df * (out_nocontrol[:-1])** 2))  # cumulative variance
# rms_nocontrol = np.flip(np.sqrt(var_nocontrol))  # RMS value of the output without control
#print(f'mean:{np.mean(rms_nocontrol)}, max:{np.max(rms_nocontrol)}, min:{np.min(rms_nocontrol)}')

#print(f'Mean displacement: {np.mean(out_nocontrol):.3e} m, min: {np.min(out_nocontrol):.3e} m, max: {np.max(out_nocontrol):.3e} m')

if __name__ == '__main__':
    #print the structure of the dataset
    #print_hdf5_structure(f)
    #print_items(dset)
    plt.figure(figsize=(14, 4))

    plt.suptitle('Amplitude spectra', fontsize=16)

    plt.subplot(1, 3, 1)
    plt.loglog(fVel, np.sqrt(psdVel), label='Velocity')
    #plt.loglog(frequencies, abs(seism), label='Seismic data', color='orange', alpha=0.7)
    plt.ylabel('Amplitude [m/s/$\sqrt{Hz}$]')
    plt.xlabel('Frequency [Hz]')
    plt.grid(which = 'both', axis = 'both')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.loglog(fZ, np.sqrt(psdZ), label='Displacement', color='darkorange')
    plt.ylabel('Amplitude [m/$\sqrt{Hz}$]')
    plt.xlabel('Frequency [Hz]')
    plt.grid(which = 'both', axis = 'both')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.loglog(fAcc, np.sqrt(psdAcc), label ='Acceleration', color='green')
    #plt.loglog(fA_interp, np.sqrt(psdAinterp), label ='Acceleration (interpolated)', color='red', alpha=0.7)
    plt.ylabel('Amplitude [m/s$^2$/$\sqrt{Hz}$]')
    plt.xlabel('Frequency [Hz]')
    plt.grid(which = 'both', axis = 'both')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('figures/amplitude_spectra.png')
    
    #plot velocity and displacement (time domain)
    plt.figure(figsize=(8, 6))
    plt.suptitle('Seisimic data (V1:ENV_CEB_SEIS_V_dec)', fontsize=16, y = 0.95)

    plt.subplot(2, 1, 1)
    plt.plot(tt_data, seism, label='Velocity')
    plt.ylabel('Amplitude [m/s]')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(tt_data, zt.real, label='Displacement', color='darkorange')
    plt.ylabel('Amplitude [m]')
    plt.legend()
    plt.grid()
    plt.xlabel('Time [s]')

    # fig = plt.figure(figsize=(8, 5))
    # plt.title('Time evolution', size=13)
    # plt.xlabel('Time [s]', size=12)
    # plt.ylabel('x [m]', size=12)
    # plt.grid(True, ls='-', alpha=0.3, lw=0.5)
    # plt.minorticks_on()

    # #plt.plot(tt, x1, linestyle='-', linewidth=1, marker='', color='steelblue', label='x1, M$_1$')
    # #plt.plot(tt, x2, linestyle='-', linewidth=1, marker='', color='black', label='x2, M$_2$')
    # #plt.plot(tt, x3, linestyle='-', linewidth=1, marker='', color='red', label='x3, M$_3$')
    # #plt.plot(tt, x4, linestyle='-', linewidth=1, marker='', color='green', label='x4, M$_4$')
    # #plt.plot(tt, x5, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x7, M$_7$')
    # plt.plot(tt_sim, x6, linestyle='-', linewidth=1, marker='',color='blue', label='x$_{out}$, M$_{out}$') #ultima massa
    # plt.legend()
    #plt.savefig('figures/time_evolution.png')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.loglog(abs(frequencies)[:half], H[2][:half], label="Theoretical TF (from model)", color="blue")
    plt.loglog(abs(frequencies)[:half], abs(trfn), label=r"Experimental TF ($\tilde{x}_{6}/\tilde{x}_{in}$)", color="red", alpha=0.9)
    plt.loglog(abs(frequencies[:half]), abs(trfn_windowed), label=r"Experimental TF ($\tilde{x}_{6}/\tilde{x}_{in}$, windowed)", color="darkred", alpha=0.7)
    #plt.loglog(abs(frequencies[:half]), abs(trfn_vel), label=r"Experimental TF ($\tilde{v}_{6}/\tilde{v}_{in}$)", color="lightgreen", alpha=0.7)
    #plt.loglog(abs(frequencies[:half]), abs(trfn_vel_windowed), label=r"Experimental TF ($\tilde{v}_{6}/\tilde{v}_{in}$, windowed)", color="darkgreen", alpha=0.7)
    #plt.loglog(abs(frequencies[1:]), abs(np.fft.fft(xout_interp[1:])/xf)*dt, label="Test", color="orange")
    plt.xlabel("Frequency [Hz]")
    #plt.ylabel("Magnitude")
    plt.grid(True, which='both')
    plt.legend()
    plt.title("Transfer Function Comparison")
    
    #plt.figure(figsize=(8, 5))
    #plt.title('SR response to seism', size=13)

    plt.yscale('log')
    plt.xscale('log')
    # plt.xlim(1e-2, 2)
    # plt.ylim(1e-15, 1e-3)
    plt.grid(True, which='both', ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()

    plt.subplot(1, 2, 2)
    plt.title('Output response', size=13)
    #plt.loglog(freq, abs(OUT), linestyle='-', linewidth=1, marker='', color='green', label=r'Output (H*$\tilde{x}_{in}$)')
    #plt.loglog(frequencies[:half], out_nocontrol_velocity, linestyle='-', linewidth=1, marker='', color='red', label=r'Output (H*$\tilde{v}_{in}$)')
    plt.loglog(frequencies[:half], abs(trfn * xf[:half]), linestyle='--', linewidth=1, marker='', color='gold')
    plt.loglog(frequencies[:half], abs(vf_out[:half]*dt), linestyle='-', linewidth=1, marker='', color='steelblue', label=r'Output ($\tilde{v}_{6}$)', alpha=0.3)
    plt.loglog(frequencies[:half], abs(xf_out[:half]*dt), linestyle='-', linewidth=1, marker='', color='lightcoral', label=r'Output ($\tilde{x}_{6}$)', alpha=0.3)
    plt.loglog(frequencies[1:half], abs(OUT), linestyle='--', linewidth=1, marker='', color='darkorange', label=r'H*ASD ($\tilde{x}_{in}$)')
    #plt.loglog(frequencies[:half], out_asd_displacement, linestyle='--', linewidth=1, marker='', color='darkblue', label=r'H*ASD ($\tilde{x}_{6}$)')
    plt.xlabel('Frequency [Hz]', size=12)
    #plt.ylabel('ASD [m/$\sqrt{Hz}$]', size =12)
    plt.grid(True, which='both', ls='-', alpha=0.3, lw=0.5)

    #plt.plot(freq[0:-1], rms_FSF, linestyle='--', linewidth=1, marker='', color='Lime', label='rms control')

    plt.legend()
    
  
       #---------------------------RMS plot----------------------------#
    fig = plt.figure(figsize=(5, 5))
    plt.title('Mirror RMS displacement', size=13)
    plt.xlabel('Frequency [Hz]', size=12)
    plt.ylabel('RMS [m]', size =12)
    plt.yscale('log')
    plt.xscale('log')
    #plt.xlim(1e-2, 2)
    #plt.ylim(1e-15, 1e-4)
    plt.grid(True, which='both', ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()
    plt.plot(frequencies[1:half-1], rms_nc, linestyle='-', linewidth=1, marker='', color='orange', label='No control')
    plt.xlim(1e-2, 2)
    plt.ylim(1e-15, 1)   
    plt.legend()
    plt.show()

