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
    """Print a hierarchical structure of an HDF5 file.
    
    Parameters:
        obj: An h5py.File or h5py.Group object.
        indent: Current indentation level (used for recursive calls).
    """
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

tower = 'SR'
channels = ['V1:Sa_' + tower + '_F0_LVDT_V_500Hz',
            'V1:Sa_' + tower + '_F1_LVDT_V_500Hz', 
            'V1:Sa_' + tower + '_F2_LVDT_V_500Hz',
            'V1:Sa_' + tower + '_F3_LVDT_V_500Hz', 
            'V1:Sa_' + tower + '_F4_LVDT_V_500Hz']                
            #not considering F7

f = h5py.File("/Users/letizia/Desktop/INFN/new model/SaSR_test.hdf5", "r")
dset = f['SR/V1:ENV_CEB_SEIS_V_dec']
seism = f['SR/V1:ENV_CEB_SEIS_V_dec'][:] #seismic data
seism = seism * 1e-6

#constants
nperseg = 2 ** 16 #samples per segment (useful for the PSD)

T = 1800 
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
    """
    Calculates the force exerted by a spring based on the displacement.

    Parameters:
        t (float): Current time (not used in the calculation, but included for the time evolution method).
        k (float): Spring constant (depends on the system).
        displacement (complex or float): The displacement of the spring. Can be a complex number.

    Returns:
        float: The real part of the force calculated as k times the real part of the displacement.
    """
    return k * np.real(displacement)

def evolution(evol_method, Nt_step, dt, physical_params, signal_params,
              F, file_name = None):
    """
    Simulates the temporal evolution of the system under the influence of an
    external force.

    Parameters:
        evol_method (function): The function used to calculate the time evolution of the system (e.g. Euler or ARMA).
        Nt_step (int): The number of temporal steps to simulate.
        dt (float): The time step size.
        physical_params (list): The list of physical parameters for the system.
        signal_params (list): The list of parameters for the external force signal.
        F (function): The function modeling the external force.
        file_name (str, optional): The name of the file to save simulation data. Default is None.

    Returns:
        tuple: A tuple containing the time grid and the arrays of velocities and positions for each mass.
    """
    #initialise the problem
    tmax = Nt_step * dt  #maximum time
    tt = np.arange(0, tmax, dt)  #time grid
    y0 = np.array(
        (0, 0, 0, 0., 0., 0.))  #initial condition: set all velocities and positions to zero
    y_t = np.copy(y0)  #create a copy to evolve it in time

    F_signal = F(tt, *signal_params)  # external force applied over time 

    #initialise lists for velocities and positions
    v1, v2, v6 = [[], [], []]
    x1, x2, x6 = [[], [], []]

    #compute the system matrices
    A, B = matrix(*physical_params)

    #time evolution when the external force is applied
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

#temporal steps (for simulation)
Nt_step = T * 1e3

#physical parameters of the system
gamma = [5, 5]  # viscous friction coeff [kg/m*s]
M = [160, 125, 82]  # filter mass [Kg]
K = [700, 1500, 564]  # spring constant [N/m]

F = force_function

wn = 2*np.pi*frequencies[:half]

physical_params = [*M, *K, *gamma, dt]

#Interpolate the acceleration onto the simulation time grid

#time vector for seismic data (real data)
tt_data = np.arange(0, T, 1 / 62.5)  
#time vector for simulation
tmax = Nt_step * dt  
tt_sim = np.arange(0, tmax, dt)

#interpolate acceleration
interp_acc = interp1d(tt_data, At, kind='linear', bounds_error=False, fill_value=0.0)
At_interp = interp_acc(tt_sim)

#interpolate displacement data
interp_displacement = interp1d(tt_data, zt, kind='linear', bounds_error=False, fill_value=0.0)
zt_interp = interp_displacement(tt_sim)

simulation_params = [AR_model, Nt_step, dt]  
signal_params = [K[0], zt_interp] 

#calculate the time evolution of the system under the seismic input
tt, v1, v2, v6, x1, x2, x6 = (
                        evolution(*simulation_params, physical_params, signal_params,
                        F, file_name = None))

#calculate the theoretical transfer function (from model, no data)
Tf, poles = TransferFunc(wn, *M, *K, *gamma)
#compute the magnitude of the transfer function (from simulation)
H = (np.real(Tf) ** 2 + np.imag(Tf) ** 2) ** (1 / 2)

#apply a Hanning window to the data to remove spectral leakage
window = np.hanning(len(seism))

#interplate signal back to match data time grid
x6_interp = interp1d(tt_sim, x6, kind='linear', bounds_error=False, fill_value=0.0)(tt_data)
v6_interp = interp1d(tt_sim, v6, kind='linear', bounds_error=False, fill_value=0.0)(tt_data)


xf_in = np.fft.fft(zt)
xf_out = np.fft.fft(x6_interp)
#only keep positive frequencies
#the frequencies array is symmetric, so we only need the first half
xf_out = xf_out[:half]

xf_out_windowed = np.fft.fft(x6_interp*window)
xf_in_windowed = np.fft.fft(zt * window) # apply the window to the input signal

vf_in = np.fft.fft(seism)  # input signal (seismic data)
vf_out = np.fft.fft(v6_interp) # output signal (v6)

vf_in_windowed = np.fft.fft(seism*window) # apply the window to the input signal
vf_out_windowed = np.fft.fft(v6_interp*window)  # apply the window to the output signal

#Compute transfer function and its magnitude from data
trfn = (xf_out / xf[:half])*dt #experimental transfer function
trfn_windowed = (xf_out_windowed[:half] / xf_in_windowed[:half])*dt  # experimental transfer function with windowing

trfn_vel = (vf_out[:half] / vf_in[:half])*dt  # experimental transfer function for velocity
trfn_vel_windowed = (vf_out_windowed[:half] / vf_in_windowed[:half])*dt  # experimental transfer function for velocity with windowing



if __name__ == '__main__':
    #print the structure of the dataset
    #print_hdf5_structure(f)
    print_items(dset)
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
    plt.xlim(0, 1000)
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(tt_data, zt.real, label='Displacement', color='darkorange')
    plt.ylabel('Amplitude [m]')
    plt.xlim(0, 1000)
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
    plt.loglog(abs(frequencies)[:half], abs(trfn), label=r"Experimental TF ($\tilde{x}_{6}/\tilde{x}_{in}$)", color="red", alpha=0.2)
    plt.loglog(abs(frequencies[:half]), abs(trfn_windowed), label=r"Experimental TF ($\tilde{x}_{6}/\tilde{x}_{in}$, windowed)", color="darkred", alpha=0.7)
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
    plt.loglog(frequencies[:half], abs(xf_out[:half]*dt), linestyle='-', linewidth=1, marker='', color='lightcoral', label=r'Output ($\tilde{x}_{6}$)', alpha=0.7)
    #plt.loglog(frequencies[1:half], abs(OUT), linestyle='--', linewidth=1, marker='', color='darkorange', label=r'H*ASD ($\tilde{x}_{in}$)')
    #plt.loglog(frequencies[:half], out_asd_displacement, linestyle='--', linewidth=1, marker='', color='darkblue', label=r'H*ASD ($\tilde{x}_{6}$)')
    plt.xlabel('Frequency [Hz]', size=12)
    #plt.ylabel('ASD [m/$\sqrt{Hz}$]', size =12)
    plt.grid(True, which='both', ls='-', alpha=0.3, lw=0.5)

    plt.legend()
    
    plt.figure(figsize=(8, 5))
    plt.plot(tt,x6, label = '$x_3, M_3$ (output)')
    plt.plot(tt, x1, color='red', alpha=0.8, label = '$x_1, M_1$')
    plt.plot(tt, x2, linestyle='--',color='black', label = '$x_2, M_2$')
    # plt.plot(np.mean(x6)*np.ones_like(x6), linestyle='--', color='red', label='Mean value')
    # plt.scatter(x6[np.argmax(np.abs(x6))], np.max(x6), color='green', label='Max value = {:.2e} m'.format(np.max(x6)))
    # plt.scatter(x6[np.argmin(np.abs(x6))], np.min(x6), color='orange', label='Min value = {:.2e} m'.format(np.min(x6)))
    plt.title('Time evolution of the output displacement (first 180 s)', size=13)
    plt.xlabel('Time [s]', size=12)
    plt.xlim(1600, 1800)
    plt.ylabel('Displacement [$\mu$m]', size=12)
    plt.grid(True, ls='-', alpha=0.3, lw=0.5)
    plt.minorticks_on()
    plt.legend()

    plt.show()

