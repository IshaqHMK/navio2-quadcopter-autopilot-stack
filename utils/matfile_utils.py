
# utils/matfile_utils.py

import scipy.io
from datetime import datetime
import numpy as np

def initialize_storage(num_samples):

    """
    Initialize storage arrays for simulation data.
    
    Args:
        num_samples (int): Total number of samples in the simulation.
        
    Returns:
        dict: A dictionary containing initialized numpy arrays for various simulation data.
            - 'sim_times': Array to store simulation time steps. Shape: (num_samples,)
            - 'omegas': Array to store motor speeds for 4 motors. Shape: (num_samples, 4)
            - 'altitudes': Array to store altitude-related data (measured, estimated position, estimated speed). Shape: (num_samples, 3)
            - 'acc_data': Array to store accelerometer data (X, Y, Z). Shape: (num_samples, 3)
            - 'gyro_data': Array to store gyroscope data (p, q, r). Shape: (num_samples, 3)
            - 'mag_data': Array to store magnetometer data (mx, my, mz). Shape: (num_samples, 3)
            - 'attitude_data': Array to store attitude data (roll, pitch, yaw in radians). Shape: (num_samples, 3)
            - 'control_input_data': Array to store control input data (U1, U2, U3, U4). Shape: (num_samples, 4)
            - 'reference_data': Array to store reference signals (Z, roll, pitch, yaw). Shape: (num_samples, 4)
            - 'theta_pid_data': Array to store PID gains for pitch (KP, KI, KD). Shape: (num_samples, 3)
            - 'rate_reference_data': Array to store desired rates for roll, pitch, and yaw (p, q, r). Shape: (num_samples, 3)
    """

    return {
        
        'sim_times': np.zeros(num_samples),
        'omegas': np.zeros((num_samples, 4)),  # 4 motors
        'altitudes': np.zeros((num_samples, 3)),  # Altitude data
        'acc_data': np.zeros((num_samples, 3)),  # Accelerometer data
        'gyro_data': np.zeros((num_samples, 3)),  # Gyroscope data
        'mag_data': np.zeros((num_samples, 3)),  # Magnetometer data
        'attitude_data': np.zeros((num_samples, 3)),  # Attitude data
        'control_input_data': np.zeros((num_samples, 4)),  # Control inputs
        'reference_data': np.zeros((num_samples, 4)),  # References
        'theta_pid_data': np.zeros((num_samples, 3)),  # Theta PID
        'rate_reference_data': np.zeros((num_samples, 3)),  # Rate references
        'adaptive_gains_roll': np.zeros((num_samples, 3)),
        'adaptive_gains_pitch': np.zeros((num_samples, 3)),
        'adaptive_gains_yaw': np.zeros((num_samples, 3)),
        'phi_adaptive_est_history': np.zeros((num_samples, 3)), # (New)
        'theta_adaptive_est_history': np.zeros((num_samples, 3)), # (New)
        'psi_adaptive_est_history': np.zeros((num_samples, 3)) # (New)
    }

def save_to_matfile(storage, gains, path_prefix="quad_exp_lat"):
    """Save simulation data and PID gains to a .mat file."""
    # Define the path with the current date and time
    now = datetime.now()
    date_time_str = now.strftime("%d_%m_%y_%H_%M_%S")
    path = f'{path_prefix}_{date_time_str}.mat'

    # Prepare data to save
    data_to_save = {**storage, **gains}

    # Save to .mat file
    scipy.io.savemat(path, data_to_save)
    print(f"Data saved to: {path}")
