
# -*- coding: utf-8 -*-
"""
IMU Tuning Code with Recursive Low-Pass Filter
Version: IMU-Tuning-v2
"""

# Standard Python Libraries
import time
import numpy as np

import sys
import os
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2/imu')  # Add `imu` directory explicitly

# Third-Party Libraries
from imu.madgwick import Madgwick  # Madgwick filter for IMU orientation estimation
from imu.imu_utils import calibrate_imu, read_imu_data  # Utility functions for IMU calibration and data handling

# Initialize IMU Sensor
from utils.navio2 import lsm9ds1_backup as lsm9ds1
imu = lsm9ds1.LSM9DS1()
imu.initialize()

# Sampling time and number of calibration samples
Ts = 0.005  # Sampling time in seconds
calibration_samples = 100

# Calibrate IMU and retrieve biases
bx, by, bz, phi_bias, theta_bias, psi_bias, m9a, m9g, m9m = calibrate_imu(imu, calibration_samples, Ts)
print(f"IMU Calibration Done: Phi Bias: {np.rad2deg(phi_bias):.2f}°, Theta Bias: {np.rad2deg(theta_bias):.2f}°, Psi Bias: {np.rad2deg(psi_bias):.2f}°")

# Recursive filter function
def apply_recursive_filter(new_value, buffer, alpha, max_buffer_length):
    """
    Apply a recursive low-pass filter to a signal.
    
    Args:
        new_value (float): The latest data point to filter.
        buffer (list): A list of previous filtered values.
        alpha (float): The low-pass filter coefficient (0 < alpha < 1).
        max_buffer_length (int): The maximum allowed length of the buffer.
        
    Returns:
        float: The filtered value.
    """
    if buffer:
        filtered_value = alpha * new_value + (1 - alpha) * buffer[-1]
    else:
        filtered_value = new_value
    buffer.append(filtered_value)
    buffer[:] = buffer[-max_buffer_length:]  # Limit buffer size
    return filtered_value

# Filter parameters
filter_cutoff = 10.0  # Cutoff frequency in Hz
alpha = Ts / (Ts + (1 / (2 * np.pi * filter_cutoff)))  # Precompute alpha

# Define Madgwick Filter Parameters for Tuning
madgwick_params = {
    "gain": 0.021,  # Recommended starting point for MARG
    "beta": 0.32,   # Recommended for general responsiveness
    "frequency": 1 / Ts
}

# Initialize Madgwick filter
madgwick = Madgwick(gyr=np.zeros((1, 3)), acc=np.zeros((1, 3)), mag=np.zeros((1, 3)), **madgwick_params)
madgwick.q0 = np.array([1, 0, 0, 0])  # Set the initial quaternion to identity

# Initialize arrays to store measurements for analysis
num_samples = int(600 / Ts)  # Run for 60 seconds
phi_meas_arr, theta_meas_arr, psi_meas_arr = [], [], []
filtered_phi_arr, filtered_theta_arr, filtered_psi_arr = [], [], []

# Initialize buffers for recursive filtering
phi_buffer, theta_buffer, psi_buffer = [], [], []

# Main Tuning Loop
print("Starting IMU Tuning...")
start_time = time.time()

for i in range(num_samples):
    # Record current time for precise control
    loop_start = time.time()

    # Read IMU data
    X_ddot, Y_ddot, Z_ddot, mx, my, mz, p, q, r, phi, theta, psi, _, _, _, _, _, q_out = read_imu_data(
        imu, Ts, 0, 0, 0, madgwick.q0, madgwick
    )
    
    # Update the quaternion in the Madgwick filter
    madgwick.q0 = q_out

    # Store raw measurements
    phi_meas_arr.append(phi)
    theta_meas_arr.append(theta)
    psi_meas_arr.append(psi)

    # Apply the recursive low-pass filter to the latest data
    filtered_phi = apply_recursive_filter(phi, phi_buffer, alpha, max_buffer_length=20)
    filtered_theta = apply_recursive_filter(theta, theta_buffer, alpha, max_buffer_length=20)
    filtered_psi = apply_recursive_filter(psi, psi_buffer, alpha, max_buffer_length=20)

    # Store filtered measurements
    filtered_phi_arr.append(filtered_phi)
    filtered_theta_arr.append(filtered_theta)
    filtered_psi_arr.append(filtered_psi)

    # Display real-time measurements
    print(f"Time: {i * Ts:.2f}s | Raw Phi: {np.rad2deg(phi):.2f}°, Filtered Phi: {np.rad2deg(filtered_phi):.2f}° | "
          f"Raw Theta: {np.rad2deg(theta):.2f}°, Filtered Theta: {np.rad2deg(filtered_theta):.2f}° | "
          f"Raw Psi: {np.rad2deg(psi):.2f}°, Filtered Psi: {np.rad2deg(filtered_psi):.2f}°")

    # Maintain precise loop timing
    loop_duration = time.time() - loop_start
    if loop_duration < Ts:
        time.sleep(Ts - loop_duration)

# End of Tuning
print("IMU Tuning Completed.")

# Analyze the Collected Data
phi_std = np.std(filtered_phi_arr)
theta_std = np.std(filtered_theta_arr)
psi_std = np.std(filtered_psi_arr)

print(f"Filtered Standard Deviations: Phi: {phi_std:.4f} rad, Theta: {theta_std:.4f} rad, Psi: {psi_std:.4f} rad")
print("Try adjusting the filter cutoff frequency to optimize performance.")
