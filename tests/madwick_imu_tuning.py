# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Madgwick Filter
Version: IMU-Tuning-v1
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
from imu.imu_utils import calibrate_imu, read_imu_data  # Utility functions for IMU calibration
from imu.imu_utils import calibrate_imu, read_imu_data, quaternion_to_euler, euler_to_quaternion  # Utility functions for IMU calibration and data handling
 

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

# Define Madgwick Filter Parameters for Tuning
# You can update `gain` and `beta` values here
madgwick_params = {
    "gain": 0.021,  # Recommended starting point for MARG
    "beta": 0.1,   # Recommended for general responsiveness
    "frequency": 1 / Ts
}

# Initialize Madgwick filter
#madgwick = Madgwick(gyr=np.zeros((1, 3)), acc=np.zeros((1, 3)), mag=np.zeros((1, 3)), **madgwick_params)


madgwick = Madgwick(beta=0.3)  # Adjust the beta value as needed for responsiveness

# Set the initial quaternion to identity
madgwick.q0 = np.array([1, 0, 0, 0])

# Initialize arrays to store measurements for analysis
num_samples = int(600 / Ts)  # Run for 60 seconds
phi_meas_arr, theta_meas_arr, psi_meas_arr = [], [], []

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

    # Store measurements
    phi_meas_arr.append(phi)
    theta_meas_arr.append(theta)
    psi_meas_arr.append(psi)

    # Display real-time measurements (optional)
    print(f"Time: {i * Ts:.2f}s | Phi: {np.rad2deg(phi):.2f}°, Theta: {np.rad2deg(theta):.2f}°, Psi: {np.rad2deg(psi):.2f}°")

    # Maintain precise loop timing
    loop_duration = time.time() - loop_start
    if loop_duration < Ts:
        time.sleep(Ts - loop_duration)

# End of Tuning
print("IMU Tuning Completed.")

# Analyze the Collected Data
phi_std = np.std(phi_meas_arr)
theta_std = np.std(theta_meas_arr)
psi_std = np.std(psi_meas_arr)

print(f"Standard Deviations: Phi: {phi_std:.4f} rad, Theta: {theta_std:.4f} rad, Psi: {psi_std:.4f} rad")
print("Try adjusting the 'gain' and 'beta' parameters to minimize these deviations.")

# Optionally save data for offline analysis
# np.save("imu_tuning_data.npz", phi=phi_meas_arr, theta=theta_meas_arr, psi=psi_meas_arr)

