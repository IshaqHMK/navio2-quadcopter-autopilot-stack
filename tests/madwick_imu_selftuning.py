# -*- coding: utf-8 -*-
"""
Self-Tuning IMU Tuning Code for Madgwick Filter
Version: IMU-Tuning-v2
"""

# Standard Python Libraries
import time
import numpy as np
import sys
import os

# Add project and imu directories to sys.path
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2/imu')

# Third-Party Libraries
from imu.madgwick import Madgwick  # Madgwick filter for IMU orientation estimation
from imu.imu_utils import calibrate_imu, read_imu_data  # Utility functions for IMU calibration

# Initialize IMU Sensor
from utils.navio2 import lsm9ds1_backup as lsm9ds1
imu = lsm9ds1.LSM9DS1()
imu.initialize()

# Sampling time and number of calibration samples
Ts = 0.005  # Sampling time in seconds
calibration_samples = 2000

# Calibrate IMU and retrieve biases
bx, by, bz, phi_bias, theta_bias, psi_bias, m9a, m9g, m9m = calibrate_imu(imu, calibration_samples, Ts)
print(f"IMU Calibration Done: Phi Bias: {np.rad2deg(phi_bias):.2f}°, Theta Bias: {np.rad2deg(theta_bias):.2f}°, Psi Bias: {np.rad2deg(psi_bias):.2f}°")

# Parameter ranges for self-tuning - TEST 1
gain_values = np.linspace(0.01, 0.1, 5)  # Range of gain values to test
beta_values = np.linspace(0.01, 0.1, 5)  # Range of beta values to test

# Define refined parameter ranges based on the best results - TEST 3
gain_values = np.linspace(0.03, 0.08, 6)  # Finer range around the best gain values
beta_values = np.linspace(0.09, 0.11, 6)  # Finer range around the best beta values

# Define finer parameter ranges around the current optimal - TEST 4
gain_values = np.linspace(0.055, 0.065, 6)  # Refine around 0.060
beta_values = np.linspace(0.105, 0.115, 6)  # Refine around 0.110

# Define finer parameter ranges around the current optimal - TEST 5
gain_values = np.linspace(0.01, 0.05, 50)  # Refine around 0.060
#beta_values = np.linspace(1, 2, 10)  # Refine around 0.110
#beta_values = np.linspace(0.01, 3, 30)  # Refine around 0.110
beta_values = np.linspace(0.01, 0.466, 50)  # Refine around 0.110

beta_values = np.linspace(1, 1, 1)  # Refine around 0.110
gain_values = np.linspace(0.1, 0.2, 20)  # Refine around 0.060

# 1.2077


# Initialize variables for storing the best parameters 
best_params = {"gain": None, "beta": None, "phi_std": float('inf'), "theta_std": float('inf'), "psi_std": float('inf')}

# Iterate over gain and beta values
for gain in gain_values:
    for beta in beta_values:
        print(f"Testing parameters: Gain={gain:.3f}, Beta={beta:.3f}")

        # Initialize Madgwick filter with current parameters
        madgwick = Madgwick(
            gyr=np.zeros((1, 3)),
            acc=np.zeros((1, 3)),
            mag=np.zeros((1, 3)),
            gain=gain,
            frequency=1 / Ts
        )
        madgwick.q0 = np.array([1, 0, 0, 0])  # Set the initial quaternion to identity

        # Initialize arrays to store measurements
        phi_meas_arr, theta_meas_arr, psi_meas_arr = [], [], []
        num_samples = int(5 / Ts)  # Test each configuration for 30 seconds

        # Collect data
        for _ in range(num_samples):
            start_time = time.time()

            # Read IMU data
            _, _, _, _, _, _, _, _, _, phi, theta, psi, _, _, _, _, _, q_out = read_imu_data(
                imu, Ts, 0, 0, 0, madgwick.q0, madgwick
            )

            # Update quaternion in Madgwick filter
            madgwick.q0 = q_out

            # Store measurements
            phi_meas_arr.append(phi)
            theta_meas_arr.append(theta)
            psi_meas_arr.append(psi)

            # Maintain precise loop timing
            elapsed_time = time.time() - start_time
            if elapsed_time < Ts:
                time.sleep(Ts - elapsed_time)

        # Analyze collected data
        phi_std = np.std(phi_meas_arr)
        theta_std = np.std(theta_meas_arr)
        psi_std = np.std(psi_meas_arr)

        print(f"Results for Gain={gain:.3f}, Beta={beta:.3f} | Phi Std: {phi_std:.4f}, Theta Std: {theta_std:.4f}, Psi Std: {psi_std:.4f}")

        # Update best parameters if current configuration is better
        if phi_std + theta_std + psi_std < best_params["phi_std"] + best_params["theta_std"] + best_params["psi_std"]:
            best_params.update({"gain": gain, "beta": beta, "phi_std": phi_std, "theta_std": theta_std, "psi_std": psi_std})

# Display the best parameters
print("\nSelf-Tuning Completed.")
print(f"Best Parameters: Gain={best_params['gain']:.3f}, Beta={best_params['beta']:.3f}")
print(f"Standard Deviations: Phi={best_params['phi_std']:.4f}, Theta={best_params['theta_std']:.4f}, Psi={best_params['psi_std']:.4f}")

# Optionally save best parameters and results
np.savez("imu_tuning_results.npz", best_params=best_params)
