# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Madgwick Filter with IMU Offset Correction
Version: IMU-Tuning-v1_Modified

This code applies acceleration correction to account for the IMU's offset from
the center of gravity (CG). The correction is based on the rigid-body kinematics:
    
    acc_cg = acc_sensor - [ (domega x r_offset) + (omega x (omega x r_offset)) ]

where:
    - acc_sensor: raw accelerometer measurement from the IMU,
    - omega: angular velocity vector from the gyro,
    - domega: angular acceleration (approximated by finite differences),
    - r_offset: the displacement vector from the CG to the sensor location.

Reference for further details:
Titterton, D. H., & Weston, J. L. (2004). Strapdown Inertial Navigation Technology.
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
from imu.imu_utils import calibrate_imu, read_imu_data, quaternion_to_euler, euler_to_quaternion  # Utility functions

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
madgwick_params = {
    "gain": 0.021,  # Recommended starting point for MARG
    "beta": 0.1,    # Recommended for general responsiveness
    "frequency": 1 / Ts
}

# Initialize Madgwick filter (adjust beta as needed)
madgwick = Madgwick(beta=0.3)
madgwick.q0 = np.array([1, 0, 0, 0])  # Set initial quaternion to identity

# Define sensor offset (IMU position relative to the center of gravity) in meters
# Adjust these values based on your hardware configuration.
r_offset = np.array([0.05, -0.03, 0.0])  # [x_offset, y_offset, z_offset]

# Initialize arrays to store measurements for analysis and EKF input
num_samples = int(0.05 / Ts)  # Number of samples (modify if needed)
phi_meas_arr, theta_meas_arr, psi_meas_arr = [], [], []
acc_cg_arr = []   # Corrected accelerometer measurements (CG)
gyro_cg_arr = []  # Gyro measurements (unchanged)
mag_cg_arr = []   # Magnetometer measurements (unchanged)

# Initialize previous gyro measurement for angular acceleration calculation
prev_gyro = np.zeros(3)

# Main Tuning Loop
print("Starting IMU Tuning...")
start_time = time.time()

for i in range(num_samples):
    loop_start = time.time()  # Record loop start time

    # Read IMU data
    X_ddot, Y_ddot, Z_ddot, mx, my, mz, p, q, r, phi, theta, psi, _, _, _, _, _, q_out = read_imu_data(
        imu, Ts, 0, 0, 0, madgwick.q0, madgwick
    )
    
    # Update the quaternion in the Madgwick filter
    madgwick.q0 = q_out

    # Assemble gyro (angular velocity) vector from measurements
    omega = np.array([p, q, r])
    
    # Compute angular acceleration (domega) using finite differences
    if i == 0:
        domega = np.zeros(3)
    else:
        domega = (omega - prev_gyro) / Ts

    # Store current gyro measurement for the next iteration
    prev_gyro = omega.copy()

    # Compute the correction terms:
    # term1: acceleration due to angular acceleration = domega x r_offset
    term1 = np.cross(domega, r_offset)
    # term2: centripetal acceleration = omega x (omega x r_offset)
    term2 = np.cross(omega, np.cross(omega, r_offset))
    correction = term1 + term2

    # Raw accelerometer measurement vector from sensor
    acc_sensor = np.array([X_ddot, Y_ddot, Z_ddot])
    
    # Corrected accelerometer measurement at the center of gravity
    acc_cg = acc_sensor - correction

    # For a rigid body, gyro and magnetometer readings are assumed to be the same at the CG.
    gyro_cg = omega
    mag_cg = np.array([mx, my, mz])

    # (Optional) Feed the corrected measurements to your EKF here.
    # For example: ekf.update(acc_cg, gyro_cg, mag_cg)

    # Store measurements for later analysis or logging
    phi_meas_arr.append(phi)
    theta_meas_arr.append(theta)
    psi_meas_arr.append(psi)
    acc_cg_arr.append(acc_cg)
    gyro_cg_arr.append(gyro_cg)
    mag_cg_arr.append(mag_cg)

    # Display real-time measurements (optional)
    print(f"Time: {i * Ts:.2f}s | Phi: {np.rad2deg(phi):.2f}°, Theta: {np.rad2deg(theta):.2f}°, Psi: {np.rad2deg(psi):.2f}°")
    print(f"Corrected Acc (CG): {acc_cg}, Gyro: {gyro_cg}, Mag: {mag_cg}")

    # Maintain precise loop timing
    loop_duration = time.time() - loop_start
    if loop_duration < Ts:
        time.sleep(Ts - loop_duration)

print("IMU Tuning Completed.")

# Analyze the Collected Data (example: compute standard deviations of Euler angles)
phi_std = np.std(phi_meas_arr)
theta_std = np.std(theta_meas_arr)
psi_std = np.std(psi_meas_arr)
print(f"Standard Deviations: Phi: {phi_std:.4f} rad, Theta: {theta_std:.4f} rad, Psi: {psi_std:.4f} rad")
print("Try adjusting the 'gain' and 'beta' parameters to minimize these deviations.")

# Optionally, save data for offline analysis
# np.savez("imu_tuning_data.npz", phi=phi_meas_arr, theta=theta_meas_arr, psi=psi_meas_arr,
#          acc_cg=acc_cg_arr, gyro_cg=gyro_cg_arr, mag_cg=mag_cg_arr)

# Reference for further reading on sensor compensation:
# Titterton, D. H., & Weston, J. L. (2004). Strapdown Inertial Navigation Technology.
