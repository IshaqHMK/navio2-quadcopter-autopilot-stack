#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Madgwick, EKF & Complementary Filters 
with Sensor Offset and Misalignment Correction
Version: IMU-Tuning-vDualFiltersPlusComp-v1
"""

# ========================
# Standard Python Libraries
# ========================
import time
import numpy as np
import math
import sys
import os
from scipy.io import savemat  # For saving results to a .mat file

import RPi.GPIO as GPIO  # Add GPIO library

GPIO.setwarnings(False)  # Prevent duplicate GPIO warnings
GPIO.setmode(GPIO.BCM)
START_SIGNAL_PIN = 26  # GP26 (Pin 37)

GPIO.setup(START_SIGNAL_PIN, GPIO.OUT)
GPIO.output(START_SIGNAL_PIN, GPIO.LOW)  # Stop signal to dSPACE
time.sleep(0)  # Simulate experiment duration

# Add your project directories if necessary
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2/imu')

# ========================
# Global Variables
# ========================
# Sensor offset from the center of gravity (in meters) in the vehicle frame.
# sensor_offset = np.array([-0.06, -0.05, 0.0])  # e.g. 10 cm offset in X and Y
sensor_offset = np.array([0, 0, 0.0])  # e.g. 10 cm offset in X and Y

# Misalignment correction angles (in radians); these rotate the sensor frame into the vehicle frame.
misalign_phi   = math.radians(0)               # Rotation about sensor X-axis.
misalign_theta = math.radians(0)   # For example, 3.4° about sensor Y-axis.
misalign_psi   = math.radians(0)               # Rotation about sensor Z-axis.

# Complementary filter gain (0 < gain < 1). Gain near 1 uses mostly gyro integration.
comp_gain = 0.98

# Global variable to hold previous gyro measurement (for computing angular acceleration).
prev_gyro = None

# ========================
# Third-Party / Custom Modules
# ========================
from imu.madgwick import Madgwick  # Madgwick filter for orientation estimation
from imu.ekf import EKF            # EKF filter for orientation estimation

# ------------------------
# Helper Functions
# ------------------------

def rotation_matrix_from_euler(phi, theta, psi):
    """
    Compute the rotation matrix from Euler angles (X-Y-Z order).
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(phi), -math.sin(phi)],
                    [0, math.sin(phi), math.cos(phi)]])
    R_y = np.array([[math.cos(theta), 0, math.sin(theta)],
                    [0, 1, 0],
                    [-math.sin(theta), 0, math.cos(theta)]])
    R_z = np.array([[math.cos(psi), -math.sin(psi), 0],
                    [math.sin(psi), math.cos(psi), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x

# Precompute the misalignment rotation matrix (constant).
R_mis = rotation_matrix_from_euler(misalign_phi, misalign_theta, misalign_psi)

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    q1, q2: Arrays/lists of the form [w, x, y, z]
    Returns:
      q = q1 * q2 (as a numpy array)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def euler_to_quaternion(phi, theta, psi):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion (w, x, y, z).
    """
    qw = math.cos(phi/2)*math.cos(theta/2)*math.cos(psi/2) + math.sin(phi/2)*math.sin(theta/2)*math.sin(psi/2)
    qx = math.sin(phi/2)*math.cos(theta/2)*math.cos(psi/2) - math.cos(phi/2)*math.sin(theta/2)*math.sin(psi/2)
    qy = math.cos(phi/2)*math.sin(theta/2)*math.cos(psi/2) + math.sin(phi/2)*math.cos(theta/2)*math.sin(psi/2)
    qz = math.cos(phi/2)*math.cos(theta/2)*math.sin(psi/2) - math.sin(phi/2)*math.sin(theta/2)*math.cos(psi/2)
    return np.array([qw, qx, qy, qz])

def quaternion_to_euler(q):
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
    """
    w, x, y, z = q
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    phi = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2*(w*y - z*x)
    # Clamp sinp to [-1,1] to avoid numerical errors.
    sinp = max(-1.0, min(1.0, sinp))
    theta = math.asin(sinp)
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    psi = math.atan2(siny_cosp, cosy_cosp)
    return phi, theta, psi

def complementary_update(q_prev, gyr, acc, mag, dt, gain):
    """
    Complementary filter update.
    q_prev: previous orientation as quaternion [w,x,y,z]
    gyr: gyro measurement (aligned) in rad/s (list or array of length 3)
    acc: accelerometer measurement (compensated) in m/s^2 (list or array of length 3)
    mag: magnetometer measurement (aligned) (list or array of length 3)
    dt: time step (seconds)
    gain: complementary filter gain (0 < gain < 1)
    
    Returns:
      q_out: updated quaternion estimate.
    """
    # 1. Gyro integration.
    delta_angle = np.array(gyr) * dt
    angle = np.linalg.norm(delta_angle)
    if angle > 1e-6:
        axis = delta_angle / angle
        delta_q = np.hstack([math.cos(angle/2), math.sin(angle/2)*axis])
    else:
        delta_q = np.array([1.0, 0.0, 0.0, 0.0])
    q_gyro = quaternion_multiply(q_prev, delta_q)
    
    # 2. Accelerometer-Magnetometer (AM) estimation.
    ax, ay, az = acc
    # Compute roll and pitch from accelerometer.
    phi_am = math.atan2(ay, az)
    theta_am = math.atan2(-ax, math.sqrt(ay**2 + az**2))
    # Tilt-compensated yaw estimation.
    mx, my, mz = mag
    cos_phi = math.cos(phi_am)
    sin_phi = math.sin(phi_am)
    cos_theta = math.cos(theta_am)
    sin_theta = math.sin(theta_am)
    Bx = mx * cos_theta + my * sin_phi * sin_theta + mz * cos_phi * sin_theta
    By = my * cos_phi - mz * sin_phi
    yaw_am = math.atan2(-By, Bx)
    q_am = euler_to_quaternion(phi_am, theta_am, yaw_am)
    
    # 3. Fuse the two estimates.
    q_blend = gain * q_gyro + (1 - gain) * q_am
    q_out = q_blend / np.linalg.norm(q_blend)
    return q_out

def _compute_tilt_compensated_yaw(mx, my, mz, roll_rad, pitch_rad):
    """Compute yaw from magnetometer with tilt compensation."""
    cos_phi = math.cos(roll_rad)
    sin_phi = math.sin(roll_rad)
    cos_theta = math.cos(pitch_rad)
    sin_theta = math.sin(pitch_rad)
    Bx = mx * cos_theta + my * sin_phi * sin_theta + mz * cos_phi * sin_theta
    By = my * cos_phi - mz * sin_phi
    yaw_rad = math.atan2(-By, Bx)
    return yaw_rad

# ------------------------
# Calibration Function
# ------------------------
def calibrate_imu(imu, calibration_samples, dt):
    """
    Calibrate the IMU by determining gyro biases and computing initial Euler angles
    from the accelerometer and magnetometer.
    (Applies misalignment correction.)
    """
    bx, by, bz = 0.0, 0.0, 0.0
    phi_sum, theta_sum, psi_sum = 0.0, 0.0, 0.0

    start_time = time.time()
    for i in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        # Convert raw data to numpy arrays.
        acc_raw = np.array(m9a)
        gyro_raw = np.array(m9g)
        mag_raw = np.array(m9m)
        # Apply sign corrections for magnetometer.
        mag_raw[1] = -mag_raw[1]
        mag_raw[2] = -mag_raw[2]
        # Apply misalignment correction.
        acc_aligned = R_mis.dot(acc_raw)
        gyro_aligned = R_mis.dot(gyro_raw)
        mag_aligned  = R_mis.dot(mag_raw)
        ax, ay, az = acc_aligned
        gx, gy, gz = gyro_aligned
        mx, my, mz = mag_aligned

        bx += gx
        by += gy
        bz += gz

        # Compute roll and pitch from accelerometer.
        phi = math.atan2(ay, math.sqrt(ax**2 + az**2))
        theta = math.atan2(-ax, math.sqrt(ay**2 + az**2))
        # Compute tilt-compensated yaw from magnetometer.
        psi = _compute_tilt_compensated_yaw(mx, my, mz, phi, theta)

        phi_sum += phi
        theta_sum += theta
        psi_sum += psi

        next_sample_time = start_time + (i + 1) * dt
        while time.time() < next_sample_time:
            pass

    bx /= calibration_samples
    by /= calibration_samples
    bz /= calibration_samples
    phi_bias = phi_sum / calibration_samples
    theta_bias = theta_sum / calibration_samples
    psi_bias = psi_sum / calibration_samples

    return bx, by, bz, phi_bias, theta_bias, psi_bias, m9a, m9g, m9m

# ------------------------
# New Dual-IMU Calibration Function
# ------------------------
def calibrate_dual_imu(imu1, imu2, calibration_samples, dt):
    """
    Calibrate two IMUs and return the average biases and initial Euler angles.
    """
    bx1, by1, bz1, phi_bias1, theta_bias1, psi_bias1, _, _, _ = calibrate_imu(imu1, calibration_samples, dt)
    bx2, by2, bz2, phi_bias2, theta_bias2, psi_bias2, _, _, _ = calibrate_imu(imu2, calibration_samples, dt)
    bx = (bx1 + bx2) / 2
    by = (by1 + by2) / 2
    bz = (bz1 + bz2) / 2
    phi_bias = (phi_bias1 + phi_bias2) / 2
    theta_bias = (theta_bias1 + theta_bias2) / 2
    psi_bias = (psi_bias1 + psi_bias2) / 2
    return bx, by, bz, phi_bias, theta_bias, psi_bias

# ------------------------
# Read IMU Data (Raw with Corrections)
# ------------------------
def read_imu_data_raw(imu, dt, prev_gyro_phi, prev_gyro_theta, prev_psi):
    """
    Read raw IMU data, apply misalignment correction and sensor offset compensation.
    Returns:
      acc_comp: Accelerometer measurement corrected to the center-of-gravity (list)
      gyro_aligned: Misalignment-corrected gyro (list)
      mag_aligned: Misalignment-corrected magnetometer (list)
      phi_acc, theta_acc: Accelerometer-based roll and pitch (radians)
      phi_gyro, theta_gyro, psi_gyro: Integrated gyro angles (for reference)
    """
    global prev_gyro, sensor_offset
    m9a, m9g, m9m = imu.getMotion9()
    acc_raw = np.array(m9a)
    gyro_raw = np.array(m9g)
    mag_raw = np.array(m9m)
    # Apply sign corrections for magnetometer.
    mag_raw[1] = -mag_raw[1]
    mag_raw[2] = -mag_raw[2]
    # Apply misalignment correction.
    acc_aligned = R_mis.dot(acc_raw)
    gyro_aligned = R_mis.dot(gyro_raw)
    mag_aligned  = R_mis.dot(mag_raw)
    # Compute sensor offset compensation for accelerometer.
    current_gyro = np.array(gyro_aligned)
    if prev_gyro is None:
        domega = np.zeros(3)
    else:
        domega = (current_gyro - prev_gyro) / dt
    prev_gyro = current_gyro
    term1 = np.cross(domega, sensor_offset)
    term2 = np.cross(current_gyro, np.cross(current_gyro, sensor_offset))
    acc_comp = acc_aligned - (term1 + term2)

    # Compute accelerometer-based roll and pitch.
    phi_acc = math.atan2(acc_comp[1], acc_comp[2])
    theta_acc = math.atan2(-acc_comp[0], math.sqrt(acc_comp[1]**2 + acc_comp[2]**2))
    # Simple gyro integration (for reference).
    phi_gyro = prev_gyro_phi + dt * gyro_aligned[0]
    theta_gyro = prev_gyro_theta + dt * gyro_aligned[1]
    psi_gyro = dt * gyro_aligned[2]
    return (acc_comp.tolist(), gyro_aligned.tolist(), mag_aligned.tolist(),
            phi_acc, theta_acc, phi_gyro, theta_gyro, psi_gyro)

# ------------------------
# New Dual-IMU Reading Function
# ------------------------
def read_dual_imu_data_raw(imu1, imu2, dt, prev_gyro_phi, prev_gyro_theta, prev_psi):
    """
    Read data from both IMUs, average their outputs, and return the averaged results.
    """
    data1 = read_imu_data_raw(imu1, dt, prev_gyro_phi, prev_gyro_theta, prev_psi)
    data2 = read_imu_data_raw(imu2, dt, prev_gyro_phi, prev_gyro_theta, prev_psi)
    # Average the accelerometer, gyro, and magnetometer measurements element-wise.
    acc_comp = list((np.array(data1[0]) + np.array(data2[0])) / 2)
    gyro_aligned = list((np.array(data1[1]) + np.array(data2[1])) / 2)
    # mag_aligned = list((np.array(data1[2]) + np.array(data2[2])) / 2)
    mag_aligned = list((np.array(data1[2]) ) / 1)
    # Average Euler angles.
    phi_acc = (data1[3] + data2[3]) / 2
    theta_acc = (data1[4] + data2[4]) / 2
    phi_gyro = (data1[5] + data2[5]) / 2
    theta_gyro = (data1[6] + data2[6]) / 2
    psi_gyro = (data1[7] + data2[7]) / 2
    return acc_comp, gyro_aligned, mag_aligned, phi_acc, theta_acc, phi_gyro, theta_gyro, psi_gyro

# ------------------------
# Main Script
# ------------------------
if __name__ == '__main__':
    from utils.navio2 import lsm9ds1_backup as lsm9ds1
    from utils.navio2 import mpu9250  # IMU sensor 1 (MPU9250)

    imu1 = lsm9ds1.LSM9DS1()
    imu2 = mpu9250.MPU9250()

    imu1.initialize()
    imu2.initialize()

    Ts = 0.005         # Sampling time in seconds.
    calibration_samples = 100

    # Calibrate the dual IMU.
    bx, by, bz, phi_bias, theta_bias, psi_bias = calibrate_dual_imu(imu1, imu2, calibration_samples, Ts)
    print(f"IMU Calibration Done: Phi Bias: {np.rad2deg(phi_bias):.2f}°, Theta Bias: {np.rad2deg(theta_bias):.2f}°, Psi Bias: {np.rad2deg(psi_bias):.2f}°")

    # Set initial quaternions from calibration.
    q0_init = euler_to_quaternion(phi_bias, theta_bias, psi_bias)
    # Initialize filter quaternions.
    q0_madw = q0_init.copy()
    q0_ekf = q0_init.copy()
    q0_comp = q0_init.copy()

    # Initialize filters.
    madgwick = Madgwick(beta=0.3)
    madgwick.q0 = q0_madw
    
    ekf = EKF(gyr=np.zeros((1, 3)), acc=np.zeros((1, 3)), mag=np.zeros((1, 3)),
          noises=[(0.9)**2, (0.001)**2, (0.1)**2])
    ekf.q0 = q0_ekf

    # For the complementary filter, we update our own q.
    q_comp = q0_comp.copy()

    # Arrays to store filter outputs.
    madw_phi_arr, madw_theta_arr, madw_psi_arr = [], [], []
    ekf_phi_arr, ekf_theta_arr, ekf_psi_arr = [], [], []
    comp_phi_arr, comp_theta_arr, comp_psi_arr = [], [], []

    # Initialize integrated gyro angles for raw sensor processing.
    prev_phi_gyro = phi_bias
    prev_theta_gyro = theta_bias
    prev_psi_gyro = psi_bias

    print("Starting IMU Tuning (Three Filters)...")
    num_samples = int(500 / Ts)  # Run for 500*Ts seconds.
    start_time = time.time()

    GPIO.output(START_SIGNAL_PIN, GPIO.HIGH)  # Send start signal to dSPACE

    for i in range(num_samples):
        loop_start = time.time()

        # Read sensor data from both IMUs and average the results.
        (acc_comp, gyro_aligned, mag_aligned,
         phi_acc, theta_acc, phi_gyro, theta_gyro, psi_gyro) = read_dual_imu_data_raw(imu1, imu2, Ts, prev_phi_gyro, prev_theta_gyro, prev_psi_gyro)
        # Update integrated gyro angles.
        prev_phi_gyro = phi_gyro
        prev_theta_gyro = theta_gyro
        prev_psi_gyro += psi_gyro

        # Update Madgwick filter.
        q_madw = madgwick.updateMARG(madgwick.q0, gyr=gyro_aligned, acc=acc_comp, mag=mag_aligned, dt=Ts)
        # q_madw = madgwick.updateIMU(madgwick.q0, gyr=gyro_aligned, acc=acc_comp, dt=Ts)
        madgwick.q0 = q_madw

        # Update EKF.
        q_ekf = ekf.update(ekf.q0, gyr=gyro_aligned, acc=acc_comp, mag=mag_aligned, dt=Ts)
        ekf.q0 = q_ekf

        # Update Complementary filter.
        q_comp = complementary_update(q_comp, gyr=gyro_aligned, acc=acc_comp, mag=mag_aligned, dt=Ts, gain=comp_gain)

        # Convert quaternions to Euler angles.
        phi_madw, theta_madw, psi_madw = quaternion_to_euler(q_madw)
        phi_ekf, theta_ekf, psi_ekf = quaternion_to_euler(q_ekf)
        phi_comp, theta_comp, psi_comp = quaternion_to_euler(q_comp)

        # Store results.
        madw_phi_arr.append(phi_madw)
        madw_theta_arr.append(theta_madw)
        madw_psi_arr.append(psi_madw)
        ekf_phi_arr.append(phi_ekf)
        ekf_theta_arr.append(theta_ekf)
        ekf_psi_arr.append(psi_ekf)
        comp_phi_arr.append(phi_comp)
        comp_theta_arr.append(theta_comp)
        comp_psi_arr.append(psi_comp)

        # Display real-time measurements.
        print(f"Time: {i*Ts:.2f}s | Madw: Phi {np.rad2deg(phi_madw):.2f}°, Theta {np.rad2deg(theta_madw):.2f}°, Psi {np.rad2deg(psi_madw):.2f}° | "
              f"EKF: Phi {np.rad2deg(phi_ekf):.2f}°, Theta {np.rad2deg(theta_ekf):.2f}°, Psi {np.rad2deg(psi_ekf):.2f}° | "
              f"Comp: Phi {np.rad2deg(phi_comp):.2f}°, Theta {np.rad2deg(theta_comp):.2f}°, Psi {np.rad2deg(psi_comp):.2f}°")

        loop_duration = time.time() - loop_start
        if loop_duration < Ts:
            time.sleep(Ts - loop_duration)

    print("IMU Tuning Completed.")
    GPIO.output(START_SIGNAL_PIN, GPIO.LOW)  # End experiment signal

    # Optionally, print standard deviations.
    print(f"Madgwick Std: Phi {np.std(madw_phi_arr):.4f}, Theta {np.std(madw_theta_arr):.4f}, Psi {np.std(madw_psi_arr):.4f} rad")
    print(f"EKF Std: Phi {np.std(ekf_phi_arr):.4f}, Theta {np.std(ekf_theta_arr):.4f}, Psi {np.std(ekf_psi_arr):.4f} rad")
    print(f"Comp. Filter Std: Phi {np.std(comp_phi_arr):.4f}, Theta {np.std(comp_theta_arr):.4f}, Psi {np.std(comp_psi_arr):.4f} rad")

    # Save results to a MATLAB .mat file.
    results = {
        'madw_phi': np.array(madw_phi_arr),
        'madw_theta': np.array(madw_theta_arr),
        'madw_psi': np.array(madw_psi_arr),
        'ekf_phi': np.array(ekf_phi_arr),
        'ekf_theta': np.array(ekf_theta_arr),
        'ekf_psi': np.array(ekf_psi_arr),
        'comp_phi': np.array(comp_phi_arr),
        'comp_theta': np.array(comp_theta_arr),
        'comp_psi': np.array(comp_psi_arr),
        'time': np.linspace(0, num_samples*Ts, num_samples)
    }
 
    # Generate timestamp in the format YY_MM_DD_HH_MM_SS
    timestamp = time.strftime("%y_%m_%d_%H_%M_%S")

    # Define the save path with the formatted filename
    save_path = f"/home/pi/Documents/Quadcopter_Control_v2/data/results/imu_test_{timestamp}.mat"
    
    # savemat(save_path, results)
    print(f"Results saved to {save_path}")
