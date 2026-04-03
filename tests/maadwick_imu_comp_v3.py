#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Madgwick Filter with Sensor Offset + Angle Offset Compensation
Version: IMU-Tuning-v1_Modified
"""

# ========================
# Standard Python Libraries
# ========================
import time
import numpy as np
import math
import sys
import os

# Add your project directories if necessary
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2/imu')

# ========================
# USER-CONFIGURABLE OFFSETS
# ========================

# (A) Translational offset of the IMU from the center of gravity [meters].
#     Adjust these values if your sensor is physically shifted in X/Y/Z.
sensor_offset = np.array([0, 0, 0])  # e.g., 10 cm in X and 10 cm in Y

# (B) Final angle offsets [degrees].
#     Adjust these if your IMU angles differ from a known reference (e.g., an encoder).
#     Example: If encoder says -38 but IMU says -48 => roll_offset_deg = +10.0
roll_offset_deg =  0   #math.radians(+1.73)  # Offset for roll  (phi)
pitch_offset_deg = 0  # +1.73 #math.radians(+1.73)   # Offset for pitch (theta)
yaw_offset_deg =  0    # Offset for yaw   (psi)

# ========================
# Global Variables
# ========================
# Global variable to hold the previous gyro measurement for computing angular acceleration.
prev_gyro = None

# ========================
# Third-Party / Custom Modules
# ========================
from imu.madgwick import Madgwick  # Madgwick filter for orientation estimation

# ------------------------
# Calibration Function
# ------------------------
def calibrate_imu(imu, calibration_samples, dt):
    """
    Calibrate the IMU by determining gyro biases and computing
    initial Euler angles from the accelerometer and magnetometer.
    """
    bx, by, bz = 0.0, 0.0, 0.0
    phi_sum, theta_sum, psi_sum = 0.0, 0.0, 0.0

    start_time = time.time()
    for i in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        ax, ay, az = m9a
        gx, gy, gz = m9g
        mx, my, mz = m9m

        # Apply sign corrections (if needed) for your setup
        my = -my
        mz = -mz

        # Sum up gyro values for bias
        bx += gx
        by += gy
        bz += gz

        # Compute roll and pitch from accelerometer
        phi = math.atan2(ay, math.sqrt(ax**2 + az**2))
        theta = math.atan2(-ax, math.sqrt(ay**2 + az**2))

        # Compute tilt-compensated yaw from magnetometer
        psi = _compute_tilt_compensated_yaw(mx, my, mz, phi, theta)

        phi_sum += phi
        theta_sum += theta
        psi_sum += psi

        # Timing to maintain calibration sample rate
        next_sample_time = start_time + (i + 1) * dt
        while time.time() < next_sample_time:
            pass  # Busy wait for precise timing

    bx /= calibration_samples
    by /= calibration_samples
    bz /= calibration_samples
    phi_bias = phi_sum / calibration_samples
    theta_bias = theta_sum / calibration_samples
    psi_bias = psi_sum / calibration_samples

    return bx, by, bz, phi_bias, theta_bias, psi_bias, m9a, m9g, m9m

def _compute_tilt_compensated_yaw(mx, my, mz, roll_rad, pitch_rad):
    """
    Compute yaw from magnetometer with tilt compensation.
    """
    cos_phi = math.cos(roll_rad)
    sin_phi = math.sin(roll_rad)
    cos_theta = math.cos(pitch_rad)
    sin_theta = math.sin(pitch_rad)

    # Tilt-compensated magnetometer components
    Bx = mx * cos_theta + my * sin_phi * sin_theta + mz * cos_phi * sin_theta
    By = my * cos_phi - mz * sin_phi
    yaw_rad = math.atan2(-By, Bx)
    return yaw_rad

# ------------------------
# Data Reading Function (with Compensation)
# ------------------------
def read_imu_data(imu, dt, prev_gyro_phi, prev_gyro_theta, prev_psi, q0, madgwick):
    """
    Read IMU data, apply sensor offset compensation on the accelerometer,
    and update the Madgwick filter.
    
    Returns raw accelerometer values, magnetometer, gyro, the computed Euler angles,
    intermediate accelerometer-based angles, integrated gyro angles, and the new quaternion.
    """
    global prev_gyro, sensor_offset
    m9a, m9g, m9m = imu.getMotion9()
    ax, ay, az = m9a
    gx, gy, gz = m9g
    mx, my, mz = m9m

    # Compute angular acceleration (finite difference of gyro)
    current_gyro = np.array([gx, gy, gz])
    if prev_gyro is None:
        domega = np.zeros(3)
    else:
        domega = (current_gyro - prev_gyro) / dt
    prev_gyro = current_gyro

    # Apply compensation to accelerometer data:
    #   acc_CG = acc_sensor - [ (dω x r) + (ω x (ω x r)) ]
    term1 = np.cross(domega, sensor_offset)
    term2 = np.cross(current_gyro, np.cross(current_gyro, sensor_offset))
    acc_cg = np.array([ax, ay, az]) - (term1 + term2)

    # Compute accelerometer-based roll & pitch (using corrected acceleration)
    phi_acc = math.atan2(acc_cg[1], acc_cg[2])
    theta_acc = math.atan2(-acc_cg[0], math.sqrt(acc_cg[1]**2 + acc_cg[2]**2))

    # Gyro integration for roll, pitch (simple integration)
    phi_gyro = prev_gyro_phi + dt * gx
    theta_gyro = prev_gyro_theta + dt * gy
    psi_gyro = dt * gz
    psi_int = prev_psi + psi_gyro

    # Correct magnetometer reading if needed
    m9m_corrected = [mx, -my, -mz]

    # Update Madgwick filter with compensated accelerometer and corrected magnetometer
    q = madgwick.updateMARG(q0, gyr=m9g, acc=acc_cg, mag=m9m_corrected, dt=dt)

    # Convert quaternion to Euler angles
    phi, theta, psi = quaternion_to_euler(q)

    return (ax, ay, az, mx, my, mz, gx, gy, gz,
            phi, theta, psi, phi_acc, theta_acc, phi_gyro, theta_gyro, psi_gyro, q)

# ------------------------
# Quaternion/Euler Conversion
# ------------------------
def quaternion_to_euler(q):
    """
    Convert a quaternion (w, x, y, z) into Euler angles (roll, pitch, yaw).
    """
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def euler_to_quaternion(phi, theta, psi):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion (w, x, y, z).
    """
    qw = (math.cos(phi/2)*math.cos(theta/2)*math.cos(psi/2) +
          math.sin(phi/2)*math.sin(theta/2)*math.sin(psi/2))
    qx = (math.sin(phi/2)*math.cos(theta/2)*math.cos(psi/2) -
          math.cos(phi/2)*math.sin(theta/2)*math.sin(psi/2))
    qy = (math.cos(phi/2)*math.sin(theta/2)*math.cos(psi/2) +
          math.sin(phi/2)*math.cos(theta/2)*math.sin(psi/2))
    qz = (math.cos(phi/2)*math.cos(theta/2)*math.sin(psi/2) -
          math.sin(phi/2)*math.sin(theta/2)*math.cos(psi/2))
    return np.array([qw, qx, qy, qz])

# ========================
# Main Script
# ========================
if __name__ == '__main__':
    # Initialize IMU Sensor (using Navio2 driver)
    from utils.navio2 import lsm9ds1_backup as lsm9ds1
    imu = lsm9ds1.LSM9DS1()
    imu.initialize()

    # Sampling time and calibration parameters
    Ts = 0.005          # 5 ms sampling period
    calibration_samples = 200

    # Calibrate the IMU and obtain biases and initial angles
    bx, by, bz, phi_bias, theta_bias, psi_bias, m9a, m9g, m9m = calibrate_imu(imu, calibration_samples, Ts)
    print(f"IMU Calibration Done.")
    print(f"Phi Bias: {np.rad2deg(phi_bias):.2f}°, Theta Bias: {np.rad2deg(theta_bias):.2f}°, Psi Bias: {np.rad2deg(psi_bias):.2f}°")

    # Set the initial quaternion based on the calibrated angles
    q0 = euler_to_quaternion(phi_bias, theta_bias, psi_bias)

    # Initialize the Madgwick filter
    madgwick = Madgwick(beta=0.3)
    madgwick.q0 = q0

    # Initialize arrays to store Euler angle measurements (if you want analysis later)
    num_samples = int(600 / Ts)  # e.g., run for 60 seconds
    phi_meas_arr, theta_meas_arr, psi_meas_arr = [], [], []

    # Initialize previous integrated gyro angles
    prev_gyro_phi = phi_bias
    prev_gyro_theta = theta_bias
    prev_psi = psi_bias

    print("Starting IMU Tuning...")
    start_time = time.time()

    for i in range(num_samples):
        loop_start = time.time()

        # Read IMU data with sensor offset compensation
        (ax, ay, az, mx, my, mz, gx, gy, gz,
         phi, theta, psi,
         phi_acc, theta_acc,
         phi_gyro, theta_gyro, psi_gyro,
         q_out) = read_imu_data(
             imu, Ts, prev_gyro_phi, prev_gyro_theta, prev_psi, madgwick.q0, madgwick
        )

        # Update the integrated gyro angles for next iteration
        prev_gyro_phi = phi_gyro
        prev_gyro_theta = theta_gyro
        prev_psi += psi_gyro

        # Update the filter's quaternion
        madgwick.q0 = q_out

        # -----------------------------
        #  Apply final angle offsets
        # -----------------------------
        phi_corrected  = phi  + math.radians(roll_offset_deg)
        theta_corrected = theta + math.radians(pitch_offset_deg)
        psi_corrected   = psi   + math.radians(yaw_offset_deg)

        # Store the corrected Euler angles for analysis
        phi_meas_arr.append(phi_corrected)
        theta_meas_arr.append(theta_corrected)
        psi_meas_arr.append(psi_corrected)

        # Print the results in degrees
        print(f"Time: {i*Ts:.2f}s | "
              f"Phi: {np.rad2deg(phi_corrected):.2f}°, "
              f"Theta: {np.rad2deg(theta_corrected):.2f}°, "
              f"Psi: {np.rad2deg(psi_corrected):.2f}°")

        # Maintain loop timing
        loop_duration = time.time() - loop_start
        if loop_duration < Ts:
            time.sleep(Ts - loop_duration)

    print("IMU Tuning Completed.")

    # Analyze the collected data
    phi_std = np.std(phi_meas_arr)
    theta_std = np.std(theta_meas_arr)
    psi_std = np.std(psi_meas_arr)
    print(f"Standard Deviations (radians):")
    print(f"   Phi:   {phi_std:.4f}  |  Theta: {theta_std:.4f}  |  Psi: {psi_std:.4f}")
    print("Adjust the 'beta' in Madgwick or your angle offsets if needed.")
