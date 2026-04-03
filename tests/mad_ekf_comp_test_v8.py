#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Madgwick & Complementary Filters 
Version: IMU-Tuning-vDualFiltersPlusComp-v1 (No EKF)
MAIN VERSION
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

import RPi.GPIO as GPIO  # For signaling
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
START_SIGNAL_PIN = 26
GPIO.setup(START_SIGNAL_PIN, GPIO.OUT)
GPIO.output(START_SIGNAL_PIN, GPIO.LOW)

# Add your project directories if necessary
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2/imu')

# ========================
# Global Variables
# ========================
# Complementary filter gain (0 < gain < 1). Close to 1 means heavier gyro weighting.
comp_gain = 0.98

# ========================
# Third-Party / Custom Modules
# ========================
from imu.madgwick import Madgwick  # Madgwick filter for orientation estimation
 
time.sleep(5)  # Simulate time before starting

# ------------------------
# Quaternion / Euler Helpers
# ------------------------
def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    q1, q2 are [w, x, y, z].
    Returns q = q1 * q2 (as a numpy array).
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
    Convert Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z].
    """
    qw = (math.cos(phi/2)*math.cos(theta/2)*math.cos(psi/2)
          + math.sin(phi/2)*math.sin(theta/2)*math.sin(psi/2))
    qx = (math.sin(phi/2)*math.cos(theta/2)*math.cos(psi/2)
          - math.cos(phi/2)*math.sin(theta/2)*math.sin(psi/2))
    qy = (math.cos(phi/2)*math.sin(theta/2)*math.cos(psi/2)
          + math.sin(phi/2)*math.cos(theta/2)*math.sin(psi/2))
    qz = (math.cos(phi/2)*math.cos(theta/2)*math.sin(psi/2)
          - math.sin(phi/2)*math.sin(theta/2)*math.cos(psi/2))
    return np.array([qw, qx, qy, qz])

def quaternion_to_euler(q):
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw).
    """
    w, x, y, z = q
    # Roll (phi)
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    phi = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (theta)
    sinp = 2.0 * (w*y - z*x)
    sinp = max(-1.0, min(1.0, sinp))  # clamp
    theta = math.asin(sinp)
    # Yaw (psi)
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    psi = math.atan2(siny_cosp, cosy_cosp)
    return phi, theta, psi

# ------------------------
# Complementary Filter
# ------------------------
def complementary_update(q_prev, gyr, acc, mag, dt, gain):
    """
    Complementary filter update (Quaternion-based).
    For minimal confusion: 
      - We only use accelerometer to correct roll & pitch.
      - We keep yaw from gyro integration alone (mag ignored).

    :param q_prev: previous orientation [w, x, y, z]
    :param gyr:    gyro [gx, gy, gz], rad/s
    :param acc:    accelerometer [ax, ay, az], m/s^2
    :param mag:    magnetometer (ignored here for yaw)
    :param dt:     time step (s)
    :param gain:   complementary gain (0 < gain < 1)
    :return:       updated orientation quaternion
    """
    # --- 1) Integrate gyro to get predicted orientation (q_gyro).
    delta_angle = np.array(gyr) * dt
    angle = np.linalg.norm(delta_angle)
    if angle > 1e-6:
        axis = delta_angle / angle
        delta_q = np.hstack([math.cos(angle/2), math.sin(angle/2)*axis])
    else:
        delta_q = np.array([1.0, 0.0, 0.0, 0.0])
    q_gyro = quaternion_multiply(q_prev, delta_q)

    # Convert q_gyro to Euler angles
    phi_g, theta_g, psi_g = quaternion_to_euler(q_gyro)

    # --- 2) Estimate roll & pitch from accelerometer.
    ax, ay, az = acc
    # Roll from acc
    phi_acc = math.atan2(ay, az)
    # Pitch from acc
    theta_acc = math.atan2(-ax, math.sqrt(ay*ay + az*az))

    # For yaw, we trust the gyro integration. (Ignore magnetometer.)
    # So final yaw = psi_g

    # --- 3) Complementary fuse for roll & pitch:
    phi_final   = gain * phi_g   + (1 - gain) * phi_acc
    theta_final = gain * theta_g + (1 - gain) * theta_acc
    psi_final   = psi_g  # purely gyro-based

    # --- 4) Convert fused Euler angles back to quaternion.
    q_fused = euler_to_quaternion(phi_final, theta_final, psi_final)
    # Normalize to avoid drift
    q_out = q_fused / np.linalg.norm(q_fused)
    return q_out

# ------------------------
# Basic Magnetometer Yaw (if needed)
# ------------------------
def _compute_tilt_compensated_yaw(mx, my, mz, roll_rad, pitch_rad):
    """
    Example function to get yaw from mag. 
    Not used in the complementary filter now (ignored for yaw).
    """
    # Simple heading ignoring tilt:
    yaw_rad = math.atan2(my, mx)
    return yaw_rad

# ------------------------
# IMU Calibration
# ------------------------
def calibrate_imu(imu, calibration_samples, dt):
    """
    Calibrate the IMU by determining gyro biases and
    computing initial Euler angles from acc + mag (rough).
    """
    bx, by, bz = 0.0, 0.0, 0.0
    phi_sum, theta_sum, psi_sum = 0.0, 0.0, 0.0

    start_time = time.time()
    for _ in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        acc_raw = np.array(m9a)
        gyro_raw = np.array(m9g)
        mag_raw = np.array(m9m)
        # Apply sign corrections for magnetometer if needed
        mag_raw[1] = -mag_raw[1]
        mag_raw[2] = -mag_raw[2]

        ax, ay, az = acc_raw
        gx, gy, gz = gyro_raw
        mx, my, mz = mag_raw

        # Accumulate gyro offsets
        bx += gx
        by += gy
        bz += gz

        # Estimate roll & pitch from accelerometer
        phi = math.atan2(ay, math.sqrt(ax*ax + az*az))
        theta = math.atan2(-ax, math.sqrt(ay*ay + az*az))

        # Rough yaw from magnetometer
        psi = _compute_tilt_compensated_yaw(mx, my, mz, phi, theta)

        phi_sum   += phi
        theta_sum += theta
        psi_sum   += psi

        # Wait the remainder of dt
        while (time.time() - start_time) < dt:
            pass
        start_time += dt

    # Average out the biases
    bx /= calibration_samples
    by /= calibration_samples
    bz /= calibration_samples

    phi_bias   = phi_sum   / calibration_samples
    theta_bias = theta_sum / calibration_samples
    psi_bias   = psi_sum   / calibration_samples

    return bx, by, bz, phi_bias, theta_bias, psi_bias

def calibrate_dual_imu(imu1, imu2, calibration_samples, dt):
    """
    Calibrate two IMUs and average the results.
    """
    bx1, by1, bz1, phi_bias1, theta_bias1, psi_bias1 = calibrate_imu(imu1, calibration_samples, dt)
    bx2, by2, bz2, phi_bias2, theta_bias2, psi_bias2 = calibrate_imu(imu2, calibration_samples, dt)

    bx = (bx1 + bx2) / 2
    by = (by1 + by2) / 2
    bz = (bz1 + bz2) / 2

    phi_bias   = (phi_bias1   + phi_bias2)   / 2
    theta_bias = (theta_bias1 + theta_bias2) / 2
    psi_bias   = (psi_bias1   + psi_bias2)   / 2

    return bx, by, bz, phi_bias, theta_bias, psi_bias

# ------------------------
# Read IMU Data
# ------------------------
def read_imu_data_raw(imu, dt, prev_gyro_phi, prev_gyro_theta, prev_psi):
    """
    Returns:
      acc_comp, gyro_aligned, mag_aligned,
      phi_acc, theta_acc,
      phi_gyro, theta_gyro, psi_gyro
    """
    m9a, m9g, m9m = imu.getMotion9()
    acc_raw = np.array(m9a)
    gyro_raw = np.array(m9g)
    mag_raw = np.array(m9m)
    # Sign corrections for magnetometer
    mag_raw[1] = -mag_raw[1]
    mag_raw[2] = -mag_raw[2]

    ax, ay, az = acc_raw
    gx, gy, gz = gyro_raw

    # No sensor offset comp in this code snippet
    acc_comp = acc_raw
    gyro_aligned = gyro_raw
    mag_aligned  = mag_raw

    # Compute Euler from accelerometer alone
    phi_acc   = math.atan2(ay, az)
    theta_acc = math.atan2(-ax, math.sqrt(ay*ay + az*az))

    # Integrate gyro for reference
    phi_gyro   = prev_gyro_phi   + dt*gx
    theta_gyro = prev_gyro_theta + dt*gy
    psi_gyro   = dt*gz  # increment

    return (
        acc_comp.tolist(),
        gyro_aligned.tolist(),
        mag_aligned.tolist(),
        phi_acc, theta_acc,
        phi_gyro, theta_gyro, psi_gyro
    )

def read_dual_imu_data_raw(imu1, imu2, dt, prev_gyro_phi, prev_gyro_theta, prev_psi):
    """
    Reads both IMUs, averages the accelerometer & gyro. 
    Magnetometer is taken from the first IMU by default.
    """
    data1 = read_imu_data_raw(imu1, dt, prev_gyro_phi, prev_gyro_theta, prev_psi)
    data2 = read_imu_data_raw(imu2, dt, prev_gyro_phi, prev_gyro_theta, prev_psi)

    acc_comp = list((np.array(data1[0]) + np.array(data2[0])) / 2.0)
    gyro_aligned = list((np.array(data1[1]) + np.array(data2[1])) / 2.0)
    mag_aligned  = data1[2]  # from first IMU

    phi_acc   = (data1[3] + data2[3]) / 2.0
    theta_acc = (data1[4] + data2[4]) / 2.0
    phi_gyro  = (data1[5] + data2[5]) / 2.0
    theta_gyro= (data1[6] + data2[6]) / 2.0
    psi_gyro  = (data1[7] + data2[7]) / 2.0

    return (
        acc_comp, gyro_aligned, mag_aligned,
        phi_acc, theta_acc, phi_gyro, theta_gyro, psi_gyro
    )

# ------------------------
# Main Script
# ------------------------
if __name__ == '__main__':
    # Example IMU imports for Navio2
    from utils.navio2 import lsm9ds1_backup as lsm9ds1
    from utils.navio2 import mpu9250

    imu1 = lsm9ds1.LSM9DS1()
    imu2 = mpu9250.MPU9250()

    imu1.initialize()
    imu2.initialize()

    Ts = 0.005  # 200 Hz
    calibration_samples = 100

    # -- Calibrate both IMUs
    bx, by, bz, phi_bias, theta_bias, psi_bias = calibrate_dual_imu(
        imu1, imu2, calibration_samples, Ts
    )
    print(f"IMU Calibration Done: "
          f"Phi Bias: {np.rad2deg(phi_bias):.2f}°, "
          f"Theta Bias: {np.rad2deg(theta_bias):.2f}°, "
          f"Psi Bias: {np.rad2deg(psi_bias):.2f}°")

    # -- Initial orientation from calibration
    q0_init = euler_to_quaternion(phi_bias, theta_bias, psi_bias)
    q_madw = q0_init.copy()
    q_comp = q0_init.copy()

    # -- Setup Madgwick Gains
    #   (Tune these for your sensor.  Example placeholders here.)
    w_gyro_error = 0.05
    w_gyro_bias_drift = 0.001
    # For demonstration, pick some simple values, or use sqrt(3/4)*...
    Beta = 0.03
 

    madgwick = Madgwick(
        beta=Beta,
        magnetic_ref=None  # or your local reference if using MARG update
    )
    madgwick.q0 = q_madw

    # Arrays to store filter outputs
    madw_phi_arr, madw_theta_arr, madw_psi_arr = [], [], []
    comp_phi_arr, comp_theta_arr, comp_psi_arr = [], [], []

    # Track integrated gyro angles for reference
    prev_phi_gyro   = phi_bias
    prev_theta_gyro = theta_bias
    prev_psi_gyro   = psi_bias

    print("Starting IMU Tuning (Madgwick + Complementary)...")
    num_samples = int(120 / Ts)  # run ~120 seconds
    start_time = time.time()

    GPIO.output(START_SIGNAL_PIN, GPIO.HIGH)  # Signal start

    for i in range(num_samples):
        loop_start = time.time()

        (acc_comp, gyro_aligned, mag_aligned,
         phi_acc, theta_acc,
         phi_gyro, theta_gyro, psi_gyro) = read_dual_imu_data_raw(
            imu1, imu2, Ts, prev_phi_gyro, prev_theta_gyro, prev_psi_gyro
        )

        # Update integrated gyro angles
        prev_phi_gyro   = phi_gyro
        prev_theta_gyro = theta_gyro
        prev_psi_gyro  += psi_gyro  # accumulate

        # --- Update Madgwick filter (IMU-only or MARG)
        # q_madw_new = madgwick.updateMARG(madgwick.q0, gyro_aligned, acc_comp, mag_aligned, Ts)
        q_madw_new = madgwick.updateIMU(madgwick.q0, gyro_aligned, acc_comp, Ts)
        madgwick.q0 = q_madw_new
        q_madw = q_madw_new

        # --- Update Complementary filter
        q_comp = complementary_update(
            q_comp, gyr=gyro_aligned, acc=acc_comp, mag=mag_aligned,
            dt=Ts, gain=comp_gain
        )

        # Convert both to Euler
        phi_madw, theta_madw, psi_madw = quaternion_to_euler(q_madw)
        phi_comp, theta_comp, psi_comp = quaternion_to_euler(q_comp)

        # Store
        madw_phi_arr.append(phi_madw)
        madw_theta_arr.append(theta_madw)
        madw_psi_arr.append(psi_madw)

        comp_phi_arr.append(phi_comp)
        comp_theta_arr.append(theta_comp)
        comp_psi_arr.append(psi_comp)

        # Display real-time
        print(f"Time: {i*Ts:.2f}s | "
              f"Madg: R {np.rad2deg(phi_madw):.2f}°, P {np.rad2deg(theta_madw):.2f}°, Y {np.rad2deg(psi_madw):.2f}° | "
              f"Comp: R {np.rad2deg(phi_comp):.2f}°, P {np.rad2deg(theta_comp):.2f}°, Y {np.rad2deg(psi_comp):.2f}°")

        # Enforce sample time
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < Ts:
            time.sleep(Ts - loop_elapsed)

    print("IMU Tuning Completed.")
    GPIO.output(START_SIGNAL_PIN, GPIO.LOW)  # Signal end

    # Print standard deviations
    print(f"Madgwick Std [rad]:  phi={np.std(madw_phi_arr):.4f}, "
          f"theta={np.std(madw_theta_arr):.4f}, psi={np.std(madw_psi_arr):.4f}")
    print(f"Comp. Std [rad]:     phi={np.std(comp_phi_arr):.4f}, "
          f"theta={np.std(comp_theta_arr):.4f}, psi={np.std(comp_psi_arr):.4f}")

    # Save results
    results = {
        'madw_phi':   np.array(madw_phi_arr),
        'madw_theta': np.array(madw_theta_arr),
        'madw_psi':   np.array(madw_psi_arr),
        'comp_phi':   np.array(comp_phi_arr),
        'comp_theta': np.array(comp_theta_arr),
        'comp_psi':   np.array(comp_psi_arr),
        'time':       np.linspace(0, num_samples*Ts, num_samples)
    }

    timestamp = time.strftime("%y_%m_%d_%H_%M_%S")
    save_path = f"/home/pi/Documents/Quadcopter_Control_v2/data/results/imu_pitch_{timestamp}.mat"
    # savemat(save_path, results)
    print(f"Results saved to {save_path}")
