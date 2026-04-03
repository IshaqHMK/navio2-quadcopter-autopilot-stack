#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Madgwick & Complementary Filters 
Version: IMU-Tuning-vDualFiltersPlusComp-v1 (No EKF)
correct
fiakzied for madqick
complemenrtary is causing problem!
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

# Global variables to hold magnetometer and yaw biases (set during calibration)
mag_bias_cal = None
psi_bias_cal = None

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

def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi].
    """
    while angle > math.pi:
        angle -= 2*math.pi
    while angle < -math.pi:
        angle += 2*math.pi
    return angle

# ------------------------
# Complementary Filter (with magnetometer yaw correction)
# ------------------------
def complementary_update(q_prev, gyr, acc, mag, dt, gain):
    """
    Complementary filter update (Quaternion-based).
    Fuse gyro with accelerometer for roll & pitch and use magnetometer for yaw.
    
    :param q_prev: previous orientation [w, x, y, z]
    :param gyr:    gyro [gx, gy, gz], rad/s
    :param acc:    accelerometer [ax, ay, az], m/s^2
    :param mag:    magnetometer [mx, my, mz] (bias corrected)
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
    phi_acc = math.atan2(ay, az)
    theta_acc = math.atan2(-ax, math.sqrt(ay*ay + az*az))

    # --- 3) Estimate yaw from magnetometer (tilt compensated)
    # Use accelerometer-based roll & pitch for tilt compensation.
    mag_yaw = _compute_tilt_compensated_yaw(mag[0], mag[1], mag[2], phi_acc, theta_acc)
    
    # --- 4) Fuse yaw: subtract the calibrated yaw bias.
    # psi_bias_cal is set during calibration.
    yaw_error = wrap_angle((-mag_yaw + psi_bias_cal) - psi_g)
    psi_final = psi_g + (1 - gain) * yaw_error

    # --- 5) Complementary fuse for roll & pitch:
    phi_final   = gain * phi_g   + (1 - gain) * phi_acc
    theta_final = gain * theta_g + (1 - gain) * theta_acc

    # --- 6) Convert fused Euler angles back to quaternion.
    q_fused = euler_to_quaternion(phi_final, theta_final, psi_final)
    q_out = q_fused / np.linalg.norm(q_fused)
    return q_out

# ------------------------
# Basic Magnetometer Yaw (Tilt Compensation)
# ------------------------
def _compute_tilt_compensated_yaw(mx, my, mz, roll_rad, pitch_rad):
    """
    Compute tilt compensated yaw from magnetometer data.
    """
    Xh = mx * math.cos(pitch_rad) + mz * math.sin(pitch_rad)
    Yh = mx * math.sin(roll_rad) * math.sin(pitch_rad) + my * math.cos(roll_rad) - mz * math.sin(roll_rad) * math.cos(pitch_rad)
    yaw_rad = math.atan2(Yh, Xh)
    return yaw_rad

# ------------------------
# IMU Calibration (including magnetometer bias)
# ------------------------
def calibrate_imu(imu, calibration_samples, dt):
    """
    Calibrate the IMU by determining gyro biases, computing initial Euler angles from acc+mag,
    and estimating the magnetometer bias.
    """
    bx, by, bz = 0.0, 0.0, 0.0
    phi_sum, theta_sum, psi_sum = 0.0, 0.0, 0.0
    mag_sum = np.array([0.0, 0.0, 0.0])

    start_time = time.time()
    for _ in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        acc_raw = np.array(m9a)
        gyro_raw = np.array(m9g)
        mag_raw = np.array(m9m)
        mag_raw[1] = -mag_raw[1]
        mag_raw[2] = -mag_raw[2]

        mag_sum += mag_raw

        ax, ay, az = acc_raw
        gx, gy, gz = gyro_raw
        mx, my, mz = mag_raw

        bx += gx
        by += gy
        bz += gz

        phi = math.atan2(ay, math.sqrt(ax*ax + az*az))
        theta = math.atan2(-ax, math.sqrt(ay*ay + az*az))
        psi = _compute_tilt_compensated_yaw(mx, my, mz, phi, theta)

        phi_sum   += phi
        theta_sum += theta
        psi_sum   += psi

        while (time.time() - start_time) < dt:
            pass
        start_time += dt

    bx /= calibration_samples
    by /= calibration_samples
    bz /= calibration_samples

    phi_bias   = phi_sum   / calibration_samples
    theta_bias = theta_sum / calibration_samples
    psi_bias   = psi_sum   / calibration_samples
    mag_bias = mag_sum / calibration_samples

    return bx, by, bz, phi_bias, theta_bias, psi_bias, mag_bias

def calibrate_dual_imu(imu1, imu2, calibration_samples, dt):
    """
    Calibrate two IMUs and average the results.
    """
    bx1, by1, bz1, phi_bias1, theta_bias1, psi_bias1, mag_bias1 = calibrate_imu(imu1, calibration_samples, dt)
    bx2, by2, bz2, phi_bias2, theta_bias2, psi_bias2, mag_bias2 = calibrate_imu(imu2, calibration_samples, dt)

    bx = (bx1 + bx2) / 2
    by = (by1 + by2) / 2
    bz = (bz1 + bz2) / 2

    phi_bias   = (phi_bias1   + phi_bias2)   / 2
    theta_bias = (theta_bias1 + theta_bias2) / 2
    psi_bias   = (psi_bias1   + psi_bias2)   / 2
    mag_bias = (mag_bias1 + mag_bias2) / 2

    return bx, by, bz, phi_bias, theta_bias, psi_bias, mag_bias

# ------------------------
# Read IMU Data (with magnetometer bias correction)
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
    mag_raw[1] = -mag_raw[1]
    mag_raw[2] = -mag_raw[2]
    global mag_bias_cal
    if mag_bias_cal is not None:
        mag_raw = mag_raw - mag_bias_cal

    ax, ay, az = acc_raw
    gx, gy, gz = gyro_raw

    acc_comp = acc_raw
    gyro_aligned = gyro_raw
    mag_aligned  = mag_raw

    phi_acc   = math.atan2(ay, az)
    theta_acc = math.atan2(-ax, math.sqrt(ay*ay + az*az))

    phi_gyro   = prev_gyro_phi   + dt*gx
    theta_gyro = prev_gyro_theta + dt*gy
    psi_gyro   = dt*gz

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
    from utils.navio2 import lsm9ds1_backup as lsm9ds1
    from utils.navio2 import mpu9250

    imu1 = lsm9ds1.LSM9DS1()
    imu2 = mpu9250.MPU9250()

    imu1.initialize()
    imu2.initialize()

    Ts = 0.005  # 200 Hz
    calibration_samples = 100

    bx, by, bz, phi_bias, theta_bias, psi_bias, mag_bias = calibrate_dual_imu(
        imu1, imu2, calibration_samples, Ts
    )
    mag_bias_cal = mag_bias
    psi_bias_cal = psi_bias

    print(f"IMU Calibration Done: "
          f"Phi Bias: {np.rad2deg(phi_bias):.2f}°, "
          f"Theta Bias: {np.rad2deg(theta_bias):.2f}°, "
          f"Psi Bias: {np.rad2deg(psi_bias):.2f}°")
    print(f"Mag Bias: {mag_bias}")

    q0_init = euler_to_quaternion(phi_bias, theta_bias, psi_bias)
    q_madw = q0_init.copy()
    q_comp = q0_init.copy()

    w_gyro_error = 0.05
    w_gyro_bias_drift = 0.001
    Beta = 0.03
 
    madgwick = Madgwick(
        beta=Beta,
        magnetic_ref=None
    )
    madgwick.q0 = q_madw

    madw_phi_arr, madw_theta_arr, madw_psi_arr = [], [], []
    comp_phi_arr, comp_theta_arr, comp_psi_arr = [], [], []

    prev_phi_gyro   = phi_bias
    prev_theta_gyro = theta_bias
    prev_psi_gyro   = psi_bias

    gyro_bias_est = np.array([bx, by, bz])

    print("Starting IMU Tuning (Madgwick + Complementary with Gyro Drift & Magnetometer Yaw Correction)...")
    num_samples = int(120 / Ts)
    start_time = time.time()

    GPIO.output(START_SIGNAL_PIN, GPIO.HIGH)

    for i in range(num_samples):
        loop_start = time.time()

        (acc_comp, gyro_aligned, mag_aligned,
         phi_acc, theta_acc,
         raw_phi_gyro, raw_theta_gyro, raw_psi_gyro) = read_dual_imu_data_raw(
            imu1, imu2, Ts, prev_phi_gyro, prev_theta_gyro, prev_psi_gyro
        )

        gyr_corrected = np.array(gyro_aligned) - gyro_bias_est
        phi_gyro_corr   = prev_phi_gyro   + Ts * gyr_corrected[0]
        theta_gyro_corr = prev_theta_gyro + Ts * gyr_corrected[1]
        psi_gyro_corr   = prev_psi_gyro   + Ts * gyr_corrected[2]

        error_phi   = w_gyro_error * (phi_acc - phi_gyro_corr)
        error_theta = w_gyro_error * (theta_acc - theta_gyro_corr)
        gyro_bias_est[0] += w_gyro_bias_drift * error_phi   * Ts
        gyro_bias_est[1] += w_gyro_bias_drift * error_theta * Ts

        prev_phi_gyro   = phi_gyro_corr
        prev_theta_gyro = theta_gyro_corr
        prev_psi_gyro   = psi_gyro_corr

        q_madw_new = madgwick.updateIMU(madgwick.q0, gyr_corrected.tolist(), acc_comp, Ts)
        madgwick.q0 = q_madw_new
        q_madw = q_madw_new

        q_comp = complementary_update(
            q_comp, gyr=gyr_corrected.tolist(), acc=acc_comp, mag=mag_aligned,
            dt=Ts, gain=comp_gain
        )

        phi_madw, theta_madw, psi_madw = quaternion_to_euler(q_madw)
        phi_comp, theta_comp, psi_comp = quaternion_to_euler(q_comp)

        madw_phi_arr.append(phi_madw)
        madw_theta_arr.append(theta_madw)
        madw_psi_arr.append(psi_madw)

        comp_phi_arr.append(phi_comp)
        comp_theta_arr.append(theta_comp)
        comp_psi_arr.append(psi_comp)

        print(f"Time: {i*Ts:.2f}s | "
              f"Madg: R {np.rad2deg(phi_madw):.2f}°, P {np.rad2deg(theta_madw):.2f}°, Y {np.rad2deg(psi_madw-psi_bias_cal):.2f}° | "
              f"Comp: R {np.rad2deg(phi_comp):.2f}°, P {np.rad2deg(theta_comp):.2f}°, Y {np.rad2deg(psi_comp):.2f}°")

        loop_elapsed = time.time() - loop_start
        if loop_elapsed < Ts:
            time.sleep(Ts - loop_elapsed)

    print("IMU Tuning Completed.")
    GPIO.output(START_SIGNAL_PIN, GPIO.LOW)

    print(f"Madgwick Std [rad]:  phi={np.std(madw_phi_arr):.4f}, "
          f"theta={np.std(madw_theta_arr):.4f}, psi={np.std(madw_psi_arr):.4f}")
    print(f"Comp. Std [rad]:     phi={np.std(comp_phi_arr):.4f}, "
          f"theta={np.std(comp_theta_arr):.4f}, psi={np.std(comp_psi_arr):.4f}")

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
