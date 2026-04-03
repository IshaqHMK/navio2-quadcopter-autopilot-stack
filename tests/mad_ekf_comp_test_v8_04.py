#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Complementary Filter Only, with automatic yaw drift threshold tuning.
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
comp_gain = 0.98
# We'll still define default values, but we'll override 'stationaryThreshold'
# after the 5-second auto-tuning loop:
windowSize = 50      # number of samples in ring buffer (~2s at 100 Hz)
stationaryThreshold = 0.001  # rad/s standard deviation threshold (will be overwritten)

# ========================
# Third-Party / Custom Modules
# ========================

time.sleep(2)  # Small pause before everything starts

# ------------------------
# Quaternion / Euler Helpers
# ------------------------
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def euler_to_quaternion(phi, theta, psi):
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
    w, x, y, z = q
    # Roll
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    phi = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = 2.0 * (w*y - z*x)
    sinp = max(-1.0, min(1.0, sinp))
    theta = math.asin(sinp)
    # Yaw
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    psi = math.atan2(siny_cosp, cosy_cosp)
    return phi, theta, psi

# ------------------------
# Complementary Filter
# ------------------------
def complementary_update(q_prev, gyr, acc, dt, gain):
    """
    Complementary filter update (Quaternion-based).
    - We only use accelerometer to correct roll & pitch.
    - We keep yaw from gyro integration alone.
    """
    delta_angle = np.array(gyr) * dt
    angle = np.linalg.norm(delta_angle)
    if angle > 1e-6:
        axis = delta_angle / angle
        delta_q = np.hstack([math.cos(angle/2), math.sin(angle/2)*axis])
    else:
        delta_q = np.array([1.0, 0.0, 0.0, 0.0])
    q_gyro = quaternion_multiply(q_prev, delta_q)

    phi_g, theta_g, psi_g = quaternion_to_euler(q_gyro)

    ax, ay, az = acc
    phi_acc = math.atan2(ay, az)
    theta_acc = math.atan2(-ax, math.sqrt(ay*ay + az*az))

    phi_final   = gain * phi_g   + (1 - gain) * phi_acc
    theta_final = gain * theta_g + (1 - gain) * theta_acc

    q_fused = euler_to_quaternion(phi_final, theta_final, psi_g)
    q_out = q_fused / np.linalg.norm(q_fused)
    return q_out

# ------------------------
# Yaw Drift Corrector
# ------------------------
class YawDriftCorrectorRingBuffer:
    """
    Implements the paper's adaptive sliding-window yaw drift removal
    using a ring buffer, with auto-updated offset when 'static.'
    """
    def __init__(self, window_size=200, stationary_threshold=0.001, init_offset=0.0):
        self.window_size = window_size
        self.stationary_threshold = stationary_threshold
        self.Db = init_offset
        self.buffer = np.zeros(window_size, dtype=float)
        self.rbIndex = 0
        self.samples_count = 0

    def update(self, raw_z):
        # Insert currentZ into ring buffer
        self.buffer[self.rbIndex] = raw_z
        self.rbIndex += 1
        if self.rbIndex >= self.window_size:
            self.rbIndex = 0
        self.samples_count += 1

        if self.samples_count >= self.window_size:
            # Check std dev in the window
            windowStd = np.std(self.buffer)
            if windowStd < self.stationary_threshold:
                self.Db = np.mean(self.buffer)
                correctedZ = raw_z - self.Db
            else:
                correctedZ = raw_z - self.Db
        else:
            correctedZ = raw_z - self.Db

        return correctedZ

# ------------------------
# Basic Mag Yaw (unused)
# ------------------------
def _compute_tilt_compensated_yaw(mx, my, mz, roll_rad, pitch_rad):
    return math.atan2(my, mx)

# ------------------------
# IMU Calibration
# ------------------------
def calibrate_imu(imu, calibration_samples, dt):
    bx = by = bz = 0.0
    phi_sum = theta_sum = psi_sum = 0.0
    start_time = time.time()
    for _ in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        acc_raw = np.array(m9a)
        gyro_raw = np.array(m9g)
        mag_raw = np.array(m9m)

        mag_raw[1] = -mag_raw[1]
        mag_raw[2] = -mag_raw[2]

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

    return bx, by, bz, phi_bias, theta_bias, psi_bias

def calibrate_dual_imu(imu1, imu2, calibration_samples, dt):
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
    m9a, m9g, m9m = imu.getMotion9()
    acc_raw = np.array(m9a)
    gyro_raw = np.array(m9g)
    mag_raw = np.array(m9m)

    mag_raw[1] = -mag_raw[1]
    mag_raw[2] = -mag_raw[2]

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
    data1 = read_imu_data_raw(imu1, dt, prev_gyro_phi, prev_gyro_theta, prev_psi)
    data2 = read_imu_data_raw(imu2, dt, prev_gyro_phi, prev_theta_gyro, prev_psi)

    acc_comp = list((np.array(data1[0]) + np.array(data2[0])) / 2.0)
    gyro_aligned = list((np.array(data1[1]) + np.array(data2[1])) / 2.0)
    mag_aligned  = data1[2]

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

    Ts = 0.01  # 100 Hz
    calibration_samples = 100

    # 1) Calibrate dual IMU for basic offsets
    bx, by, bz, phi_bias, theta_bias, psi_bias = calibrate_dual_imu(imu1, imu2, calibration_samples, Ts)
    print(f"IMU Calibration Done: "
          f"Phi Bias: {np.rad2deg(phi_bias):.2f}°, "
          f"Theta Bias: {np.rad2deg(theta_bias):.2f}°, "
          f"Psi Bias: {np.rad2deg(psi_bias):.2f}°")

    # 2) We do an additional 5-second loop to measure the noise on gyro Z
    #    and auto-set the 'stationaryThreshold' for the YawDriftCorrector.
    print("Starting 5-second pre-experiment loop to auto-tune yaw drift threshold...")
    pre_duration = 5.0
    num_pre = int(pre_duration / Ts)
    Z_samples = []

    # We'll keep device presumably still in these 5s:
    prev_phi_gyro = phi_bias
    prev_theta_gyro = theta_bias
    prev_psi_gyro = psi_bias

    t_start_pre = time.time()
    for i in range(num_pre):
        loop_start = time.time()
        (acc_comp, gyro_aligned, mag_aligned,
         phi_acc, theta_acc, phi_gyro, theta_gyro, psi_gyro) = read_dual_imu_data_raw(
            imu1, imu2, Ts, prev_phi_gyro, prev_theta_gyro, prev_psi_gyro
        )
        prev_phi_gyro   = phi_gyro
        prev_theta_gyro = theta_gyro
        prev_psi_gyro  += psi_gyro

        Z_samples.append(gyro_aligned[2])  # store the Z-axis
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < Ts:
            time.sleep(Ts - loop_elapsed)

    # Compute standard deviation
    z_std = np.std(Z_samples)

    auto_stationary_threshold = z_std
    print(f"Measured Z-axis std = {z_std:.6f} rad/s, "
          f"Auto-set stationaryThreshold = {auto_stationary_threshold:.6f} rad/s")

    # Now we override the global stationaryThreshold with this new auto value
    stationaryThreshold = auto_stationary_threshold

    # 3) Create YawDriftCorrector with auto threshold
    yaw_corrector = YawDriftCorrectorRingBuffer(window_size=windowSize,
                                                stationary_threshold=stationaryThreshold,
                                                init_offset=0.0)

    # 4) Setup for main run
    q0_init = euler_to_quaternion(phi_bias, theta_bias, 0*psi_bias)
    q_comp = q0_init.copy()

    comp_phi_arr, comp_theta_arr, comp_psi_arr = [], [], []
    sensor_acc_arr, sensor_gyro_arr, sensor_mag_arr = [], [], []
    sensor_phi_acc_arr, sensor_theta_acc_arr = [], []
    sensor_phi_gyro_arr, sensor_theta_gyro_arr, sensor_psi_gyro_arr = [], [], []

    # Reset these for main run
    prev_phi_gyro   = phi_bias
    prev_theta_gyro = theta_bias
    prev_psi_gyro   = psi_bias

    print("Starting MAIN IMU Tuning (Complementary Only + Yaw Drift Correction)...")
    main_duration = 50.0  # run 50 seconds
    num_samples = int(main_duration / Ts)
    start_time = time.time()

    GPIO.output(START_SIGNAL_PIN, GPIO.HIGH)  # Signal start

    for i in range(num_samples):
        loop_start = time.time()

        (acc_comp, gyro_aligned, mag_aligned,
         phi_acc, theta_acc, phi_gyro, theta_gyro, psi_gyro) = read_dual_imu_data_raw(
            imu1, imu2, Ts, prev_phi_gyro, prev_theta_gyro, prev_psi_gyro
        )

        # Store sensor data
        sensor_acc_arr.append(acc_comp)
        sensor_gyro_arr.append(gyro_aligned)
        sensor_mag_arr.append(mag_aligned)
        sensor_phi_acc_arr.append(phi_acc)
        sensor_theta_acc_arr.append(theta_acc)
        sensor_phi_gyro_arr.append(phi_gyro)
        sensor_theta_gyro_arr.append(theta_gyro)
        sensor_psi_gyro_arr.append(psi_gyro)

        # Update integrated gyro angles
        prev_phi_gyro   = phi_gyro
        prev_theta_gyro = theta_gyro
        prev_psi_gyro  += psi_gyro

        # --- Apply Yaw Drift Correction on Z-axis
        gz_corrected = yaw_corrector.update(gyro_aligned[2])  # raw Z
        gyro_aligned[2] = gz_corrected  # override with corrected

        # --- Update Complementary filter
        q_comp = complementary_update(q_comp, gyr=gyro_aligned, acc=acc_comp, dt=Ts, gain=comp_gain)

        phi_c, theta_c, psi_c = quaternion_to_euler(q_comp)
        comp_phi_arr.append(phi_c)
        comp_theta_arr.append(theta_c)
        comp_psi_arr.append(psi_c)

        # Print real-time
        print(f"Time: {i*Ts:.2f}s | Comp: R {np.rad2deg(phi_c):.2f}°, "
              f"P {np.rad2deg(theta_c):.2f}°, Y {np.rad2deg(psi_c):.2f}°")

        loop_elapsed = time.time() - loop_start
        if loop_elapsed < Ts:
            time.sleep(Ts - loop_elapsed)

    print("IMU Tuning Completed.")
    GPIO.output(START_SIGNAL_PIN, GPIO.LOW)  # Signal end

    # Print standard deviations
    print(f"Complementary Std [rad]: "
          f"phi={np.std(comp_phi_arr):.4f}, "
          f"theta={np.std(comp_theta_arr):.4f}, "
          f"psi={np.std(comp_psi_arr):.4f}")

    # Save results
    results = {
        'comp_phi':   np.array(comp_phi_arr),
        'comp_theta': np.array(comp_theta_arr),
        'comp_psi':   np.array(comp_psi_arr),
        'sensor_acc': np.array(sensor_acc_arr),
        'sensor_gyro': np.array(sensor_gyro_arr),
        'sensor_mag': np.array(sensor_mag_arr),
        'sensor_phi_acc': np.array(sensor_phi_acc_arr),
        'sensor_theta_acc': np.array(sensor_theta_acc_arr),
        'sensor_phi_gyro': np.array(sensor_phi_gyro_arr),
        'sensor_theta_gyro': np.array(sensor_theta_gyro_arr),
        'sensor_psi_gyro': np.array(sensor_psi_gyro_arr),
        'time':       np.linspace(0, num_samples*Ts, num_samples)
    }

    timestamp = time.strftime("%y_%m_%d_%H_%M_%S")
    save_path = f"/home/pi/Documents/Quadcopter_Control_v2/data/results/imu_comp_{timestamp}.mat"
    savemat(save_path, results)
    print(f"Results saved to {save_path}")
