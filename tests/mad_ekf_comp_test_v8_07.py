#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Complementary Filter Only, with automatic drift threshold
and external integration for roll, pitch, yaw.
Minimal changes from your script, with an added Kalman filter stage to smooth
the complementary filter output.
"""

import time
import numpy as np
import math
import sys
import os
from scipy.io import savemat
import RPi.GPIO as GPIO

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
START_SIGNAL_PIN = 26
GPIO.setup(START_SIGNAL_PIN, GPIO.OUT)
GPIO.output(START_SIGNAL_PIN, GPIO.LOW)

sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2/imu')


# Global Parameters
comp_gain = 0.90

stationaryThreshold = 0.001  # rad/s std (will be auto-set after 5s pre-run)
Ts = 0.005  # 100 Hz
windowSize = int(1/Ts)  # This converts the float to an integer.

# ------------------------
# Complementary: No Internal Gyro Integration
# ------------------------
def complementary_update(phi_old, theta_old, psi_old, acc, gain):
    """
    A simpler complementary filter step that:
      - Takes *existing* roll/pitch/yaw (from external integration).
      - Computes roll/pitch from accelerometer.
      - Fuses them with the old roll/pitch using 'gain'.
      - Leaves yaw unchanged (psi_new = psi_old).

    Returns (phi_new, theta_new, psi_new).
    """
    ax, ay, az = acc
    # Estimate roll/pitch from accelerometer
    phi_acc = math.atan2(ay, az)
    theta_acc = math.atan2(-ax, math.sqrt(ay*ay + az*az))
    # Complementary fuse roll, pitch
    phi_new   = gain * phi_old   + (1 - gain) * phi_acc
    theta_new = gain * theta_old + (1 - gain) * theta_acc
    psi_new   = psi_old  # purely from external integration
    return phi_new, theta_new, psi_new

# ------------------------
# Kalman Filter Update (scalar) for Euler angles
# ------------------------
def kalman_filter_update_scalar(x_prev, P_prev, z, Q, R):
    """
    A simple scalar Kalman filter update.
    x_prev: previous state (angle)
    P_prev: previous covariance (scalar)
    z: measurement
    Q: process noise covariance (scalar)
    R: measurement noise covariance (scalar)
    Returns (x_new, P_new)
    """
    # Prediction step (assume constant model)
    x_pred = x_prev
    P_pred = P_prev + Q
    # Kalman gain
    K = P_pred / (P_pred + R)
    # Update step
    x_new = x_pred + K * (z - x_pred)
    P_new = (1 - K) * P_pred
    return x_new, P_new

# ------------------------
# Drift Corrector for All Three Axes
# ------------------------
class GyroDriftCorrector3Axis:
    """
    Maintains three ring buffers (gx, gy, gz) to detect stationary intervals.
    If all three std devs are below threshold => update offsets to average.
    Otherwise keep the old offsets.
    """
    def __init__(self, window_size=50, stationary_threshold=0.001):
        self.window_size = window_size
        self.stationary_threshold = stationary_threshold
        self.buffer_x = np.zeros(window_size)
        self.buffer_y = np.zeros(window_size)
        self.buffer_z = np.zeros(window_size)
        self.idx = 0
        self.count = 0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0

    def update(self, gx_raw, gy_raw, gz_raw):
        # Insert into ring buffers
        self.buffer_x[self.idx] = gx_raw
        self.buffer_y[self.idx] = gy_raw
        self.buffer_z[self.idx] = gz_raw
        self.idx += 1
        if self.idx >= self.window_size:
            self.idx = 0
        self.count += 1

        # If buffer is full, check if "all axes" are near stationary
        if self.count >= self.window_size:
            std_x = np.std(self.buffer_x)
            std_y = np.std(self.buffer_y)
            std_z = np.std(self.buffer_z)
            # If all three are below threshold => treat as stationary
            if (std_x < self.stationary_threshold and
                std_y < self.stationary_threshold and
                std_z < self.stationary_threshold):
                self.offset_x = np.mean(self.buffer_x)
                self.offset_y = np.mean(self.buffer_y)
                self.offset_z = np.mean(self.buffer_z)

        # Subtract offsets
        gx_corr = gx_raw - self.offset_x
        gy_corr = gy_raw - self.offset_y
        gz_corr = gz_raw - self.offset_z
        return gx_corr, gy_corr, gz_corr

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
    data2 = read_imu_data_raw(imu2, dt, prev_gyro_phi, prev_gyro_theta, prev_psi)  # For simplicity, using imu1 twice
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

###############################################################################
# MAIN SCRIPT
###############################################################################
if __name__ == '__main__':
    from utils.navio2 import lsm9ds1_backup as lsm9ds1
    from utils.navio2 import mpu9250

    imu1 = lsm9ds1.LSM9DS1()
    imu2 = mpu9250.MPU9250()
    imu1.initialize()
    imu2.initialize()

    calibration_samples = 100
    bx, by, bz, phi_bias, theta_bias, psi_bias = calibrate_dual_imu(
        imu1, imu2, calibration_samples, Ts
    )
    print(f"IMU Calibration: Roll bias={np.rad2deg(phi_bias):.2f}°, "
          f"Pitch bias={np.rad2deg(theta_bias):.2f}°, Yaw bias={np.rad2deg(psi_bias):.2f}°")

    # 2) 5-second pre-run to measure overall gyro noise => auto threshold
    print("Pre-run 5s to auto-tune drift threshold for all axes...")
    pre_duration = 5.0
    num_pre = int(pre_duration / Ts)
    gx_list, gy_list, gz_list = [], [], []

    prev_phi_gyro = phi_bias
    prev_theta_gyro = theta_bias
    prev_psi_gyro = psi_bias

    for i in range(num_pre):
        start_loop = time.time()
        (acc_comp, gyro_aligned, mag_aligned,
         phi_acc, theta_acc, phi_gyr, theta_gyr, psi_gyr) = read_dual_imu_data_raw(
            imu1, imu2, Ts, prev_phi_gyro, prev_theta_gyro, prev_psi_gyro
        )
        prev_phi_gyro   = phi_gyr
        prev_theta_gyro = theta_gyr
        prev_psi_gyro  += psi_gyr

        gx_list.append(gyro_aligned[0])
        gy_list.append(gyro_aligned[1])
        gz_list.append(gyro_aligned[2])

        elapsed = time.time() - start_loop
        if elapsed < Ts:
            time.sleep(Ts - elapsed)

    gx_std = np.std(gx_list)
    gy_std = np.std(gy_list)
    gz_std = np.std(gz_list)
    overall_std = (gx_std + gy_std + gz_std) / 3.0
    auto_threshold = 5.0 * overall_std
    print(f"Measured stds => gx={gx_std:.5f}, gy={gy_std:.5f}, gz={gz_std:.5f}. "
          f"Auto threshold = {auto_threshold:.5f}")

    # 3) Create 3-axis drift corrector
    drift_corrector = GyroDriftCorrector3Axis(window_size=windowSize,
                                              stationary_threshold=auto_threshold)

    # 4) We'll do our own roll/pitch/yaw integration externally
    #    Start from the bias angles
    phi_gyro = phi_bias
    theta_gyro = theta_bias
    psi_gyro = 0 * psi_bias

    # Initialize Kalman filter variables for each angle (scalar filters)
    x_phi = phi_bias;  P_phi = 1e-3
    x_theta = theta_bias;  P_theta = 1e-3
    x_psi = 0 * psi_bias;  P_psi = 1e-3
    Q = 1e-1  # process noise covariance
    R = 1e-3  # measurement noise covariance

    # 5) Start main loop
    GPIO.output(START_SIGNAL_PIN, GPIO.HIGH)
    main_duration = 100.0
    num_main = int(main_duration / Ts)
    time_arr = np.linspace(0, main_duration, num_main)

    phi_comp_arr, theta_comp_arr, psi_comp_arr = [], [], []
    sensor_acc_arr, sensor_gyro_arr = [], []
    t0 = time.time()

    for i in range(num_main):
        loop_start = time.time()
        (acc_comp, gyro_aligned, _,  # ignoring mag
         phi_acc, theta_acc, phi_gyr, theta_gyr, psi_gyr) = read_dual_imu_data_raw(
            imu1, imu2, Ts, phi_gyro, theta_gyro, psi_gyro
        )

        sensor_acc_arr.append(acc_comp)
        sensor_gyro_arr.append(gyro_aligned)

        # 5a) Apply drift corrector to Gx, Gy, Gz
        gx_corr, gy_corr, gz_corr = drift_corrector.update(
            gyro_aligned[0], gyro_aligned[1], gyro_aligned[2]
        )

        # 5b) Integrate externally => roll/pitch/yaw in radians
        phi_gyro   += gx_corr * Ts
        theta_gyro += gy_corr * Ts
        psi_gyro   += gz_corr * Ts

        # 5c) Minimal complementary => fuse roll/pitch with accelerometer
        #     Keep yaw from our integrated yaw_gyro
        phi_comp, theta_comp, psi_comp = complementary_update(
            phi_gyro, theta_gyro, psi_gyro, acc_comp, comp_gain
        )

        # 5d) Apply simple Kalman filter updates to each Euler angle
        def kalman_filter_update_scalar(x_prev, P_prev, z, Q, R):
            x_pred = x_prev
            P_pred = P_prev + Q
            K = P_pred / (P_pred + R)
            x_new = x_pred + K * (z - x_pred)
            P_new = (1 - K) * P_pred
            return x_new, P_new

        x_phi, P_phi = kalman_filter_update_scalar(x_phi, P_phi, phi_comp, Q, R)
        x_theta, P_theta = kalman_filter_update_scalar(x_theta, P_theta, theta_comp, Q, R)
        x_psi, P_psi = kalman_filter_update_scalar(x_psi, P_psi, psi_comp, Q, R)

        # Use the filtered values
        phi_comp = x_phi
        theta_comp = x_theta
        psi_comp = x_psi

        # Overwrite the integrated angles with the fused (and filtered) roll/pitch/yaw
        phi_gyro   = phi_comp
        theta_gyro = theta_comp
        psi_gyro   = psi_comp

        phi_comp_arr.append(phi_comp)
        theta_comp_arr.append(theta_comp)
        psi_comp_arr.append(psi_comp)

        print(f"Time={i*Ts:.2f}s | Roll={np.rad2deg(phi_comp):.2f}°, "
              f"Pitch={np.rad2deg(theta_comp):.2f}°, Yaw={np.rad2deg(psi_comp):.2f}°")

        elapsed = time.time() - loop_start
        if elapsed < Ts:
            time.sleep(Ts - elapsed)

    GPIO.output(START_SIGNAL_PIN, GPIO.LOW)
    print("IMU Tuning Completed.")

    results = {
        'phi_comp': np.array(phi_comp_arr),
        'theta_comp': np.array(theta_comp_arr),
        'psi_comp': np.array(psi_comp_arr),
        'acc': np.array(sensor_acc_arr),
        'gyro': np.array(sensor_gyro_arr),
        'time': time_arr
    }
    timestamp = time.strftime("%y_%m_%d_%H_%M_%S")
    save_path = f"/home/pi/Documents/Quadcopter_Control_v2/data/results/imu_comp_{timestamp}.mat"
    savemat(save_path, results)
    print(f"Results saved to {save_path}")
