#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Custom EKF Filter for Navio2 with Sensor Offset and Misalignment Correction
Version: CustomEKF-v1

We add an AFTER-EKF "Adaptive Sliding-Window" yaw-drift method,
just like your EXACT_PAPER_ADAPTIVE_SLIDING_YAW_SCRIPT in MATLAB.

The original EKF code and main loop structure are NOT changed or removed.
We only add minimal lines at the end of the loop to do post-processing.
"""

# ========================
# Standard Python Libraries
# ========================
import time
import numpy as np
import math
import sys
import os
from scipy.io import savemat  # For saving results in a .mat file

# Add your project directories if necessary
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2/imu')

# ========================
# Global Variables
# ========================
# Sensor offset from the center of gravity (in meters), defined in vehicle frame.
sensor_offset = np.array([0, 0, 0.0])  # e.g., 10 cm off in X and Y

# Misalignment correction angles (in radians) – adjust these as needed.
misalign_phi   = 0.0               # rotation about X-axis
misalign_theta = math.radians(0)   # rotation about Y-axis
misalign_psi   = 0.0               # rotation about Z-axis

# Global variable to hold previous gyro measurement (if needed).
prev_gyro = None

# ========================
# Third-Party / Custom Modules
# ========================
# (We are not using these filters in our custom implementation, but they are kept for reference.)
# from imu.madgwick import Madgwick
# from imu.ekf import EKF
from utils.navio2 import lsm9ds1_backup as lsm9ds1

# ------------------------
# Euler and Quaternion Conversion Functions
# ------------------------
def euler_to_quaternion(phi, theta, psi):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    qw = math.cos(phi/2)*math.cos(theta/2)*math.cos(psi/2) + math.sin(phi/2)*math.sin(theta/2)*math.sin(psi/2)
    qx = math.sin(phi/2)*math.cos(theta/2)*math.cos(psi/2) - math.cos(phi/2)*math.sin(theta/2)*math.sin(psi/2)
    qy = math.cos(phi/2)*math.sin(theta/2)*math.cos(psi/2) + math.sin(phi/2)*math.cos(theta/2)*math.sin(psi/2)
    qz = math.cos(phi/2)*math.cos(theta/2)*math.sin(psi/2) - math.sin(phi/2)*math.sin(theta/2)*math.cos(psi/2)
    return np.array([qw, qx, qy, qz])

def quaternion_to_euler(q):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    phi = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    theta = math.asin(sinp)
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    psi = math.atan2(siny_cosp, cosy_cosp)
    return phi, theta, psi

# ------------------------
# Custom EKF Class Definition
# ------------------------
class CustomEKF:
    def __init__(self, q0=None):
        """
        Initialize the custom EKF.
        If q0 is provided (as a quaternion), convert it to Euler angles.
        Otherwise, initialize state to zeros.
        State vector: x = [phi; theta; psi; bp; bq; br]
        """
        if q0 is not None:
            phi, theta, psi = quaternion_to_euler(q0)
        else:
            phi = theta = psi = 0.0
        self.x = np.array([phi, theta, psi, 0.0, 0.0, 0.0])
        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        gyroNoiseVar = 1 #1e-2
        biasNoiseVar = 1e-6
        self.Q = np.diag([gyroNoiseVar, gyroNoiseVar, gyroNoiseVar,
                          biasNoiseVar, biasNoiseVar, biasNoiseVar])
        accNoiseVar = 0.9
        self.R = np.diag([accNoiseVar, accNoiseVar, accNoiseVar])
        self.g = 9.81
        # Gains for additional bias adaptation for roll and pitch:
        self.k_phi_bias = 0.1
        self.k_theta_bias = 0.1
        # Innovation gating threshold (in m/s^2)
        self.gating_threshold = 1.0

    def update(self, q0, gyr, acc, mag, dt):
        """
        Update the filter state using new sensor measurements.
        Inputs:
          q0  : current quaternion estimate (unused for propagation; provided for interface consistency)
          gyr : gyro measurement as [p, q, r] in rad/s
          acc : accelerometer measurement as [ax, ay, az] in m/s^2
          mag : magnetometer measurement (unused, placeholder)
          dt  : sampling interval in seconds
        Returns:
          Updated quaternion estimate.
        """
        # Extract current state:
        phi, theta, psi, bp, bq, br = self.x

        # Bias-corrected gyro measurements:
        p_corr = gyr[0] - bp
        q_corr = gyr[1] - bq
        r_corr = gyr[2] - br

        # RK4 integration for Euler angles:
        def T(phi, theta):
            return np.array([
                [1, math.sin(phi)*math.tan(theta), math.cos(phi)*math.tan(theta)],
                [0, math.cos(phi), -math.sin(phi)],
                [0, math.sin(phi)/math.cos(theta), math.cos(phi)/math.cos(theta)]
            ])
        u_corr = np.array([p_corr, q_corr, r_corr])
        k1 = T(phi, theta).dot(u_corr)
        phi2 = phi + 0.5 * dt * k1[0]
        theta2 = theta + 0.5 * dt * k1[1]
        psi2 = psi + 0.5 * dt * k1[2]
        k2 = T(phi2, theta2).dot(u_corr)
        phi3 = phi + 0.5 * dt * k2[0]
        theta3 = theta + 0.5 * dt * k2[1]
        psi3 = psi + 0.5 * dt * k2[2]
        k3 = T(phi3, theta3).dot(u_corr)
        phi4 = phi + dt * k3[0]
        theta4 = theta + dt * k3[1]
        psi4 = psi + dt * k3[2]
        k4 = T(phi4, theta4).dot(u_corr)
        phi_new = phi + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        theta_new = theta + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        psi_new = psi + (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        # Biases remain unchanged during prediction:
        bp_new, bq_new, br_new = bp, bq, br

        # Predicted state vector:
        x_pred = np.array([phi_new, theta_new, psi_new, bp_new, bq_new, br_new])

        # Process Model Jacobian F (using bias-corrected gyro terms)
        tan_theta = math.tan(theta)
        sec2_theta = 1/math.cos(theta)**2
        F = np.eye(6)
        # Replace (q - bq) with q_corr and (r - br) with r_corr:
        F[0,0] = 1 + dt*( math.cos(phi)*tan_theta*(q_corr) - math.sin(phi)*tan_theta*(r_corr) )
        F[0,1] = dt*( math.sin(phi)*sec2_theta*(q_corr) + math.cos(phi)*sec2_theta*(r_corr) )
        F[0,3] = -dt
        F[0,4] = -dt*math.sin(phi)*tan_theta
        F[0,5] = -dt*math.cos(phi)*tan_theta
        F[1,0] = dt*(-math.sin(phi)*(q_corr) - math.cos(phi)*(r_corr))
        F[1,4] = -dt*math.cos(phi)
        F[1,5] = dt*math.sin(phi)
        F[2,0] = dt*( (math.cos(phi)/math.cos(theta))*(q_corr) - (math.sin(phi)/math.cos(theta))*(r_corr) )
        F[2,1] = dt*( math.sin(phi)*math.sin(theta)/math.cos(theta)**2*(q_corr)
                      + math.cos(phi)*math.sin(theta)/math.cos(theta)**2*(r_corr) )
        F[2,4] = -dt*(math.sin(phi)/math.cos(theta))
        F[2,5] = -dt*(math.cos(phi)/math.cos(theta))

        # Covariance prediction:
        self.P = F.dot(self.P).dot(F.T) + self.Q

        # Measurement update using accelerometer:
        h_x = np.array([
            -self.g * math.sin(theta_new),
            self.g * math.sin(phi_new) * math.cos(theta_new),
            self.g * math.cos(phi_new) * math.cos(theta_new)
        ])
        y_meas = np.array(acc) - h_x

        # Measurement Jacobian H (only phi and theta affect the accelerometer measurement)
        H = np.zeros((3,6))
        H[0,1] = -self.g * math.cos(theta_new)
        H[1,0] =  self.g * math.cos(phi_new) * math.cos(theta_new)
        H[1,1] = -self.g * math.sin(phi_new) * math.sin(theta_new)
        H[2,0] = -self.g * math.sin(phi_new) * math.cos(theta_new)
        H[2,1] = -self.g * math.cos(phi_new) * math.sin(theta_new)
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        # Innovation gating: if norm(acc) deviates too much from g, skip update.
        if abs(np.linalg.norm(acc) - self.g) > self.gating_threshold:
            x_updated = x_pred
        else:
            x_updated = x_pred + K.dot(y_meas)
            self.P = (np.eye(6) - K.dot(H)).dot(self.P)
            # --- Additional Bias Adaptation for Roll and Pitch ---
            phi_acc = math.atan2(acc[1], acc[2])
            theta_acc = -math.asin(np.clip(acc[0]/self.g, -1.0, 1.0))
            x_updated[3] = x_updated[3] + dt * self.k_phi_bias * (phi_acc - x_updated[0])
            x_updated[4] = x_updated[4] + dt * self.k_theta_bias * (theta_acc - x_updated[1])

        self.x = x_updated
        q_updated = euler_to_quaternion(self.x[0], self.x[1], self.x[2])
        return q_updated


# ================================
# Main Script: Using the CustomEKF
# ================================
if __name__ == '__main__':
    import time
    from utils.navio2 import lsm9ds1_backup as lsm9ds1

    # --------------------------
    # (A) Setup: IMU + EKF
    # --------------------------
    imu = lsm9ds1.LSM9DS1()
    imu.initialize()

    Ts = 0.005  # Sampling time in seconds
    num_samples = int(60 / Ts)  # Run for 60 seconds

    # Calibration: Read a few gyro samples for bias estimation.
    calibration_samples = 200
    gyro_calib = []
    for i in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        gyro_calib.append(m9g)
        time.sleep(Ts)
    gyro_calib = np.array(gyro_calib)
    gyro_bias = np.mean(gyro_calib, axis=0)  # [bias_x, bias_y, bias_z]

    # Set initial quaternion from calibration (assuming zero Euler angles)
    q0 = euler_to_quaternion(0, 0, 0)

    # Instantiate our custom EKF filter and initialize its biases:
    custom_ekf = CustomEKF(q0=q0)
    custom_ekf.x[3:6] = gyro_bias  # initialize [bp, bq, br]

    ekf_phi_arr = []
    ekf_theta_arr = []
    ekf_psi_arr = []

    print("Starting Custom EKF Filter on Navio2...")

    # --------------------------
    # (B) Prepare Sliding-Window Yaw Correction (Minimal Additions)
    # --------------------------
    # EXACT PAPER method: We'll track raw gyro z-axis in a ring buffer,
    # if stationary => update offset => integrate to get corrected yaw.
    windowSize = 10              # ~2 seconds if 100 Hz
    stationaryThreshold = 0.01   # threshold for "stationary in yaw"
    slidingWindow = []
    db = 0.0                      # drift offset
    yaw_sliding_deg = 0.0         # integrated yaw in degrees
    yaw_sliding_arr = []          # store final corrected yaw

    # --------------------------
    # (C) Main loop
    # --------------------------
    for i in range(num_samples):
        loop_start = time.time()

        # (C1) Read IMU data
        m9a, m9g, m9m = imu.getMotion9()
        acc_raw = np.array(m9a)
        gyro_raw = np.array(m9g)
        mag_raw = np.array(m9m)

        # (C2) EKF update (UNMODIFIED)
        acc_comp = acc_raw.tolist()
        gyro_aligned = gyro_raw.tolist()
        mag_aligned = mag_raw.tolist()

        q_updated = custom_ekf.update(q0, gyr=gyro_aligned, acc=acc_comp, mag=mag_aligned, dt=Ts)
        q0 = q_updated

        # (C3) Extract Euler from EKF to store
        phi, theta, psi = quaternion_to_euler(q_updated)
        ekf_phi_arr.append(phi)
        ekf_theta_arr.append(theta)
        ekf_psi_arr.append(psi)

        # (C4) Print EKF yaw
        print(f"Time: {i*Ts:.2f}s | EKF: Roll {np.rad2deg(phi):.2f}°, "
              f"Pitch {np.rad2deg(theta):.2f}°, Yaw {np.rad2deg(psi):.2f}°")

        # -----------------------------
        # (D) Sliding-Window Yaw Correction 
        # (added AFTER the EKF step)
        # -----------------------------
        # 1) Add the newest raw yaw-rate (Z) to the ring buffer
        Z = gyro_raw[2]  # rad/s
        slidingWindow.append(Z)
        if len(slidingWindow) > windowSize:
            slidingWindow.pop(0)

        # 2) Check if stationary => update drift offset db
        if len(slidingWindow) == windowSize:
            current_std = np.std(slidingWindow)
            if current_std < stationaryThreshold:
                db = np.mean(slidingWindow)

        # 3) Corrected yaw-rate, integrate
        Zc = Z - db                   # rad/s
        yaw_sliding_deg += (Zc * Ts * (180.0 / math.pi))  # in degrees
        yaw_sliding_arr.append(yaw_sliding_deg)

        # (Optional) Could also print the "sliding-window corrected yaw"
        print(f"   --> Sliding Yaw: {yaw_sliding_deg:.2f}°")

        # (C5) Sleep to maintain loop rate
        loop_duration = time.time() - loop_start
        if loop_duration < Ts:
            time.sleep(Ts - loop_duration)

    print("Custom EKF filtering completed.")

    # --------------------------
    # (E) Save Results
    # --------------------------
    results = {
        'ekf_phi':      np.array(ekf_phi_arr),
        'ekf_theta':    np.array(ekf_theta_arr),
        'ekf_psi':      np.array(ekf_psi_arr),      # in radians
        'sliding_yaw':  np.array(yaw_sliding_arr),  # in degrees
        'time':         np.linspace(0, num_samples*Ts, num_samples)
    }
    savemat("custom_ekf_results.mat", results)
    print("Results saved to custom_ekf_results.mat")
    print("Done.")
