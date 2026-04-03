#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Custom EKF Filter (Navio2) with 
Sensor Offset + Misalignment Correction + 
ADAPTIVE SLIDING-WINDOW YAW DRIFT using a Virtual Measurement.

We do NOT remove or replace any lines from your original EKF class – 
we only ADD code for the sliding-window logic and virtual measurement.
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
# Global Variables (unchanged)
# ========================
sensor_offset = np.array([0, 0, 0.0])
misalign_phi   = 0.0
misalign_theta = math.radians(0)
misalign_psi   = 0.0
prev_gyro = None

# ========================
# Third-Party / Custom Modules
# ========================
from utils.navio2 import lsm9ds1_backup as lsm9ds1

# ------------------------
# Euler and Quaternion Conversion Functions
# ------------------------
def euler_to_quaternion(phi, theta, psi):
    qw = math.cos(phi/2)*math.cos(theta/2)*math.cos(psi/2) + math.sin(phi/2)*math.sin(theta/2)*math.sin(psi/2)
    qx = math.sin(phi/2)*math.cos(theta/2)*math.cos(psi/2) - math.cos(phi/2)*math.sin(theta/2)*math.sin(psi/2)
    qy = math.cos(phi/2)*math.sin(theta/2)*math.cos(psi/2) + math.sin(phi/2)*math.cos(theta/2)*math.sin(psi/2)
    qz = math.cos(phi/2)*math.cos(theta/2)*math.sin(psi/2) - math.sin(phi/2)*math.sin(theta/2)*math.cos(psi/2)
    return np.array([qw, qx, qy, qz])

def quaternion_to_euler(q):
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
# Your Existing EKF Class
# ------------------------
class CustomEKF:
    def __init__(self, q0=None):
        """
        Unchanged code: 6-state filter: x = [phi, theta, psi, bp, bq, br].
        """
        if q0 is not None:
            phi, theta, psi = quaternion_to_euler(q0)
        else:
            phi = theta = psi = 0.0
        self.x = np.array([phi, theta, psi, 0.0, 0.0, 0.0])
        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        gyroNoiseVar = 1  # can tune
        biasNoiseVar = 1e-6
        self.Q = np.diag([gyroNoiseVar, gyroNoiseVar, gyroNoiseVar,
                          biasNoiseVar, biasNoiseVar, biasNoiseVar])
        accNoiseVar = 0.9
        self.R = np.diag([accNoiseVar, accNoiseVar, accNoiseVar])
        self.g = 9.81
        self.k_phi_bias = 0.1
        self.k_theta_bias = 0.1
        self.gating_threshold = 1.0

    def update(self, q0, gyr, acc, mag, dt):
        # --- PREDICTION STEP (RK4) ---
        phi, theta, psi, bp, bq, br = self.x
        p_corr = gyr[0] - bp
        q_corr = gyr[1] - bq
        r_corr = gyr[2] - br

        def T(phi, theta):
            return np.array([
                [1, math.sin(phi)*math.tan(theta), math.cos(phi)*math.tan(theta)],
                [0, math.cos(phi), -math.sin(phi)],
                [0, math.sin(phi)/math.cos(theta), math.cos(phi)/math.cos(theta)]
            ])
        u_corr = np.array([p_corr, q_corr, r_corr])
        k1 = T(phi, theta).dot(u_corr)
        phi2 = phi + 0.5*dt*k1[0]
        theta2 = theta + 0.5*dt*k1[1]
        psi2 = psi + 0.5*dt*k1[2]
        k2 = T(phi2, theta2).dot(u_corr)
        phi3 = phi + 0.5*dt*k2[0]
        theta3 = theta + 0.5*dt*k2[1]
        psi3 = psi + 0.5*dt*k2[2]
        k3 = T(phi3, theta3).dot(u_corr)
        phi4 = phi + dt*k3[0]
        theta4 = theta + dt*k3[1]
        psi4 = psi + dt*k3[2]
        k4 = T(phi4, theta4).dot(u_corr)
        phi_new   = phi   + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        theta_new = theta + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        psi_new   = psi   + (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        bp_new, bq_new, br_new = bp, bq, br
        x_pred = np.array([phi_new, theta_new, psi_new, bp_new, bq_new, br_new])

        # --- JACOBIAN (F) & PREDICT ---
        tan_theta = math.tan(theta)
        sec2_theta = 1/math.cos(theta)**2
        F = np.eye(6)
        F[0,0] = 1 + dt*( math.cos(phi)*tan_theta*(q_corr) - math.sin(phi)*tan_theta*(r_corr) )
        F[0,1] = dt*( math.sin(phi)*sec2_theta*(q_corr) + math.cos(phi)*sec2_theta*(r_corr) )
        F[0,3] = -dt
        F[0,4] = -dt*math.sin(phi)*tan_theta
        F[0,5] = -dt*math.cos(phi)*tan_theta
        F[1,0] = dt*(-math.sin(phi)*(q_corr) - math.cos(phi)*(r_corr))
        F[1,4] = -dt*math.cos(phi)
        F[1,5] = dt*math.sin(phi)
        F[2,0] = dt*( (math.cos(phi)/math.cos(theta))*q_corr - (math.sin(phi)/math.cos(theta))*r_corr )
        F[2,1] = dt*( math.sin(phi)*math.sin(theta)/math.cos(theta)**2*q_corr
                      + math.cos(phi)*math.sin(theta)/math.cos(theta)**2*r_corr )
        F[2,4] = -dt*( math.sin(phi)/math.cos(theta) )
        F[2,5] = -dt*( math.cos(phi)/math.cos(theta) )
        self.P = F.dot(self.P).dot(F.T) + self.Q

        # --- ACCELEROMETER UPDATE (H, K) ---
        h_x = np.array([
            -self.g * math.sin(theta_new),
            self.g * math.sin(phi_new) * math.cos(theta_new),
            self.g * math.cos(phi_new) * math.cos(theta_new)
        ])
        y_meas = np.array(acc) - h_x
        H = np.zeros((3,6))
        H[0,1] = -self.g * math.cos(theta_new)
        H[1,0] =  self.g * math.cos(phi_new)*math.cos(theta_new)
        H[1,1] = -self.g * math.sin(phi_new)*math.sin(theta_new)
        H[2,0] = -self.g * math.sin(phi_new)*math.cos(theta_new)
        H[2,1] = -self.g * math.cos(phi_new)*math.sin(theta_new)
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        if abs(np.linalg.norm(acc) - self.g) > self.gating_threshold:
            x_updated = x_pred
        else:
            x_updated = x_pred + K.dot(y_meas)
            self.P = (np.eye(6) - K.dot(H)).dot(self.P)
            # Additional roll/pitch bias adaptation
            phi_acc = math.atan2(acc[1], acc[2])
            theta_acc = -math.asin(np.clip(acc[0]/self.g, -1.0, 1.0))
            x_updated[3] += dt*self.k_phi_bias  * (phi_acc - x_updated[0])
            x_updated[4] += dt*self.k_theta_bias * (theta_acc - x_updated[1])

        self.x = x_updated
        q_updated = euler_to_quaternion(self.x[0], self.x[1], self.x[2])
        return q_updated


# ================================
# Main Script
# ================================
if __name__ == '__main__':
    import time
    from utils.navio2 import lsm9ds1_backup as lsm9ds1

    # (A) Sliding-Window Setup
    windowSize = 200
    stationaryThreshold = 0.001
    slidingWindow = []
    db_yaw = 0.0  # Our drift estimate for yaw bias

    # (B) Initialize IMU
    imu = lsm9ds1.LSM9DS1()
    imu.initialize()
    Ts = 0.005
    num_samples = int(60 / Ts)

    # (B1) Gyro calibration
    calibration_samples = 200
    gyro_calib = []
    for i in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        gyro_calib.append(m9g)
        time.sleep(Ts)
    gyro_calib = np.array(gyro_calib)
    gyro_bias = np.mean(gyro_calib, axis=0)

    # (B2) EKF init
    q0 = euler_to_quaternion(0,0,0)
    custom_ekf = CustomEKF(q0=q0)
    custom_ekf.x[3:6] = gyro_bias  # [bp, bq, br]

    ekf_phi_arr = []
    ekf_theta_arr = []
    ekf_psi_arr = []

    # For the "virtual measurement" of br
    R_yawBias = 1e-4  # or bigger if you trust it less

    print("Starting Custom EKF Filter on Navio2...")
    for i in range(num_samples):
        loop_start = time.time()
        m9a, m9g, m9m = imu.getMotion9()
        acc_raw = np.array(m9a)
        gyro_raw = np.array(m9g)
        mag_raw = np.array(m9m)

        # (C) Sliding-window logic for r (yaw rate)
        slidingWindow.append(gyro_raw[2])
        if len(slidingWindow) > windowSize:
            slidingWindow.pop(0)

        # If we have a full buffer & it's "stationary" in yaw => update db_yaw
        if len(slidingWindow) == windowSize:
            std_yaw = np.std(slidingWindow)
            if std_yaw < stationaryThreshold:
                db_yaw = np.mean(slidingWindow)

        # (D) EKF update with *raw* gyro
        q_updated = custom_ekf.update(q0,
                                      gyr=gyro_raw.tolist(),
                                      acc=acc_raw.tolist(),
                                      mag=mag_raw.tolist(),
                                      dt=Ts)
        q0 = q_updated
        phi, theta, psi = quaternion_to_euler(q_updated)
        ekf_phi_arr.append(phi)
        ekf_theta_arr.append(theta)
        ekf_psi_arr.append(psi)

        # --------------------------------------------------------
        # (E) Virtual Measurement of br = db_yaw (1D Kalman update)
        # --------------------------------------------------------
        # We do this if we think the system is stationary in yaw
        # (i.e., std_yaw < threshold) so that db_yaw is meaningful.
        if len(slidingWindow) == windowSize and std_yaw < stationaryThreshold:
            # Current br in the state:
            br_current = custom_ekf.x[5]

            # Residual: (db_yaw - br_current)
            residual = db_yaw - br_current

            # H_yaw = [0, 0, 0, 0, 0, 1]
            H_yaw = np.array([[0,0,0,0,0,1]], dtype=float)
            # S_yaw = H_yaw * P * H_yaw^T + R_yawBias
            S_yaw = H_yaw.dot(custom_ekf.P).dot(H_yaw.T) + R_yawBias
            # K_yaw = P * H_yaw^T * inv(S_yaw)
            K_yaw = custom_ekf.P.dot(H_yaw.T) / S_yaw[0,0]

            # State update
            custom_ekf.x = custom_ekf.x + K_yaw.flatten() * residual
            # Cov update
            I_6 = np.eye(6)
            custom_ekf.P = (I_6 - K_yaw.dot(H_yaw)).dot(custom_ekf.P)

        print(f"Time: {i*Ts:.2f}s | Roll {np.rad2deg(phi):.2f}°, "
              f"Pitch {np.rad2deg(theta):.2f}°, Yaw {np.rad2deg(psi):.2f}°")

        # (F) Loop timing
        loop_duration = time.time() - loop_start
        if loop_duration < Ts:
            time.sleep(Ts - loop_duration)

    print("Custom EKF filtering completed.")

    # (G) Save results
    results = {
        'ekf_phi': np.array(ekf_phi_arr),
        'ekf_theta': np.array(ekf_theta_arr),
        'ekf_psi': np.array(ekf_psi_arr),
        'time': np.linspace(0, num_samples*Ts, num_samples)
    }
    savemat("custom_ekf_results.mat", results)
    print("Results saved to custom_ekf_results.mat")
