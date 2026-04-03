#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Custom EKF Filter for Navio2 with Sensor Offset and Misalignment Correction
Version: CustomEKF-v1
used in mainv23 10 april 2025
"""

# APPROVED!!
# Implemnet Experimentally!

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
        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
        gyroNoiseVar = 1e-5 #1e-2 # 1e-5 for best rol pitch
        biasNoiseVar = 1e-12
        self.Q = np.diag([gyroNoiseVar, gyroNoiseVar, gyroNoiseVar, biasNoiseVar, biasNoiseVar, biasNoiseVar])
        accNoiseVar = 0.1
        self.R = np.diag([accNoiseVar, accNoiseVar, accNoiseVar])
        self.g = 9.81
        # Gains for additional bias adaptation for roll and pitch:
        self.k_phi_bias = 0.01
        self.k_theta_bias = 0.01
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
        phi_gyro = phi + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        theta_gyro = theta + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        psi_gyro = psi + (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        # Biases remain unchanged during prediction:
        bp_new, bq_new, br_new = bp, bq, br

        # Predicted state vector:
        x_pred = np.array([phi_gyro, theta_gyro, psi_gyro, bp_new, bq_new, br_new])

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
        F[2,1] = dt*( math.sin(phi)*math.sin(theta)/math.cos(theta)**2*(q_corr) + math.cos(phi)*math.sin(theta)/math.cos(theta)**2*(r_corr) )
        F[2,4] = -dt*(math.sin(phi)/math.cos(theta))
        F[2,5] = -dt*(math.cos(phi)/math.cos(theta))

        # Covariance prediction:
        self.P = F.dot(self.P).dot(F.T) + self.Q

        # Measurement update using accelerometer:
        h_x = np.array([
            -self.g * math.sin(theta_gyro),
            self.g * math.sin(phi_gyro) * math.cos(theta_gyro),
            self.g * math.cos(phi_gyro) * math.cos(theta_gyro)
        ])
        y_meas = np.array(acc) - h_x

        # Measurement Jacobian H (only phi and theta affect the accelerometer measurement)
        H = np.zeros((3,6))
        H[0,1] = -self.g * math.cos(theta_gyro)
        H[1,0] = self.g * math.cos(phi_gyro) * math.cos(theta_gyro)
        H[1,1] = -self.g * math.sin(phi_gyro) * math.sin(theta_gyro)
        H[2,0] = -self.g * math.sin(phi_gyro) * math.cos(theta_gyro)
        H[2,1] = -self.g * math.cos(phi_gyro) * math.sin(theta_gyro)
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        # Innovation gating: if norm(acc) deviates too much from g, skip update.
        #if abs(np.linalg.norm(acc) - self.g) > self.gating_threshold:
        #    x_updated = x_pred
        #else:
        x_updated = x_pred + K.dot(y_meas)
        self.P = (np.eye(6) - K.dot(H)).dot(self.P)
        # --- Bias Adaptation for Roll and Pitch ---
        phi_acc = math.atan2(acc[1], acc[2])
        theta_acc = math.atan2(-acc[0], math.sqrt(acc[1]**2 + acc[2]**2))

        x_updated[3] = x_updated[3] + dt * self.k_phi_bias * (phi_acc - phi_gyro)
        x_updated[4] = x_updated[4] + dt * self.k_theta_bias * (theta_acc - theta_gyro)
        
        self.x = x_updated
        q_updated = euler_to_quaternion(self.x[0], self.x[1], self.x[2])
        return q_updated, self.x



# Function to apply recursive filtering with buffer size limit
def apply_recursive_filter(new_value, filtered_list, alpha, max_buffer_length):
    if filtered_list:
        new_filtered_value = alpha * new_value + (1 - alpha) * filtered_list[-1]
    else:
        new_filtered_value = new_value
    filtered_list.append(new_filtered_value)
    # Limit buffer size to prevent growth
    if len(filtered_list) > max_buffer_length:
        filtered_list.pop(0)
    return new_filtered_value


filtered_phi = []
filtered_theta = []
filtered_psi = []

# ================================
# Main Script: Using the CustomEKF
# ================================
if __name__ == '__main__':
    import time
    from utils.navio2 import lsm9ds1_backup as lsm9ds1

    # Initialize IMU (using MPU9250/LSM9DS1)
    imu = lsm9ds1.LSM9DS1()
    imu.initialize()
    
    Ts = 0.005  # Sampling time in seconds
    num_samples = int(60 / Ts)  # Run for 60 seconds
    
    # step 1: Calibration: Read a few gyro samples for bias estimation.
    calibration_samples = 200
    gyro_calib = []
    for i in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        gyro_calib.append(m9g)
        time.sleep(Ts)
    gyro_calib = np.array(gyro_calib)
    gyro_bias = np.mean(gyro_calib, axis=0)  # [bias_x, bias_y, bias_z]
    gyro_min = np.min(gyro_calib, axis=0)
    gyro_max = np.max(gyro_calib, axis=0)
    
    # Set initial quaternion from calibration (assuming zero Euler angles)
    q0 = euler_to_quaternion(0, 0, 0)
    
    # Instantiate our custom EKF filter and initialize its biases:
    custom_ekf = CustomEKF(q0=q0)
    custom_ekf.x[3:6] = gyro_bias  # initialize [bp, bq, br]
    
    # Precompute recursive low-pass filter coefficient
    filter_cutoff = 20  # Hz
    alpha = Ts / (Ts + (1 / (2 * np.pi * filter_cutoff)))
    
    ekf_phi_arr, ekf_theta_arr, ekf_psi_arr = [], [], []
    ekf_state_arr = []  # Save complete EKF state (including biases)
    sensor_acc_arr, sensor_gyro_arr, sensor_mag_arr, sim_time = [], [], [], []
    
    prev_acc = None # delet later
    prev_gyro = None # delet later
    filter_cutoff_sensor = 20  # Hz
    alpha_sensor = Ts / (Ts + (1 / (2 * np.pi * filter_cutoff_sensor)))
    

    print("Starting Custom EKF Filter on Navio2...")
    for i in range(num_samples):
        t0 = time.time()
        sim_time.append(i * Ts)
        
        # IMU reading
        m9a, m9g, m9m = imu.getMotion9()

        m9a = np.array(m9a)
        m9g = np.array(m9g)
        if prev_acc is None:
            filtered_acc = m9a
            filtered_gyro = m9g
        else:
            filtered_acc = alpha_sensor * m9a + (1 - alpha_sensor) * prev_acc
            filtered_gyro = alpha_sensor * m9g + (1 - alpha_sensor) * prev_gyro
        prev_acc = filtered_acc
        prev_gyro = filtered_gyro
        # m9a = filtered_acc.tolist()  # Replace m9a with filtered values
        m9g = filtered_gyro.tolist()  # Replace m9g with filtered values

        # Adjust gyro readings: if within [min, max] then use bias value
        gyro_tol = 0.1  # 10% tolerance
        for j in range(3):
            if abs(m9g[j] - gyro_bias[j]) < abs(gyro_bias[j]) * gyro_tol:
                m9g[j] = gyro_bias[j]

        q_updated, x_state = custom_ekf.update(q0, gyr=m9g, acc=m9a, mag=m9m, dt=Ts)
        q0 = q_updated
        
        # Get Euler angles and apply recursive low-pass filter
        phi, theta, psi = quaternion_to_euler(q_updated)
        if i == 0:
            filtered_phi, filtered_theta, filtered_psi = phi, theta, psi
        else:
            filtered_phi = alpha * phi + (1 - alpha) * filtered_phi
            filtered_theta = alpha * theta + (1 - alpha) * filtered_theta
            filtered_psi = alpha * psi + (1 - alpha) * filtered_psi
        phi, theta, psi = filtered_phi, filtered_theta, filtered_psi

        ekf_state_arr.append(x_state)
        sensor_acc_arr.append(m9a)
        sensor_gyro_arr.append(m9g)
        sensor_mag_arr.append(m9m)
        ekf_phi_arr.append(phi)
        ekf_theta_arr.append(theta)
        ekf_psi_arr.append(psi)
        
        print(f"Time: {i*Ts:.2f}s | EKF: Roll {np.rad2deg(phi):.3f}°, Pitch {np.rad2deg(theta):.3f}°, Yaw {np.rad2deg(psi):.3f}°, bx {(x_state[3]):.3f}°, by {(x_state[4]):.3f}°,")
        
        t_elapsed = time.time() - t0
        if t_elapsed < Ts:
            time.sleep(Ts - t_elapsed)
    
    print("Custom EKF filtering completed.")
    
    results = {
        'ekf_phi': np.array(ekf_phi_arr),
        'ekf_theta': np.array(ekf_theta_arr),
        'ekf_psi': np.array(ekf_psi_arr),
        'ekf_state': np.array(ekf_state_arr),
        'sensor_data': {
            'acc': np.array(sensor_acc_arr),
            'gyro': np.array(sensor_gyro_arr),
            'mag': np.array(sensor_mag_arr)
        },
        'sim_time': np.array(sim_time)
    }
    
    # Define the save path with the formatted filename
    timestamp = time.strftime("%y_%m_%d_%H_%M_%S")
    save_path = f"/home/pi/Documents/Quadcopter_Control_v2/data/results/ekf_results_lsm_003_{timestamp}.mat"
    savemat(save_path, results)
    print(f"Results saved to {save_path}")
    
    # Compute and display standard deviation of EKF estimates
    std_phi = np.std(ekf_phi_arr)
    std_theta = np.std(ekf_theta_arr)
    std_psi = np.std(ekf_psi_arr)
    print(f"StD (ekf_est)| Roll: {np.rad2deg(std_phi):.2f}° Pitch: {np.rad2deg(std_theta):.2f}° Yaw: {np.rad2deg(std_psi):.2f}°")
