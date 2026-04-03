#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Tuning Code for Custom EKF Filter for Navio2  
Method discussed in Meeting

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


def euler_to_quaternion(phi, theta, psi):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    qw = math.cos(phi/2) * math.cos(theta/2) * math.cos(psi/2) + math.sin(phi/2) * math.sin(theta/2) * math.sin(psi/2)
    qx = math.sin(phi/2) * math.cos(theta/2) * math.cos(psi/2) - math.cos(phi/2) * math.sin(theta/2) * math.sin(psi/2)
    qy = math.cos(phi/2) * math.sin(theta/2) * math.cos(psi/2) + math.sin(phi/2) * math.cos(theta/2) * math.sin(psi/2)
    qz = math.cos(phi/2) * math.cos(theta/2) * math.sin(psi/2) - math.sin(phi/2) * math.sin(theta/2) * math.cos(psi/2)
    return np.array([qw, qx, qy, qz])

def quaternion_to_euler(q):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    phi = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    theta = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    psi = math.atan2(siny_cosp, cosy_cosp)
    return phi, theta, psi

class CustomEKF:
    def __init__(self, gyroBias, gyroVar, q0=None):
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

        self.x = np.array([phi, theta, psi, gyroBias[0], gyroBias[1], gyroBias[2]])
        # Initial covariance matrix 
        self.P = np.diag([0.1, 0.1, 0.1, 0.2, 0.1, 0.0])
        
        # Process noise covariance matrix Q
        gyroNoiseVar = 1e-5
        biasNoiseVar = 1e-12

        self.Q = np.diag([gyroNoiseVar, gyroNoiseVar, gyroNoiseVar,
                          biasNoiseVar, biasNoiseVar, biasNoiseVar])
        #self.Q = np.diag([gyroVar[0], gyroVar[1], gyroVar[2],
        #                  biasNoiseVar, biasNoiseVar, biasNoiseVar])
        
        # Measurement noise covariance R
        accNoiseVar = 1e-1
        self.R = np.diag([accNoiseVar, accNoiseVar, accNoiseVar])
        
        self.g = 9.81  # gravitational acceleration

    def update(self, q0, gyr, acc, mag, dt):
        """
        Update the EKF state using new sensor measurements.
        
        Inputs:
          q0  : Current quaternion estimate (unused in prediction; provided for interface consistency)
          gyr : Gyro measurement as [p, q, r] in rad/s (use measurement from previous timestep)
          acc : Accelerometer measurement as [ax, ay, az] in m/s^2
          mag : Magnetometer measurement (unused, placeholder)
          dt  : Sampling interval in seconds
          
        Returns:
          Updated quaternion estimate and updated state vector.
        """
        # Extract the current state: [phi, theta, psi, bp, bq, br]
        phi, theta, psi, bp, bq, br = self.x
        
        # Use gyro measurements (assumed to be from the previous timestep)
        p = gyr[0]
        q = gyr[1]
        r = gyr[2]
        
        # Build the T6 matrix as in the Matlab implementation
        # Note: sec(theta) = 1 / cos(theta)
        T6 = np.array([
            [1.0, math.sin(phi) * math.tan(theta),  math.cos(phi) * math.tan(theta), -1.0, -math.sin(phi) * math.tan(theta), -math.cos(phi) * math.tan(theta)],
            [0.0, math.cos(phi),                   -math.sin(phi),                    0.0, -math.cos(phi),                    math.sin(phi)],
            [0.0, math.sin(phi) / math.cos(theta),  math.cos(phi) / math.cos(theta),  0.0, -math.sin(phi) / math.cos(theta), -math.cos(phi) / math.cos(theta)],
            [0.0, 0.0,                             0.0,                             0.0,  0.0,                             0.0],
            [0.0, 0.0,                             0.0,                             0.0,  0.0,                             0.0],
            [0.0, 0.0,                             0.0,                             0.0,  0.0,                             0.0]
        ])
        
        # Construct input vector: [p; q; r; bp; bq; br]
        u = np.array([p, q, r, bp, bq, br])
        
        # Compute time derivative of the state
        xdot = T6.dot(u)
        
        # Prediction step: Euler integration
        x_pred = self.x + dt * xdot
        
        # Process Model Jacobian F (6x6), computed using the original state values
        F = np.zeros((6, 6))
        
        # Row 1:
        F[0, 0] = 1.0 - dt * ( bq * math.cos(phi) * math.tan(theta) - q * math.cos(phi) * math.tan(theta)
                               - br * math.sin(phi) * math.tan(theta) + r * math.sin(phi) * math.tan(theta) )
        F[0, 1] = - ( dt * ( br * math.cos(phi) - r * math.cos(phi) + bq * math.sin(phi) - q * math.sin(phi) ) ) / (math.cos(theta)**2)
        F[0, 2] = 0.0
        F[0, 3] = -dt
        F[0, 4] = -dt * math.sin(phi) * math.tan(theta)
        F[0, 5] = -dt * math.cos(phi) * math.tan(theta)
        
        # Row 2:
        F[1, 0] = dt * ( br * math.cos(phi) - r * math.cos(phi) + bq * math.sin(phi) - q * math.sin(phi) )
        F[1, 1] = 1.0
        F[1, 2] = 0.0
        F[1, 3] = 0.0
        F[1, 4] = -dt * math.cos(phi)
        F[1, 5] = dt * math.sin(phi)
        
        # Row 3:
        F[2, 0] = - ( dt * ( bq * math.cos(phi) - q * math.cos(phi) - br * math.sin(phi) + r * math.sin(phi) ) ) / math.cos(theta)
        F[2, 1] = - ( dt * math.sin(theta) * ( br * math.cos(phi) - r * math.cos(phi) + bq * math.sin(phi) - q * math.sin(phi) ) ) / (math.cos(theta)**2)
        F[2, 2] = 1.0
        F[2, 3] = 0.0
        F[2, 4] = -dt * math.sin(phi) / math.cos(theta)
        F[2, 5] = -dt * math.cos(phi) / math.cos(theta)
        
        # Rows 4-6: Bias dynamics (identity)
        F[3, :] = [0, 0, 0, 1, 0, 0]
        F[4, :] = [0, 0, 0, 0, 1, 0]
        F[5, :] = [0, 0, 0, 0, 0, 1]
        
        # Covariance prediction:
        self.P = F.dot(self.P).dot(F.T) + self.Q
        
        # Measurement Update Step
        # Predicted accelerometer measurement from the predicted state x_pred:
        # h(x) = [ -g*sin(theta_pred);
        #           g*sin(phi_pred)*cos(theta_pred);
        #           g*cos(phi_pred)*cos(theta_pred) ]
        phi_pred = x_pred[0]
        theta_pred = x_pred[1]
        h_x = np.array([
            -self.g * math.sin(theta_pred),
             self.g * math.sin(phi_pred) * math.cos(theta_pred),
             self.g * math.cos(phi_pred) * math.cos(theta_pred)
        ])
        
        # Measurement residual (innovation)
        y_meas = np.array(acc) - h_x
        
 # Compute measurement Jacobian H (3x6) exactly as given in MATLAB:
        H = np.array([
            [0, -self.g * math.cos(theta_pred), 0, 0, 0, 0],
            [self.g * math.cos(phi_pred) * math.cos(theta_pred), -self.g * math.sin(phi_pred) * math.sin(theta_pred), 0, 0, 0, 0],
            [-self.g * math.sin(phi_pred) * math.cos(theta_pred), -self.g * math.cos(phi_pred) * math.sin(theta_pred), 0, 0, 0, 0]
           ])

        # Kalman Gain calculation:
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        
        # Update state estimate:
        self.x = x_pred + K.dot(y_meas)
        
        # Update covariance:
        self.P = (np.eye(6) - K.dot(H)).dot(self.P)
        
        # Convert the updated Euler angles to quaternion for the output.
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
    # Initialize IMU (using MPU9250/LSM9DS1)
    imu = lsm9ds1.LSM9DS1()
    imu.initialize()
    
    Ts = 0.005  # Sampling time in seconds
    num_samples = int(100 / Ts)  # Adjust as needed

    # Step 1: Calibration – Read a few gyro samples for bias estimation.
    calibration_samples = 200
    gyro_calib = []
    for i in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        gyro_calib.append(m9g)
        time.sleep(Ts)
    gyro_calib = np.array(gyro_calib)
    
    # Compute gyro bias (mean) and bias variance (using variance)
    gyro_mean = np.mean(gyro_calib, axis=0)         # [bias_x, bias_y, bias_z]
    gyro_var = np.var(gyro_calib, axis=0)         # variance for each axis
    
    # Set initial quaternion from calibration (assuming zero Euler angles)
    q0 = euler_to_quaternion(0, 0, 0)
    
    # Instantiate our custom EKF filter and initialize its biases
    custom_ekf = CustomEKF(gyro_mean, gyro_var, q0)

    # Precompute recursive low-pass filter coefficient
    filter_cutoff = 5  # Hz
    alpha = Ts / (Ts + (1 / (2 * np.pi * filter_cutoff)))
    
    ekf_phi_arr, ekf_theta_arr, ekf_psi_arr = [], [], []
    ekf_state_arr = []  # Save complete EKF state (including biases)
    sensor_acc_arr, sensor_gyro_arr, sensor_mag_arr, sim_time = [], [], [], []
    
    prev_acc = None #  
    prev_gyro = None # 
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
        m9a = filtered_acc.tolist()  # Replace m9a with filtered values
        m9g = filtered_gyro.tolist()  # Replace m9g with filtered values

        # Adjust gyro readings: if within [min, max] then use bias value
        gyro_tol = 0.1  # 10% tolerance
        for j in range(3):
            if abs(m9g[j] - gyro_mean[j]) < abs(gyro_mean[j]) * gyro_tol:
                m9g[j] = gyro_mean[j]

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
        
        print(f"Time: {i*Ts:.2f}s | EKF: Roll {np.rad2deg(phi):.3f}°, Pitch {np.rad2deg(theta):.3f}°, Yaw {np.rad2deg(psi):.3f}°, bx {(x_state[3]):.3f}°, by {(x_state[4]):.3f}°, bz {(x_state[5]):.10f}°")
        #print(f"Time: {i*Ts:.2f}s | EKF Cov: Roll {(p_cov[3,3]):.10f}°, Pitch {(p_cov[4,4]):.10f}°, Yaw {(p_cov[5,5]):.3f}°")
        
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
    #savemat(save_path, results)
    print(f"Results saved to {save_path}")
    
    # Compute and display standard deviation of EKF estimates
    std_phi = np.std(ekf_phi_arr)
    std_theta = np.std(ekf_theta_arr)
    std_psi = np.std(ekf_psi_arr)
    print(f"StD (ekf_est)| Roll: {np.rad2deg(std_phi):.5f}° Pitch: {np.rad2deg(std_theta):.5f}° Yaw: {np.rad2deg(std_psi):.5f}°")
