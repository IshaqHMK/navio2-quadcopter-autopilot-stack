# IH 2025
# version 7
# Note:  Madwick filter and corrected sampling time issue

# Standard Python Libraries
import time  # Time-related functions
import math  # Mathematical operations
from datetime import datetime  # Used to save data with timestamps

# Third-Party Libraries
import numpy as np  # Numerical computations
import scipy.io  # For saving data in MATLAB format
import serial  # Serial communication for the altitude sensor
import matplotlib.pyplot as plt  # Plotting library

# Add the project directory to the Python path (for relative imports)
import sys
import os
# This allows importing modules from the project directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Analysis Modules
from analysis.display_controller_metrics import display_controller_metrics  # Display RMSE and error metrics
from analysis.plot_results import plot_results  # Plot experimental results
from analysis.plot_results import plot_rate_results  # Plot rate control results

# Utility Modules for Navio2
from utils.navio2 import util  # Navio utility functions (e.g., hardware checks)
from utils.navio2 import mpu9250  # IMU sensor 1 (MPU9250)
from utils.navio2 import lsm9ds1_backup as lsm9ds1  # IMU sensor 2 (LSM9DS1)
from utils.navio2 import pwm  # Functions to send PWM signals to motors
from utils.navio2.leds import Led  # LED control for debugging or system state indication

# Add the imu directory to the Python path (for Madgwick filter and related modules)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imu'))
# IMU Modules
from imu.madgwick import Madgwick  # Madgwick filter for IMU orientation estimation
# Madgwick filter also relies on the following files: orientation.py, core.py, mathfuncs.py, constants.py
from imu.imu_utils import calibrate_imu, read_imu_data, quaternion_to_euler, euler_to_quaternion  # Utility functions for IMU calibration and data handling

# Altitude Sensor Utilities
from utils.sensors.altitude_utils import get_altitude_bias, read_lidar  # Functions to handle altitude data from lidar sensors

# from utils.navio2.pwm_utils import check_apm_and_initialize_pwm, send_speed_command  # Importing PWM functions from pwm_utils
from utils.matfile_utils import initialize_storage, save_to_matfile

from utils.data_processing import extract_and_display_all


# PWM Functions: 
def check_apm_and_initialize_pwm():
    """Checks for correct autopilot hardware and initializes PWM outputs."""
    #navio2.util.check_apm()
    util.check_apm()  # Use the updated path
    pwm_outputs = []
    for channel in PWM_OUTPUTS:
        pwm_out = pwm.PWM(channel)
        pwm_out.initialize()
        pwm_out.set_period(50)
        pwm_out.enable()
        pwm_outputs.append(pwm_out)
    return pwm_outputs

def send_pulse_length(pwm_outputs, pulse_lengths):
    # Sends specified pulse lengths to all motor ESCs.
    for i, pwm_out in enumerate(pwm_outputs):
        pwm_out.set_duty_cycle(pulse_lengths[i])
 
# Updated function
def send_speed_command(pwm_outputs, omega, MIN_PULSE_LENGTH, MAX_PULSE_LENGTH):
    # Convert omega (angular velocity) to PWM signals
    pwm_signals = omega_to_pwm(*omega)  # Unpack omega array to individual omega values
    
    # Sends specified pulse lengths (PWM signals) to all motor ESCs
    send_pulse_length(pwm_outputs, pwm_signals)

# New function to convert omega (angular velocity) to PWM signals
def omega_to_pwm(omega_1, omega_2, omega_3, omega_4):
    # Parameters for each motor (a_i and b_i)
    # a = np.array([0.731542, 0.744994, 0.751595, 0.726769])
    a = np.array([731.542497, 744.994274, 751.594551, 726.769124])
    b = np.array([-789.014715, -806.217581, -813.664533, -790.667045])

    # Calculate PWM signals based on the inverse of the mapping
    pwm_signals = (np.array([omega_1, omega_2, omega_3, omega_4]) - b) / a
    
    # Ensure that the PWM signals are within the allowed bounds
    pwm_signals = np.clip(pwm_signals, MIN_PULSE_LENGTH, MAX_PULSE_LENGTH)
    
    return pwm_signals



# Reference Functions: 
def reference_generator(time, initial_Z, initial_phi, initial_theta, initial_psi, signal=4):
    
    if signal == 1:
        T1 = 20  # Duration of the first transition
        
        T2 = 40  # Total duration until the second transition completes
        Z_mid = -0.1 #-0.5 # Mid altitude in meters
        Z_final = 0.0  # Final altitude in meters
        initial_Z = 0
            
        if time < T1:
            # First phase: from initial_Z to Z_mid
            Z_des_GF = initial_Z + (Z_mid - initial_Z) * 0.5 * (1 - np.cos(np.pi * time / T1))
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * time / T1))
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * time / T1))
        elif time < T2:
            # Second phase: from Z_mid to Z_final
            adjusted_time = time - T1
            duration = T2 - T1
            Z_des_GF = Z_mid + (Z_final - Z_mid) * 0.5 * (1 - np.cos(np.pi * adjusted_time / duration))
            phi_des = 0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des = 0 * (np.pi / 180)  # Remain at 0 degre
        else:
            Z_des_GF = Z_final
            phi_des = 0 * (np.pi / 180)  
            theta_des = 0 * (np.pi / 180)  
        psi_des =  0 * (np.pi / 180)  
        
        
    elif signal == 2:  ## Smooth transition using half sinusoid
        smooth_factor = 1
        T = 10
        if time < T: 
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * time / T))
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * time / T))
            psi_des = initial_psi * 0.5 * (1 + np.cos(np.pi * time / T))
        elif time < 20:
            phi_des =    0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =  0 * (np.pi / 180)  # Remain at 0 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        elif time < 40:
            phi_des =       5 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =     0 * (np.pi / 180)  # Remain at 5 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        elif time < 60:
            phi_des =    0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =  0 * (np.pi / 180)  # Remain at 0 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        elif time < 80:
            phi_des =      -5 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =    -0 * (np.pi / 180)  # Remain at -5 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        elif time < 100:
            phi_des =    0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =  0 * (np.pi / 180)  # Remain at 0 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        elif time < 120:
            phi_des =       0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =     5 * (np.pi / 180)  # Remain at 5 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        elif time < 140:
            phi_des =    0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =  0 * (np.pi / 180)  # Remain at 0 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        elif time < 160:
            phi_des =      -0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =    -5 * (np.pi / 180)  # Remain at -5 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        elif time < 180:
            phi_des =    0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =  0 * (np.pi / 180)  # Remain at 0 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        else:
            phi_des =    0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        Z_des_GF = 0
        
    elif signal == 3:  ## Smooth transition using half sinusoid
        smooth_factor = 1
        T1 = 3
        T2 = 20+T1
        T3 = 60  
        T4 = 110
        f_theta = 0.1
        f_Z = 0.05
        initial_Z = 0 
        Z_amp = -0.1 #-0.5 #  altitude in meters
        Z_final = 0.0  # Final altitude in meters
        
        if time < T1: 
            Z_des_GF = initial_Z  
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * time / T1))
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * time / T1))
            psi_des = initial_psi * 0.5 * (1 + np.cos(np.pi * time / T1))
        elif time < T2:
            #Z_des_GF = # goes from initial_Z  to Z_amp and then back to initial_Z maybe use (1- cos).. for this and  f_Z = 0.1 Hz
            Z_des_GF = initial_Z + Z_amp * (1 - np.cos(2 * np.pi * f_Z * (time - T1))) / 2
            phi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =  0 * (np.pi / 180)  # Remain at 0 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
        elif time < T3:
            Z_des_GF = 0 
            phi_des =  0 * np.sin(2 * np.pi * f_theta * (time - T2)) * (np.pi / 180)   
            theta_des = 0 * np.sin(2 * np.pi * f_theta * (time - T2)) * (np.pi / 180)
            psi_des =  0 * (np.pi / 180)   
        elif time < T4:
            Z_des_GF = 0
            phi_des =  0 * np.sin(2 * np.pi * f_theta * (time - 20)) * (np.pi / 180)   
            theta_des = 0 * np.sin(2 * np.pi * f_theta * (time - 20)) * (np.pi / 180)
            psi_des =  0 * (np.pi / 180)  
        else:
            Z_des_GF = 0
            phi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des =  0 * (np.pi / 180)  # Remain at 0 degrees
            psi_des =  0 * (np.pi / 180)  # Remain at 0 degrees
            
    elif signal == 4:  ## Smooth transition using half sinusoid
        smooth_factor = 1
        T1 = 3
        T2 = T1 +3
        T3 = T2 + 20  
        T4 = T3 + 3
        f_theta = 0.1
        initial_Z = 0 
        Z_amp = -0.05  # Altitude in meters
        Z_final = 0.0  # Final altitude in meters

        if time < T1: 
            # Initial smooth transition to hover state
            Z_des_GF = initial_Z  
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * time / T1))
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * time / T1))
            psi_des = initial_psi * 0.5 * (1 + np.cos(np.pi * time / T1))
        elif time < T2:
            # Transition from initial_Z to Z_amp
            Z_des_GF = initial_Z + 0.5 * (1 - np.cos(np.pi * (time - T1) / (T2 - T1))) * (Z_amp - initial_Z)
            phi_des = 0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des = 0 * (np.pi / 180)  # Remain at 0 degrees
            psi_des = 0 * (np.pi / 180)  # Remain at 0 degrees
        elif time < T3:
            # Maintain altitude at Z_amp
            Z_des_GF = Z_amp 
            phi_des = 0 * np.sin(2 * np.pi * f_theta * (time - T2)) * (np.pi / 180)
            theta_des = 0 * np.sin(2 * np.pi * f_theta * (time - T2)) * (np.pi / 180)
            psi_des = 0 * (np.pi / 180)   
        elif time < T4:
            # Transition from Z_amp back to initial_Z
            Z_des_GF = Z_amp - 0.5 * (1 - np.cos(np.pi * (time - T3) / (T4 - T3))) * (Z_amp - initial_Z)
            phi_des = 0 * np.sin(2 * np.pi * f_theta * (time - T3)) * (np.pi / 180)
            theta_des = 0 * np.sin(2 * np.pi * f_theta * (time - T3)) * (np.pi / 180)
            psi_des = 0 * (np.pi / 180)
        else:
            # End of maneuver, stabilize at initial_Z
            Z_des_GF = initial_Z
            phi_des = 0 * (np.pi / 180)  # Remain at 0 degrees
            theta_des = 0 * (np.pi / 180)  # Remain at 0 degrees
            psi_des = 0 * (np.pi / 180)  # Remain at 0 degrees




    elif signal == 5:  ## Smooth transition using half sinusoid
        T = 3  # Duration for the half sinusoid
        deltaTime = 10  # Duration for each step in the repeating sequence
        pattern = [0, 5, 0, -5]  # Repeating pattern values in degrees
        if time < T:
            # Half-sinusoidal wave for the first 10 seconds
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * time / T))
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * time / T))
        else:
            # Time since the half-sinusoid completed
            time_since_half_sinusoid = time - T
            # Find the index in the pattern based on how many deltaTime periods have passed
            index = int((time_since_half_sinusoid // deltaTime) % len(pattern))
            # Get the desired degree from the pattern and convert to radians
            current_degree = pattern[index] * (np.pi / 180)
            phi_des = current_degree
            theta_des = current_degree
        
        Z_des_GF = 0
        psi_des = 0
        
        
         
        
    return Z_des_GF, phi_des, theta_des, psi_des

# To tune p, q, r
def rate_reference_generator(time):
    # Time thresholds
    T1 = 5
    T2 = 40
    
    # Constants
    deg_ps_to_rad_ps = np.pi / 180  # Conversion from degree/s to rad/s
    
    # Sinusoidal parameters
    Amp = 2 * deg_ps_to_rad_ps  # Amplitude in radians per second
    f_pqr = 0.1  # Frequency in Hz
    
    # Generate the reference rates
    if time < T1:
        # First phase: 0 deg/s (static)
        p_desired = 0
        q_desired = 0
        r_desired = 0
    elif time < T2:
        # Second phase: Sinusoidal roll rate, 0 for others
        p_desired = 0*Amp * np.sin(2 * np.pi * f_pqr * time)
        q_desired = Amp * np.sin(2 * np.pi * f_pqr * time)
        r_desired = 0
    else:
        # Third phase: 0 deg/s (static)
        p_desired = 0
        q_desired = 0
        r_desired = 0
    
    return p_desired, q_desired, r_desired

# Conrol Functions:
def attitude_PID(Z_des_GF, phi_des, theta_des, psi_des, Z_meas, phi_meas, theta_meas, psi_meas, p_meas, q_meas, r_meas, 
                 Z_KP, Z_KI, Z_KD, Phi_KP, Phi_KI, Phi_KD, Theta_KP, Theta_KI, Theta_KD, Psi_KP, Psi_KI, Psi_KD, Ts, m, g):
    global z_error_sum, phi_error_sum, theta_error_sum, psi_error_sum
    global previous_z_error, previous_phi_error, previous_theta_error, previous_psi_error

    # Z Position PID Controller/Altitude Controller
    z_error = Z_des_GF - Z_meas
    z_error_sum += z_error * Ts
    z_error_dot = (z_error - previous_z_error) / Ts

    cp = Z_KP * z_error
    ci = Z_KI * z_error_sum
    cd = Z_KD * z_error_dot
    #U1 = -(cp + ci + cd) / (np.cos(theta_meas) * np.cos(phi_meas)) + (m * g) / (np.cos(theta_meas) * np.cos(phi_meas))
    # U1 = -(cp + ci + cd) / (np.cos(theta_meas) * np.cos(theta_meas)) + (m * g) / (np.cos(theta_meas) * np.cos(phi_meas))
    U1 = -(cp + ci + cd)  
    #U1 = -(cp + ci + cd) + (m * g) 
    U1 = min(U1_max, max(U1_min, U1))  # Ensure U1 is between U1_min and U1_max
    
    previous_z_error = z_error
    
    # Roll PID Controller
    phi_error = phi_des - phi_meas
    phi_error_sum += phi_error * Ts
    phi_error_dot = (phi_error - previous_phi_error) / Ts

    cp = Phi_KP * phi_error
    ci = Phi_KI * phi_error_sum
    cd = Phi_KD * phi_error_dot
    p_desired = cp + ci + cd

    previous_phi_error = phi_error

    # Pitch PID Controller
    theta_error = theta_des - theta_meas
    theta_error_sum += theta_error * Ts
    theta_error_dot = (theta_error - previous_theta_error) / Ts

    cp = Theta_KP * theta_error
    ci = Theta_KI * theta_error_sum
    cd = Theta_KD * theta_error_dot
    q_desired = cp + ci + cd
    
    previous_theta_error = theta_error

    # Yaw PID Controller
    psi_error = psi_des - psi_meas
    psi_error_sum += psi_error * Ts
    psi_error_dot = (psi_error - previous_psi_error) / Ts

    cp = Psi_KP * psi_error
    ci = Psi_KI * psi_error_sum
    cd = Psi_KD * psi_error_dot
    r_desired = cp + ci + cd

    previous_psi_error = psi_error

    return U1, p_desired, q_desired, r_desired

def rate_PID(p_desired, q_desired, r_desired, p_meas, q_meas, r_meas, 
             P_KP, P_KI, P_KD, Q_KP, Q_KI, Q_KD, R_KP, R_KI, R_KD, Ts, PQR_PID_Enable):
    global p_error_sum, q_error_sum, r_error_sum
    global previous_p_error, previous_q_error, previous_r_error

    if PQR_PID_Enable:
        # Roll Rate PID Controller
        p_error = p_desired - p_meas
        p_error_sum += p_error * Ts
        p_error_dot = (p_error - previous_p_error) / Ts

        cp = P_KP * p_error
        ci = P_KI * p_error_sum
        cd = P_KD * p_error_dot
        U2 = cp + ci + cd

        previous_p_error = p_error

        # Pitch Rate PID Controller
        q_error = q_desired - q_meas
        q_error_sum += q_error * Ts
        q_error_dot = (q_error - previous_q_error) / Ts

        cp = Q_KP * q_error
        ci = Q_KI * q_error_sum
        cd = Q_KD * q_error_dot
        U3 = cp + ci + cd

        previous_q_error = q_error

        # Yaw Rate PID Controller
        r_error = r_desired - r_meas
        r_error_sum += r_error * Ts
        r_error_dot = (r_error - previous_r_error) / Ts

        cp = R_KP * r_error
        ci = R_KI * r_error_sum
        cd = R_KD * r_error_dot
        U4 = cp + ci + cd

        previous_r_error = r_error

    else:
        # Direct Control (PID Disabled)
        U2 = p_desired
        U3 = q_desired
        U4 = r_desired
        
        # Apply limits
        U2 = min(U2_max, max(U2_min, U2))  
        U3 = min(U3_max, max(U3_min, U3))  
        U4 = min(U4_max, max(U4_min, U4))  
    
    return U2, U3, U4

def motor_speed(U1, U2, U3, U4, KT, Kd, l, max_motor_speed, min_motor_speed):
    
    # Calculate motor speeds (rad/s)^2
    w1 = U1 / (4 * KT) + U3 / (2 * KT * l) + U4 / (4 * Kd)
    w2 = U1 / (4 * KT) - U2 / (2 * KT * l) - U4 / (4 * Kd)
    w3 = U1 / (4 * KT) - U3 / (2 * KT * l) + U4 / (4 * Kd)
    w4 = U1 / (4 * KT) + U2 / (2 * KT * l) - U4 / (4 * Kd)
    
    # Apply realistic motor speed limits
    max_speed_squared = max_motor_speed ** 2
    min_speed_squared = min_motor_speed ** 2
    
    
    w1 = min(max(w1, min_speed_squared), max_speed_squared)
    w2 = min(max(w2, min_speed_squared), max_speed_squared)
    w3 = min(max(w3, min_speed_squared), max_speed_squared)
    w4 = min(max(w4, min_speed_squared), max_speed_squared)

    # Compute motor speeds
    omega_1 = np.sqrt(w1)  # Front Motor
    omega_2 = np.sqrt(w2)  # Right Motor
    omega_3 = np.sqrt(w3)  # Rear Motor
    omega_4 = np.sqrt(w4)  # Left Motor

    return omega_1, omega_2, omega_3, omega_4

# Safety Exit: 
# def on_press(key):
    global exit_flag
    if key == Key.space:  # If the space bar is pressed
        exit_flag = True
        
# PID Tuning Function 2: using keys
# def create_adjustment_handler(param_name, initial_value, increment):
    params = {param_name: initial_value}

    def adjust_parameter(key):
        if key == Key.up:
            params[param_name] += increment
            # print(f"{param_name} increased to {params[param_name]:.2f}")
        elif key == Key.down:
            params[param_name] -= increment
            # print(f"{param_name} decreased to {params[param_name]:.2f}")

        return params[param_name]

    return adjust_parameter
    
# PID Tuning Function 1:
def linear_time_variant(val_min, val_max, t_min, t_max, current_time):
    """
    Linearly changes theta_KD from val_min to val_max between t_min and t_max.

    Args:
    val_min (float): Initial value of theta_KD at t_min.
    val_max (float): Final value of theta_KD at t_max.
    t_min (float): Time at which the value starts changing.
    t_max (float): Time at which the value stops changing.
    current_time (float): The current simulation time.

    Returns:
    float: The value of theta_KD at the current time.
    """
    if current_time < t_min:
        return val_min
    elif current_time > t_max:
        return val_max
    else:
        slope = (val_max - val_min) / (t_max - t_min)
        return val_min + slope * (current_time - t_min)

# ------------------------------------------------------------------- #
# Initialization ------------------------------------------------------
# ------------------------------------------------------------------- #

# Sensors Setup:
# 1. Altitude Sensor Setup
ser = serial.Serial("/dev/serial0", 115200, timeout=1)  # Setup serial connection
# 2. Initialize the MPU9250 IMU sensor
imu = mpu9250.MPU9250()
# (alt) Initialize the LSM9DS1 IMU sensor
#imu = lsm9ds1.LSM9DS1()
imu.initialize()

# ------------------------------------------------------------------- #

# Initialize LED 
led = Led()
led.setColor('Green')

# ------------------------------------------------------------------- #
 
# PWM Settings
PWM_OUTPUTS = [0, 1, 2, 3]  # Assign PWM Motor Pins
# Calibrated pulse lengths and speed bounds (DO NOT CHANGE)
MIN_PULSE_LENGTH = np.array([1.140, 1.142, 1.141, 1.152]) # (DO NOT CHANGE)
MAX_PULSE_LENGTH = np.array([2.0, 2.0, 2.0, 2.0]) # (DO NOT CHANGE)
 
# Maximum Speeds in rad/s
MIN_OMEGA = np.array([30, 30, 30, 30]) # rad/s corresponds to 286.4789 rpm
MAX_OMEGA = np.array([526.4262, 526.4262, 526.4262, 526.4262])  # rad/s corresponds to 5027 rpm (based on new propeller)

NOSPEED_PULSE_LENGTHS = np.array([1.0, 1.0, 1.0, 1.0]) 

# ------------------------------------------------------------------- #

# Quadrotor Physical Parameters
m = 1 #2.1680   # Quadrotor mass(kg)
g = 9.80665  # Gravity (m/s^2)
l = .25 #    # Distance from the center of mass to the each motor(m)
KT = 0.000021;    # Thrust force coeffecient (kg-m) (T-motor 12 inch prop)
Kd = 1.3858e-6;    # Drag torque coeffecient (kg-m^2) (SHOULD BE CHANGED)
max_motor_speed_rpm = 5027  # rpm ~~ 14.8V x 800 rpm/V

# -------------------------------------------------------------------
# KV = 800  # RPM per volt for the DJI 2312 motor
# Calculate the torque constant Kt in Nm/A
# Kt_nm_per_a = 60 / (2 * math.pi * KV)
# Conversion factor from Nm to kg.m
# conversion_factor_nm_to_kgm = 0.102
# Calculate Kt in kg.m
# KT = Kt_nm_per_a * conversion_factor_nm_to_kgm
# -------------------------------------------------------------------

min_motor_speed = MIN_OMEGA[0]  # rad/s also (motors lower limit (can't spin in reverse))
max_motor_speed = MAX_OMEGA[0]  # 1.3650  is max # rad/s   950.43 rad/s (original)

print(f"Max Motor Speed: {max_motor_speed:.2f} rad/s") # MIN_OMEGA[0]

# ------------------------------------------------------------------- #
# Defining limits for control inputs
U1_max = KT * 4 * max_motor_speed ** 2
U1_min =  KT * 4 * min_motor_speed ** 2
U2_max = KT * l * max_motor_speed ** 2
U2_min = -KT * l * max_motor_speed ** 2   
U3_max = KT * l * max_motor_speed ** 2
U3_min = -KT * l * max_motor_speed ** 2  # Same assumption as U2
U4_max = Kd * 2 * max_motor_speed ** 2
U4_min = -Kd * 2 * max_motor_speed ** 2  

print(f"Min Control Input U1: {U1_min:.2f} and Max Control Input U1: {U1_max:.2f}  ")  
print(f"Min Control Input U2: {U2_min:.2f} and Max Control Input U2: {U2_max:.2f}  ")  
print(f"Min Control Input U3: {U3_min:.2f} and Max Control Input U3: {U3_max:.2f}  ")  
print(f"Min Control Input U4: {U4_min:.2f} and Max Control Input U4: {U4_max:.2f}  ") 
 
# ---------------------- PID Parameters  ---------------------- #
Z_KP, Z_KI, Z_KD =  0, 0, 0
Phi_KP, Phi_KI, Phi_KD = 0, 0, 0   
Theta_KP, Theta_KI, Theta_KD =  0, 0, 0   
Psi_KP, Psi_KI, Psi_KD = 0, 0, 0

PQR_PID_Enable = False
P_KP, P_KI, P_KD = 0, 0, 0  
Q_KP, Q_KI, Q_KD = 0, 0, 0  
R_KP, R_KI, R_KD = 0, 0, 0

# ---------------------- Simulation settings ---------------------- #
Ts = 0.005  # Sampling time in seconds (increased to 10ms)
total_simulation_time = 10 # Simulation time in seconds  

# ---------------------- Calibration Process ---------------------- #
calibration_samples = 10
# get IMU bias 
bx, by, bz, phi_bias, theta_bias, m9a, m9g, m9m = calibrate_imu(imu, calibration_samples, Ts)
# print and check if it match our system: 
print(f"Phi (Acc): {np.rad2deg(phi_bias):.2f}°, Theta (Acc): {np.rad2deg(theta_bias):.2f}°")
# ---------------------- Madgwick Filter Parameter ---------------------- #
#madgwick = Madgwick(beta=0.1)  # Adjust the beta value as needed for responsiveness
# Ensure that imu data is an Nx3 array where each row is a 3D magnetometer sample
m9a = np.array(m9a).reshape(-1, 3)  # Reshape to (N, 3) if necessary
m9g = np.array(m9g).reshape(-1, 3)  # Reshape to (N, 3) if necessary
# m9m = np.array(m9m).reshape(-1, 3)  # Reshape to (N, 3) if necessary

# Initialize Madgwick filter without magnetometer data
# madgwick = Madgwick(gyr=m9g, acc=m9a, frequency=100.0)   

# Initialize Madgwick filter with magnetometer data
#madgwick = Madgwick(gyr=m9g, acc=m9a, mag=m9m, frequency=100.0)

# from source code: beta = 0.033, 0.041, 0.05, 0.06, up to 0.1 
# Madgwick(gyr=gyro_data, acc=acc_data, mag=mag_data, gain_imu=0.01, gain_marg=0.05)
# madgwick = Madgwick(gyr=m9g, acc=m9a, mag=m9m, gain_imu=0.033, gain_marg=0.05)

# Begin with the typical beta value (0.033 for IMU or 0.041 for MARG)
#madgwick_stable = Madgwick(gyr=m9g, acc=m9a, mag=m9m, gain=0.033, beta=0.033)

# madgwick = Madgwick(gyr=m9g, acc=m9a, mag=m9m, gain=0.05, beta=0.1, frequency=1/Ts)
 
# Initialize Madgwick filter with custom parameters
madgwick = Madgwick(gyr=m9g, acc=m9a, gain=0.05, beta=0.1, frequency=1/Ts)  # Adjust parameters here
# madgwick = Madgwick(gyr=m9g, acc=m9a)

 
# gain: Increase the gain if you need faster response to changes in orientation, 
# but lower it if you want a more stable, smoother output in noisy environments.
  
# beta: Increase beta to make the filter more sensitive to gyroscope 
# errors and quicker to react, or decrease it to smooth out the response 
# and reduce sensitivity to sensor noise.


# ---------------------- Measure altitude bias ---------------------- #
alt_bias = get_altitude_bias(calibration_samples, ser)

# Define global variables for persistent storage
z_error_sum = 0
phi_error_sum = 0
theta_error_sum = 0
psi_error_sum = 0
previous_z_error = 0
previous_phi_error = 0
previous_theta_error = 0
previous_psi_error = 0
p_error_sum = 0
q_error_sum = 0
r_error_sum = 0
previous_p_error = 0
previous_q_error = 0
previous_r_error = 0

# ------------------------------------------------------------------- #

def main():

    # --------- initialize PWM --------- #
    pwm_outputs = check_apm_and_initialize_pwm()
    
    # --------- initialize altitude related variables --------- #
    last_valid_altitude = read_lidar(ser) - alt_bias  # Initialize with 0 m
    delta_altitude_threshold = 0.15  # Define a threshold for jerk detection (delta m)
    
    # --------- initialize state variables for altitude integration --------- # (IP - check)
    Z_dot_est = 0.0         # Initial vertical speed
    Z_est = 0.0             # Initial position
    alpha_dot_est = 0.25    # Complementary filter coefficient for altitude speed
    
    # --------- initialize IMU related variables --------- #
    X_ddot_cur = Y_ddot_cur = Z_ddot_cur=  0
    mx = my = mz = 0
    p_cur = q_cur = r_cur = 0
    phi_mf = theta_mf = psi_mf = 0
    phi_acc_cur = theta_acc_cur = 0
    phi_gyro_cur = theta_gyro_cur = psi_gyro_cur = 0
    prev_phi_gyro = prev_theta_gyro = prev_psi_gyro = 0

    # --------- initialize bias phi theta angles as current state --------- #
    prev_phi = phi_bias
    prev_theta = theta_bias
    prev_psi = 0
    
    # --------- initialize madgwick related variables --------- #
    # Convert current angles to quaternion
    initial_quaternion = euler_to_quaternion(prev_phi, prev_theta, prev_psi)
    # Set the initial quaternion
    madgwick.q0 = initial_quaternion
    
    # LED Color white:  all initialization completed
    led.setColor('White')  # (0, 1, 0)
    
    # ---------------------------------------------------------------
    # Send no-speed pulse lengths for 5 seconds before the experiment
    # ---------------------------------------------------------------
    print("Starting Experiment in 5 seconds...")
    start_no_speed_time = time.time()
    while time.time() - start_no_speed_time < 5:
        send_pulse_length(pwm_outputs, NOSPEED_PULSE_LENGTHS)
        
        # ---------- measure altitude (before experimenting) ----------
        Z_meas = 0 #read_lidar(ser)  
        if Z_meas is not None:
            Z_meas -= alt_bias  # Subtract bias to make initial point exactly 0
            if abs(Z_meas - last_valid_altitude) <= delta_altitude_threshold:
                last_valid_altitude = Z_meas  # Update last valid altitude
        else:
            Z_meas = last_valid_altitude
        
        # ---------- measure IMU (before experimenting) ----------
        X_ddot_cur, Y_ddot_cur, Z_ddot_cur, mx, my, mz, p_cur, q_cur, r_cur, phi_mf, theta_mf, psi_mf, phi_acc_cur, theta_acc_cur, phi_gyro_cur, theta_gyro_cur, psi_gyro_cur, q = read_imu_data(
            imu, Ts, prev_phi_gyro, prev_theta_gyro, prev_psi_gyro, madgwick.q0, madgwick)
        #
        prev_phi_gyro = phi_gyro_cur
        prev_theta_gyro = theta_gyro_cur
        prev_psi_gyro = psi_gyro_cur

        # assign previous measurements for IMU madgwick processing:
        madgwick.q0 = q
        #
        # Interchange & assign to match our confg. and process (does NOT involve MF):  
        X_ddot_meas, Y_ddot_meas, Z_ddot_meas = X_ddot_cur, Y_ddot_cur, Z_ddot_cur
        p_meas, q_meas, r_meas = p_cur, q_cur, r_cur
        phi_meas, theta_meas, psi_meas = phi_mf, theta_mf, psi_mf
        phi_gyro, theta_gyro = phi_gyro_cur, theta_gyro_cur 
        phi_acc, theta_acc = phi_acc_cur, theta_acc_cur
        # Note: No Interchanging needed! 7-1-2025
        # -----------------------------
        time.sleep(Ts)
        
    # Set Initial Theta Value (exp will start from current measured attitude and altitude)
    initial_Z = Z_meas # in meters
    initial_phi = phi_meas # in radian 
    initial_theta = theta_meas # in radian 
    initial_psi = psi_meas # in radian
    
    print(f"Initial IMU Measurments: X_ddot:  {X_ddot_meas:.2f}, Y_ddot {Y_ddot_meas:.2f}, Z_ddot {Z_ddot_meas:.2f} | "
        f"Phi: {np.rad2deg(phi_meas):.2f}°, Theta: {np.rad2deg(theta_meas):.2f}°, Psi: {np.rad2deg(psi_meas):.2f}°. ")
        
    # initialize arrays to store mesurements: 
    num_samples = int(total_simulation_time / Ts)
    storage = initialize_storage(num_samples)

    # --------------------------------------------------------------- #
    # --------------------------------------------------------------- #

    start_time = time.perf_counter()  # High-resolution timer for simulation time
    last_time = 0  # Initialize last_time for execution timing

    # Color red: Experiment started
    led.setColor('Red')  
    
    for i in range(num_samples):
        
        # Get the current time:
        current_time = time.perf_counter()  

        sim_time = current_time - start_time # sim time starts from 0
        
        val_min = 0.9    
        val_max = 1.12  
        time_min = 6
        time_max = total_simulation_time 
        # Theta_KD = adjust_gain(None) # for small incremental tuning
        # Phi_KD= linear_time_variant(val_min, val_max, time_min, time_max, sim_time)
        LinearTuned = Phi_KD
        
        # --------------------------------------------------------------- #
        # Step 1: measure altitude
        Z_meas = read_lidar(ser)
        if Z_meas is not None:
            Z_meas -= alt_bias  # Subtract bias to make initial point exactly 0
            if abs(Z_meas - last_valid_altitude) <= delta_altitude_threshold:
                last_valid_altitude = Z_meas  # Update last valid altitude
        else:
            # Use previous measurement if no new data
            Z_meas = last_valid_altitude

        # (IP) Use a complementary filter for vertical speed and position
        Z_dot_est = alpha_dot_est * (Z_dot_est + (Z_ddot_meas - g) * Ts) + (1 - alpha_dot_est) * ((Z_meas - Z_est) / Ts)
        Z_est += Z_dot_est * Ts
        
        # Step 2: measure imu     
        X_ddot_cur, Y_ddot_cur, Z_ddot_cur, mx, my, mz, p_cur, q_cur, r_cur, phi_mf, theta_mf, psi_mf, phi_acc_cur, theta_acc_cur, phi_gyro_cur, theta_gyro_cur, psi_gyro_cur, q = read_imu_data(
            imu, Ts, prev_phi_gyro, prev_theta_gyro, prev_psi_gyro, madgwick.q0, madgwick)
        #
        prev_phi_gyro = phi_gyro_cur
        prev_theta_gyro = theta_gyro_cur
        prev_psi_gyro = psi_gyro_cur
        #
        # assign previous measurements for IMU madgwick processing:
        madgwick.q0 = q
        #        
        # Interchange if needed to match our confg. and process (does NOT involve MF): 
        X_ddot_meas, Y_ddot_meas, Z_ddot_meas = X_ddot_cur, Y_ddot_cur, Z_ddot_cur
        p_meas, q_meas, r_meas = p_cur, q_cur, r_cur
        phi_meas, theta_meas, psi_meas = phi_mf, theta_mf, psi_mf
        phi_gyro, theta_gyro = phi_gyro_cur, theta_gyro_cur 
        phi_acc, theta_acc = phi_acc_cur, theta_acc_cur
        # --------------------------------------------------------------- #
        # Step 3. Define Reference Signals 
        Z_des_GF, phi_des, theta_des, psi_des = reference_generator(sim_time, initial_Z, initial_phi, initial_theta, initial_psi)
        
        # Step 4. Implement Attitude Controller 
        U1, p_desired, q_desired, r_desired = attitude_PID(Z_des_GF, phi_des, theta_des, psi_des, Z_meas, phi_meas, theta_meas, psi_meas, p_meas, q_meas, r_meas, Z_KP, Z_KI, Z_KD, Phi_KP, Phi_KI, Phi_KD, Theta_KP, Theta_KI, Theta_KD, Psi_KP, Psi_KI, Psi_KD, Ts, m, g)
        
        # Tune Rate Controller: (comment when done)
        # p_desired, q_desired, r_desired = rate_reference_generator(sim_time) # COMMENTED
        
        # Step 5. Implement Rate Controller (if enabled)
        U2, U3, U4 = rate_PID(p_desired, q_desired, r_desired, p_meas, q_meas, r_meas, P_KP, P_KI, P_KD, Q_KP, Q_KI, Q_KD, R_KP, R_KI, R_KD, Ts, PQR_PID_Enable)

        # Step 6. Calculate Desired Motor Speeds in rad/s
        omega_1, omega_2, omega_3, omega_4  = motor_speed(U1, U2, U3, U4, KT, Kd, l, max_motor_speed, min_motor_speed)

        # Step 5. Send PWM signals corresponding to the calculated angular velocities
        omega = np.array([omega_1, omega_2, omega_3, omega_4])
        send_speed_command(pwm_outputs, omega, MIN_PULSE_LENGTH, MAX_PULSE_LENGTH)

        # Store current data: populate storage dictionary
        storage['sim_times'][i] = sim_time
        storage['omegas'][i, :] = omega
        storage['altitudes'][i] = [Z_meas, Z_est, Z_dot_est]
        storage['acc_data'][i] = [X_ddot_meas, Y_ddot_meas, Z_ddot_meas]
        storage['gyro_data'][i] = [p_meas, q_meas, r_meas]
        storage['mag_data'][i] = [mx, my, mz]
        storage['attitude_data'][i] = [phi_meas, theta_meas, psi_meas]
        storage['control_input_data'][i] = [U1, U2, U3, U4]
        storage['reference_data'][i] = [Z_des_GF, phi_des, theta_des, psi_des]
        storage['rate_reference_data'][i] = [p_desired, q_desired, r_desired]
        
        # Measure execution time for this iteration
        execution_time = time.perf_counter() - current_time
        
        # Display on terminal
        print(f"ET: {execution_time:.4f}s, ST: {sim_time:.2f}s | Z: {Z_des_GF:.2f}m  / {(Z_meas):.2f}m| "
        f"Phi: {np.rad2deg(phi_des):.2f}° / {np.rad2deg(phi_meas):.2f}° | Theta: {np.rad2deg(theta_des):.2f}° / {np.rad2deg(theta_meas):.2f}° | Psi: {np.rad2deg(psi_des):.2f}° / {np.rad2deg(psi_meas):.2f}° || "
        f"w1: {omega_1:.0f} rad/s, w3: {omega_3:.0f} rad/s | w2: {omega_2:.0f} rad/s, w4: {omega_4:.0f} rad/s, LinearTuned: {LinearTuned:.2f}   ")
        
        # --------------------------------------------------------------- #
        # Future TO DO: Safety Stop: check if space bar is pressed for exit
        # --------------------------------------------------------------- #
        
        # Delay for precise sampling time
        next_time = current_time + Ts
        while time.perf_counter() < next_time:
          pass
        # ----------------------- end of for loop ----------------------- #
            
    # Experiment ended - Blue Color LED
    led.setColor('Blue')  # (1, 1, 0)
    
    # --------- save controller gains --------- #
    gains = {
        'Z_PID_Gains': {'Z_KP': Z_KP, 'Z_KI': Z_KI, 'Z_KD': Z_KD},
        'Phi_PID_Gains': {'Phi_KP': Phi_KP, 'Phi_KI': Phi_KI, 'Phi_KD': Phi_KD},
        'Theta_PID_Gains': {'Theta_KP': Theta_KP, 'Theta_KI': Theta_KI, 'Theta_KD': Theta_KD},
        'Psi_PID_Gains': {'Psi_KP': Psi_KP, 'Psi_KI': Psi_KI, 'Psi_KD': Psi_KD},
        'P_PID_Gains': {'P_KP': P_KP, 'P_KI': P_KI, 'P_KD': P_KD},
        'Q_PID_Gains': {'Q_KP': Q_KP, 'Q_KI': Q_KI, 'Q_KD': Q_KD},
        'R_PID_Gains': {'R_KP': R_KP, 'R_KI': R_KI, 'R_KD': R_KD}
    }

    # --------- save data to .mat file --------- #
    save_to_matfile(storage, gains, path_prefix='/home/pi/Documents/Quadcopter_Control_v2/data/results/quad_exp_J_on_Y')
    # --------- saving data completed --------- #
    led.setColor('Green')  # (1, 1, 1)
    
    print("Experiment completed. Setting zero speed.")
    # Send no-speed pulse lengths for 2 seconds after the experiment
    start_no_speed_time = time.time()
    while time.time() - start_no_speed_time < 5:
        send_pulse_length(pwm_outputs, NOSPEED_PULSE_LENGTHS)
        time.sleep(Ts)  # Maintain timing control
    
    # Extract and display all saved variables from the storage dictionary
    extract_and_display_all(storage, time_min, time_max, Ts)

    # Shutting down processes
    led.setColor('Black')  # (1, 1, 1)
    # PWM shut down
    for pwm_out in pwm_outputs:
       pwm_out.disable()   
       
    ser.close()  # Close serial port
    
if __name__ == "__main__":
    main()
    
 
