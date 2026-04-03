# IH 2025
# version 21 is the main version
# Note: copied from version 21  
# here we replace madwick with our custom ekf 
# ekf_test_6_april_2024_LSM_T01VFT2.py
# adaptive grad

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
from utils import system_utils 


# New Feedback Linearization Function  
def feedback_linearization_control(U1_star, U2_star, U3_star, U4_star, 
                                   zdot, p, q, r, theta_meas, phi_meas, gyroscopic_term_omega, 
                                   params, K1, K2, K3, K4):
    """
    Transforms the raw control inputs (U1_star, U2_star, U3_star, U4_star) into actual control inputs using feedback linearization.
    
    Args:
        U1_star, U2_star, U3_star, U4_star: Desired control inputs.
        zdot: Measured vertical velocity.
        p, q, r: Measured angular velocities.
        gyroscopic_term_omega: Gyroscopic term.
        params: Dictionary containing mass (m) and inertias (Jx, Jy, Jz, Jp).
        K1, K2, K3, K4: Feedback gains for altitude, roll, pitch, and yaw.

    Returns:
        U1, U2, U3, U4: Transformed control inputs.
    """
    m = params['m']
    g = params['g']
    Jx = params['Jx']
    Kdz = params['Kdz']
    Jy = params['Jy']
    Jz = params['Jz']
    Jp = params['Jp']

    U1 = ((m * - K1 * zdot) + (Kdz * zdot) + g + U1_star) / ((math.cos(theta_meas) * math.cos(phi_meas)))
    U2 = Jx * (- K2 * p + (U2_star / Jx) - (((Jy - Jz) / Jx) * q * r) + (Jp * q * gyroscopic_term_omega))
    U3 = Jy * (- K3 * q + (U3_star / Jy) - (((Jz - Jx) / Jy) * p * r) - (Jp * p * gyroscopic_term_omega))
    U4 = Jz * (- K4 * r + (U4_star / Jz) - (((Jx - Jy) / Jz) * p * q))

    return U1, U2, U3, U4


# New function to calculate for Euler Rates 
def euler_angle_rates(p, q, r, phi, theta):
    """
    Computes the Euler angle rates (phi_dot, theta_dot, psi_dot) from
    body rates p, q, r and current angles phi, theta.

    Reference formulas:
      phi_dot   = p + q sin(phi) tan(theta) + r cos(phi) tan(theta)
      theta_dot = q cos(phi) - r sin(phi)
      psi_dot   = (q sin(phi) + r cos(phi)) / cos(theta)

    The small eps (1e-12) helps avoid division-by-zero if cos(theta) ~ 0.
    """
    phi_dot_calc = p + (q * math.sin(phi) + r * math.cos(phi)) * math.tan(theta)
    theta_dot_calc = q * math.cos(phi) - r * math.sin(phi)
    # Avoid division by zero near theta = +/- 90 degrees:
    denom = max(math.cos(theta), 1e-12) if math.cos(theta) >= 0 else min(math.cos(theta), -1e-12)
    psi_dot_calc = (q * math.sin(phi) + r * math.cos(phi)) / denom
    return phi_dot_calc, theta_dot_calc, psi_dot_calc

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

# Initialize PWM Functions: 
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
 
# PWM Updated function
def send_speed_command(pwm_outputs, omega, MIN_PULSE_LENGTH, MAX_PULSE_LENGTH):
    # Convert omega (angular velocity) to PWM signals
    pwm_signals = omega_to_pwm(*omega)  # Unpack omega array to individual omega values
    
    # Sends specified pulse lengths (PWM signals) to all motor ESCs
    send_pulse_length(pwm_outputs, pwm_signals)

# Function to convert omega (angular velocity) to PWM signals
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
def reference_generator(time, initial_Z, initial_phi, initial_theta, initial_psi, signal=5):
    
    if signal == 1:
        T1 = 20  # Duration of the first transition (initial altitude to mid-altitude)
        T2 = 40  # Total duration until the second transition completes (mid-altitude to final altitude)
        Z_mid = -0.1  # Mid altitude in meters (intermediate descent point)
        Z_final = 0.0  # Final altitude in meters (hover at ground level)
        initial_Z = 0  # Initial altitude in meters

        if time < T1:
            # **Phase 1: Smooth transition from initial altitude to mid-altitude**
            Z_des_GF = initial_Z + (Z_mid - initial_Z) * 0.5 * (1 - np.cos(np.pi * time / T1))
            # Cosine-based smooth transition for roll and pitch
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * time / T1))  # Roll angle gradually reduced to zero
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * time / T1))  # Pitch angle gradually reduced to zero
        elif time < T2:
            # **Phase 2: Smooth transition from mid-altitude to final altitude**
            adjusted_time = time - T1  # Adjust time relative to the start of this phase
            duration = T2 - T1  # Duration of this phase
            Z_des_GF = Z_mid + (Z_final - Z_mid) * 0.5 * (1 - np.cos(np.pi * adjusted_time / duration))
            phi_des = 0 * (np.pi / 180)  # Maintain roll at 0 degrees
            theta_des = 0 * (np.pi / 180)  # Maintain pitch at 0 degrees
        else:
            # **Final Phase: Stabilize at final altitude**
            Z_des_GF = Z_final  # Maintain altitude at the final value (hover at ground level)
            phi_des = 0 * (np.pi / 180)  # Maintain roll at 0 degrees
            theta_des = 0 * (np.pi / 180)  # Maintain pitch at 0 degrees
        
        # Yaw angle remains constant at zero throughout the signal
        psi_des = 0 * (np.pi / 180)  

        
        
    elif signal == 2:  ## Smooth transition using half sinusoid
        smooth_factor = 1
        T = 10  # Time duration for initial smooth transition

        if time < T: 
            # Smoothly reduce roll, pitch, and yaw angles to zero using a half-cosine function
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * time / T))  # Roll angle smoothly transitions to 0
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * time / T))  # Pitch angle smoothly transitions to 0
            psi_des = initial_psi * 0.5 * (1 + np.cos(np.pi * time / T))  # Yaw angle smoothly transitions to 0
        elif time < 20:
            # Hold roll, pitch, and yaw angles at zero
            phi_des = 0 * (np.pi / 180)
            theta_des = 0 * (np.pi / 180)
            psi_des = 0 * (np.pi / 180)
        elif time < 40:
            # Set roll to 5 degrees, pitch to 0 degrees, yaw to 0 degrees
            phi_des = 5 * (np.pi / 180)  # Roll
            theta_des = 0 * (np.pi / 180)  # Pitch
            psi_des = 0 * (np.pi / 180)  # Yaw
        elif time < 60:
            # Reset roll, pitch, and yaw to zero
            phi_des = 0 * (np.pi / 180)
            theta_des = 0 * (np.pi / 180)
            psi_des = 0 * (np.pi / 180)
        elif time < 80:
            # Set roll to -5 degrees, pitch to 0 degrees, yaw to 0 degrees
            phi_des = -5 * (np.pi / 180)  # Roll
            theta_des = 0 * (np.pi / 180)  # Pitch
            psi_des = 0 * (np.pi / 180)  # Yaw
        elif time < 100:
            # Reset roll, pitch, and yaw to zero
            phi_des = 0 * (np.pi / 180)
            theta_des = 0 * (np.pi / 180)
            psi_des = 0 * (np.pi / 180)
        elif time < 120:
            # Set pitch to 5 degrees, roll and yaw remain at zero
            phi_des = 0 * (np.pi / 180)  # Roll
            theta_des = 5 * (np.pi / 180)  # Pitch
            psi_des = 0 * (np.pi / 180)  # Yaw
        elif time < 140:
            # Reset roll, pitch, and yaw to zero
            phi_des = 0 * (np.pi / 180)
            theta_des = 0 * (np.pi / 180)
            psi_des = 0 * (np.pi / 180)
        elif time < 160:
            # Set pitch to -5 degrees, roll and yaw remain at zero
            phi_des = 0 * (np.pi / 180)  # Roll
            theta_des = -5 * (np.pi / 180)  # Pitch
            psi_des = 0 * (np.pi / 180)  # Yaw
        elif time < 180:
            # Reset roll, pitch, and yaw to zero
            phi_des = 0 * (np.pi / 180)
            theta_des = 0 * (np.pi / 180)
            psi_des = 0 * (np.pi / 180)
        else:
            # Stabilize at zero angles for roll, pitch, and yaw
            phi_des = 0 * (np.pi / 180)
            theta_des = 0 * (np.pi / 180)
            psi_des = 0 * (np.pi / 180)

        Z_des_GF = 0  # Maintain altitude at zero

        
    elif signal == 3:  ## Smooth transition using half sinusoid
        smooth_factor = 1
        T1 = 3  # Time duration for initial smooth transition
        T2 = 20 + T1  # Time for sinusoidal transition between initial_Z and Z_amp
        T3 = 60  # Time duration to maintain zero altitude
        T4 = 110  # Time duration to maintain oscillations in roll and pitch
        f_theta = 0.1  # Frequency for roll and pitch oscillations
        f_Z = 0.05  # Frequency for altitude sinusoidal oscillations
        initial_Z = 0  # Initial altitude
        Z_amp = -0.1  # Target altitude amplitude in meters
        Z_final = 0.0  # Final altitude in meters

        if time < T1: 
            # Initial smooth transition for attitude control
            Z_des_GF = initial_Z  # Maintain the initial altitude
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * time / T1))  # Smoothly reduce roll to zero
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * time / T1))  # Smoothly reduce pitch to zero
            psi_des = initial_psi * 0.5 * (1 + np.cos(np.pi * time / T1))  # Smoothly reduce yaw to zero
        elif time < T2:
            # Smooth sinusoidal transition of altitude
            Z_des_GF = initial_Z + Z_amp * (1 - np.cos(2 * np.pi * f_Z * (time - T1))) / 2  # Sinusoidal oscillation of altitude
            phi_des = 0 * (np.pi / 180)  # Maintain roll at 0 degrees
            theta_des = 0 * (np.pi / 180)  # Maintain pitch at 0 degrees
            psi_des = 0 * (np.pi / 180)  # Maintain yaw at 0 degrees
        elif time < T3:
            # Maintain zero altitude with oscillations in roll and pitch
            Z_des_GF = 0  # Keep altitude at zero
            phi_des = 0 * np.sin(2 * np.pi * f_theta * (time - T2)) * (np.pi / 180)  # Sinusoidal oscillation in roll
            theta_des = 0 * np.sin(2 * np.pi * f_theta * (time - T2)) * (np.pi / 180)  # Sinusoidal oscillation in pitch
            psi_des = 0 * (np.pi / 180)  # Maintain yaw at 0 degrees
        elif time < T4:
            # Maintain zero altitude with oscillations in roll and pitch
            Z_des_GF = 0  # Keep altitude at zero
            phi_des = 0 * np.sin(2 * np.pi * f_theta * (time - 20)) * (np.pi / 180)  # Sinusoidal oscillation in roll
            theta_des = 0 * np.sin(2 * np.pi * f_theta * (time - 20)) * (np.pi / 180)  # Sinusoidal oscillation in pitch
            psi_des = 0 * (np.pi / 180)  # Maintain yaw at 0 degrees
        else:
            # Stabilize the system at zero altitude and zero angles
            Z_des_GF = 0  # Keep altitude at zero
            phi_des = 0 * (np.pi / 180)  # Stabilize roll at 0 degrees
            theta_des = 0 * (np.pi / 180)  # Stabilize pitch at 0 degrees
            psi_des = 0 * (np.pi / 180)  # Stabilize yaw at 0 degrees



    elif signal == 4:  ## Smooth transition using half sinusoid
        smooth_factor = 1
        T1 = 3  # Time duration for the initial smooth transition
        T2 = T1 + 3  # Time duration for the transition from initial altitude to desired amplitude
        T3 = T2 + 20  # Time duration to maintain the desired altitude
        T4 = T3 + 3  # Time duration for the smooth transition back to the initial altitude
        f_theta = 0.1  # Frequency factor for oscillations in angles
        initial_Z = 0  # Initial altitude
        Z_amp = -0.05  # Target altitude amplitude (in meters)
        Z_final = 0.0  # Final altitude after maneuver (in meters)

        if time < T1: 
            # Initial smooth transition to hover state
            Z_des_GF = initial_Z  # Maintain initial altitude
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * time / T1))  # Smoothly reduce roll angle to zero
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * time / T1))  # Smoothly reduce pitch angle to zero
            psi_des = initial_psi * 0.5 * (1 + np.cos(np.pi * time / T1))  # Smoothly reduce yaw angle to zero
        elif time < T2:
            # Transition from initial altitude to desired amplitude using a half-sinusoidal profile
            Z_des_GF = initial_Z + 0.5 * (1 - np.cos(np.pi * (time - T1) / (T2 - T1))) * (Z_amp - initial_Z)
            phi_des = 0 * (np.pi / 180)  # Maintain zero roll
            theta_des = 0 * (np.pi / 180)  # Maintain zero pitch
            psi_des = 0 * (np.pi / 180)  # Maintain zero yaw
        elif time < T3:
            # Maintain altitude at the desired amplitude with oscillations in roll and pitch
            Z_des_GF = Z_amp  # Keep altitude constant at Z_amp
            phi_des = 0 * np.sin(2 * np.pi * f_theta * (time - T2)) * (np.pi / 180)  # Small sinusoidal oscillations in roll
            theta_des = 0 * np.sin(2 * np.pi * f_theta * (time - T2)) * (np.pi / 180)  # Small sinusoidal oscillations in pitch
            psi_des = 0 * (np.pi / 180)  # Maintain zero yaw
        elif time < T4:
            # Smooth transition back from the desired amplitude to the initial altitude
            Z_des_GF = Z_amp - 0.5 * (1 - np.cos(np.pi * (time - T3) / (T4 - T3))) * (Z_amp - initial_Z)
            phi_des = 0 * np.sin(2 * np.pi * f_theta * (time - T3)) * (np.pi / 180)  # Small sinusoidal oscillations in roll
            theta_des = 0 * np.sin(2 * np.pi * f_theta * (time - T3)) * (np.pi / 180)  # Small sinusoidal oscillations in pitch
            psi_des = 0 * (np.pi / 180)  # Maintain zero yaw
        else:
            # Stabilize at the initial altitude after the maneuver
            Z_des_GF = initial_Z  # Return to the initial altitude
            phi_des = 0 * (np.pi / 180)  # Stabilize roll at zero
            theta_des = 0 * (np.pi / 180)  # Stabilize pitch at zero
            psi_des = 0 * (np.pi / 180)  # Stabilize yaw at zero


    elif signal == 5:  ## Smooth transition to 0, hold, then step psi to 30 degrees
        smooth_factor = 1
        T = 5  # Time duration for initial smooth transition
        hold_time = T + 5  # Hold at zero for 5 seconds after smooth transition
        step_time = hold_time + 10  # Time when psi steps to 30 degrees

        if time < T: 
            # Smoothly reduce roll, pitch, and yaw angles to zero using a half-cosine function
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * time / T))  # Roll smoothly transitions to 0
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * time / T))  # Pitch smoothly transitions to 0
            psi_des = initial_psi * 0.5 * (1 + np.cos(np.pi * time / T))  # Yaw smoothly transitions to 0
        elif time < hold_time:
            # Hold roll, pitch, and yaw angles at zero
            phi_des = 0 * (np.pi / 180)  # Maintain roll at 0 degrees
            theta_des = 0 * (np.pi / 180)  # Maintain pitch at 0 degrees
            psi_des = 0 * (np.pi / 180)  # Maintain yaw at 0 degrees
        elif time < step_time:
            # Step psi to 30 degrees while roll and pitch remain at zero
            phi_des = 0 * (np.pi / 180)  # Maintain roll at 0 degrees
            theta_des = 0 * (np.pi / 180)  # Maintain pitch at 0 degrees
            psi_des = 0 * (np.pi / 180)  # Step yaw to 30 degrees
        else:
            # Stabilize with roll and pitch at zero and yaw at 30 degrees
            phi_des = 0 * (np.pi / 180)  # Maintain roll at 0 degrees
            theta_des = 0 * (np.pi / 180)  # Maintain pitch at 0 degrees
            psi_des = 0 * (np.pi / 180)  # Maintain yaw at 30 degrees

        Z_des_GF = 0  # Maintain altitude at zero

    elif signal == 6:  ## Smooth transition using half sinusoid
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
        q_desired = 0*Amp * np.sin(2 * np.pi * f_pqr * time)
        r_desired = 0
    else:
        # Third phase: 0 deg/s (static)
        p_desired = 0
        q_desired = 0
        r_desired = 0
    
    return p_desired, q_desired, r_desired

# Conrol Functions:
def attitude_PID(Z_des_GF, phi_des, theta_des, psi_des, 
                 Z_meas, phi_meas, theta_meas, psi_meas, 
                 p_meas, q_meas, r_meas, 
                 Z_KP, Z_KI, Z_KD, 
                 Phi_KP, Phi_KI, Phi_KD, 
                 Theta_KP, Theta_KI, Theta_KD, 
                 Psi_KP, Psi_KI, Psi_KD, 
                 Ts, m, g):
    global z_error_sum, phi_error_sum, theta_error_sum, psi_error_sum
    global previous_z_error, previous_phi_error, previous_theta_error, previous_psi_error
    # NEW globals for the desired reference derivative for each axis:
    global previous_phi_des, previous_theta_des, previous_psi_des

    # Z Position PID Controller/Altitude Controller
    z_error = Z_des_GF - Z_meas
    z_error_sum += z_error * Ts
    z_error_dot = (z_error - previous_z_error) / Ts

    cp = Z_KP * z_error
    ci = Z_KI * z_error_sum
    cd = Z_KD * z_error_dot
    U1 = cp + ci + cd
    U1 = min(U1_max, max(U1_min, U1))  # Ensure U1 is within limits

    previous_z_error = z_error

    # --- Compute Euler-angle rates from sensor measurements ---
    phi_dot_calc, theta_dot_calc, psi_dot_calc = euler_angle_rates(p_meas, q_meas, r_meas, phi_meas, theta_meas)

    # Roll PID Controller
    phi_error = phi_des - phi_meas
    phi_error_sum += phi_error * Ts
    if previous_phi_des is None:
        phi_des_dot = 0.0
    else:
        phi_des_dot = (phi_des - previous_phi_des) / Ts
    phi_error_dot = phi_des_dot - phi_dot_calc
    previous_phi_des = phi_des
    cp = Phi_KP * phi_error
    ci = Phi_KI * phi_error_sum
    cd = Phi_KD * phi_error_dot
    p_desired = cp + ci + cd
    previous_phi_error = phi_error

    # Pitch PID Controller
    theta_error = theta_des - theta_meas
    theta_error_sum += theta_error * Ts
    if previous_theta_des is None:
        theta_des_dot = 0.0
    else:
        theta_des_dot = (theta_des - previous_theta_des) / Ts
    theta_error_dot = theta_des_dot - theta_dot_calc
    previous_theta_des = theta_des
    cp = Theta_KP * theta_error
    ci = Theta_KI * theta_error_sum
    cd = Theta_KD * theta_error_dot
    q_desired = cp + ci + cd
    previous_theta_error = theta_error

    # Yaw PID Controller
    psi_error = psi_des - psi_meas
    psi_error_sum += psi_error * Ts
    if previous_psi_des is None:
        psi_des_dot = 0.0
    else:
        psi_des_dot = (psi_des - previous_psi_des) / Ts
    psi_error_dot = psi_des_dot - psi_dot_calc
    previous_psi_des = psi_des
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
    # w1 = U1 / (4 * KT) + U3 / (2 * KT * l) + U4 / (4 * Kd)
    # w2 = U1 / (4 * KT) - U2 / (2 * KT * l) - U4 / (4 * Kd)
    # w3 = U1 / (4 * KT) - U3 / (2 * KT * l) + U4 / (4 * Kd)
    # w4 = U1 / (4 * KT) + U2 / (2 * KT * l) - U4 / (4 * Kd)

    # Calculate motor speeds (rad/s)^2 based on U1 U2 U3 and U4
    w1_squared = U1 / (4 * KT) - U3 / (2 * KT * l) - U4 / (4 * Kd)
    w2_squared = U1 / (4 * KT) - U2 / (2 * KT * l) + U4 / (4 * Kd)
    w3_squared = U1 / (4 * KT) + U3 / (2 * KT * l) - U4 / (4 * Kd)
    w4_squared = U1 / (4 * KT) + U2 / (2 * KT * l) + U4 / (4 * Kd)

    # Apply realistic motor speed limits
    max_speed_squared = max_motor_speed ** 2
    min_speed_squared = min_motor_speed ** 2
    
    w1 = min(max(w1_squared, min_speed_squared), max_speed_squared)
    w2 = min(max(w2_squared, min_speed_squared), max_speed_squared)
    w3 = min(max(w3_squared, min_speed_squared), max_speed_squared)
    w4 = min(max(w4_squared, min_speed_squared), max_speed_squared)

    # Compute motor speeds
    omega_1 = np.sqrt(w1)  # Front Motor
    omega_2 = np.sqrt(w2)  # Right Motor
    omega_3 = np.sqrt(w3)  # Rear Motor
    omega_4 = np.sqrt(w4)  # Left Motor

    return omega_1, omega_2, omega_3, omega_4

# EKF
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




# -------------------------- Generic Adaptive Gains Class -------------------------- #
class AdaptiveGains:
    def __init__(self, initial_Kp, initial_Ki, initial_Kd, alpha=0.1, gamma=0.1, 
                 bounds_Kp=(0, 1), bounds_Ki=(0, 1), bounds_Kd=(0, 1)):
        self.Kp = initial_Kp
        self.Ki = initial_Ki
        self.Kd = initial_Kd
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Momentum factor
        self.bounds_Kp = bounds_Kp
        self.bounds_Ki = bounds_Ki
        self.bounds_Kd = bounds_Kd
        self.vKp = 0.0
        self.vKi = 0.0
        self.vKd = 0.0
        self.last_grad_y_u = None         # NEW: store last valid grad_y_u
        self.prev_error = None
        self.integral = 0.0
        self.prev_control = None
        self.prev_output = None

    def update(self, error, control, output, dt):
        """
        Updates the PID gains using a Nesterov momentum scheme.
        
        Args:
            error   : Current error (desired - measured) for the axis.
            control : Previous control signal for the axis.
            output  : Previous measured output for the axis.
            dt      : Time step.
            
        Returns:
            Updated gains (Kp, Ki, Kd).
        """
        self.integral += error * dt
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt

        # Estimate sensitivity (avoid division by zero)
        if self.prev_control is not None and abs(control - self.prev_control) > 1e-6:
            grad_y_u = (output - self.prev_output) / (control - self.prev_control)
            self.last_grad_y_u = grad_y_u  # update last valid value
        else:
            grad_y_u = self.last_grad_y_u if self.last_grad_y_u is not None else 0.0

        # dL/dy for 1/2*(r - y)^2 is -error
        dL_dy = -error

        # For PID:
        gradKp = dL_dy * grad_y_u * error      # partial derivative with respect to Kp
        gradKi = dL_dy * grad_y_u * self.integral
        gradKd = dL_dy * grad_y_u * derivative

        # Gradients for cost function J = 0.5 * error^2
        gradKp = - error * grad_y_u * error
        gradKi = - error * grad_y_u * self.integral
        gradKd = - error * grad_y_u * derivative

        # Momentum updates
        self.vKp = self.gamma * self.vKp - self.alpha * gradKp
        self.vKi = self.gamma * self.vKi - self.alpha * gradKi
        self.vKd = self.gamma * self.vKd - self.alpha * gradKd

        # Apply updates to the parameters
        self.Kp += self.vKp
        self.Ki += self.vKi
        self.Kd += self.vKd

        # Clamp gains to bounds
        self.Kp = max(min(self.Kp, self.bounds_Kp[1]), self.bounds_Kp[0])
        self.Ki = max(min(self.Ki, self.bounds_Ki[1]), self.bounds_Ki[0])
        self.Kd = max(min(self.Kd, self.bounds_Kd[1]), self.bounds_Kd[0])
        
        self.prev_error = error
        self.prev_control = control
        self.prev_output = output

        return self.Kp, self.Ki, self.Kd


def estimate_attitude_and_rates(imu, Ts, alpha_sensor, alpha_attitude, q0, custom_ekf, 
                                gyro_bias, prev_acc, prev_gyro, filtered_phi, filtered_theta, 
                                filtered_psi, iteration_index):
    """
    Reads sensor data from the IMU, filters the raw measurements,
    applies the EKF update, and computes the filtered Euler angles.

    Args:
        imu: IMU sensor object with a getMotion9() method.
        Ts: Sampling time.
        alpha_sensor: Low-pass filter coefficient for accelerometer and gyro.
        alpha_attitude: Low-pass filter coefficient for Euler angles.
        q0: Current quaternion estimate.
        custom_ekf: Instance of the custom EKF filter.
        gyro_bias: The gyro bias vector.
        prev_acc: Previous filtered accelerometer data (or None if first iteration).
        prev_gyro: Previous filtered gyro data (or None if first iteration).
        filtered_phi, filtered_theta, filtered_psi: Previous filtered Euler angles.
        iteration_index: Current iteration index to condition the first run.

    Returns:
        A tuple containing:
          - Accelerometer measurements: X_ddot_meas, Y_ddot_meas, Z_ddot_meas
          - Gyro measurements: p_meas, q_meas, r_meas
          - Filtered Euler angles: phi_meas, theta_meas, psi_meas
          - Updated q0 (quaternion),
          - Updated prev_acc and prev_gyro,
          - Updated filtered Euler angles (filtered_phi, filtered_theta, filtered_psi)
          - x_state: The EKF state vector from the update.
    """
    # Read raw sensor data.
    m9a, m9g, m9m = imu.getMotion9()
    m9a = np.array(m9a)
    m9g = np.array(m9g)
    
    # Apply low-pass filter to accelerometer and gyro data.
    if prev_acc is None:
        filtered_acc = m9a
        filtered_gyro = m9g
    else:
        filtered_acc = alpha_sensor * m9a + (1 - alpha_sensor) * prev_acc
        filtered_gyro = alpha_sensor * m9g + (1 - alpha_sensor) * prev_gyro
    # Update previous filtered values.
    prev_acc = filtered_acc
    prev_gyro = filtered_gyro
    
    # Convert to list for further processing if necessary.
    m9a_list = filtered_acc.tolist()
    m9g_list = filtered_gyro.tolist()
    
    # Adjust gyro readings using the bias if the deviation is within tolerance.
    gyro_tol = 0.1  # 10% tolerance
    for j in range(3):
        if abs(m9g_list[j] - gyro_bias[j]) < abs(gyro_bias[j]) * gyro_tol:
            m9g_list[j] = gyro_bias[j]
    
    # Update the EKF with the filtered sensor readings.
    q_updated, x_state = custom_ekf.update(q0, gyr=m9g_list, acc=m9a_list, mag=m9m, dt=Ts)
    q0 = q_updated

    # Convert the updated quaternion to Euler angles.
    phi_ekf, theta_ekf, psi_ekf = quaternion_to_euler(q_updated)
    
    # Apply a recursive low-pass filter to the Euler angle estimates.
    # Apply a recursive low-pass filter to the Euler angle estimates.
    if iteration_index == 0 or filtered_phi is None:
        filtered_phi = phi_ekf
        filtered_theta = theta_ekf
        filtered_psi = psi_ekf
    else:
        filtered_phi = alpha_attitude * phi_ekf + (1 - alpha_attitude) * filtered_phi
        filtered_theta = alpha_attitude * theta_ekf + (1 - alpha_attitude) * filtered_theta
        filtered_psi = alpha_attitude * psi_ekf + (1 - alpha_attitude) * filtered_psi

    
    # Set the final measured Euler angles.
    phi_meas, theta_meas, psi_meas = filtered_phi, filtered_theta, filtered_psi
    # Extract accelerometer and gyro measurements.
    X_ddot_meas, Y_ddot_meas, Z_ddot_meas = m9a_list[0], m9a_list[1], m9a_list[2]
    p_meas, q_meas, r_meas = m9g_list[0], m9g_list[1], m9g_list[2]
    
    return (X_ddot_meas, Y_ddot_meas, Z_ddot_meas,
            p_meas, q_meas, r_meas,
            phi_meas, theta_meas, psi_meas,
            q0,
            prev_acc, prev_gyro,
            filtered_phi, filtered_theta, filtered_psi,
            x_state)


# Safety Exit: Future Implementation
 
# --------------------------- Initialization ---------------------------- #

# ---------------------------- PWM Related ------------------------------ #
system_utils.stop_idle_signal()
time.sleep(2)
# ----------------------- Configuration Parameters ----------------------- #
TEMP_THRESHOLD = 50  # Maximum allowable CPU temperature (°C)

# ---------------------------- System Check ------------------------------ #
print("Checking system parameters...")
temp = system_utils.measure_temp()
freq = system_utils.measure_cpu_freq()
volts = system_utils.measure_volts()
voltage = system_utils.monitor_voltage()

if voltage and voltage < 1.0:  
    print("Warning: Voltage is too low! Check your power supply.")

while not system_utils.check_temperature_threshold(TEMP_THRESHOLD):
    print(f"Temperature exceeds safe threshold of {TEMP_THRESHOLD}°C! Waiting for temperature to drop...")
    time.sleep(5)  # Wait for 5 seconds before rechecking

print(f"System Parameters: Temperature={temp}°C, Frequency={freq}MHz, Voltage={volts}V")

# ------------------------ System Optimizations -------------------------- #
system_utils.set_realtime_priority()           # Set real-time priority
system_utils.set_cpu_affinity([0, 1, 2, 3])    # Assign specific CPU cores
system_utils.lock_memory()                     # Lock memory to prevent paging
system_utils.disable_power_management()        # Disable power-saving features
system_utils.optimize_disk_io()                # Optimize disk I/O operations
system_utils.disable_io_buffering()            # Disable I/O buffering for performance
system_utils.wait_for_low_cpu(threshold=1, interval=1)  # Wait for low CPU usage
# 1% usage and check every 1 sec for stability
system_utils.reduce_logging_overhead()         # Reduce logging overhead


# ------------------------ Sensors Setup -------------------------- #
# Altitude Sensor Setup
ser = serial.Serial("/dev/serial0", 115200, timeout=1)  # Setup serial connection

# Initialize the MPU9250 IMU sensor - sensor has mag reading issues
# imu = mpu9250.MPU9250()
# (alt) Initialize the LSM9DS1 IMU sensor
imu = lsm9ds1.LSM9DS1()
imu.initialize()

 # -------- Initialize Filters -------- #
filter_cutoff = 10.0  # Cutoff frequency in Hz

# Initialize LED 
led = Led()
led.setColor('Green')

# ------------------------ PWM and Thrusters Configuration -------------------------- #
PWM_OUTPUTS = [0, 1, 2, 3]  # Assign PWM Motor Pins

# Calibrated pulse lengths and speed bounds (DO NOT CHANGE)
MIN_PULSE_LENGTH = np.array([1.140, 1.142, 1.141, 1.152])# (DO NOT CHANGE)
MAX_PULSE_LENGTH = np.array([2.0, 2.0, 2.0, 2.0]) # (DO NOT CHANGE)
NOSPEED_PULSE_LENGTHS = np.array([1.0, 1.0, 1.0, 1.0]) 

# Maximum Speeds in rad/s (MN3508 with P12*4 Propellers)
MIN_OMEGA = np.array([30, 30, 30, 30]) # rad/s corresponds to 286.4789 rpm
#MAX_OMEGA = np.array([526.4262, 526.4262, 526.4262, 526.4262])  # rad/s corresponds to 5027 rpm  
MAX_OMEGA = np.array([700, 700, 700, 700])  # rad/s corresponds to 5027 rpm  

# ------------------------ Quadrotor Physical Parameters -------------------------- #
# Quadcopter Physical Parameters 
Quad_without_prop = 1.780 # quadcopter mass (Kg) without stand and propellers - DEC2024
Quad_base = 0.119 # quadcopter base frame (Kg)  (3D Print) DEC2024
Quad_rod = 0.221 # quadcopter base frame (Kg)  (Linear Support) DEC2024
Quad_t_mot_prop = 4 * 0.012  # quadcopter 4 propellers (Kg) JAN2025
Quad_4s_lipo = 0.341  # old 4s battery 4 propellers (Kg) JAN2025
Quad_6s_lipo = 0.495 # new 6s battery 4 propellers (Kg) JAN2025
# total mass with stand: 
m = Quad_without_prop + Quad_base + Quad_rod + Quad_t_mot_prop + Quad_6s_lipo - Quad_4s_lipo
# m is Mass (kg) =  2.3220
print(f"Total mass with stand: {m} kg")
g = 9.80665  # Gravity (m/s^2)
l = 0.225  # Distance from the center to each motor (m)
KT = 0.000022  # Thrust coefficient (N/(rad/s)^2)  JAN2025 RCbenchmark
Kd = l * KT  # Drag torque coefficient (N·m/(rad/s)^2)

# Parameters identified using Biflir Pendulum:
Jx = 0.0206;  # Moment of inertia about X axis (kg-m^2)
Jy = 0.0210;  # Moment of inertia about Y axis (kg-m^2)
Jz = 0.0361;  # Moment of inertia about Z axis (kg-m^2)
Kdx = 0.0002;  # Translational drag force coeffecient (kg/s)
Kdy = 0.0003;  # Translational drag force coeffecient (kg/s)
Kdz = 0.0057 ;  # 0.0005    % Translational drag force coeffecient (kg/s)

# J_p Calculations for Gyroscopic effect:
prop_diameter = 12 * 0.0254  # Propeller diameter in meters
prop_radius = prop_diameter / 2  # Radius in meters
prop_mass = 14.5 / 1000  # Propeller mass in kg  
motor_mass = 82 / 1000  # Motor mass in kg 
rotor_mass = 0.5 * motor_mass  # Approximate rotor mass
rotor_radius = 0.04   # Rotor radius (shaft diameter assumed as 4mm)
# Calculate gyroscopic inertia for a single propeller and motor
Jp_propeller = 0.5 * prop_mass * prop_radius ** 2  #  Gyroscopic inertia of propeller
Jp_rotor = 0.5 * rotor_mass * rotor_radius ** 2  # Gyroscopic inertia of rotor
# Total gyroscopic inertia for one motor-propeller system
Jp = Jp_propeller + Jp_rotor

g = 9.80665     # Gravity (m/s^2)
l = .225        # Distance from the center of mass to the each motor(m)
KT = 0.000022   # Thrust force coeffecient (N/(rad/s)^2 ) (T-motor 12 inch prop)
Kd = l*KT       # Drag torque coeffecient (N m / (rad/s)^2 ) 

min_motor_speed = MIN_OMEGA[0]  # rad/s (minimum limit after caclulating motor speeds)
max_motor_speed = MAX_OMEGA[0]  # rad/s (maxmimum limit after caclulating motor speeds)

print(f"--> Min Motor Speed: {min_motor_speed:.2f} rad/s")  
print(f"--> Max Motor Speed: {max_motor_speed:.2f} rad/s")  
 
# Defining limits for control inputs
U1_max = KT * 4 * max_motor_speed ** 2
U1_min =  KT * 4 * min_motor_speed ** 2
print(f"--> Min Control Input U1: {U1_min:.2f} and Max Control Input U1: {U1_max:.2f}  ")  
U2_max = KT * l * max_motor_speed ** 2
U2_min = -KT * l * max_motor_speed ** 2 
print(f"--> Min Control Input U2: {U2_min:.2f} and Max Control Input U2: {U2_max:.2f}  ")  
U3_max = KT * l * max_motor_speed ** 2
U3_min = -KT * l * max_motor_speed ** 2 
print(f"--> Min Control Input U3: {U3_min:.2f} and Max Control Input U3: {U3_max:.2f}  ")  
U4_max = Kd * 2 * max_motor_speed ** 2
U4_min = -Kd * 2 * max_motor_speed ** 2  
print(f"--> Min Control Input U4: {U4_min:.2f} and Max Control Input U4: {U4_max:.2f}  ") 
 
# ---------------------- PID Parameters  ---------------------- #
# Feedback linearization gains (for roll, pitch, yaw)
K1 = 0  # for altitude
K2 = 20 # for phi
K3 = 20 # for theta
K4 = 0  # for psi

print(f"KPVAL: {(K2**2)/(4*(1/Jx))} ")

#Z_KP, Z_KI, Z_KD = (K1**2)/(4*(1/m)), 0, 0
Z_KP, Z_KI, Z_KD = 0, 0, 0
#Phi_KP, Phi_KI, Phi_KD = (K2**2)/(4*(1/Jx)), 0, 0
#Phi_KP, Phi_KI, Phi_KD = 2.5, 4, 3.25
#Theta_KP, Theta_KI, Theta_KD = (K3**2)/(4*(1/Jy)), 0, 0
#Theta_KP, Theta_KI, Theta_KD = 2.5, 4, 3.5
# Psi_KP, Psi_KI, Psi_KD = (K4**2)/(4*(1/Jz)), 0, 0
Psi_KP, Psi_KI, Psi_KD = 0, 0, 0

PQR_PID_Enable = False
P_KP, P_KI, P_KD = 0.1, 0, 0.01  
Q_KP, Q_KI, Q_KD = 0.1, 0, 0.01  
R_KP, R_KI, R_KD = 0.1, 0, 0.01

# ---------------------- Simulation settings ---------------------- #
Ts = 0.005  # Sampling time in seconds (increased to 10ms)
total_simulation_time = 100 # Simulation time in seconds  

# ---------------------- Calibration Process ---------------------- #
calibration_samples = 10
# Calibrate IMU and get biases
bx, by, bz, phi_bias, theta_bias, psi_bias, m9a, m9g, m9m = calibrate_imu(imu, calibration_samples, Ts)

# Print and check if it matches the system
print(f"--> Phi (Acc): {np.rad2deg(phi_bias):.2f}°, Theta (Acc): {np.rad2deg(theta_bias):.2f}°, Psi (Mag): {np.rad2deg(psi_bias):.2f}°")

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
previous_phi_des = None
previous_theta_des = None
previous_psi_des = None


# Parameters dictionary for feedback linearization
parameters = {
    'm': m,
    'g': g,
    'l': l,
    'KT': KT,
    'Kd': Kd,
    'Kdx': Kdx,    # drag coefficient in X
    'Kdy': Kdy,    # drag coefficient in Y
    'Kdz': Kdz,    # drag coefficient in Z
    'Jx': Jx,    # Moment of inertia about X
    'Jy': Jy,    # Moment of inertia about Y
    'Jz': Jz,  # Moment of inertia about Z
    'Jp': Jp   # Propeller moment of inertia
}

# ---------------------- End of all configurations ---------------------- #
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
    
    # --------- Step 1 of Init Custom EKF Filter  --------- #
    # Read a few gyro samples for bias estimation.
    calibration_samples = 200
    gyro_calib = []
    for i in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        gyro_calib.append(m9g)
        time.sleep(Ts)
    gyro_calib = np.array(gyro_calib)
    gyro_bias = np.mean(gyro_calib, axis=0)  # [bias_x, bias_y, bias_z]

    # Set initial quaternion from calibration (assuming zero Euler angles) - we need to fix 
    q0 = euler_to_quaternion(phi_bias, theta_bias, 0*psi_bias)
    #
    # Instantiate our custom EKF filter and initialize its biases:
    custom_ekf = CustomEKF(q0=q0)
    custom_ekf.x[3:6] = gyro_bias  # initialize [bp, bq, br]
    #
    # Precompute recursive low-pass filter coefficient
    filter_cutoff_attitude = 5 # 20  # Hz
    alpha_attitude = Ts / (Ts + (1 / (2 * np.pi * filter_cutoff_attitude)))
    # --------- initialize filtered variables --------- #
    filtered_phi = None
    filtered_theta = None
    filtered_psi = None

    # 
    # Filter accelerometer/gyro 
    prev_acc = None # 
    prev_gyro = None # to filter gyro measurements
    filter_cutoff_sensor = 20  # Hz
    alpha_sensor = Ts / (Ts + (1 / (2 * np.pi * filter_cutoff_sensor)))
    #
    # --------- End of: Step 1 of Custom EKF Filter  --------- #

    # Initialize gyroscopic omega for feedback linearization term
    gyroscopic_term_omega = 0.0 # gyroscopic term omega 


    
    # ------------- Adaptive PID Part -----------------
    #
    # Learning Rate Alpha [0.01, 0.2] 
    # high alpha = destabilize by overshooting + fast gain update
    # low alpha = Stable but slow to adapt (sluggish).
    #
    # Momentum Factor Gamma [0.5,0.95]
    # high gamma = Smooth updates, improves stability, slows adaptation slightly.
    # low gamma = Fast, but may cause oscillations.
    # start with: alpha =0.05, gamma = 0.9

    #Phi_KP, Phi_KI, Phi_KD = 2.5, 4, 3.25
    Phi_KP, Phi_KI, Phi_KD = 2.9, 2.9, 2.8
    #Theta_KP, Theta_KI, Theta_KD = 2.5, 4, 3.5
    Theta_KP, Theta_KI, Theta_KD =  2.6, 2.9, 2.8

    # Initialize AdaptiveGains for each axis
    # adaptive_z = AdaptiveGains(initial_Kp=Z_KP, initial_Ki=Z_KI, initial_Kd=Z_KD, alpha=0.07, gamma=0.9, bounds_Kp=(0,30), bounds_Ki=(0,20), bounds_Kd=(0,10))
    adaptive_phi = AdaptiveGains(initial_Kp=Phi_KP, initial_Ki=Phi_KI, initial_Kd=Phi_KD, alpha=0.005, gamma=0.90, bounds_Kp=(0.5,3), bounds_Ki=(0.5,3), bounds_Kd=(0.5,3))
    adaptive_theta = AdaptiveGains(initial_Kp=Theta_KP, initial_Ki=Theta_KI, initial_Kd=Theta_KD, alpha=0.005, gamma=0.90, bounds_Kp=(0.5,3), bounds_Ki=(0.5,3), bounds_Kd=(0.5,3))
    # adaptive_psi = AdaptiveGains(initial_Kp=Psi_KP, initial_Ki=Psi_KI, initial_Kd=Psi_KD, alpha=0.01, gamma=0.9, bounds_Kp=(0,10), bounds_Ki=(0,5), bounds_Kd=(0,5))
    

    # LED Color white:  all initialization completed
    led.setColor('White')  # (0, 1, 0)
    
    # ---------------------------------------------------------------
    # Send no-speed pulse lengths for 5 seconds before the experiment
    # Warm-up motors in the last 3 seconds of the initial 5-second delay
    # ---------------------------------------------------------------
    print("======== Starting Experiment in 5 seconds ========")
    start_no_speed_time = time.time()
    total_delay = 5  # Total delay before starting the experiment

    while time.time() - start_no_speed_time < total_delay:
        elapsed_time = time.time() - start_no_speed_time
        send_pulse_length(pwm_outputs, MIN_PULSE_LENGTH)
 
        # ---------- measure altitude (before experimenting) ----------
        Z_meas = 0 #read_lidar(ser)  
        if Z_meas is not None:
            Z_meas -= alt_bias  # Subtract bias to make initial point exactly 0
            if abs(Z_meas - last_valid_altitude) <= delta_altitude_threshold:
                last_valid_altitude = Z_meas  # Update last valid altitude
        else:
            Z_meas = last_valid_altitude

        # +++++++++++ estimate attitude (before experimenting - EKF ) +++++++++++ #
        (
            X_ddot_meas, Y_ddot_meas, Z_ddot_meas,
            p_meas, q_meas, r_meas,
            phi_meas, theta_meas, psi_meas,
            q0,
            prev_acc, prev_gyro,
            filtered_phi, filtered_theta, filtered_psi,
            x_state
        ) = estimate_attitude_and_rates(
                imu=imu, Ts=Ts, alpha_sensor=alpha_sensor, alpha_attitude=alpha_attitude,
                q0=q0, custom_ekf=custom_ekf,
                gyro_bias=gyro_bias, prev_acc=prev_acc, prev_gyro=prev_gyro,
                filtered_phi=filtered_phi, filtered_theta=filtered_theta, filtered_psi=filtered_psi,
                iteration_index=i  # Use the current iteration index
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
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

    # Color red: Experiment started
    led.setColor('Red')  
    
    for i in range(num_samples):
        
        # Get the current time:
        current_time = time.perf_counter()  
        sim_time = current_time - start_time # sim time starts from 0
        
        time_min = 6
        time_max = total_simulation_time 

        #  ================== State Estimation =============== #
        # Step 1: measure altitude
        Z_meas = read_lidar(ser)
        if Z_meas is not None:
            Z_meas -= alt_bias  # Subtract bias to make initial point exactly 0
            if abs(Z_meas - last_valid_altitude) <= delta_altitude_threshold:
                last_valid_altitude = Z_meas  # Update last valid altitude
        else:
            # Use previous measurement if no new data
            Z_meas = last_valid_altitude

        # Use a complementary filter for vertical speed and position ----> IP
        Z_dot_est = alpha_dot_est * (Z_dot_est + (Z_ddot_meas - g) * Ts) + (1 - alpha_dot_est) * ((Z_meas - Z_est) / Ts)
        Z_est += Z_dot_est * Ts
        z_dot_meas = Z_dot_est
        
        # Step 2: measure imu and estimate using EKF
        # +++++++++++ estimate attitude (before experimenting - EKF ) +++++++++++ #
        (
            X_ddot_meas, Y_ddot_meas, Z_ddot_meas,
            p_meas, q_meas, r_meas,
            phi_meas, theta_meas, psi_meas,
            q0,
            prev_acc, prev_gyro,
            filtered_phi, filtered_theta, filtered_psi,
            x_state
        ) = estimate_attitude_and_rates(
                imu=imu, Ts=Ts, alpha_sensor=alpha_sensor, alpha_attitude=alpha_attitude,
                q0=q0, custom_ekf=custom_ekf, gyro_bias=gyro_bias,
                prev_acc=prev_acc, prev_gyro=prev_gyro,
                filtered_phi=filtered_phi, filtered_theta=filtered_theta, filtered_psi=filtered_psi,
                iteration_index=i  # Use the current iteration index
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        #  ================== REFERENCE SIGNAL =============== #
        Z_des_GF, phi_des, theta_des, psi_des = reference_generator(sim_time, initial_Z, initial_phi, initial_theta, initial_psi)
        
        # ------------- Adaptive GD PID Part 1 - begin ----------------- #
        # Compute errors for each axis 
        error_phi = phi_des - phi_meas
        error_theta = theta_des - theta_meas
        
        
        # For iterations beyond the first, update adaptive gains using previous iteration's control and output.
        if i > 0:
            Phi_KP, Phi_KI, Phi_KD = adaptive_phi.update(error_phi, p_desired_prev, phi_meas_prev, Ts)
            Theta_KP, Theta_KI, Theta_KD = adaptive_theta.update(error_theta, q_desired_prev, theta_meas_prev, Ts)
            # Psi_KP, Psi_KI, Psi_KD = adaptive_psi.update(error_psi, r_desired_prev, psi_meas_prev, Ts)
        # ------------- Adaptive GD PID Part 1 - end ----------------- #
        #Phi_KP, Phi_KI, Phi_KD = 2, 1, 1
        #Theta_KP, Theta_KI, Theta_KD = 2.5, 4, 3.5
        #Theta_KP, Theta_KI, Theta_KD = 2,1 , 1
            
        #  ================== ATTITUDE CONTROLLER =============== #
        U1, p_desired, q_desired, r_desired = attitude_PID(Z_des_GF, phi_des, theta_des, psi_des, Z_meas, phi_meas, theta_meas, psi_meas, p_meas, q_meas, r_meas, Z_KP, Z_KI, Z_KD, Phi_KP, Phi_KI, Phi_KD, Theta_KP, Theta_KI, Theta_KD, Psi_KP, Psi_KI, Psi_KD, Ts, m, g)
        

        # ------------- Adaptive PID Part 2 - Begin----------------- #
        # Save current control outputs and measured values for next iteration. 
        # U1_prev = U1
        p_desired_prev = p_desired
        q_desired_prev = q_desired
        # r_desired_prev = r_desired
        # z_meas_prev = z_meas
        phi_meas_prev = phi_meas
        theta_meas_prev = theta_meas
        # psi_meas_prev = psi_meas

        # ------------- Adaptive PID Part 2 Ends Here -----------------  #

        #  ================== RATE CONTROLLER =============== #
        # Tune Rate Controller: (comment when done)
        # p_desired, q_desired, r_desired = rate_reference_generator(sim_time) # COMMENTED
        # Step 5. Implement Rate Controller (if enabled)
        U2, U3, U4 = rate_PID(p_desired, q_desired, r_desired, p_meas, q_meas, r_meas, P_KP, P_KI, P_KD, Q_KP, Q_KI, Q_KD, R_KP, R_KI, R_KD, Ts, PQR_PID_Enable)
        
        #  ================== FEEDBACK LINEARIZATION =============== #
        enable_feedback_linearization = False  # set to False to disable feedback linearization
        if enable_feedback_linearization:
            # Save raw control inputs (before feedback linearization)
            U1_star = U1
            U2_star = U2
            U3_star = U3
            U4_star = U4
            # Transform U2, U3, U4 using the feedback linearization function.
            U1, U2, U3, U4 = feedback_linearization_control(U1_star, U2_star, U3_star, U4_star, 
                                                            z_dot_meas, p_meas, q_meas, r_meas,
                                                            theta_meas, phi_meas, gyroscopic_term_omega, 
                                                            parameters, K1, K2, K3, K4)
            
        U1 = 0 # disable altitude control for now (for feedback linearization)

        #  ================== Calculate Desired Motor Speed =============== #
        # Step 6. Calculate Desired Motor Speeds in rad/s
        omega_1, omega_2, omega_3, omega_4  = motor_speed(U1, U2, U3, U4, KT, Kd, l, max_motor_speed, min_motor_speed)
        gyroscopic_term_omega =  0 * (omega_2 + omega_4 - omega_3 - omega_1)  # for gyroscopic term in feedback lineartization

        # Step 5. Send PWM signals corresponding to the calculated angular velocities
        omega = np.array([omega_1, omega_2, omega_3, omega_4])
        send_speed_command(pwm_outputs, omega, MIN_PULSE_LENGTH, MAX_PULSE_LENGTH)

        # Store current data: populate storage dictionary
        storage['sim_times'][i] = sim_time
        storage['omegas'][i, :] = omega
        storage['altitudes'][i] = [Z_meas, Z_est, Z_dot_est]
        storage['acc_data'][i] = [X_ddot_meas, Y_ddot_meas, Z_ddot_meas]
        storage['gyro_data'][i] = [p_meas, q_meas, r_meas]
        storage['mag_data'][i] = [m9m[0], m9m[1], m9m[2] ]
        storage['attitude_data'][i] = [phi_meas, theta_meas, psi_meas]
        storage['control_input_data'][i] = [U1, U2, U3, U4]
        storage['reference_data'][i] = [Z_des_GF, phi_des, theta_des, psi_des]
        storage['rate_reference_data'][i] = [p_desired, q_desired, r_desired]
        storage['adaptive_gains_roll'][i] = [Phi_KP, Phi_KI, Phi_KD]
        storage['adaptive_gains_pitch'][i] = [Theta_KP, Theta_KI, Theta_KD]
        storage['adaptive_gains_yaw'][i] = [Psi_KP, Psi_KI, Psi_KD]

        # Measure execution time for this iteration
        execution_time = time.perf_counter() - current_time
        
        # Display on terminal
        print(f"ET: {execution_time:.4f}s, ST: {sim_time:.2f}s | Z: {Z_des_GF:.2f}m  / {(Z_meas):.2f}m| "
        f"Phi: {np.rad2deg(phi_des):.2f}° / {np.rad2deg(phi_meas):.2f}° | Theta: {np.rad2deg(theta_des):.2f}° / {np.rad2deg(theta_meas):.2f}° | Psi: {np.rad2deg(psi_des):.2f}° / {np.rad2deg(psi_meas):.2f}° || "
        f"w1: {omega_1:.0f} rad/s, w3: {omega_3:.0f} rad/s | w2: {omega_2:.0f} rad/s, w4: {omega_4:.0f} rad/s ")
        
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
    save_to_matfile(storage, gains, path_prefix='/home/pi/Documents/Quadcopter_Control_v2/data/results/quad_AGD_')
    # --------- saving data completed --------- #
    led.setColor('Green')  # (1, 1, 1)

    print("Experiment completed. Gradually reducing motor speeds to zero.")
    # Gradually reduce motor speeds to NOSPEED_PULSE_LENGTHS
    current_time = time.time()
    reduce_duration = 1  # Duration to smoothly reduce speeds (seconds)
    end_time = current_time + reduce_duration
    while time.time() < end_time:
       # Calculate the remaining fraction of time
       elapsed_time = time.time() - current_time
       progress = elapsed_time / reduce_duration  # From 0 to 1
       # Interpolate omega to smoothly transition to MIN_OMEGA
       omega_current = omega * (1 - progress) + MIN_OMEGA * progress
       # Send the interpolated speed commands
       #send_speed_command(pwm_outputs, omega_current, MIN_PULSE_LENGTH, MAX_PULSE_LENGTH)
       send_speed_command(pwm_outputs, omega_current, MIN_PULSE_LENGTH, MIN_PULSE_LENGTH)
       time.sleep(Ts)  # Maintain timing control
    # Ensure motors are completely stopped at the end
    time.sleep(1)  # Maintain timing control
    #send_speed_command(pwm_outputs, NOSPEED_PULSE_LENGTHS, NOSPEED_PULSE_LENGTHS, MAX_PULSE_LENGTH)
    send_pulse_length(pwm_outputs, NOSPEED_PULSE_LENGTHS)
    
    print("Motors stopped.")

 
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
    system_utils.restart_idle_signal()
    
 
