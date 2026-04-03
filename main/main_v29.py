# IH 2025
# version 21 is the main version
# Note: copied from version 27  
# here we replace madwick with our custom ekf 
# ekf_test_16_april_2024_LSM_T01VFT4.py
# adaptive grad
# sinusoidal step 1~ 5 degrees attitude control = safe test!


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


# =========== Quadcopter Functions =========== #

# Feedback Linearization Function  
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

# function to calculate for Euler Rates 
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

# Reference Functions: 5 - 04 - 2025
def reference_generator(sim_time, initial_Z, initial_phi, initial_theta, initial_psi, dt, signal):
    """
    Generates reference signals for altitude and attitude.
    signal == 1: Smooth transition then continuous sinusoid.
    signal == 2: Smooth transition then alternating steps.
    signal == 3: Smooth transition, then φ smooth step (half-sine up, hold, half-sine down).
    signal == 4: Smooth transition, then φ and θ each smooth step with their own timings.
    """
    T = 5              # Duration of smooth transitions (ramp up/down)
    hold_time = T + 5  # Hold at zero after initial transition

    # Amplitudes
    A_Z     = 0
    A_phi   = 0 * (np.pi / 180)   # 0 degrees
    A_theta = 0 * (np.pi / 180)
    A_psi   = 0 * (np.pi / 180)

    # Sinusoid frequency (only for signal 1 & 2)
    freq = 0.1
    omega = 2 * np.pi * freq

    # Signal-specific timings (Signal 3)
    t3_start = 40
    t3_end   = 80

    # Signal-specific timings (Signal 4)
    t4_phi_start = 40
    t4_phi_end   = 120 # 80
    t4_theta_start = 80 # 60
    t4_theta_end   = 160 #100
    
    # Signal-specific timings (Signal 5)
    t5_start = 40
    t5_end = 80

    def zero_all():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # 1) Initial smooth transition to zero
    if sim_time < T:
        cos_t = np.cos(np.pi * sim_time / T)
        sin_t = np.sin(np.pi * sim_time / T)
        coeff = 0.5 * (1 + cos_t)
        speed = - (np.pi / (2 * T)) * sin_t

        z_des       = initial_Z      * coeff
        z_dot_des   = initial_Z      * speed
        phi_des     = initial_phi    * coeff
        phi_dot_des = initial_phi    * speed
        theta_des     = initial_theta  * coeff
        theta_dot_des = initial_theta  * speed
        psi_des     = initial_psi    * coeff
        psi_dot_des = initial_psi    * speed

        return (z_des, phi_des, theta_des, psi_des,
                z_dot_des, phi_dot_des, theta_dot_des, psi_dot_des)

    # 2) Hold at zero
    if sim_time < hold_time:
        return zero_all()

    # 3) After hold, per‐signal behavior
    if signal == 1:
        t = sim_time - hold_time
        z_des       = A_Z     * np.sin(omega * t)
        z_dot_des   = A_Z     * omega * np.cos(omega * t)
        phi_des     = A_phi   * np.sin(omega * t)
        phi_dot_des = A_phi   * omega * np.cos(omega * t)
        theta_des     = A_theta * np.sin(omega * t)
        theta_dot_des = A_theta * omega * np.cos(omega * t)
        psi_des     = A_psi   * np.sin(omega * t)
        psi_dot_des = A_psi   * omega * np.cos(omega * t)

    elif signal == 2:
        t = sim_time - hold_time
        step_period = 3
        phase = int(t / step_period) % 2
        if phase == 0:
            z_des = A_Z;     phi_des = A_phi
            theta_des = A_theta; psi_des = A_psi
        else:
            z_des = phi_des = theta_des = psi_des = 0.0

        prev = sim_time - dt
        if prev < hold_time:
            z_prev = phi_prev = theta_prev = psi_prev = 0.0
        else:
            tp = prev - hold_time
            ph = int(tp / step_period) % 2
            if ph == 0:
                z_prev = A_Z;     phi_prev = A_phi
                theta_prev = A_theta; psi_prev = A_psi
            else:
                z_prev = phi_prev = theta_prev = psi_prev = 0.0

        z_dot_des     = (z_des     - z_prev)     / dt
        phi_dot_des   = (phi_des   - phi_prev)   / dt
        theta_dot_des = (theta_des - theta_prev) / dt
        psi_dot_des   = (psi_des   - psi_prev)   / dt

    elif signal == 3:
        
        A_phi   = 5 * (np.pi / 180)   # 5 degrees

        # φ smooth step: ramp-up, hold, ramp-down
        if sim_time < t3_start:
            phi_des = phi_dot_des = 0.0
        elif sim_time < t3_start + T:
            tau = sim_time - t3_start
            phi_des     = A_phi * 0.5 * (1 - np.cos(np.pi * tau / T))
            phi_dot_des = A_phi * (np.pi / (2 * T)) * np.sin(np.pi * tau / T)
        elif sim_time < t3_end:
            phi_des = A_phi
            phi_dot_des = 0.0
        elif sim_time < t3_end + T:
            tau = sim_time - t3_end
            phi_des     = A_phi * 0.5 * (1 + np.cos(np.pi * tau / T))
            phi_dot_des = -A_phi * (np.pi / (2 * T)) * np.sin(np.pi * tau / T)
        else:
            phi_des = phi_dot_des = 0.0

        z_des = z_dot_des = 0.0
        theta_des = theta_dot_des = 0.0
        psi_des = psi_dot_des = 0.0

    elif signal == 4:
        A_phi   = 5 * (np.pi / 180)   # 1 degrees
        A_theta = 5 * (np.pi / 180)
        # φ smooth step
        if sim_time < t4_phi_start:
            phi_des = phi_dot_des = 0.0
        elif sim_time < t4_phi_start + T:
            tau_phi = sim_time - t4_phi_start
            phi_des     = A_phi * 0.5 * (1 - np.cos(np.pi * tau_phi / T))
            phi_dot_des = A_phi * (np.pi / (2 * T)) * np.sin(np.pi * tau_phi / T)
        elif sim_time < t4_phi_end:
            phi_des = A_phi
            phi_dot_des = 0.0
        elif sim_time < t4_phi_end + T:
            tau_phi = sim_time - t4_phi_end
            phi_des     = A_phi * 0.5 * (1 + np.cos(np.pi * tau_phi / T))
            phi_dot_des = -A_phi * (np.pi / (2 * T)) * np.sin(np.pi * tau_phi / T)
        else:
            phi_des = phi_dot_des = 0.0

        # θ smooth step
        if sim_time < t4_theta_start:
            theta_des = theta_dot_des = 0.0
        elif sim_time < t4_theta_start + T:
            tau_theta = sim_time - t4_theta_start
            theta_des     = A_phi * 0.5 * (1 - np.cos(np.pi * tau_theta / T))
            theta_dot_des = A_phi * (np.pi / (2 * T)) * np.sin(np.pi * tau_theta / T)
        elif sim_time < t4_theta_end:
            theta_des = A_phi
            theta_dot_des = 0.0
        elif sim_time < t4_theta_end + T:
            tau_theta = sim_time - t4_theta_end
            theta_des     = A_phi * 0.5 * (1 + np.cos(np.pi * tau_theta / T))
            theta_dot_des = -A_phi * (np.pi / (2 * T)) * np.sin(np.pi * tau_theta / T)
        else:
            theta_des = theta_dot_des = 0.0

        z_des = z_dot_des = 0.0
        psi_des = psi_dot_des = 0.0


    # ψ smooth step (signal 5)
    elif signal == 5:
        A_psi_local = 5 * (np.pi / 180)
        if sim_time < t5_start:
            psi_des = psi_dot_des = 0.0
        elif sim_time < t5_start + T:
            tau = sim_time - t5_start
            psi_des     = A_psi_local * 0.5 * (1 - np.cos(np.pi * tau / T))
            psi_dot_des = A_psi_local * (np.pi / (2 * T)) * np.sin(np.pi * tau / T)
        elif sim_time < t5_end:
            psi_des = A_psi_local
            psi_dot_des = 0.0
        elif sim_time < t5_end + T:
            tau = sim_time - t5_end
            psi_des     = A_psi_local * 0.5 * (1 + np.cos(np.pi * tau / T))
            psi_dot_des = -A_psi_local * (np.pi / (2 * T)) * np.sin(np.pi * tau / T)
        else:
            psi_des = psi_dot_des = 0.0

        z_des = z_dot_des = 0.0
        phi_des = phi_dot_des = 0.0
        theta_des = theta_dot_des = 0.0


    else:
        return zero_all()

    return (
        z_des,     phi_des,     theta_des,     psi_des,
        z_dot_des, phi_dot_des, theta_dot_des, psi_dot_des
    )

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
def attitude_altitude_PID(
    z_des, phi_des, theta_des, psi_des,
    z_dot_des, phi_dot_des, theta_dot_des, psi_dot_des,
    z_meas, phi_meas, theta_meas, psi_meas,
    z_dot_meas, phi_dot_meas, theta_dot_meas, psi_dot_meas,
    p_meas, q_meas, r_meas,
    Z_KP, Z_KI, Z_KD,
    Phi_KP, Phi_KI, Phi_KD,
    Theta_KP, Theta_KI, Theta_KD,
    Psi_KP, Psi_KI, Psi_KD,
    Ts, m, g ):
    """
    Attitude PID controller: computes control input U1 and desired angular rates.
    """
    global z_error_sum, phi_error_sum, theta_error_sum, psi_error_sum
    global previous_z_error, previous_phi_error, previous_theta_error, previous_psi_error

    # Altitude PID
    z_error = z_des - z_meas
    z_error_sum += z_error * Ts
    z_error_dot = z_dot_des - z_dot_meas
    cp = Z_KP * z_error
    ci = Z_KI * z_error_sum
    cd = Z_KD * z_error_dot
    U1 = cp + ci + cd
    previous_z_error = z_error

    # Roll PID
    phi_error = phi_des - phi_meas
    phi_error_sum += phi_error * Ts
    phi_error_dot = phi_dot_des - phi_dot_meas
    cp = Phi_KP * phi_error
    ci = Phi_KI * phi_error_sum
    cd = Phi_KD * phi_error_dot
    p_desired = cp + ci + cd
    previous_phi_error = phi_error

    # Pitch PID
    theta_error = theta_des - theta_meas
    theta_error_sum += theta_error * Ts
    theta_error_dot = theta_dot_des - theta_dot_meas
    cp = Theta_KP * theta_error
    ci = Theta_KI * theta_error_sum
    cd = Theta_KD * theta_error_dot
    q_desired = cp + ci + cd
    previous_theta_error = theta_error

    # Yaw PID
    psi_error = psi_des - psi_meas
    psi_error_sum += psi_error * Ts
    psi_error_dot = psi_dot_des - psi_dot_meas
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

# =========== Quadcopter Classes =========== #
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
        accNoiseVar = 4e-0
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

# -------------------- Adaptive Gain Tuner Class -------------------- #
class AdaptiveGains:
    """
    This class adapts the PID gains (Kp, Ki, Kd) using gradient descent on a
    cost function of the form:

       L = 1/2 * e^2         for the PID error   

    We apply the chain rule exactly as shown in Eqn (8):

       dL/dKp = dL/dy * dy/du * du/dKp

    Then incorporate a simple momentum term:
       v <- gamma * v - alpha * grad
       param <- param + v
    """

    def __init__(
        self,
        initial_Kp, initial_Ki, initial_Kd,
        alpha=0.0, gamma=0.0,  # alpha = learning rate, gamma = momentum
        bounds_Kp=(0,1), bounds_Ki=(0,1), bounds_Kd=(0,1),
        axis_name="altitude"
    ):
        # PID gains
        self.Kp = initial_Kp
        self.Ki = initial_Ki
        self.Kd = initial_Kd

        # Learning/momentum rates
        self.alpha = alpha
        self.gamma = gamma

        # Bounds
        self.bounds_Kp = bounds_Kp
        self.bounds_Ki = bounds_Ki
        self.bounds_Kd = bounds_Kd

        # Momentum velocities for each parameter
        self.vKp = 0.0
        self.vKi = 0.0
        self.vKd = 0.0

        # Corrected: store grad_y_u
        self.last_grad_y_u = None         # NEW: store last valid grad_y_u
        self.grad_history = []             # NEW: record grad_y_u values over iterations

        # For PID derivative & integral
        self.prev_error   = None
        self.integral     = 0.0

        # For chain-rule approximation of dy/du
        self.prev_control = None
        self.prev_output  = None

        # NEW: Store previous desired reference for computing desired derivative
        self.prev_desired = None

        self.axis_name    = axis_name

    # def update(self, error, control, output, dt, rate_val=0.0):
    def update(self, error, control, output, dt, desired=None, desired_dot=None, measured_rate=None):
    
        """
        Performs one update step for Kp, Ki, Kd, Kn.

        Args:
          error   : (r - y)  the tracking error in the PID sense
          control : previous control signal (u_{t-1}) used in chain rule
          output  : previous measured output (y_{t-1}) used in chain rule
          dt      : sample time
          rate_val: a "rate" (z_dot, p, q, r) for feedback-lin cost = 1/2*(rate^2)

        Returns:
          Updated (Kp, Ki, Kd, Kn).
        """

        # ============= 1) Compute partial derivatives for PID =============
        #
        # We want dL/dKp = dL/dy * dy/du * du/dKp.
        #
        #   - dL/dy, for L = 1/2 * e^2, is:  dL/dy = (dL/de)*(de/dy) = e * (-1) = - e
        #       because e = r - y => de/dy = -1
        #
        #   - dy/du is approx (output - prev_output)/(control - prev_control), i.e. grad_y_u
        #
        #   - du/dKp = partial of [u = Kp*error + Ki*∫error + Kd*d(error)/dt ] wrt Kp => error
        #
        # => dL/dKp = [ - e ] * [ grad_y_u ] * [ e ] = - e^2 * grad_y_u
        #
        # Similarly:
        #   dL/dKi = [ - e ] * [ grad_y_u ] * [ integral_of_error ]
        #   dL/dKd = [ - e ] * [ grad_y_u ] * [ d(error)/dt ]
        #
        # ============= 2) Compute partial derivatives for feedback-lin =============
        #
        # Suppose we define L_{Kn} = 1/2 * (rate_val)^2, so partial wrt (rate_val) = rate_val.
        # Then the chain:  dL/dKn = dL/d(rate_val) * d(rate_val)/du * du/dKn
        # In practice, we do a similar approximate approach:
        #   dL/dKn = - rate_val * grad_y_u * rate_val   (mirroring the code below)
        #
        # ============= 3) Momentum update =============
        #
        #   vKp <- gamma * vKp - alpha * (dL/dKp)
        #   Kp  <- Kp + vKp
        #
        # Then clamp to bounds, etc.

        # Accumulate integral for Ki
        self.integral += error * dt

        # Compute derivative of error ----------- ***
        # 1) if caller supplied desired_dot, use it directly
        if desired_dot is not None and measured_rate is not None:
            derivative = desired_dot - measured_rate

        # 2) otherwise if they passed desired but no desired_dot, approximate it
        elif desired is not None and measured_rate is not None:
            print("No desired_dot provided; approximating via finite difference")
            if self.prev_desired is None:
                approx_desired_dot = 0.0
            else:
                approx_desired_dot = (desired - self.prev_desired) / dt
            derivative = approx_desired_dot - measured_rate
            self.prev_desired = desired

        # 3) otherwise fall back to derivative of error
        else:
            print("No desired or desired_dot given; using error derivative")
            if self.prev_error is None:
                derivative = 0.0
            else:
                derivative = (error - self.prev_error) / dt

        # Corrected: Approx partial y/partial u
        if self.prev_control is not None and abs(control - self.prev_control) > 1e-50:
            grad_y_u = (output - self.prev_output) / (control - self.prev_control)
            self.last_grad_y_u = grad_y_u  # update last valid value
        else:
            grad_y_u = self.last_grad_y_u if self.last_grad_y_u is not None else 0.0
        self.grad_history.append(grad_y_u)  # NEW: save for later plotting


        # dL/dy for  1/2*(r - y)^2  is  - error
        dL_dy = -error

        # -----------
        # For PID:
        # partial u wrt Kp => error
        # partial u wrt Ki => integral
        # partial u wrt Kd => derivative
        # => chain: dL/dKp = dL/dy * dy/du * error
        gradKp = dL_dy * grad_y_u * error      # eqn (8) in screenshot
        gradKi = dL_dy * grad_y_u * self.integral
        gradKd = dL_dy * grad_y_u * derivative

        # ============= Momentum updates =============
        self.vKp = self.gamma*self.vKp - self.alpha*gradKp
        self.vKi = self.gamma*self.vKi - self.alpha*gradKi
        self.vKd = self.gamma*self.vKd - self.alpha*gradKd
   
        # ============= Apply updates to the parameters =============
        self.Kp += self.vKp
        self.Ki += self.vKi
        self.Kd += self.vKd


        # ============= Bounds =============
        self.Kp = max(min(self.Kp, self.bounds_Kp[1]), self.bounds_Kp[0])
        self.Ki = max(min(self.Ki, self.bounds_Ki[1]), self.bounds_Ki[0])
        self.Kd = max(min(self.Kd, self.bounds_Kd[1]), self.bounds_Kd[0])

        # Store for next iteration
        self.prev_error   = error
        self.prev_control = control
        self.prev_output  = output

        return self.Kp, self.Ki, self.Kd

def routh_hurwitz_stable(Kp, Ki, Kd, I):
    """
    Check 1‑axis PID stability by the three Routh–Hurwitz inequalities:
      Kp > (I*Ki)/Kd
      Ki < (Kp*Kd)/I
      Kd > (I*Ki)/Kp
    """
    return (
        Kp  > (I * Ki) / Kd
        and Ki  < (Kp * Kd) / I
        and Kd  > (I * Ki) / Kp
    )

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
          - Accelerometer measurements: x_ddot_meas, y_ddot_meas, z_ddot_meas
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
    x_ddot_meas, y_ddot_meas, z_ddot_meas = m9a_list[0], m9a_list[1], m9a_list[2]
    p_meas, q_meas, r_meas = m9g_list[0], m9g_list[1], m9g_list[2]
    
    return (x_ddot_meas, y_ddot_meas, z_ddot_meas,
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

 # -------- Initialize -------- #
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
K2 = 0 # for phi
K3 = 0 # for theta
K4 = 0  # for psi

# PID gains 
Z_KP, Z_KI, Z_KD = 0, 0, 0

PQR_PID_Enable = False
P_KP, P_KI, P_KD = 0.1, 0, 0.01  
Q_KP, Q_KI, Q_KD = 0.1, 0, 0.01  
R_KP, R_KI, R_KD = 0.1, 0, 0.01

# ---------------------- Simulation settings ---------------------- #
Ts = 0.005  # Sampling time in seconds (increased to 10ms)
total_simulation_time = 10 #200 # Simulation time in seconds  

# ---------------------- Get Initial Attitude from Accelrometer and Mag ---------------------- #
get_init_att_samples = 10
# Calibrate IMU and get biases
bx, by, bz, phi_bias, theta_bias, psi_bias, m9a, m9g, m9m = calibrate_imu(imu, get_init_att_samples, Ts)

# Print and check if it matches the system
print(f"--> Phi (Acc): {np.rad2deg(phi_bias):.2f}°, Theta (Acc): {np.rad2deg(theta_bias):.2f}°, Psi (Mag): {np.rad2deg(psi_bias):.2f}°")

# ---------------------- Measure altitude bias ---------------------- #
alt_bias = get_altitude_bias(get_init_att_samples, ser)

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

    # ========= INITIALIZATION =========  #
    #
    # --------- initialize PWM --------- #
    pwm_outputs = check_apm_and_initialize_pwm()
    #
    # --------- initialize altitude related variables --------- #
    last_valid_altitude = read_lidar(ser) - alt_bias  # Initialize with 0 m
    delta_altitude_threshold = 0.15  # Define a threshold for jerk detection (delta m)
    #
    # --------- initialize state variables for altitude integration --------- # (IP - check)
    z_dot_est = 0.0         # Initial vertical speed
    z_est = 0.0             # Initial position
    alpha_dot_est = 0.25    # Complementary filter coefficient for altitude speed
    #
    # --------- initialize Custom EKF Filter  --------- #
    # Read a few gyro samples for gyro mean and variance estimation.
    calibration_samples = 200
    gyro_calib = []
    for i in range(calibration_samples):
        m9a, m9g, m9m = imu.getMotion9()
        gyro_calib.append(m9g)
        time.sleep(Ts)
    gyro_calib = np.array(gyro_calib)
    # Compute gyro mean and bias variance 
    gyro_mean = np.mean(gyro_calib, axis=0)         # [bias_x, bias_y, bias_z]
    gyro_var = np.var(gyro_calib, axis=0)         # variance for each axis
    #
    # Set initial quaternion from calibration (assuming zero Euler angles)
    q0 = euler_to_quaternion(phi_bias, theta_bias, 0*psi_bias)
    #
    # Instantiate our custom EKF filter and initialize its biases
    custom_ekf = CustomEKF(gyro_mean, gyro_var, q0)
    #
    # Precompute recursive low-pass filter coefficient
    filter_cutoff_attitude = 5 #1 #5 # 20  # Hz
    alpha_attitude = Ts / (Ts + (1 / (2 * np.pi * filter_cutoff_attitude)))
    #
    # --------- initialize filtered variables --------- #
    filtered_phi = None
    filtered_theta = None
    filtered_psi = None
    # 
    # Filter accelerometer/gyro 
    prev_acc = None # 
    prev_gyro = None # to filter gyro measurements
    filter_cutoff_sensor = 10  # Hz
    alpha_sensor = Ts / (Ts + (1 / (2 * np.pi * filter_cutoff_sensor)))
    #
    # --------- End of: Step 1 of Custom EKF Filter  --------- #
    #
    # Initialize gyroscopic omega for feedback linearization term
    gyroscopic_term_omega = 0.0 # gyroscopic term omega 
    #
    # ------------- Adaptive PID Part -----------------
    #
    # Learning Rate Alpha [0.01, 0.2] 
    # high alpha = destabilize by overshooting + fast gain update
    # low alpha = Stable but slow to adapt (sluggish).
    #
    # Momentum Factor Gamma [0.5, 0.95]
    # high gamma = Smooth updates, improves stability, slows adaptation slightly.
    # low gamma = Fast, but may cause oscillations.
    # start with: alpha = 0.05, gamma = 0.9


    # initial gains: 
    Phi_KP, Phi_KI, Phi_KD = 4.2, 1.7, 3.7
    Theta_KP, Theta_KI, Theta_KD =  4.5, 3, 4
    Psi_KP, Psi_KI, Psi_KD = 10, 3, 0.1

    Phi_KP, Phi_KI, Phi_KD = 4, 1, 3.7
    Theta_KP, Theta_KI, Theta_KD =  4, 1, 4
    Psi_KP, Psi_KI, Psi_KD = 10, 1, 0.01


    Phi_KP, Phi_KI, Phi_KD = 4, 1, 3.6
    Theta_KP, Theta_KI, Theta_KD =  4, 1, 3.9
    Theta_KP, Theta_KI, Theta_KD =  3.9, 1, 3.9
    Phi_KP, Phi_KI, Phi_KD = 4, 1, 3.5

    Phi_KP, Phi_KI, Phi_KD = 4, 0.9, 3.5 
    Theta_KP, Theta_KI, Theta_KD =  3.9, 0.9, 3.9

    # Phi_KP, Phi_KI, Phi_KD = 4.023927175939060, 1.068629481622863, 3.637579725136221
    # Theta_KP, Theta_KI, Theta_KD = 4.035878494890100, 1.137221831343394, 3.966582628055617
    # Psi_KP, Psi_KI, Psi_KD = 9.963381946344220, 1.020900317739730, 0.103978532224160

    # back to original gains:
    Phi_KP, Phi_KI, Phi_KD = 4, 1, 3.7
    Theta_KP, Theta_KI, Theta_KD =  4, 1, 4
    Psi_KP, Psi_KI, Psi_KD = 10, 1, 0.01

    alpha_rpy = 0.03  
    gamma_rpy = 0.4  #   <---  change this

    # Initialize AdaptiveGains for each axis    
    adaptive_phi = AdaptiveGains(
        initial_Kp=Phi_KP, initial_Ki=Phi_KI, initial_Kd=Phi_KD,
        alpha=alpha_rpy, gamma=gamma_rpy,
        bounds_Kp=(3.5, 4.5),
        bounds_Ki=(0.7, 4),
        bounds_Kd=(3, 4.5),
        axis_name="roll"
    )

    adaptive_theta = AdaptiveGains(
        initial_Kp=Theta_KP, initial_Ki=Theta_KI, initial_Kd=Theta_KD,
        alpha=alpha_rpy, gamma=gamma_rpy,
        bounds_Kp=(3.5, 5),
        bounds_Ki=(0.7, 4),
        bounds_Kd=(3, 4.5),
        axis_name="pitch"
    )

    adaptive_psi = AdaptiveGains(
        initial_Kp=Psi_KP, initial_Ki=Psi_KI, initial_Kd=Psi_KD,
        alpha=alpha_rpy, gamma=gamma_rpy,
        bounds_Kp=(7, 11),
        bounds_Ki=(0.7, 4),
        bounds_Kd=(0.0001, 1),
        axis_name="yaw"
    )

    # LED Color white:  all initialization completed
    led.setColor('White')  # (0, 1, 0)
    
    # ---------------------------------------------------------------
    # Send no-speed pulse lengths for 5 seconds before the experiment
    # Warm-up motors in the last 3 seconds of the initial 5-second delay
    # ---------------------------------------------------------------
    print("======== Starting Experiment in 5 seconds ========")
    start_no_speed_time = time.time()
    total_delay = 5  # Total delay before starting the experiment
    #
    while time.time() - start_no_speed_time < total_delay:
        elapsed_time = time.time() - start_no_speed_time
        send_pulse_length(pwm_outputs, MIN_PULSE_LENGTH)
    #
        # ---------- measure altitude (before experimenting) ----------
        z_meas = 0 #read_lidar(ser)  
        if z_meas is not None:
            z_meas -= alt_bias  # Subtract bias to make initial point exactly 0
            if abs(z_meas - last_valid_altitude) <= delta_altitude_threshold:
                last_valid_altitude = z_meas  # Update last valid altitude
        else:
            z_meas = last_valid_altitude
    #
        # +++++++++++ estimate attitude (before experimenting - EKF ) +++++++++++ #
        (
            x_ddot_meas, y_ddot_meas, z_ddot_meas,
            p_meas, q_meas, r_meas,
            phi_meas, theta_meas, psi_meas,
            q0,
            prev_acc, prev_gyro,
            filtered_phi, filtered_theta, filtered_psi,
            x_state
        ) = estimate_attitude_and_rates(
                imu=imu, Ts=Ts, alpha_sensor=alpha_sensor, alpha_attitude=alpha_attitude,
                q0=q0, custom_ekf=custom_ekf,
                gyro_bias=gyro_mean, prev_acc=prev_acc, prev_gyro=prev_gyro,
                filtered_phi=filtered_phi, filtered_theta=filtered_theta, filtered_psi=filtered_psi,
                iteration_index=i  # Use the current iteration index
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        time.sleep(Ts)
    #
    # Set Initial Theta Value (exp will start from current measured attitude and altitude)
    initial_Z = z_meas # in meters
    initial_phi = phi_meas # in radian 
    initial_theta = theta_meas # in radian 
    initial_psi = psi_meas # in radian
    #
    print(f"Initial IMU Measurments: X_ddot:  {x_ddot_meas:.2f}, Y_ddot {y_ddot_meas:.2f}, Z_ddot {z_ddot_meas:.2f} | "
        f"Phi: {np.rad2deg(initial_phi):.2f}°, Theta: {np.rad2deg(initial_theta):.2f}°, Psi: {np.rad2deg(initial_psi):.2f}°. ")
    #
    # initialize arrays to store mesurements: 
    num_samples = int(total_simulation_time / Ts)
    storage = initialize_storage(num_samples)
    #
    # --------------------------------------------------------------- #
    start_time = time.perf_counter()  # High-resolution timer for simulation time
    #
    # Color red: Experiment started
    led.setColor('Red')  
    #
    for i in range(num_samples):
        #
        # Get the current time:
        current_time = time.perf_counter()  
        sim_time = current_time - start_time # sim time starts from 0
        #
        time_min = 6
        time_max = total_simulation_time 
        #
        #  ================== State Estimation =============== #
        # Step 1: measure altitude
        z_meas = read_lidar(ser)
        if z_meas is not None:
            z_meas -= alt_bias  # Subtract bias to make initial point exactly 0
            if abs(z_meas - last_valid_altitude) <= delta_altitude_threshold:
                last_valid_altitude = z_meas  # Update last valid altitude
        else:
            # Use previous measurement if no new data
            z_meas = last_valid_altitude
        #
        # Use a complementary filter for vertical speed and position ----> IP
        z_dot_est = alpha_dot_est * (z_dot_est + (z_ddot_meas - g) * Ts) + (1 - alpha_dot_est) * ((z_meas - z_est) / Ts)
        z_est += z_dot_est * Ts
        z_dot_meas = z_dot_est
        #
        # Step 2: measure imu and estimate using EKF
        # +++++++++++ estimate attitude (before experimenting - EKF ) +++++++++++ #
        (
            x_ddot_meas, y_ddot_meas, z_ddot_meas,
            p_meas, q_meas, r_meas,
            phi_meas, theta_meas, psi_meas,
            q0,
            prev_acc, prev_gyro,
            filtered_phi, filtered_theta, filtered_psi,
            x_state
        ) = estimate_attitude_and_rates(
                imu=imu, Ts=Ts, alpha_sensor=alpha_sensor, alpha_attitude=alpha_attitude,
                q0=q0, custom_ekf=custom_ekf, gyro_bias=gyro_mean,
                prev_acc=prev_acc, prev_gyro=prev_gyro,
                filtered_phi=filtered_phi, filtered_theta=filtered_theta, filtered_psi=filtered_psi,
                iteration_index=i  # Use the current iteration index
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #
        #  ================== REFERENCE SIGNAL =============== #
        z_des, phi_des, theta_des, psi_des, z_dot_des, phi_dot_des, theta_dot_des, psi_dot_des = reference_generator(
            sim_time, initial_Z, initial_phi, initial_theta, initial_psi, Ts, signal=4)
        #
        ### ----------------- Adaptive PID Part 1 --------------- ###
        # Compute errors for each axis 
        error_z = z_des - z_meas
        error_phi = phi_des - phi_meas
        error_theta = theta_des - theta_meas
        error_psi = psi_des - psi_meas
        # measure attitude rates
        phi_dot_meas, theta_dot_meas, psi_dot_meas = euler_angle_rates(p_meas, q_meas, r_meas, phi_meas, theta_meas)
        
        # Adapt altitude gains (Kp,Ki,Kd)
        # For iterations beyond the first, update adaptive gains using previous iteration's control and output.
        if i>0:

            Phi_KP, Phi_KI, Phi_KD = adaptive_phi.update(
                error = error_phi,
                control= p_desired_prev,
                output = phi_meas_prev, # phi_prev,
                dt     = Ts,
                desired=phi_des,           # Current desired roll angle
                desired_dot = phi_dot_des,
                measured_rate=phi_dot_meas  # Measured  rate   
            )
            if not routh_hurwitz_stable(Phi_KP, Phi_KI, Phi_KD, Jx):
                Phi_KP, Phi_KI, Phi_KD = prev_Phi_gains
                print("instable phi")
            else:
                prev_Phi_gains = (Phi_KP, Phi_KI, Phi_KD)

            Theta_KP, Theta_KI, Theta_KD = adaptive_theta.update(
                error = error_theta,
                control= q_desired_prev,
                output = theta_meas_prev,
                dt     = Ts,
                desired=theta_des,           # Current desired pitch angle
                desired_dot = theta_dot_des,
                measured_rate=theta_dot_meas  # Measured  rate  
            )
            if not routh_hurwitz_stable(Theta_KP, Theta_KI, Theta_KD, Jy):
                Theta_KP, Theta_KI, Theta_KD = prev_Theta_gains
                print("instable theta")
            else:
                prev_Theta_gains = (Theta_KP, Theta_KI, Theta_KD)

            Psi_KP, Psi_KI, Psi_KD = adaptive_psi.update(
                error = error_psi,
                control= r_desired_prev,
                output = psi_meas_prev,
                dt     = Ts,
                desired=psi_des,      # Current desired psi angle
                desired_dot = psi_dot_des,
                measured_rate=psi_dot_meas  # Measured  rate 
            )
            if not routh_hurwitz_stable(Psi_KP, Psi_KI, Psi_KD, Jz): # if unstable
                Psi_KP, Psi_KI, Psi_KD = prev_Psi_gains
                print("instable psi")
            else:
                prev_Psi_gains = (Psi_KP, Psi_KI, Psi_KD)

        # ------------- Adaptive GD PID Part 1 - end ----------------- #

        #Phi_KP, Phi_KI, Phi_KD = 6, 1, 3.5 # 
        #Theta_KP, Theta_KI, Theta_KD = 7, 1 , 3.25
        #Psi_KP, Psi_KI, Psi_KD = 12, 2, 3
        #Phi_KP, Phi_KI, Phi_KD = 4, 2, 4.211842902863446
        #Theta_KP, Theta_KI, Theta_KD =5.054176070041542, 2.440842622881349, 5.0

        #Theta_KP, Theta_KI, Theta_KD =  5.005687226464850, 2.584760389724215, 4.935138985982476
        #Psi_KP, Psi_KI, Psi_KD =20.012539351376663, 4.021138590290899, 2.378723676476598
        #Phi_KP, Phi_KI, Phi_KD =  4.268117245885182, 3.398894134632787, 2.974725283043803

        #Theta_KP, Theta_KI, Theta_KD = 4.910366077290426, 2.336701887802864, 5.396521850604633
        #Psi_KP, Psi_KI, Psi_KD = 20.014960264437020, 4.722594626468815, 1.825655753048644
        #Theta_KP, Theta_KI, Theta_KD = 6.910366077290426, 2.336701887802864, 5.396521850604633

        #Phi_KP, Phi_KI, Phi_KD = 4.3, 0.5, 2
        #Theta_KP, Theta_KI, Theta_KD = 5, 0.5, 2
        #Psi_KP, Psi_KI, Psi_KD = 40, 1, 0.5

        #Phi_KP, Phi_KI, Phi_KD = 4.3, 0.5, 3.75
        #Theta_KP, Theta_KI, Theta_KD = 5, 0.5, 4.1
        # Psi_KP, Psi_KI, Psi_KD = 15, 1, 0.5



        #  ================== ATTITUDE CONTROLLER =============== #
        U1, p_desired, q_desired, r_desired = attitude_altitude_PID(z_des, phi_des, theta_des, psi_des, 
                                                           z_dot_des, phi_dot_des, theta_dot_des, psi_dot_des,
                                                           z_meas, phi_meas, theta_meas, psi_meas, 
                                                           z_dot_meas, phi_dot_meas, theta_dot_meas, psi_dot_meas,
                                                           p_meas, q_meas, r_meas, 
                                                           Z_KP, Z_KI, Z_KD, 
                                                           Phi_KP, Phi_KI, Phi_KD, 
                                                           Theta_KP, Theta_KI, Theta_KD, 
                                                           Psi_KP, Psi_KI, Psi_KD, 
                                                           Ts, m, g)
        

        ### ----------------- Adaptive PID Part 2 --------------- ###
        # Save current control outputs for next iteration
        U1_prev, p_desired_prev, q_desired_prev, r_desired_prev = U1, p_desired, q_desired, r_desired
        # Save current measured states for next iteration
        z_meas_prev, phi_meas_prev, theta_meas_prev, psi_meas_prev = z_meas, phi_meas, theta_meas, psi_meas
        # ------------- Adaptive PID Part Ends Here -----------------

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
        storage['altitudes'][i] = [z_meas, z_est, z_dot_est]
        storage['acc_data'][i] = [x_ddot_meas, y_ddot_meas, z_ddot_meas]
        storage['gyro_data'][i] = [p_meas, q_meas, r_meas]
        storage['mag_data'][i] = [m9m[0], m9m[1], m9m[2] ]
        storage['attitude_data'][i] = [phi_meas, theta_meas, psi_meas]
        storage['control_input_data'][i] = [U1, U2, U3, U4]
        # storage['reference_data'][i] = [z_des, phi_des, theta_des, psi_des, z_dot_des, phi_dot_des, theta_dot_des, psi_dot_des ]
        storage['reference_data'][i] = [z_des, phi_des, theta_des, psi_des ]
        storage['rate_reference_data'][i] = [p_desired, q_desired, r_desired]
        storage['adaptive_gains_roll'][i] = [Phi_KP, Phi_KI, Phi_KD]
        storage['adaptive_gains_pitch'][i] = [Theta_KP, Theta_KI, Theta_KD]
        storage['adaptive_gains_yaw'][i] = [Psi_KP, Psi_KI, Psi_KD]

        # Measure execution time for this iteration
        execution_time = time.perf_counter() - current_time
        
        # Display on terminal
        print(f"ET: {execution_time:.4f}s, ST: {sim_time:.2f}s | Z: {z_des:.2f}/{(z_meas):.2f}m| "
        f"Phi: {np.rad2deg(phi_des):.2f}/{np.rad2deg(phi_meas):.2f}° | Theta: {np.rad2deg(theta_des):.2f}/{np.rad2deg(theta_meas):.2f}° | Psi:{np.rad2deg(psi_des):.2f}/{np.rad2deg(psi_meas):.2f} || "
        f"w1:{omega_1:.0f}, w3:{omega_3:.0f} | w2:{omega_2:.0f}, w4:{omega_4:.0f} rad/s")
        
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
    
 
