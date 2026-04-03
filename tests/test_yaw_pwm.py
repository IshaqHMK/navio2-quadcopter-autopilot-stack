# IH 2025 - Yaw PWM Test
# Version 7: Reflecting Original Code Structure

# Standard Python Libraries
import time
import numpy as np

# Add the project directory to the Python path
import sys
import os
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')

# Utility Modules for Navio2
from utils.navio2 import pwm  # Functions to send PWM signals to motors
from utils.navio2.util import check_apm  # Navio utility functions (e.g., hardware checks)

from utils import system_utils 


# ---------------------------- PWM Related ------------------------------ #

system_utils.stop_idle_signal()
time.sleep(3)

# PWM Functions
def check_apm_and_initialize_pwm():
    """Checks for correct autopilot hardware and initializes PWM outputs."""
    check_apm()
    pwm_outputs = []
    for channel in PWM_OUTPUTS:
        pwm_out = pwm.PWM(channel)
        pwm_out.initialize()
        pwm_out.set_period(50)
        pwm_out.enable()
        pwm_outputs.append(pwm_out)
    return pwm_outputs

def send_pulse_length(pwm_outputs, pulse_lengths):
    """Sends specified pulse lengths to all motor ESCs."""
    for i, pwm_out in enumerate(pwm_outputs):
        pwm_out.set_duty_cycle(pulse_lengths[i])

def omega_to_pwm(omega_1, omega_2, omega_3, omega_4):
    """Converts angular velocity (omega) to PWM signals."""
    a = np.array([731.542497, 744.994274, 751.594551, 726.769124])
    b = np.array([-789.014715, -806.217581, -813.664533, -790.667045])
    pwm_signals = (np.array([omega_1, omega_2, omega_3, omega_4]) - b) / a
    pwm_signals = np.clip(pwm_signals, MIN_PULSE_LENGTH, MAX_PULSE_LENGTH)
    return pwm_signals

# Constants
PWM_OUTPUTS = [0, 1, 2, 3]  # Motor channels
MIN_PULSE_LENGTH = np.array([1.140, 1.142, 1.141, 1.152])
MAX_PULSE_LENGTH = np.array([2.0, 2.0, 2.0, 2.0])
NOSPEED_PULSE_LENGTHS = np.array([1.0, 1.0, 1.0, 1.0]) 

MIN_OMEGA = np.array([40, 40, 40, 40])  # Minimum angular velocity (rad/s)
MAX_OMEGA = np.array([526.4262, 526.4262, 526.4262, 526.4262])  # Maximum angular velocity (rad/s)

MAX_OMEGA = MAX_OMEGA/4


Ts = 0.001  # Sampling time in seconds

# Main Function
def main():
    # Initialize PWM
    pwm_outputs = check_apm_and_initialize_pwm()
    print("Starting Yaw PWM Test...")

    # Start motors at no speed for safety
    start_no_speed_time = time.time()
    while time.time() - start_no_speed_time < 3:
        send_pulse_length(pwm_outputs, NOSPEED_PULSE_LENGTHS)

    # Simulation parameters
    yaw_sim_time = 10  # Simulation time for each yaw phase (positive and negative)
    hold_time = 5      # Hold time at MIN_OMEGA after each yaw phase
    num_yaw_samples = int(yaw_sim_time / Ts)
    num_hold_samples = int(hold_time / Ts)

    # Positive Yaw Phase
    print("Starting Positive Yaw Phase...")
    start_time = time.perf_counter()
    for i in range(num_yaw_samples):
        current_time = time.perf_counter()
        sim_time = current_time - start_time

        # Ramp-up for negative yaw
        omega_ramp = np.linspace(MIN_OMEGA[0], MAX_OMEGA[0], num_yaw_samples)
        omega_1, omega_3 = omega_ramp[i], omega_ramp[i]
        omega_2, omega_4 = MIN_OMEGA[0], MIN_OMEGA[0]

        # Convert omega to PWM signals
        pwm_signals = omega_to_pwm(omega_1, omega_2, omega_3, omega_4)

        # Send PWM signals
        send_pulse_length(pwm_outputs, pwm_signals)

        # Log and display test values
        print(f"ET: {time.perf_counter() - current_time:.4f}s, ST: {sim_time:.2f}s | "
              f"Omega: [w1: {omega_1:.2f}, w2: {omega_2:.2f}, w3: {omega_3:.2f}, w4: {omega_4:.2f}] | "
              f"PWM: {pwm_signals}")

        # Wait for the next sampling interval
        next_time = current_time + Ts
        while time.perf_counter() < next_time:
            pass

    # Hold Phase After negative Yaw
    print("Holding All Motors at MIN_OMEGA...")
    for _ in range(num_hold_samples):
        send_pulse_length(pwm_outputs, omega_to_pwm(*MIN_OMEGA))
        time.sleep(Ts)

    # Positive Yaw Phase
    print("Starting Negative Yaw Phase...")
    start_time = time.perf_counter()
    for i in range(num_yaw_samples):
        current_time = time.perf_counter()
        sim_time = current_time - start_time

        # Ramp-up for positive yaw
        omega_ramp = np.linspace(MIN_OMEGA[0], MAX_OMEGA[0], num_yaw_samples)
        omega_2, omega_4 = omega_ramp[i], omega_ramp[i]
        omega_1, omega_3 = MIN_OMEGA[0], MIN_OMEGA[0]

        # Convert omega to PWM signals
        pwm_signals = omega_to_pwm(omega_1, omega_2, omega_3, omega_4)

        # Send PWM signals
        send_pulse_length(pwm_outputs, pwm_signals)

        # Log and display test values
        print(f"ET: {time.perf_counter() - current_time:.4f}s, ST: {sim_time:.2f}s | "
              f"Omega: [w1: {omega_1:.2f}, w2: {omega_2:.2f}, w3: {omega_3:.2f}, w4: {omega_4:.2f}] | "
              f"PWM: {pwm_signals}")

        # Wait for the next sampling interval
        next_time = current_time + Ts
        while time.perf_counter() < next_time:
            pass

    # Hold Phase After Positive Yaw
    print("Holding All Motors at MIN_OMEGA...")
    for _ in range(num_hold_samples):
        send_pulse_length(pwm_outputs, omega_to_pwm(*MIN_OMEGA))
        time.sleep(Ts)

    # Stop motors at the end of the simulation
    send_pulse_length(pwm_outputs, MIN_PULSE_LENGTH)
    print("Test completed. Motors stopped.")

    # Disable PWM outputs
    for pwm_out in pwm_outputs:
        pwm_out.disable()

if __name__ == "__main__":
    main()
    system_utils.restart_idle_signal()
