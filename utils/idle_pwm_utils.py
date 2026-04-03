# pwm_utils.py
# Created by Ishaq Hafez (31-DEC-2024)
# Utility functions to manage PWM initialization and control for motor ESCs.

import time
from navio2 import pwm  # Import the PWM module from Navio2

def initialize_pwm_outputs(pwm_channels, frequency=50):
    """
    Initialize and configure PWM outputs for the specified channels.

    Args:
        pwm_channels (list): List of PWM channels to initialize (e.g., [0, 1, 2, 3]).
        frequency (int): PWM frequency in Hz. Default is 50Hz.

    Returns:
        list: List of initialized PWM output objects.

    Steps:
    1. Iterate over the list of `pwm_channels`.
    2. Create a `PWM` object for each channel.
    3. Initialize and configure the PWM output:
        - Set the period to match the specified `frequency`.
        - Enable the PWM channel.
    4. Append each configured PWM object to the `pwm_outputs` list.
    """
    pwm_outputs = []
    for channel in pwm_channels:
        pwm_out = pwm.PWM(channel)  # Create PWM object for the channel
        pwm_out.initialize()  # Initialize the PWM channel
        pwm_out.set_period(frequency)  # Set PWM frequency (default: 50Hz)
        pwm_out.enable()  # Enable the PWM output
        pwm_outputs.append(pwm_out)  # Add the PWM object to the outputs list
    return pwm_outputs  # Return the list of configured PWM outputs

def send_pulse_length_continuous(pwm_outputs, pulse_length, stop_signal):
    """
    Continuously send the current pulse length to all motor ESCs.

    Args:
        pwm_outputs (list): List of initialized PWM output objects.
        pulse_length (dict): Dictionary holding the current pulse length, 
                             e.g., {"value": 1.5}.
        stop_signal (threading.Event): Event to signal stopping the operation.

    Steps:
    1. Enter a loop that runs until the `stop_signal` is set.
    2. Check if the current pulse length (`pulse_length["value"]`) is not None.
    3. Send the pulse length to all PWM outputs by calling `set_duty_cycle()`.
    4. Exit the loop when the `stop_signal` is triggered.
    """
    while not stop_signal.is_set():  # Loop until the stop signal is set
        if pulse_length["value"] is not None:  # Check if a pulse length is set
            for pwm_out in pwm_outputs:  # Iterate over all PWM outputs
                pwm_out.set_duty_cycle(pulse_length["value"])  # Apply the pulse length
        time.sleep(0.5)  # Add a small delay to prevent high CPU usage
