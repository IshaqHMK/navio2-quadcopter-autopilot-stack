import sys
import os
import time

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required modules from utils folder
from utils.navio2 import pwm, util

# Check if running on an APM (ArduPilot Mega) system
util.check_apm()

# PWM Configuration
PWM_OUTPUT = 0  # Output channel for PWM
SERVO_MIN = 1.250  # Minimum duty cycle in milliseconds
SERVO_MAX = 1.750  # Maximum duty cycle in milliseconds

# Initialize PWM
with pwm.PWM(PWM_OUTPUT) as pwm_controller:
    pwm_controller.set_period(50)  # Set PWM frequency to 50Hz
    pwm_controller.enable()

    while True:
        # Set duty cycle to minimum
        pwm_controller.set_duty_cycle(SERVO_MIN)
        time.sleep(1)  # Wait for 1 second

        # Set duty cycle to maximum
        pwm_controller.set_duty_cycle(SERVO_MAX)
        time.sleep(1)  # Wait for 1 second
