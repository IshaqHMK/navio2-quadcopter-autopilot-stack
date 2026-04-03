# CAN BE DELETED

import numpy as np
from utils.navio2 import lsm9ds1
from utils.navio2.leds import Led
from imu.madgwick import Madgwick
from imu.madgwick import Madgwick  # Madgwick filter for IMU orientation estimation
# Madgwick filter also relies on the following files: orientation.py, core.py, mathfuncs.py, constants.py
from imu.imu_utils import calibrate_imu, read_imu_data, quaternion_to_euler, euler_to_quaternion  # Utility functions for IMU calibration and data handling
from utils.sensors.altitude_utils import get_altitude_bias

# Altitude Sensor Setup
SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 115200

# Initialize Altitude Sensor
def initialize_altitude_sensor():
    import serial
    return serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# IMU Sensor Initialization
def initialize_imu():
    imu = lsm9ds1.LSM9DS1()
    #imu = mpu9250.MPU9250()
    imu.initialize()
    return imu

# LED Initialization
def initialize_led():
    led = Led()
    led.setColor('Green')
    return led

# PWM Settings
PWM_OUTPUTS = [0, 1, 2, 3]
MIN_PULSE_LENGTH = np.array([1.189, 1.1894, 1.189, 1.1847])
MAX_PULSE_LENGTH = np.array([1.753, 1.7534, 1.767, 1.75])
MIN_OMEGA_RPM = np.array([4030, 4030, 4030, 4030])
MAX_OMEGA_RPM = np.array([21000, 21000, 21000, 21000])

# Convert RPM to rad/s
MIN_OMEGA = (MIN_OMEGA_RPM / 2) * ((2 * np.pi) / 60)
MAX_OMEGA = (MAX_OMEGA_RPM / 2) * ((2 * np.pi) / 60)
NOSPEED_PULSE_LENGTHS = np.array([1.0, 1.0, 1.0, 1.0])

# Quadrotor Physical Parameters
m = 1.2  # Quadrotor mass (kg)
g = 9.80665  # Gravity (m/s^2)
l = 0.25  # Distance from the center of mass to the motors (m)
KT = 1.3328e-5  # Thrust force coefficient
Kd = 1.3858e-6  # Drag torque coefficient

# PID Parameters
PID_GAINS = {
    'Z': {'KP': 20, 'KI': 0, 'KD': 1},
    'Phi': {'KP': 7.5083, 'KI': 3.3677, 'KD': 0.9},
    'Theta': {'KP': 9, 'KI': 2.7023, 'KD': 1.3},
    'Psi': {'KP': 25.0013, 'KI': 16.3225, 'KD': 1.4375},
}

# Madgwick Filter Initialization
def initialize_madgwick():
    return Madgwick(beta=0.048)

# Calibration Process
def calibrate_and_get_bias(imu, samples, Ts):
    return calibrate_imu(imu, samples, Ts)

# Simulation Parameters
Ts = 0.005  # Sampling time (s)
total_simulation_time = 10  # Total simulation time (s)
