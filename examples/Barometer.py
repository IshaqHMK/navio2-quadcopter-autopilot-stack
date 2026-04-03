"""
Provided to you by Emlid Ltd (c) 2014.
twitter.com/emlidtech || www.emlid.com || info@emlid.com

Example: Get pressure from MS5611 barometer onboard of Navio shield for Raspberry Pi

To run this example navigate to the directory containing it and run the following command:
sudo python3 Barometer_example.py
"""

import time
import sys

# Add the project directory to the Python path
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')

# Import MS5611 barometer and utility functions
from utils.navio2 import ms5611, util

# Check for the presence of APM (Autopilot Manager) hardware
util.check_apm()

# Initialize the MS5611 barometer
baro = ms5611.MS5611()
baro.initialize()

# Main loop for reading and displaying barometer data
while True:
    # Refresh pressure data
    baro.refreshPressure()
    time.sleep(0.01)  # Waiting for pressure data to be ready (10ms)
    baro.readPressure()

    # Refresh temperature data
    baro.refreshTemperature()
    time.sleep(0.01)  # Waiting for temperature data to be ready (10ms)
    baro.readTemperature()

    # Calculate and retrieve pressure and temperature values
    baro.calculatePressureAndTemperature()

    # Display the results
    print(f"Temperature (C): {baro.TEMP:.6f}, Pressure (millibar): {baro.PRES:.6f}")

    # Delay before the next reading
    time.sleep(1)
