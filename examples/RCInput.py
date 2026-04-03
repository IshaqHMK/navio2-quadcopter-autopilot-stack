import sys
import os
import time

# Dynamically add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required modules from the utils folder
from utils.navio2 import rcinput, util

# Check if running on an APM (ArduPilot Mega) system
util.check_apm()

# Initialize RCInput
rcin = rcinput.RCInput()

# Continuously read and display RC channel input
while True:
    # Read the signal period from channel 2
    period = rcin.read(2)
    print(period)  # Print the period (Python 3 uses parentheses for print)
    time.sleep(1)  # Wait for 1 second
