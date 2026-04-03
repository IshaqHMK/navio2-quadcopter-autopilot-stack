import sys
import time

# Add the project directory to the Python path
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')
# Import ADC and utility functions from the utils.navio2 folder
from utils.navio2 import adc, util

# Check for the presence of APM (Autopilot Manager) hardware
util.check_apm()

# Initialize the ADC (Analog-to-Digital Converter)
adc = adc.ADC()
results = [0] * adc.channel_count

# Continuously read ADC values and display them
while True:
    s = ''
    for i in range(adc.channel_count):
        results[i] = adc.read(i)
        s += 'A{0}: {1:6.4f}V '.format(i, results[i] / 1000)
    print(s)
    time.sleep(0.5)
