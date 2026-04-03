import sys
import time

# Add the project directory to the Python path
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')

# Import Navio utility and LED modules
from utils.navio2 import leds, util

# Check for the presence of APM (Autopilot Manager) hardware
util.check_apm()

# Initialize the LED object
led = leds.Led()

# Set the initial LED color to yellow
led.setColor('Yellow')
print("LED is yellow")
time.sleep(1)

# Infinite loop to cycle through LED colors
while True:
    led.setColor('Green')
    print("LED is green")
    time.sleep(1)

    led.setColor('Cyan')
    print("LED is cyan")
    time.sleep(1)

    led.setColor('Blue')
    print("LED is blue")
    time.sleep(1)

    led.setColor('Magenta')
    print("LED is magenta")
    time.sleep(1)

    led.setColor('Red')
    print("LED is red")
    time.sleep(1)

    led.setColor('Yellow')
    print("LED is yellow")
    time.sleep(1)
