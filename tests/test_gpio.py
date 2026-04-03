import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)  # Prevent duplicate GPIO warnings
GPIO.setmode(GPIO.BCM)
START_SIGNAL_PIN = 26  # GP26 (Pin 37)

GPIO.setup(START_SIGNAL_PIN, GPIO.OUT)

print("Starting Experiment: HIGH signal (3.3V)")
GPIO.output(START_SIGNAL_PIN, GPIO.HIGH)  # Start signal to dSPACE
time.sleep(0.1)  # Simulate experiment duration

print("Ending Experiment: LOW signal (0V)")
GPIO.output(START_SIGNAL_PIN, GPIO.LOW)  # Stop signal to dSPACE

GPIO.cleanup()  # Properly reset GPIO at the end

