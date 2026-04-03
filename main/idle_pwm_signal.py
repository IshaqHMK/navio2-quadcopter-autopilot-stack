
import time
import threading
import sys
 
# Add the project directory to the Python path
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')

from utils.idle_pwm_utils import initialize_pwm_outputs, send_pulse_length_continuous
 
# Configuration Parameters
MIN_PULSE_LENGTH = 1.0  # Low PWM pulse for idle
PWM_OUTPUTS = [0, 1, 2, 3]  # PWM channels for ESCs

def main():
    pwm_outputs = initialize_pwm_outputs(PWM_OUTPUTS)
    pulse_length = {"value": MIN_PULSE_LENGTH}  # Always send low signal
    stop_signal = threading.Event()

    # Start thread to continuously send low PWM signal
    signal_thread = threading.Thread(
        target=send_pulse_length_continuous,
        args=(pwm_outputs, pulse_length, stop_signal)
    )
    signal_thread.start()

    print("Idle PWM signal running. Enter: sudo pkill -f idle_pwm_signal.py to exit.")
    print("Enter: sudo python3 /home/pi/Documents/Quadcopter_Control_v2/main/idle_pwm_signal.py &  to restart and press ctrl+c")

    try:
        while True:
            time.sleep(5)  # Keep the script running
    except KeyboardInterrupt:
        print("Stopping idle PWM signal...")
        stop_signal.set()
        signal_thread.join()
        for pwm_out in pwm_outputs:
            pwm_out.disable()

if __name__ == "__main__":
    main()
