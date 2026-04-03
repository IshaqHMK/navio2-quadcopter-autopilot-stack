# pwm_utils.py - Contains functions for PWM control
# CANCELLED - FILE CAN BE DELETED





from utils.navio2 import pwm  # Importing the pwm module from Navio2



def check_apm_and_initialize_pwm(PWM_OUTPUTS, util):
    """Checks for correct autopilot hardware and initializes PWM outputs."""
    util.check_apm()  # Use the updated path to check APM
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

def send_speed_command(pwm_outputs, omega, MIN_PULSE_LENGTH, MAX_PULSE_LENGTH, MIN_OMEGA, MAX_OMEGA):
    """Converts speed command to pulse lengths and sends to motors."""
    pulse_lengths = MIN_PULSE_LENGTH + (MAX_PULSE_LENGTH - MIN_PULSE_LENGTH) * (omega - MIN_OMEGA) / (MAX_OMEGA - MIN_OMEGA)
    send_pulse_length(pwm_outputs, pulse_lengths)
