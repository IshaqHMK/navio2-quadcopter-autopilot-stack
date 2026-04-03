# sensors/altitude_utils.py
import time  # For timing during sensor calibration and measurement

def get_altitude_bias(N, ser):
    """
    Calculate the average bias for the altitude sensor over N samples.
    
    Args:
        N (int): Number of samples for averaging.
        ser (serial.Serial): Serial object for communicating with the LiDAR sensor.

    Returns:
        float: Calculated altitude bias.
    """
    alt_sum = 0.0
    for _ in range(N):
        Z_meas = read_lidar(ser)
        if Z_meas is not None:
            alt_sum += Z_meas
        time.sleep(0.05)
    return alt_sum / N

def read_lidar(ser):
    """
    Read a single distance measurement from the LiDAR sensor.
    
    Args:
        ser (serial.Serial): Serial object for communicating with the LiDAR sensor.
    
    Returns:
        float: Distance in meters (negative for upward Z).
    """
    if ser.in_waiting >= 9:
        recv = ser.read(9)
        ser.reset_input_buffer()
        if recv[0] == 0x59 and recv[1] == 0x59:
            distance_cm = recv[2] + recv[3] * 256
            distance_meters = distance_cm / 100.0  # Convert to meters
            return -distance_meters  # Z is negative upwards
    return None
