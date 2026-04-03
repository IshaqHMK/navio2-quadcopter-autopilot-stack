# imu/imu_utils.py

import math
import numpy as np
import time  # For calibration timing


# IMU Functions:
def calibrate_imu(imu, calibration_samples, dt):
    """
    Calibrate the IMU by determining gyro and accelerometer biases, incorporating sampling time.
    """
    # Initialize variables to store sums
    bx, by, bz = 0.0, 0.0, 0.0
    phi_sum, theta_sum, psi_sum = 0.0, 0.0, 0.0

    start_time = time.time()

    # Calibration loop
    for i in range(calibration_samples):
        
        # Gather gyro, accelerometer, and magnetometer data
        m9a, m9g, m9m = imu.getMotion9()
        ax, ay, az = m9a
        gx, gy, gz = m9g
        mx, my, mz = m9m
        
        # Apply the negative transformation to my and mz - 27 jan 2024
        my = -my
        mz = -mz
        
        # Accumulate gyro data for bias calculation
        bx += gx
        by += gy
        bz += gz

        # Calculate and accumulate angle biases
        phi = math.atan2(ay, math.sqrt(ax**2 + az**2))
        theta = math.atan2(-ax, math.sqrt(ay**2 + az**2))
        psi = _compute_tilt_compensated_yaw(mx, my, mz, phi, theta)  # Use tilt-compensated yaw calculation

        phi_sum += phi
        theta_sum += theta
        psi_sum += psi

        # Calculate and accumulate angle biases
        # phi_sum += math.atan2(ay, math.sqrt(ax**2 + az**2))
        # theta_sum += math.atan2(-ax, math.sqrt(ay**2 + az**2))
        
        
        next_sample_time = start_time + (i + 1) * dt
        while time.time() < next_sample_time:
            pass  # Busy wait for precise timing

    # Compute averages
    bx /= calibration_samples
    by /= calibration_samples
    bz /= calibration_samples
    phi_bias = phi_sum / calibration_samples
    theta_bias = theta_sum / calibration_samples
    psi_bias = psi_sum / calibration_samples

    return bx, by, bz, phi_bias, theta_bias, psi_bias, m9a, m9g, m9m


def _compute_tilt_compensated_yaw(mx, my, mz, roll_rad, pitch_rad):
    """
    Compute yaw from magnetometer with tilt compensation
    using the current roll, pitch (in radians).
    
    For a reference, one possible approach:
      Bx = mx * cos(pitch) + my * sin(roll)*sin(pitch) + mz * cos(roll)*sin(pitch)
      By = my * cos(roll) - mz * sin(roll)
      yaw = atan2(-By, Bx)

    You might see variations depending on axis orientation or sign conventions.
    """
    cos_phi = math.cos(roll_rad)
    sin_phi = math.sin(roll_rad)
    cos_theta = math.cos(pitch_rad)
    sin_theta = math.sin(pitch_rad)

    # Tilt-compensated mag components
    Bx = mx * cos_theta + my * sin_phi * sin_theta + mz * cos_phi * sin_theta
    By = my * cos_phi - mz * sin_phi

    # Yaw from magnetometer with tilt compensation
    yaw_rad = math.atan2(-By, Bx)
    return yaw_rad



def read_imu_data(imu, dt, prev_gyro_phi, prev_gyro_theta, prev_psi, q0, madgwick):
    """
    Read IMU data using pre-calibrated biases and return current readings and angles.
    """
    m9a, m9g, m9m = imu.getMotion9()
    ax, ay, az = m9a
    gx, gy, gz = m9g
    mx, my, mz = m9m
    
    phi_acc = math.atan2(ay, az)  
    theta_acc = math.atan2(-ax, math.sqrt(ay**2 + az**2))  

    phi_gyro = prev_gyro_phi + dt * gx # p
    theta_gyro = prev_gyro_theta + dt * gy  # q
    psi_gyro = dt * gz  # r
    
    psi_int = prev_psi + psi_gyro  # Yaw (psi) accumulation over time
    
    
    # Update Madgwick filter and get the current orientation as a quaternion
    # q = madgwick.updateMARG(q=madgwick.q0, gyr=gyr, acc=acc, mag=m9m, dt=dt) # uses magnitude data
    # q = madgwick.updateIMU(q0, gyr=m9g, acc=m9a, dt=dt) # dont use magnitude data

    m9m_corrected = [mx, -my, -mz]  # correct the magnetif field direction 27_jan_2024
    q = madgwick.updateMARG(q0, gyr=m9g, acc=m9a, mag=m9m_corrected, dt=dt) # uses magnitude data
    #q = ekf.update(q0, gyr=m9g, acc=m9a, mag=m9m_corrected, dt=dt) 
    # q = madgwick.updateIMU(q0, gyr=m9g, acc=m9a, dt=dt) # dont use magnitude data
    
    phi, theta, psi = quaternion_to_euler(q)
        
    return ax, ay, az, mx, my, mz, gx, gy, gz, phi, theta, psi, phi_acc, theta_acc, phi_gyro, theta_gyro, psi_gyro, q

def quaternion_to_euler(q):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).
    q: a 4-element array representing the quaternion (w, x, y, z)
    """
    w, x, y, z = q

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    phi = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        theta = math.copysign(np.pi / 2, sinp)
    else:
        theta = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    psi = math.atan2(siny_cosp, cosy_cosp)

    return phi, theta, psi

def euler_to_quaternion(phi, theta, psi):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).
    Angles must be in radians.
    """
    qw = math.cos(phi/2) * math.cos(theta/2) * math.cos(psi/2) + math.sin(phi/2) * math.sin(theta/2) * math.sin(psi/2)
    qx = math.sin(phi/2) * math.cos(theta/2) * math.cos(psi/2) - math.cos(phi/2) * math.sin(theta/2) * math.sin(psi/2)
    qy = math.cos(phi/2) * math.sin(theta/2) * math.cos(psi/2) + math.sin(phi/2) * math.cos(theta/2) * math.sin(psi/2)
    qz = math.cos(phi/2) * math.cos(theta/2) * math.sin(psi/2) - math.sin(phi/2) * math.sin(theta/2) * math.cos(psi/2)
    return np.array([qw, qx, qy, qz])

