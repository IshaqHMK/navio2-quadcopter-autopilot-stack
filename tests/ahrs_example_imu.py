import time
import math
import socket
import sys
import os

# Add the root directory to the system path
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')

from utils.navio2 import lsm9ds1_backup as lsm9ds1
# Constants
G_SI = 9.80665
PI = 3.14159


class AHRS:
    def __init__(self, imu):
        self.sensor = imu
        self.q0 = 1.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        self.twoKi = 0.0  # Integral gain
        self.twoKp = 2.0  # Proportional gain
        self.integralFBx = 0.0
        self.integralFBy = 0.0
        self.integralFBz = 0.0

    def inv_sqrt(self, x):
        """Fast inverse square root."""
        return 1.0 / math.sqrt(x)

    def updateIMU(self, dt):
        """Update quaternion using accelerometer and gyroscope data."""
        #ax, ay, az = self.sensor.read_accelerometer()
        #gx, gy, gz = self.sensor.read_gyroscope()
        # Read accelerometer, gyroscope, and magnetometer data
        m9a, m9g, m9m = self.sensor.getMotion9()
        ax, ay, az = m9a
        gx, gy, gz = m9g
        mx, my, mz = m9m

        # Normalize accelerometer measurements
        norm = self.inv_sqrt(ax * ax + ay * ay + az * az)
        ax *= norm
        ay *= norm
        az *= norm

        # Estimated direction of gravity
        halfvx = self.q1 * self.q3 - self.q0 * self.q2
        halfvy = self.q0 * self.q1 + self.q2 * self.q3
        halfvz = self.q0 * self.q0 - 0.5 + self.q3 * self.q3

        # Error is the cross product between the estimated and measured gravity
        halfex = (ay * halfvz - az * halfvy)
        halfey = (az * halfvx - ax * halfvz)
        halfez = (ax * halfvy - ay * halfvx)

        # Apply integral feedback
        if self.twoKi > 0.0:
            self.integralFBx += self.twoKi * halfex * dt
            self.integralFBy += self.twoKi * halfey * dt
            self.integralFBz += self.twoKi * halfez * dt
            gx += self.integralFBx
            gy += self.integralFBy
            gz += self.integralFBz
        else:
            self.integralFBx = 0.0
            self.integralFBy = 0.0
            self.integralFBz = 0.0

        # Apply proportional feedback
        gx += self.twoKp * halfex
        gy += self.twoKp * halfey
        gz += self.twoKp * halfez

        # Integrate rate of change of quaternion
        gx *= 0.5 * dt
        gy *= 0.5 * dt
        gz *= 0.5 * dt
        qa = self.q0
        qb = self.q1
        qc = self.q2
        self.q0 += -qb * gx - qc * gy - self.q3 * gz
        self.q1 += qa * gx + qc * gz - self.q3 * gy
        self.q2 += qa * gy - qb * gz + self.q3 * gx
        self.q3 += qa * gz + qb * gy - qc * gx

        # Normalize quaternion
        norm = self.inv_sqrt(self.q0 * self.q0 + self.q1 * self.q1 + self.q2 * self.q2 + self.q3 * self.q3)
        self.q0 *= norm
        self.q1 *= norm
        self.q2 *= norm
        self.q3 *= norm

    def get_euler(self):
        """Get roll, pitch, and yaw angles in degrees."""
        roll = math.atan2(2 * (self.q0 * self.q1 + self.q2 * self.q3), 1 - 2 * (self.q1 * self.q1 + self.q2 * self.q2)) * 180.0 / PI
        pitch = math.asin(2 * (self.q0 * self.q2 - self.q3 * self.q1)) * 180.0 / PI
        yaw = math.atan2(2 * (self.q0 * self.q3 + self.q1 * self.q2), 1 - 2 * (self.q2 * self.q2 + self.q3 * self.q3)) * 180.0 / PI
        return roll, pitch, yaw


class Socket:
    def __init__(self, ip="127.0.0.1", port=7000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = (ip, port)

    def send(self, w, x, y, z, rate):
        message = f"{w:.6f} {x:.6f} {y:.6f} {z:.6f} {rate}Hz\n"
        self.sock.sendto(message.encode(), self.server_address)


def main():
    # Initialize the IMU
    imu = lsm9ds1.LSM9DS1()
    imu.initialize()
    

    # Initialize AHRS and Socket
    ahrs = AHRS(imu)
    sock = Socket()

    # Set initial time for loop
    prev_time = time.time()

    while True:
        # Calculate delta time
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        # Update AHRS and get Euler angles
        ahrs.updateIMU(dt)
        roll, pitch, yaw = ahrs.get_euler()

        # Print the results
        print(f"ROLL: {roll:.2f}, PITCH: {pitch:.2f}, YAW: {yaw:.2f}, RATE: {1/dt:.2f} Hz")

        # Send quaternion data over socket
        sock.send(ahrs.q0, ahrs.q1, ahrs.q2, ahrs.q3, int(1 / dt))

        # Sleep to maintain loop rate
        time.sleep(0.001)


if __name__ == "__main__":
    main()
