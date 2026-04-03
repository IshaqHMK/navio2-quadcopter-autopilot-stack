"""
MS5611 driver code is placed under the BSD license.
Copyright (c) 2014, Emlid Limited, www.emlid.com
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the Emlid Limited nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL EMLID LIMITED BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import spidev
import time
import argparse
import sys
import os

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Navio2 modules
from utils.navio2 import mpu9250, util, lsm9ds1_backup as lsm9ds1

# Check for APM presence
util.check_apm()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    help="Sensor selection: -i [sensor name]. Sensors names: mpu is MPU9250, lsm is LSM9DS1"
)

if len(sys.argv) == 1:
    print("Enter parameter")
    parser.print_help()
    sys.exit(1)
elif len(sys.argv) == 2:
    sys.exit("Enter sensor name: mpu or lsm")

args = parser.parse_args()

# Initialize the IMU
if args.i == 'mpu':
    print("Selected: MPU9250")
    imu = mpu9250.MPU9250()
elif args.i == 'lsm':
    print("Selected: LSM9DS1")
    imu = lsm9ds1.LSM9DS1()
else:
    print("Wrong sensor name. Select: mpu or lsm")
    sys.exit(1)

# Test IMU connection
if imu.testConnection():
    print("Connection established: True")
else:
    sys.exit("Connection established: False")

imu.initialize()
time.sleep(1)

# Read sensor data in a loop
while True:
    m9a, m9g, m9m = imu.getMotion9()

    print(
        "Acc:",
        "{:+7.3f}".format(m9a[0]),
        "{:+7.3f}".format(m9a[1]),
        "{:+7.3f}".format(m9a[2]),
        " Gyr:",
        "{:+8.3f}".format(m9g[0]),
        "{:+8.3f}".format(m9g[1]),
        "{:+8.3f}".format(m9g[2]),
        " Mag:",
        "{:+7.3f}".format(m9m[0]),
        "{:+7.3f}".format(m9m[1]),
        "{:+7.3f}".format(m9m[2])
    )

    time.sleep(0.5)
