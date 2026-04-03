# system_utils.py
# Created by Ishaq Hafez (31-DEC-2024)
# Utility functions to manage system-level configurations like CPU affinity,
# real-time priority, and memory locking for improved performance in real-time applications.

import os  # Provides OS-level functionalities like getting process ID and managing priorities
import psutil  # Enables process management, including setting CPU affinity
import ctypes  # Provides access to C standard library for memory locking
import time
import subprocess
import subprocess


# ---------------------------- PWM Related ------------------------------ #
# Stop the idle signal script   
def stop_idle_signal():
    try:
        os.system("pkill -f idle_pwm_signal.py")
        print("Idle PWM signal stopped.")
    except Exception as e:
        print(f"Failed to stop idle PWM signal: {e}")

 
# Restart the idle signal script  
def restart_idle_signal():
    """
    Restart the idle signal script to send a low PWM signal when the main program is not running.
    """
    try:
        os.system("python3 /home/pi/Documents/Quadcopter_Control_v2/main/idle_pwm_signal.py &")
        print("Idle PWM signal restarted.")
    except Exception as e:
        print(f"Failed to restart idle PWM signal: {e}")

# ---------------------------- CPU Related ------------------------------ #

def measure_cpu_freq():
    """
    Measure the current CPU frequency using vcgencmd.
    Returns:
        float: CPU frequency in MHz.
    """
    try:
        freq_output = subprocess.check_output(["vcgencmd", "measure_clock", "arm"]).decode()
        freq = int(freq_output.split("=")[1]) / 1_000_000  # Convert Hz to MHz
        print(f"Current CPU Frequency: {freq} MHz")
        return freq
    except Exception as e:
        print(f"Failed to measure CPU frequency: {e}")
        return None

def measure_volts():
    """
    Measure the current core voltage using vcgencmd.
    Returns:
        float: Voltage in volts.
    """
    try:
        volts_output = subprocess.check_output(["vcgencmd", "measure_volts", "core"]).decode()
        volts = float(volts_output.replace("volt=", "").replace("V", "").strip())
        print(f"Current Core Voltage: {volts} V")
        return volts
    except Exception as e:
        print(f"Failed to measure voltage: {e}")
        return None
    
def measure_temp():
    """
    Measure the current CPU temperature using vcgencmd.
    Returns:
        float: Temperature in Celsius.
    """
    try:
        temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        temp = float(temp_output.replace("temp=", "").replace("'C", "").strip())
        print(f"Current Temperature: {temp}°C")
        return temp
    except Exception as e:
        print(f"Failed to measure temperature: {e}")
        return None

def check_temperature_threshold(threshold=70):
    """
    Check if the CPU temperature exceeds the specified threshold.
    Args:
        threshold (float): Maximum allowable temperature in Celsius.
    Returns:
        bool: True if the temperature is below the threshold, False otherwise.
    """
    temp = measure_temp()
    if temp is not None and temp > threshold:
        print(f"Temperature exceeds threshold of {threshold}°C! Current: {temp}°C")
        return False
    return True

def monitor_voltage():
    """
    Monitor the voltage of the Raspberry Pi core and return its value.
    """
    import subprocess
    try:
        volts = subprocess.check_output(["vcgencmd", "measure_volts", "core"]).decode().strip()
        return float(volts.split('=')[1].replace('V', ''))
    except Exception as e:
        print(f"Failed to measure voltage: {e}")
        return None



def enable_dynamic_turbo():
    """
    Enable turbo mode dynamically by writing directly to /sys files.
    """
    try:
        with open("/sys/devices/system/cpu/cpufreq/boost", "w") as boost_file:
            boost_file.write("1")
        print("Turbo mode dynamically enabled.")
    except PermissionError:
        print("Permission denied. Run the script as root or with sudo.")
    except FileNotFoundError:
        print("Turbo mode file not found. Your system might not support dynamic turbo.")
    except Exception as e:
        print(f"Unexpected error enabling turbo: {e}")


def disable_dynamic_turbo():
    """
    Disable turbo mode dynamically by writing directly to /sys files.
    """
    try:
        with open("/sys/devices/system/cpu/cpufreq/boost", "w") as boost_file:
            boost_file.write("0")
        print("Turbo mode dynamically disabled.")
    except PermissionError:
        print("Permission denied. Run the script as root or with sudo.")
    except FileNotFoundError:
        print("Turbo mode file not found. Your system might not support dynamic turbo.")
    except Exception as e:
        print(f"Unexpected error disabling turbo: {e}")


def enable_hugepages(pages=128):
    """
    Enable HugePages for better memory access performance.
    Args:
        pages (int): Number of HugePages to allocate.
    """
    try:
        os.system(f"sudo sysctl vm.nr_hugepages={pages}")
        print(f"Allocated {pages} HugePages.")
    except Exception as e:
        print(f"Failed to enable HugePages: {e}")

def optimize_disk_io():
    """
    Optimize disk I/O by disabling journaling on mounted filesystems.
    """
    try:
        os.system("sudo tune2fs -O ^has_journal /dev/mmcblk0p2")  # Adjust based on your mount point
        print("Journaling disabled for mounted filesystem.")
    except Exception as e:
        print(f"Failed to optimize disk I/O: {e}")

def disable_power_management():
    """
    Disable power management for better real-time performance.
    """
    try:
        os.system("sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target")
        print("Power management disabled.")
    except Exception as e:
        print(f"Failed to disable power management: {e}")


def reduce_logging_overhead():
    """
    Reduce logging overhead for improved performance.
    """
    try:
        os.system("sudo systemctl stop syslog.service")
        os.system("sudo systemctl stop rsyslog.service")
        print("Logging services stopped to reduce overhead.")
    except Exception as e:
        print(f"Failed to reduce logging overhead: {e}")


def set_max_cpu_frequency():
    """
    Force the CPU to run at its maximum frequency to improve performance.
    Requires root privileges.
    """
    try:
        os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
        print("CPU frequency scaling set to 'performance'.")
    except Exception as e:
        print(f"Failed to set CPU frequency to maximum: {e}")

def disable_io_buffering():
    """
    Disable I/O buffering to ensure faster and more consistent writes.
    """
    os.system("sudo sysctl vm.dirty_ratio=1")
    os.system("sudo sysctl vm.dirty_background_ratio=1")
    print("I/O buffering minimized.")

def wait_for_low_cpu(threshold=10, interval=1):
    """
    Wait until the CPU usage is below a specified threshold.
    
    Args:
        threshold (int): CPU usage percentage threshold (default is 10%).
        interval (float): Time in seconds to wait between checks (default is 1 second).
    """
    print(f"Waiting for CPU usage to drop below {threshold}%...")
    while True:
        cpu_usage = psutil.cpu_percent(interval=0.1)  # Get CPU usage over a short interval
        if cpu_usage < threshold:
            print(f"CPU usage is now {cpu_usage}%. Proceeding with the experiment.")
            break
        print(f"Current CPU usage: {cpu_usage}%. Retrying in {interval} seconds...")
        time.sleep(interval)



def set_cpu_affinity(cores):
    """
    Assign the current process to run only on specific CPU cores.
    This ensures consistent CPU usage and avoids unnecessary context switching.

    Args:
        cores (list): List of core IDs to assign the process (e.g., [0, 1, 2, 3]).
    
    Steps:
    1. Get the current process ID using `os.getpid()`.
    2. Use `psutil.Process` to access the process object.
    3. Set the CPU affinity using `cpu_affinity()` method.
    """
    pid = os.getpid()  # Get the current process ID
    p = psutil.Process(pid)  # Get the process object for the current PID
    p.cpu_affinity(cores)  # Restrict the process to specified cores
    print(f"CPU Affinity set to cores: {cores}")  # Inform which cores are assigned

def set_realtime_priority():
    """
    Set the process to a real-time priority, which gives it precedence over other processes.
    This reduces jitter and ensures consistent execution timing.

    Requires:
    - Root privileges to modify process priorities.

    Steps:
    1. Use `os.nice(-20)` to set the highest priority (nice value of -20).
    2. Catch any `PermissionError` if the script is not run with sudo.
    """
    try:
        os.nice(-20)  # Set process priority to the highest possible
        print("Process priority set to real-time (-20).")  # Confirm successful priority change
    except PermissionError:  # Handle cases where root privileges are not available
        print("Failed to set real-time priority. Run the script with sudo.")  # Inform user about the failure

def lock_memory():
    """
    Lock the process's memory into physical RAM to prevent paging or swapping.
    This ensures the process's memory is always available and improves real-time performance.

    Requires:
    - Root privileges to lock memory.

    Steps:
    1. Load the C standard library using `ctypes.CDLL("libc.so.6")`.
    2. Call `mlockall()` with the flags `MCL_CURRENT | MCL_FUTURE` (value 3).
       - `MCL_CURRENT`: Lock all current pages in the process's address space.
       - `MCL_FUTURE`: Lock any future memory allocations.
    3. Catch exceptions and inform the user if memory locking fails.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")  # Load the C standard library
        libc.mlockall(3)  # Lock all current and future memory allocations
        print("Memory locked to RAM.")  # Confirm successful memory locking
    except Exception as e:  # Handle errors if locking fails
        print(f"Failed to lock memory: {e}")  # Inform user about the error
