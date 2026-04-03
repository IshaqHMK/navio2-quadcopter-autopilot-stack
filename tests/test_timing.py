"""
Test script for precise timing using time.perf_counter().
This script ensures consistent execution intervals and logs execution times.
"""

# Standard Libraries
import time

# Constants
SAMPLE_TIME = 0.0000000001  # Desired sample time in seconds (5 ms)
TEST_DURATION = 2.0  # Test duration in seconds

def precise_timing_test():
    """
    Demonstrates a precise timing mechanism using time.perf_counter().
    Logs the actual execution intervals and execution time for each iteration.
    """
    # Initialize timing variables
    start_time = time.perf_counter()
    next_time = start_time
    elapsed_time = 0
    iteration = 0

    print("Starting precise timing test...")

    while elapsed_time < TEST_DURATION:
        # Record the current time
        iteration_start = time.perf_counter()

        # Compute elapsed time since start
        elapsed_time = iteration_start - start_time

        # Perform a simulated task (can be replaced with actual operations)
        print(f"Iteration {iteration}: Elapsed Time = {elapsed_time:.6f} seconds")

        # Wait until the next sample time
        next_time += SAMPLE_TIME
        while time.perf_counter() < next_time:
            pass

        # Record the time after completing the iteration
        iteration_end = time.perf_counter()
        execution_time = iteration_end - iteration_start  # Calculate the execution time
        print(f"Iteration {iteration} Execution Time: {execution_time:.6f} seconds")
        
        iteration += 1

    print("Precise timing test completed.")

if __name__ == "__main__":
    precise_timing_test()
