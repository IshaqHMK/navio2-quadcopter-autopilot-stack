import numpy as np
import matplotlib.pyplot as plt

 

def display_controller_metrics(time_data, reference_data, measured_data, omegas, sim_break_time, dt, description, start_time, end_time):
 
    """
    Calculate and display IAE, ITAE, ISE, RMSE, maximum overshoot, and settling time,
    and plot reference and measured data for a given control parameter.
    """
    
    print(" ---------------------------------------------------------")
    
    print(f"{description} - Error summary: ")
    
    time_mask_1 = (time_data >  0) & (time_data < sim_break_time)
    time_data_all = time_data[time_mask_1]
    reference_data_all = reference_data[time_mask_1]
    measured_data_all = measured_data[time_mask_1]
    
    # Ensure time_data is numpy array for diff to work correctly
    time_data_all = np.array(time_data_all)

    # Calculate errors
    errors_all = reference_data_all - measured_data_all
    abs_errors_all = np.abs(errors_all)
    squared_errors_all = np.square(errors_all)

    # IAE: Integral of Absolute Errors
    iae_all = np.sum(abs_errors_all) * dt
    print(f"{description} - Integral of Absolute Errors (IAE): {iae_all:.4f}")

    # ITAE: Integral of Time-weighted Absolute Errors
    itae_all = np.sum(time_data_all * abs_errors_all) * dt
    print(f"{description} - Integral Time-weighted Absolute Errors (ITAE): {itae_all:.4f}")

    # ISE: Integral of Squared Errors
    ise_all = np.sum(squared_errors_all) * dt
    print(f"{description} - Integral of Squared Errors (ISE): {ise_all:.4f}")

    # RMSE: Root Mean Squared Error
    rmse_all = np.sqrt(np.mean(squared_errors_all))
    print(f"{description} - Root Mean Squared Error (RMSE): {rmse_all:.4f}")
    
  
    # Filter the data for the specified time range
    if start_time >= np.min(time_data) and end_time <= np.max(time_data):
        time_mask = (time_data >= start_time) & (time_data <= end_time)
        filtered_time_data = time_data[time_mask]
        filtered_reference_data = reference_data[time_mask]
        filtered_measured_data = measured_data[time_mask]
    
        # Maximum Overshoot
        max_measured = np.max(filtered_measured_data)
        final_reference = filtered_reference_data[-1]
        overshoot = ((max_measured - final_reference) / abs(final_reference)) * 100 if final_reference != 0 else 0
        print(f"{description} - Maximum Percentage Overshoot: {overshoot:.2f}% (from {start_time}s to {end_time}s)")

        # Settling Time
        tolerance = 0.25 * abs(final_reference)  # 25% tolerance of the final reference value
        lower_bound = final_reference - tolerance
        upper_bound = final_reference + tolerance
        within_tolerance = (filtered_measured_data >= lower_bound) & (filtered_measured_data <= upper_bound)
        last_outside_index = np.where(~within_tolerance)[0][-1] if np.any(~within_tolerance) else 0
        settling_time = filtered_time_data[last_outside_index] if last_outside_index < len(filtered_time_data) else filtered_time_data[-1]
        print(f"{description} - Settling Time: {settling_time-start_time:.2f}s (from {start_time}s to {end_time}s)")
    
        # Calculate mean motor speeds
        filtered_omegas = omegas[time_mask]
        mean_motor_speeds = np.mean(filtered_omegas, axis=0)
        print(f"Mean motor speeds from {start_time} to {end_time}: {mean_motor_speeds}")
    
        # Mean Peak-to-Peak Error
        peak_to_peak_errors = np.ptp(filtered_reference_data - filtered_measured_data, axis=0)
        mean_peak_to_peak_error = np.mean(peak_to_peak_errors)
        print(f"Mean Peak-to-Peak Error (from {start_time}s to {end_time}s): {mean_peak_to_peak_error:.4f}")
        
    print(" ---------------------------------------------------------")
 
# Example usage:
# display_controller_metrics_and_plot(time_data, reference_data, measured_data, 'System Response')

