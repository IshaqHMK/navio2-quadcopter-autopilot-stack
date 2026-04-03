import numpy as np
from analysis.display_controller_metrics import display_controller_metrics
from analysis.plot_results import plot_results

def extract_and_display_all(storage, time_min, time_max, Ts):
    """
    Extracts all variables from the storage dictionary and displays relevant metrics.

    Args:
    - storage: Dictionary containing all saved experiment data.
    - time_min: Minimum time for display.
    - time_max: Maximum time for display.
    - Ts: Sampling time.
    """
    # Extract variables from the storage dictionary (all saved variables)
    sim_times = storage["sim_times"]
    omegas = storage["omegas"]
    altitudes = storage["altitudes"]
    acc_data = storage["acc_data"]
    gyro_data = storage["gyro_data"]
    mag_data = storage["mag_data"]
    attitude_data = storage["attitude_data"]
    control_input_data = storage["control_input_data"]
    reference_data = storage["reference_data"]
    rate_reference_data = storage["rate_reference_data"]

    # Extract references and measurements
    altitude_reference = reference_data[:, 0]
    altitude_measured = altitudes[:, 0]
    phi_reference = reference_data[:, 1]
    phi_measured = attitude_data[:, 0]
    theta_reference = reference_data[:, 2]
    theta_measured = attitude_data[:, 1]
    psi_reference = reference_data[:, 3]
    psi_measured = attitude_data[:, 2]

    # Display metrics for each control variable
    display_controller_metrics(sim_times, altitude_reference, altitude_measured, omegas, sim_times[-1], Ts, "Altitude Control", time_min, time_max)
    display_controller_metrics(sim_times, np.rad2deg(phi_reference), np.rad2deg(phi_measured), omegas, sim_times[-1], Ts, "Phi Control", time_min, time_max)
    display_controller_metrics(sim_times, np.rad2deg(theta_reference), np.rad2deg(theta_measured), omegas, sim_times[-1], Ts, "Theta Control", time_min, time_max)
    display_controller_metrics(sim_times, np.rad2deg(psi_reference), np.rad2deg(psi_measured), omegas, sim_times[-1], Ts, "Psi Control", time_min, time_max)

    # Optional: Uncomment for rate control analysis
    # display_controller_metrics(sim_times, np.rad2deg(rate_reference_data[:, 0]), np.rad2deg(gyro_data[:, 0]), omegas, sim_times[-1], Ts, "p Control", time_min, time_max)
    # display_controller_metrics(sim_times, np.rad2deg(rate_reference_data[:, 1]), np.rad2deg(gyro_data[:, 1]), omegas, sim_times[-1], Ts, "q Control", time_min, time_max)

    # Plot the results
    plot_results(sim_times, reference_data, altitudes[:, 0], attitude_data, control_input_data)
 
