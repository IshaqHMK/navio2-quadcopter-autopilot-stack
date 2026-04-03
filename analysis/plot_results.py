import numpy as np
import matplotlib.pyplot as plt

 

# Example usage:
# plot_results(sim_times, reference_data, altitudes[:, 0], attitude_data)


def plot_results(sim_times, reference_data, measured_altitude, attitude_data, control_input_data):
    fig, axs = plt.subplots(4, 3, figsize=(15, 20))  # 4 rows and 3 columns

    # Altitude Plot
    axs[0, 0].plot(sim_times, measured_altitude, 'b-', label='Measured Altitude', linewidth=1)
    axs[0, 0].plot(sim_times, reference_data[:, 0], 'r--', label='Reference Altitude', linewidth=1)
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Altitude (m)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Altitude Error Plot
    altitude_error = np.abs(measured_altitude - reference_data[:, 0])
    axs[0, 1].plot(sim_times, altitude_error, 'g-', label='Absolute Error', linewidth=1)
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Altitude Error (m)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Control Input Plot
    axs[0, 2].plot(sim_times, control_input_data[:,0], 'g-', label='Z Control Input', linewidth=1)
    axs[0, 2].set_xlabel('Time (s)')
    axs[0, 2].set_ylabel('Control Input (U1)')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Phi Plot
    axs[1, 0].plot(sim_times, np.rad2deg(attitude_data[:, 0]), 'b-', label='Measured Phi', linewidth=1)
    axs[1, 0].plot(sim_times, np.rad2deg(reference_data[:, 1]), 'r--', label='Reference Phi', linewidth=1)
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Phi (degrees)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Phi Error Plot
    phi_error = np.abs(np.rad2deg(attitude_data[:, 0]) - np.rad2deg(reference_data[:, 1]))
    axs[1, 1].plot(sim_times, phi_error, 'g-', label='Absolute Error', linewidth=1)
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Phi Error (degrees)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    
    # Control Input Plot
    axs[1, 2].plot(sim_times, control_input_data[:,1], 'g-', label='Phi Control Input', linewidth=1)
    axs[1, 2].set_xlabel('Time (s)')
    axs[1, 2].set_ylabel('Control Input (U2)')
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    # Theta Plot
    axs[2, 0].plot(sim_times, np.rad2deg(attitude_data[:, 1]), 'b-', label='Measured Theta', linewidth=1)
    axs[2, 0].plot(sim_times, np.rad2deg(reference_data[:, 2]), 'r--', label='Reference Theta', linewidth=1)
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Theta (degrees)')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # Theta Error Plot
    theta_error = np.abs(np.rad2deg(attitude_data[:, 1]) - np.rad2deg(reference_data[:, 2]))
    axs[2, 1].plot(sim_times, theta_error, 'g-', label='Absolute Error', linewidth=1)
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].set_ylabel('Theta Error (degrees)')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    # Control Input Plot
    axs[2, 2].plot(sim_times, control_input_data[:,2], 'g-', label='Theta Control Input', linewidth=1)
    axs[2, 2].set_xlabel('Time (s)')
    axs[2, 2].set_ylabel('Control Input (U3)')
    axs[2, 2].legend()
    axs[2, 2].grid(True)

    # Psi Plot
    axs[3, 0].plot(sim_times, np.rad2deg(attitude_data[:, 2]), 'b-', label='Measured Psi', linewidth=1)
    axs[3, 0].plot(sim_times, np.rad2deg(reference_data[:, 3]), 'r--', label='Reference Psi', linewidth=1)
    axs[3, 0].set_xlabel('Time (s)')
    axs[3, 0].set_ylabel('Psi (degrees)')
    axs[3, 0].legend()
    axs[3, 0].grid(True)

    # Psi Error Plot
    psi_error = np.abs(np.rad2deg(attitude_data[:, 2]) - np.rad2deg(reference_data[:, 3]))
    axs[3, 1].plot(sim_times, psi_error, 'g-', label='Absolute Error', linewidth=1)
    axs[3, 1].set_xlabel('Time (s)')
    axs[3, 1].set_ylabel('Psi Error (degrees)')
    axs[3, 1].legend()
    axs[3, 1].grid(True)
    
    
    # Control Input Plot
    axs[3, 2].plot(sim_times, control_input_data[:,3], 'g-', label='Theta Control Input', linewidth=1)
    axs[3, 2].set_xlabel('Time (s)')
    axs[3, 2].set_ylabel('Control Input (U4)')
    axs[3, 2].legend()
    axs[3, 2].grid(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
 
def plot_rate_results(sim_times, rate_reference_data, gyro_data):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # 3 subplots in one column

    # P - Roll Rate Plot
    axs[0].plot(sim_times, np.rad2deg(gyro_data[:, 0]), 'b-', label='Measured P (Roll Rate)', linewidth=1)
    axs[0].plot(sim_times, np.rad2deg(rate_reference_data[:, 0]), 'r--', label='Reference P (Roll Rate)', linewidth=1)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('P (degrees/sec)')
    axs[0].legend()
    axs[0].grid(True)

    # Q - Pitch Rate Plot
    axs[1].plot(sim_times, np.rad2deg(gyro_data[:, 1]), 'b-', label='Measured Q (Pitch Rate)', linewidth=1)
    axs[1].plot(sim_times, np.rad2deg(rate_reference_data[:, 1]), 'r--', label='Reference Q (Pitch Rate)', linewidth=1)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Q (degrees/sec)')
    axs[1].legend()
    axs[1].grid(True)

    # R - Yaw Rate Plot
    axs[2].plot(sim_times, np.rad2deg(gyro_data[:, 2]), 'b-', label='Measured R (Yaw Rate)', linewidth=1)
    axs[2].plot(sim_times, np.rad2deg(rate_reference_data[:, 2]), 'r--', label='Reference R (Yaw Rate)', linewidth=1)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('R (degrees/sec)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

# Example usage:
# sim_times = np.array([...])  # your simulation time array
# rate_reference_data = np.array([...])  # your reference data for p, q, r
# gyro_data = np.array([...])  # your measured gyro data for p, q, r
# plot_rate_results(sim_times, rate_reference_data, gyro_data)

