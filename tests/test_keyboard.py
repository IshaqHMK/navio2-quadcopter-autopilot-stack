import sys
import select
import termios
import tty
import time  # Importing time module for delays

def check_exit_key(key_to_check='\x1b'):  # Default to 'Esc' key
    """
    Check if a specific key (default: Esc) is pressed for safety exit.

    Args:
        key_to_check (str): The key to check for. Default is '\x1b' (Esc key).

    Returns:
        bool: True if the specified key is pressed, otherwise False.
    """
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            key = sys.stdin.read(1)
            return key == key_to_check
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return False


def main():
    print("Press 'Esc' to stop the program.")

    # Simulate a continuous loop (e.g., for your control logic)
    while True:
        print("Running... (Press Esc to exit)")
        
        # Check for the exit key
        if check_exit_key():  # Default is Esc key
            print("Safety Stop: Esc key pressed. Exiting program.")
            break
        
        # Simulate a task in the loop (e.g., control logic)
        time.sleep(0.5)  # Sleep for 0.5 seconds to simulate work

    print("Program stopped safely.")


if __name__ == "__main__":
    main()
