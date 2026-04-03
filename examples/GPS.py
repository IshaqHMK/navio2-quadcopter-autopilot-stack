import sys
import time

# Add the project directory to the Python path
sys.path.append('/home/pi/Documents/Quadcopter_Control_v2')

# Import Navio utilities and uBlox module
from utils.navio2 import util, ublox

# Check for the presence of APM (Autopilot Manager) hardware
util.check_apm()

if __name__ == "__main__":
    # Initialize the uBlox GPS module
    ubl = ublox.UBlox("spi:0.0", baudrate=5000000, timeout=2)

    # Configure the GPS module
    ubl.configure_poll_port()
    ubl.configure_poll(ublox.CLASS_CFG, ublox.MSG_CFG_USB)

    ubl.configure_port(port=ublox.PORT_SERIAL1, inMask=1, outMask=0)
    ubl.configure_port(port=ublox.PORT_USB, inMask=1, outMask=1)
    ubl.configure_port(port=ublox.PORT_SERIAL2, inMask=1, outMask=0)
    ubl.configure_poll_port()
    ubl.configure_poll_port(ublox.PORT_SERIAL1)
    ubl.configure_poll_port(ublox.PORT_SERIAL2)
    ubl.configure_poll_port(ublox.PORT_USB)
    ubl.configure_solution_rate(rate_ms=1000)

    ubl.set_preferred_dynamic_model(None)
    ubl.set_preferred_usePPP(None)

    # Configure message rates
    ubl.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_POSLLH, 1)
    ubl.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_PVT, 1)
    ubl.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_STATUS, 1)
    ubl.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_SOL, 1)
    ubl.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_VELNED, 1)
    ubl.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_SVINFO, 1)
    ubl.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_VELECEF, 1)
    ubl.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_POSECEF, 1)
    ubl.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_RAW, 1)
    ubl.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_SFRB, 1)
    ubl.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_SVSI, 1)
    ubl.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_ALM, 1)
    ubl.configure_message_rate(ublox.CLASS_RXM, ublox.MSG_RXM_EPH, 1)
    ubl.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_TIMEGPS, 5)
    ubl.configure_message_rate(ublox.CLASS_NAV, ublox.MSG_NAV_CLOCK, 5)

    # Main loop to read messages
    while True:
        msg = ubl.receive_message()
        if msg is None:
            print("No message received")
            break

        # Process and display messages
        if msg.name() == "NAV_POSLLH":
            outstr = str(msg).split(",")[1:]
            outstr = "".join(outstr)
            print("Position:", outstr)
        elif msg.name() == "NAV_STATUS":
            outstr = str(msg).split(",")[1:2]
            outstr = "".join(outstr)
            print("Status:", outstr)
