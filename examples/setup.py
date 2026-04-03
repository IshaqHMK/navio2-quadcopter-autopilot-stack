from setuptools import setup, find_packages

setup(
    name="Quadcopter_Control_v2",  # Name of your project
    version="1.0.0",
    license="BSD",
    description="Quadcopter control software with Navio2 device drivers",
    author="IshaqHafez",  #  
    author_email="ihafez@aus.edu",  #  
    url="https://github.com/ishaqhmk/Quadcopter_Control_v2",  # Replace with your repo URL
    packages=find_packages(include=['utils', 'utils.*', 'analysis', 'analysis.*', 'imu', 'imu.*']),  # Include your packages
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # List any dependencies your project needs
        "numpy",
        "scipy",
        "matplotlib",
        "spidev",
        "pyserial",
    ],
    python_requires='>=3.7',
)
