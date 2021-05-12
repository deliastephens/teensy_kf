import time, sys

import numpy as np
np.set_printoptions(linewidth=130)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 6]

import ipywidgets as widgets
from IPython.display import display

from jupyterplot import ProgressPlot # see https://github.com/lvwerra/jupyterplot

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from sympy import Matrix
from scipy.spatial.transform import Rotation

import teensyimu as ti


class IMU:
    def __init__(self, Fs=100):
        self.Fs = Fs # sample rate
        
        self.latestimu = None
        
        self._connect()
    
    def __del__(self):
        self._disconnect()
    
    def _connect(self):
        """
        Connect to the IMU via the teensyimu driver.
        
        A callback is registered that fires each time an IMU sample is received.
        """
        self.port = ti.tools.find_teensy_or_die() # finds the port teensy is connected to
        self.driver = ti.SerialDriver(self.port) # start the serial driver to get IMU
        time.sleep(0.1) # requisite 100ms wait period for everything to get setup
        # Request a specific IMU sample rate.
        # the max sample rates of accel and gyro are 4500 Hz and 9000 Hz, respectively.
        # However, the sample rate requested is for the entire device. Thus, if a sample
        # rate of 9000 Hz is requested, every received data packet will have a new gyro
        # sample but repeated accelerometer samples
        self.driver.sendRate(self.Fs) # Hz
        time.sleep(0.1) # requisite 100ms wait period for everything to get setup

        # everytime an IMU msg is received, call the imu_cb function with the data
        self.driver.registerCallbackIMU(self._callback)
        
    def _disconnect(self):
        # make sure to clean up to avoid threading / deadlock errors
        self.driver.unregisterCallbacks()
    
    def _callback(self, msg):
        """
        IMU Callback

        Every new IMU sample received causes this function to be called.

        Parameters
        ----------
        msg : ti.SerialIMUMsg
            has fields: ['t_us', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        """
        # unpack data
        self.latestimu = {
            't': msg.t_us*1e-6,
            'acc': np.array([msg.accel_x, msg.accel_y, msg.accel_z]),
            'gyr': np.array([msg.gyro_x, msg.gyro_y, msg.gyro_z])
        }
        
    def reset(self):
        """
        Resets the IMU connection, which resets time
        """
        self._disconnect()
        self._connect()
        
    def get_time(self):
        """
        Get latest timestamp from IMU
        
        Returns
        -------
        t : float
            time (in seconds) of latest IMU measurement
        """
        return self.latestimu['t'] if 't' in self.latestimu else None
        
    def get_acc(self):
        """
        Get latest acceleration measurement from IMU
        
        Returns
        -------
        acc : (3,) np.array
            x, y, z linear acceleration
        """
        return self.latestimu['acc'] if 'acc' in self.latestimu else None
    
    def get_gyr(self):
        """
        Get latest gyro measurement from IMU
        
        Returns
        -------
        gyr : (3,) np.array
            x, y, z angular velocity
        """
        return self.latestimu['gyr'] if 'gyr' in self.latestimu else None
    
    def get_mag(self):
        """
        Gets latest magnetometer measurement from IMU
        
        Returns
        -------
        mag : (3,) np.array
            x, y, z magnetic vector sensed from body frame
        """
        return self.latestimu['gyr'] if 'gyr' in self.latestimu else None