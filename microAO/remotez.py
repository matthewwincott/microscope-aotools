import os
from re import search
import time
from functools import partial
import glob

import numpy as np
import scipy
import h5py

from cockpit import depot, events
from cockpit.util import logger, userConfig
from microAO.gui.remoteFocus import RF_DATATYPES
from microAO.gui.sensorlessViewer import SensorlessResultsViewer
from microAO.events import *

RF_DATATYPES = ["zernike", "actuator"]

class RemoteZ():
    def __init__(self, device):
        # Store reference to cockpit device
        self._device = device

        # Store state
        self.datapoints = []
        self.z_lookup = {}

        self._control_matrix = self._device.proxy.get_controlMatrix()
        self._n_actuators = self._control_matrix.shape[0]
        self._n_modes = self._control_matrix.shape[1]

        self.update_calibration()

    def set_control_matrix(self, control_matrix):
        self._control_matrix = control_matrix
        self._n_actuators = control_matrix.shape[0]
        self._n_modes = control_matrix.shape[1]

    def calibrate(self, zstage, zpos, output_dir=None, defocus_modes=[4,11], other_modes=np.asarray([22, 5, 6, 7, 8, 9, 10]), start_from_flat=False):
        mover = depot.getHandlerWithName("{}".format(zstage.name))

        if start_from_flat:
            zero_position = np.asarray(userConfig.getValue("dm_sys_flat"))
        else:
            zero_position = self._device.proxy.get_last_actuator_values()

        for i, z in enumerate(zpos):
            # Set modes to 0
            self._device.set_phase(np.zeros(self._n_modes), zero_position)

            # Calculate motion time and move
            z_prev = zstage.getPosition()
            motion_time, stabilise_time = mover.getMovementTime(z_prev,z)
            total_move_time = motion_time + stabilise_time + 1
            
            move = partial(mover.moveAbsolute,z)
            events.executeAndWaitForOrTimeout(
                "{} {}".format(events.STAGE_STOPPED, zstage.name),
                move,
                total_move_time / 1000,
            )

            # Correct defocus
            modes = np.array(defocus_modes)
            action = self.sensorless_correct(modes)

            # Correct other aberrations
            modes = np.array(other_modes)
            self.sensorless_correct(modes)

            # Save datapoint
            values = self._device.proxy.get_last_actuator_values()
            datapoint = {
                'datatype': 'actuator',
                'z': z,
                'values': values
            }

            self.add_datapoint(datapoint)

    def sensorless_correct(self, modes, start_values = None):
        # Get current sensorless params
        params_prev = self._device.sensorless_params.copy()

        try:
            self._device.sensorless_params["nollZernike"] = modes


            camera = self._device.getCamera()
            action = partial(self._device.correctSensorlessSetup, camera)

            if camera is None:
                return

            # Create results viewer
            window = SensorlessResultsViewer(None, None)
            window.Show()
            logger.log.debug("Start sensorless")
            # Start sensorless AO
            events.executeAndWaitForOrTimeout(
                PUBSUB_SENSORLESS_COMPLETE,
                action,
                5
            )
            # action(camera)
            logger.log.debug("Done sensorless")
            window.Close()

        except Exception as e:
            print('error', e)
        
        finally:
            # Return sensorless params to previous
            self._device.sensorless_parms = params_prev

    def zstack(self, zmin, zmax, zstepsize, camera=None, imager=None):
        zpositions = np.linspace(
            zmin,
            zmax,
            int((zmax - zmin) / zstepsize) + 1
        )

        if camera is None:
            camera = self._device.getCamera()
        if imager is None:
            imager = self._device.getImager()

        if camera is None or imager is None:
            logger.log.info(
                f"Aborting because no camera or imager found. Camera: {camera}"
                f". Imager: {imager}."
            )
            return

        images = []

        for z in zpositions:
            # Move in remote z
            self.set_z(z)

            # Capture image
            im = self._device.captureImage(camera, imager)
            images.append(im)

        # Go back to the zero position
        self.set_z(0)

        return images

    def add_datapoint(self, datapoint):
        self.datapoints.append(datapoint)
        self.datapoints.sort(key=lambda d: d["z"])
        self.update_calibration()
        return self.datapoints

    def remove_datapoint(self, datapoint):
        self.datapoints.remove(datapoint)
        self.datapoints.sort(key=lambda d: d["z"])
        self.update_calibration()
        return self.datapoints

    def update_calibration(self, datatypes=None):
        # Get data
        if datatypes is None:
            datatypes = RF_DATATYPES

        if type(datatypes) is not list:
            datatypes = [datatypes]

        for datatype in datatypes:
            points = [a for a in self.datapoints if a["datatype"].lower() == datatype]
            z = np.array([point["z"] for point in points])
            values = np.array([point["values"] for point in points])
            
            # Calculate regression
            try:
                n_measurements = values.shape[0]
                n_values = values.shape[1]
            except IndexError:
                n_measurements = 0
                n_values = 0
            slopes = np.zeros(n_values)
            intercepts = np.zeros(n_values)

            self.z_lookup[datatype] = []

            # Continue of more than one value
            if n_measurements > 1:
                for i in range(n_values):
                    slope, intercept, r, p, se = scipy.stats.linregress(z, values[:,i])
                    slopes[i] = slope
                    intercepts[i] = intercept
                    coef = [slope, intercept]
                    # coef = np.polyfit(z,values[:,i],1)

                    self.z_lookup[datatype].append(np.poly1d(coef)) 

    def calc_shape(self, z, datatype="zernike"):
        if self._control_matrix is None:
            print('no control matrix')

            return

        actuator_pos = None

        try:
            if len(self.z_lookup[datatype]) < 2:
                print('no remotez calib')
                return

            if datatype == "zernike":
                values = np.array([self.z_lookup[datatype][i](z) for i in range(0,self._n_modes)])
                self._device.set_correction("remotez", modes=values)
            elif datatype == "actuator":
                values = np.array([self.z_lookup[datatype][i](z) for i in range(0,self._n_actuators)])
                self._device.set_correction("remotez", actuator_values=values)
            

        except IndexError:
            # No lookup data
            pass

        # Get list of last applied corrections
        corrections_list = self._device.proxy.get_last_corrections_list()

        # Add remotez correction
        corrections_list = list(set(corrections_list + ["remotez"]))

        # Get shape
        actuator_pos = self._device.proxy.calc_shape(corrections_list)

        return actuator_pos

    def set_z(self, z, datatype="actuator"):
        if self._control_matrix is None:
            print('no control matrix')

            return

        actuator_pos = None
        corrections = None

        try:
            if len(self.z_lookup[datatype]) < 2:
                print('no remotez calib')
                return

            if datatype == "zernike":
                values = np.array([self.z_lookup[datatype][i](z) for i in range(0,self._n_modes)])
                self._device.set_correction("remotez", modes=values)
            elif datatype == "actuator":
                values = np.array([self.z_lookup[datatype][i](z) for i in range(0,self._n_actuators)])
                self._device.set_correction("remotez", actuator_values=values)
            
            actuator_pos, corrections = self._device.refresh_corrections(corrections=["remotez"])

        except IndexError:
            # No lookup data
            pass

        return actuator_pos, corrections
    
    def save_datapoints(self, output_dir):
        for datapoint in self.datapoints:
            fname = "{}-{}.h5".format(datapoint["datatype"], str(datapoint["z"]).replace('.','_'))
            fpath = os.path.join(output_dir, fname)

            with h5py.File(fpath, 'w') as f:
                for key, val in datapoint.items():
                    try:
                        f.create_dataset(key, data=val)
                    except Exception as e:
                        print('Failed to write: {}'.format(key), e)

    def load_datapoints(self, input_dir):
        search_string = str(os.path.join(input_dir, '*.h5'))

        for fpath in glob.glob(search_string):
            with h5py.File(fpath, 'r') as f:
                datapoint = {}

                # Keys
                for key in f.keys():
                    if key == 'datatype':
                        datapoint[key] = f[key][()].decode('utf-8')
                    else:
                        datapoint[key] = f[key][()]                   

                self.add_datapoint(datapoint)