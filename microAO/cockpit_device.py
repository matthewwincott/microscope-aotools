#!/usr/bin/python
# -*- coding: utf-8

## Copyright (C) 2021 Matthew Wincott <matthew.wincott@eng.ox.ac.uk>
## Copyright (C) 2021 David Miguel Susano Pinto <david.pinto@bioch.ox.ac.uk>
## Copyright (C) 2019 Ian Dobbie <ian.dobbie@bioch.ox.ac.uk>
## Copyright (C) 2019 Mick Phillips <mick.phillips@gmail.com>
## Copyright (C) 2019 Nick Hall <nicholas.hall@dtc.ox.ac.uk>
##
## This file is part of Cockpit.
##
## Cockpit is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## Cockpit is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Cockpit.  If not, see <http://www.gnu.org/licenses/>.

"""Device file for a microscope-aotools AdaptiveOpticsDevice.

This file provides the cockpit end of the driver for a deformable
mirror as currently mounted on DeepSIM in Oxford.

"""
import os
import time
import queue
import pathlib
import json
import decimal

import cockpit.devices
import cockpit.devices.device
import cockpit.interfaces.imager
import cockpit.handlers.stagePositioner
from cockpit import depot
import numpy as np
import Pyro4
import wx
from cockpit import events
from cockpit.util import logger, userConfig
import h5py
import tifffile
import microscope.devices

from microAO.events import *
from microAO.gui.main import MicroscopeAOCompositeDevicePanel
from microAO.aoAlg import AdaptiveOpticsFunctions
from microAO.remotez import RemoteZ

aoAlg = AdaptiveOpticsFunctions()

def _get_timestamped_log_path(prefix):
    dirname = wx.GetApp().Config["log"].getpath("dir")
    timestamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    basename = prefix + "_" + timestamp    
    path = os.path.join(dirname, basename)

    return path

def _np_save_with_timestamp(data, basename_prefix):
    fpath = _get_timestamped_log_path(basename_prefix)
    np.save(fpath, data)


def log_correction_applied(
    image_stack,
    zernike_applied,
    nollZernike,
    sensorless_correct_coef,
    actuator_offset,
    metric_stack,
    z_steps
):
    # Save full stack of images used
    # _np_save_with_timestamp(
    #     np.asarray(image_stack),
    #     "sensorless_AO_correction_stack",
    # )

    # _np_save_with_timestamp(
    #     zernike_applied,
    #     "sensorless_AO_zernike_applied",
    # )

    # _np_save_with_timestamp(nollZernike, "sensorless_AO_nollZernike")
    # _np_save_with_timestamp(
    #     sensorless_correct_coef,
    #     "sensorless_correct_coef",
    # )

    # _np_save_with_timestamp(actuator_offset, "ac_pos_sensorless")
    # _np_save_with_timestamp(metric_stack, "sensorless_AO_metric_stack")

    ao_log_filepath = os.path.join(
        wx.GetApp().Config["log"].getpath("dir"),
        "sensorless_AO_logger.txt",
    )
    with open(ao_log_filepath, "a+") as fh:
        fh.write(
            "Time stamp: %s\n" % time.strftime("%Y/%m/%d %H:%M", time.gmtime())
        )
        fh.write("Aberrations measured: %s\n" % (sensorless_correct_coef))
        fh.write("Actuator positions applied: %s\n" % (str(actuator_offset)))

    # write data to hdf5 file
    ao_log_filepath = _get_timestamped_log_path('sensorless_AO_logger')+'.h5'

    # Write data to file
    with h5py.File(ao_log_filepath, 'w') as f:
        # Assemble data to write
        data = [('timestamp', time.strftime("%Y%m%d_%H%M", time.gmtime())),
                ('image_stack',np.asarray(image_stack)),
                ('zernike_applied',zernike_applied),
                ('nollZernike',nollZernike),
                ('sensorless_correct_coef',sensorless_correct_coef),
                ('actuator_offset',actuator_offset),
                ('metric_stack',metric_stack),
                ('z_steps',z_steps),
            ]
        
        # Write assembled data
        for datum in data:
            try:
                f.create_dataset(datum[0], data=datum[1])
            except Exception as e:
                print('Failed to write: {}'.format(datum[0]), e)
        

def mask_circular(dims, radius=None, centre=None):
    # Init radius and centre if necessary
    if centre is None:
        centre = (dims / 2).astype(int)
    if radius is None:
        # Largest circle that could fit in the dimensions
        radius = min(centre, dims - centre)
    # Create a meshgrid
    meshgrid = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))
    # Calculate distances from the centre element
    dist = np.sqrt(((meshgrid - centre.reshape(-1, 1, 1)) ** 2).sum(axis=0))
    # Return a binary mask for the specified radius
    return dist <= radius

class MicroscopeAOCompositeDevice(cockpit.devices.device.Device):
    RF_DEFAULT_LIMITS = (-1, 1)  # micrometres
    RF_POSHAN_NAME = "2 remote focus"
    RF_POSHAN_GNAME = "2 stage motion"
    RF_DURATION_TRAVEL = decimal.Decimal(0.001)
    RF_DURATION_STABILISATION = decimal.Decimal(0.010)

    def __init__(self, name: str, config={}) -> None:
        super().__init__(name, config)
        self.proxy = None

    def initialize(self):
        self.proxy = Pyro4.Proxy(self.uri)
        # self.proxy.set_trigger(cp_ttype="FALLING_EDGE", cp_tmode="ONCE")
        self.no_actuators = self.proxy.get_n_actuators()

        # Need initial values for system flat calculations
        self.sys_flat_parameters = {
            "num_it" : 10,
            "error_thresh" : np.inf,
            "nollZernike" : np.linspace(
                start=4, stop=68, num=65, dtype=int
            )
        }

        # Need intial values for sensorless AO
        self.sensorless_params = {
            "numMes": 9,
            "num_it": 2,
            "z_max": 1.5,
            "z_min": -1.5,
            "nollZernike": np.asarray([11, 22, 5, 6, 7, 8, 9, 10]),
            "start_from_flat" : True
        }

        # Shared state for the new image callbacks during sensorless
        self.sensorless_data = {}

        # Calibration parameters and data
        self.calibration_params = {
            "poke_min": 0.25,
            "poke_max": 0.75,
            "poke_steps": 5
        }
        self._calibration_data = {
            "output_filename": "",
            "running": False,
            "iteration_index": 0,
            "image_queue": queue.Queue(),
            "actuator_patterns": []
        }

        # Handle abort events
        self._abort = {
            "calib_data": False,
            "calib_calc": False
        }
        events.subscribe(events.USER_ABORT, self._on_abort)

        # Excercise the DM to remove residual static and then set to 0 position
        if self.config.get('exercise_on_startup', 'false').lower() in ['true', 't', 'yes', 'y', 1]:
            for _ in range(50):
                self.send(np.random.rand(self.no_actuators))
                time.sleep(0.01)
        
        # Reset the DM
        self.reset()

        # Load values from config
        try:
            self.updateROI()
        except Exception:
            pass

        try:
            controlMatrix = np.asarray(userConfig.getValue("dm_controlMatrix"))
            self.update_control_matrix(controlMatrix)
        except Exception:
            pass

        try:
            sys_flat = np.asarray(userConfig.getValue("dm_sys_flat"))
            self.set_system_flat(sys_flat)
        except Exception:
            pass

        # Initialise RemoteZ instance
        self.remotez = RemoteZ(self)

    def _on_abort(self):
        for key in self._abort:
            self._abort[key] = True

    def getHandlers(self):
        # Determine the hard limits
        limits = self.RF_DEFAULT_LIMITS
        limits_string = self.config.get("remote_focus_limits")
        if limits_string is not None:
            limits = tuple(
                [
                    int(limit_str.strip())
                    for limit_str in limits_string.split(",")
                ]
            )
        # Determine if the device is driven by an executor
        exp_elig = False
        t_handler = None
        t_source = self.config.get("triggersource", None)
        t_line = self.config.get("triggerline", None)
        if t_source:
            exp_elig = True
            t_handler = depot.getHandler(t_source, depot.EXECUTOR)
        # Return the handler
        return [
            cockpit.handlers.stagePositioner.PositionerHandler(
                self.RF_POSHAN_NAME,
                self.RF_POSHAN_GNAME,
                exp_elig,
                {
                    "getMovementTime": lambda *_: self._rf_get_movement_time(),
                    "getPosition": lambda _: self.remotez.get_z(),
                    "moveAbsolute": lambda _, position: self._rf_move_absolute(position),
                    "moveRelative": lambda _, delta: self._rf_move_relative(delta),
                    "setupDigitalStack": lambda *args: self._rf_setup_exp_zstack(*args)
                },
                2,
                limits,
                trigHandler=t_handler,
                trigLine=t_line
            )
        ]

    def makeUI(self, parent):
        return MicroscopeAOCompositeDevicePanel(parent, self)

    def acquireRaw(self):
        return self.proxy.acquire_raw()

    def acquireUnwrappedPhase(self):
        return self.proxy.acquire_unwrapped_phase()

    def getZernikeModes(self, image_unwrap, noZernikeModes):
        return self.proxy.getzernikemodes(image_unwrap, noZernikeModes)

    def wavefrontRMSError(self, phase_map):
        return self.proxy.wavefront_rms_error(phase_map)

    def updateROI(self):
        circle_parameters = userConfig.getValue("dm_circleParams")
        self.proxy.set_roi(*circle_parameters)

        # Check we have the interferogram ROI
        try:
            self.proxy.get_roi()
        except Exception as e:
            try:
                self.proxy.set_roi(*circle_parameters)
            except Exception:
                raise e

        # Update local aoAlg instance
        aoAlg.make_mask(int(np.round(circle_parameters[2])))

    def checkFourierFilter(self):
        circle_parameters = userConfig.getValue("dm_circleParams")
        try:
            self.proxy.get_fourierfilter()
        except Exception as e:
            try:
                test_image = self.proxy.acquire()
                self.proxy.set_fourierfilter(
                    test_image=test_image,
                    window_dim=50,
                    mask_di=int((2 * circle_parameters[2]) * (3.0 / 16.0)),
                )
                # Update local aoAlg instance
                aoAlg.make_fft_filter(
                    test_image,
                    window_dim=50,
                    mask_di=int((2 * circle_parameters[2]) * (3.0 / 16.0)),
                )
            except Exception:
                raise e

    def checkIfCalibrated(self):
        try:
            self.proxy.get_controlMatrix()
        except Exception as e:
            try:
                controlMatrix = np.asarray(
                    userConfig.getValue("dm_controlMatrix")
                )
                self.update_control_matrix(controlMatrix)
            except Exception:
                raise e

    def calibrationGetData(self, parent):
        # Select the camera which will be used to capture images
        camera = self.getCamera()
        if camera is None:
            logger.log.error(
                "Failed to start calibration because no active cameras were "
                "found."
            )
            return
        # Select the file to which the image stack will be written
        default_directory = ""
        if isinstance(self._calibration_data["output_filename"], pathlib.Path):
            default_directory = self._calibration_data["output_filename"].parent
        with wx.FileDialog(
            parent,
            message="Save calibration image stack",
            defaultDir=default_directory,
            wildcard="TIFF images (*.tif; *.tiff)|*.tif;*.tiff",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as file_dialog:
            if file_dialog.ShowModal() != wx.ID_OK:
                return
            self._calibration_data["output_filename"] = pathlib.Path(
                file_dialog.GetPath()
            )
        # Generate actuator patterns
        self._calibration_data["actuator_patterns"] = np.zeros(
            (
                self.no_actuators * self.calibration_params["poke_steps"],
                self.no_actuators
            )
        ) + 0.5
        pokeSteps = np.linspace(
            self.calibration_params["poke_min"],
            self.calibration_params["poke_max"],
            self.calibration_params["poke_steps"]
        )
        for i in range(self.no_actuators):
            for j in range(pokeSteps.shape[0]):
                self._calibration_data["actuator_patterns"][
                    (pokeSteps.shape[0] * i) + j, i
                ] = pokeSteps[j]
        # Start a saving thread
        self._calibration_data["running"] = True
        self._calibration_data["iteration_index"] = 0
        self._calibrationImageSaver()
        # Subscribe to new image event
        events.subscribe(
            events.NEW_IMAGE % camera.name,
            lambda image, _: self._calibrationOnImage(camera.name, image)
        )
        # Apply the first actuator pattern and wait for it to settle
        self.send(self._calibration_data["actuator_patterns"][0])
        time.sleep(0.1)
        # Take the first image
        wx.CallAfter(wx.GetApp().Imager.takeImage)

    @cockpit.util.threads.callInNewThread
    def _calibrationImageSaver(self):
        with tifffile.TiffWriter(self._calibration_data["output_filename"]) as fo:
            while self._calibration_data["running"]:
                try:
                    image = self._calibration_data["image_queue"].get(5)
                    fo.write(image, contiguous=True)
                except queue.Empty:
                    continue
        with self._calibration_data["output_filename"].with_name(
            self._calibration_data["output_filename"].stem + ".json"
        ).open("w", encoding="utf-8") as fo:
            json.dump(
                self._calibration_data["actuator_patterns"].tolist(),
                fo,
                sort_keys=True,
                indent=4
            )

    def _calibrationOnImage(self, camera_name, image):
        total_images = len(self._calibration_data["actuator_patterns"])
        # Put image on the queue and update the statue bar if there is no abort
        # request
        if not self._abort["calib_data"]:
            self._calibration_data["image_queue"].put(image)
            events.publish(
                events.UPDATE_STATUS_LIGHT,
                "image count",
                "AO calibration, data acquisition: image "
                f"{self._calibration_data['iteration_index'] + 1}/"
                f"{total_images}."
            )
        # Update the iteration counter
        self._calibration_data["iteration_index"] += 1
        # Decide to continue or to exit
        if (
            self._calibration_data["iteration_index"] < total_images
            and not self._abort["calib_data"]
        ):
            # Apply new pattern, wait a bit, and then take another image
            self.send(
                self._calibration_data["actuator_patterns"][
                    self._calibration_data["iteration_index"]
                ]
            )
            time.sleep(0.1)
            wx.CallAfter(wx.GetApp().Imager.takeImage)
        else:
            events.unsubscribe(
                events.NEW_IMAGE % camera_name,
                self._calibrationOnImage,
            )
            events.publish(events.UPDATE_STATUS_LIGHT, "image count", "")
            self._calibration_data["running"] = False
            self._abort["calib_data"] = False
            self._calibration_data["image_queue"].join()

    def unwrap_phase(self, image):
        # Crop the image if necessary
        roi = self.proxy.get_roi()
        image_cropped = np.zeros(
            (roi[2] * 2, roi[2] * 2), dtype=float
        )
        image_cropped[:, :] = image[
            roi[0] - roi[2] : roi[0] + roi[2],
            roi[1] - roi[2] : roi[1] + roi[2],
        ]
        if np.any(aoAlg.mask) is None:
            aoAlg.make_mask(self.roi[2])
            image = image_cropped
        else:
            image = image_cropped * aoAlg.mask
        # Unwrap the phase
        return aoAlg.unwrap_interferometry(image)

    @cockpit.util.threads.callInNewThread
    def calculateControlMatrix(self, actuator_values, file_path_image):
        self.updateROI()
        self.checkFourierFilter()
        # Process images and calculate Zernike mode coefficients
        zernike_coefficients = np.zeros_like(actuator_values)
        with tifffile.TiffFile(file_path_image) as stack:
            image_index = 0
            for page in stack.pages:
                # Check for abort requests
                if self._abort["calib_calc"]:
                    break
                # Update the status bar
                events.publish(
                    events.UPDATE_STATUS_LIGHT,
                    "image count",
                    "AO calibration, control matrix calculation: image "
                    f"{image_index + 1}/{actuator_values.shape[0]}."
                )
                # Get the image data
                image = page.asarray()
                # Phase unwrap
                image_unwrapped = self.unwrap_phase(image)
                # Check for discontinuities
                image_unwrapped_diff = (
                    abs(np.diff(np.diff(image_unwrapped, axis=1), axis=0))
                    * mask_circular(
                        np.array(image_unwrapped.shape),
                        radius=min(image_unwrapped.shape) / 2 - 3
                    )[:-1, :-1]
                )
                no_discontinuities = np.shape(
                    np.where(image_unwrapped_diff > 2 * np.pi)
                )[1]
                if no_discontinuities > np.prod(image_unwrapped.shape) / 1000.0:
                    logger.error(
                        f"Unwrapped phase for image {image_index} contained "
                        "discontinuites. Aborting calibration..."
                    )
                else:
                    # Calculate Zernike coefficients
                    zernike_coefficients[image_index] = aoAlg.get_zernike_modes(
                        image_unwrapped,
                        self.no_actuators
                    )
                # Increment the image index
                image_index += 1
        if not self._abort["calib_calc"]:
            # Calculate the control matrix
            control_matrix = aoAlg.create_control_matrix(
                zernikeAmps=zernike_coefficients,
                pokeSteps=actuator_values,
                numActuators=self.no_actuators
            )
            # Save the matrix to the user config
            userConfig.setValue(
                "dm_controlMatrix", np.ndarray.tolist(control_matrix)
            )
            # Propagate the control matrix to the microscope device
            self.update_control_matrix(control_matrix)
        else:
            # Clear the flag
            self._abort["calib_calc"] = False
        # Clear status bar
        events.publish(events.UPDATE_STATUS_LIGHT, "image count", "")

    def characterise(self):
        self.updateROI()
        self.checkFourierFilter()
        self.checkIfCalibrated()
        assay = self.proxy.assess_character()

        if np.mean(assay[1:, 1:]) < 0:
            controlMatrix = self.proxy.get_controlMatrix()
            self.update_control_matrix(-1 * controlMatrix)
            assay = assay * -1
            userConfig.setValue(
                "dm_controlMatrix", np.ndarray.tolist(controlMatrix)
            )

        # The default system corrections should be for the zernike
        # modes we can accurately recreate.
        self.sys_flat_parameters["sysFlatNollZernike"] = ((np.diag(assay) > 0.75).nonzero()[0]) + 1

        return assay

    def sysFlatCalc(self):
        self.updateROI()
        self.checkFourierFilter()
        self.checkIfCalibrated()

        control_matrix = self.proxy.get_controlMatrix()
        n_actuators = control_matrix.shape[0]
        n_modes = control_matrix.shape[1]

        z_ignore = np.zeros(n_modes)
        if self.sys_flat_parameters["sysFlatNollZernike"] is not None:
            z_ignore[self.sys_flat_parameters["sysFlatNollZernike"] - 1] = 1
        sys_flat_values, best_z_amps_corrected = self.proxy.flatten_phase(
            iterations=self.sys_flat_parameters["num_it"],
            error_thresh=self.sys_flat_parameters["error_thresh"],
            z_modes_ignore=z_ignore,
        )

        self.set_system_flat(sys_flat_values)

        return sys_flat_values, best_z_amps_corrected

    def reset(self):
        self.proxy.reset()

    def applySysFlat(self):
        sys_flat_values = self.get_system_flat()
        self.send(sys_flat_values)

    def applyLastPattern(self):
        last_ac = self.proxy.get_last_actuator_values()
        self.send(last_ac)

    def correctSensorlessSetup(self, camera):
        logger.log.info("Performing sensorless AO setup")
        # Note: Default is to correct Primary and Secondary Spherical
        # aberration and both orientations of coma, astigmatism and
        # trefoil.

        # Check for calibration
        self.checkIfCalibrated()

        # Get control matrix details
        control_matrix = self.proxy.get_controlMatrix()
        n_actuators = control_matrix.shape[0]
        n_modes = control_matrix.shape[1]


        # Shared state for the new image callbacks during sensorless
        self.sensorless_data = {
            "actuator_offset" : None,
            "camera" : camera,
            "image_stack" : [],
            "metric_stack" : [],
            "correction_stack" : [],
            "sensorless_correct_coef" : np.zeros(n_modes),          # Measured aberrations
            "z_steps" : np.linspace(self.sensorless_params["z_min"], self.sensorless_params["z_max"], self.sensorless_params["numMes"]),
            "zernike_applied" : np.zeros((0, n_modes)), # Array of all z aberrations to apply during experiment
            "metric_calculated" : np.array(())
        }

        if self.sensorless_params.get("start_from_flat", False):
            self.sensorless_data["actuator_offset"] = self.get_system_flat()
        else:
            self.sensorless_data["actuator_offset"] = self.proxy.get_last_actuator_values()

        logger.log.debug("Subscribing to camera events")
        # Subscribe to camera events
        events.subscribe(
            events.NEW_IMAGE % self.sensorless_data["camera"].name, self.correctSensorlessImage
        )

        for ii in range(self.sensorless_params["num_it"]):
            it_zernike_applied = np.zeros(
                (self.sensorless_params["numMes"] * self.sensorless_params["nollZernike"].shape[0], self.no_actuators)
            )
            for noll_ind in self.sensorless_params["nollZernike"]:
                ind = np.where(self.sensorless_params["nollZernike"] == noll_ind)[0][0]
                it_zernike_applied[
                    ind * self.sensorless_params["numMes"] : (ind + 1) * self.sensorless_params["numMes"], noll_ind - 1
                ] = self.sensorless_data["z_steps"]
            self.sensorless_data["zernike_applied"] = np.concatenate(
                [self.sensorless_data["zernike_applied"], it_zernike_applied]
            )

        logger.log.info("Applying the first Zernike mode")
        # Apply the first Zernike mode
        logger.log.debug(self.sensorless_data["zernike_applied"][len(self.sensorless_data["image_stack"]), :])
        self.set_phase(
            self.sensorless_data["zernike_applied"][len(self.sensorless_data["image_stack"]), :],
            offset=self.sensorless_data["actuator_offset"],
        )

        # Take image. This will trigger the iterative sensorless AO correction
        wx.CallAfter(wx.GetApp().Imager.takeImage)

    def correctSensorlessImage(self, image, timestamp):
        del timestamp
        if len(self.sensorless_data["image_stack"]) < self.sensorless_data["zernike_applied"].shape[0]:
            logger.log.info(
                "Correction image %i/%i"
                % (
                    len(self.sensorless_data["image_stack"]) + 1,
                    self.sensorless_data["zernike_applied"].shape[0],
                )
            )
            events.publish(
                events.UPDATE_STATUS_LIGHT,
                "image count",
                "Sensorless AO: image %s/%s, mode %s, meas. %s"
                % (
                    len(self.sensorless_data["image_stack"]) + 1,
                    self.sensorless_data["zernike_applied"].shape[0],
                    self.sensorless_params["nollZernike"][
                        len(self.sensorless_data["image_stack"])
                        // self.sensorless_params["numMes"]
                        % len(self.sensorless_params["nollZernike"])
                    ],
                    (len(self.sensorless_data["image_stack"]) + 1) % self.sensorless_params["numMes"] + 1,
                ),
            )
            # Store image for current applied phase
            self.sensorless_data["image_stack"].append(np.ndarray.tolist(image))
            wx.CallAfter(self.correctSensorlessProcessing)
        else:
            logger.log.error(
                "Failed to unsubscribe to camera events. Trying again."
            )
            events.unsubscribe(
                events.NEW_IMAGE % self.sensorless_data["camera"].name,
                self.correctSensorlessImage,
            )
            events.publish(events.UPDATE_STATUS_LIGHT, "image count", "")

    def correctSensorlessAberation(self):
        pixelSize = wx.GetApp().Objectives.GetPixelSize() * 10 ** -6

        # Find aberration amplitudes and correct
        ind = int(len(self.sensorless_data["image_stack"]) / self.sensorless_params["numMes"])
        nollInd = (
            np.where(self.sensorless_data["zernike_applied"][len(self.sensorless_data["image_stack"]) - 1, :])[
                0
            ][0]
            + 1
        )
        logger.log.debug("Current Noll index being corrected: %i" % nollInd)
        current_stack = np.asarray(self.sensorless_data["image_stack"])[
            (ind - 1) * self.sensorless_params["numMes"] : ind * self.sensorless_params["numMes"], :, :
        ]
        (
            amp_to_correct,
            ac_pos_correcting,
            metrics_calculated,
        ) = self.proxy.correct_sensorless_single_mode(
            image_stack=current_stack,
            zernike_applied=self.sensorless_data["z_steps"],
            nollIndex=nollInd,
            offset=self.sensorless_data["actuator_offset"],
            wavelength=500 * 10 ** -9,
            NA=1.1,
            pixel_size=pixelSize,
        )
        self.sensorless_data["actuator_offset"] = ac_pos_correcting
        self.sensorless_data["sensorless_correct_coef"][nollInd - 1] += amp_to_correct
        self.sensorless_data["metric_calculated"] = metrics_calculated
        for metric in metrics_calculated:
            self.sensorless_data["metric_stack"].append(metric.astype('float'))

    def correctSensorlessProcessing(self):
        logger.log.info("Processing sensorless image")
        if len(self.sensorless_data["image_stack"]) < self.sensorless_data["zernike_applied"].shape[0]:
            if len(self.sensorless_data["image_stack"]) % self.sensorless_params["numMes"] == 0:
                self.correctSensorlessAberation()

            # Advance counter by 1 and apply next phase
            self.set_phase(
                self.sensorless_data["zernike_applied"][len(self.sensorless_data["image_stack"]), :],
                offset=self.sensorless_data["actuator_offset"],
            )

        else:
            # Once all images have been obtained, unsubscribe
            logger.log.debug(
                "Unsubscribing to camera %s events" % self.sensorless_data["camera"].name
            )
            events.unsubscribe(
                events.NEW_IMAGE % self.sensorless_data["camera"].name,
                self.correctSensorlessImage,
            )
            events.publish(events.UPDATE_STATUS_LIGHT, "image count", "")

            self.correctSensorlessAberation()

            log_correction_applied(
                self.sensorless_data["image_stack"],
                self.sensorless_data["zernike_applied"],
                self.sensorless_params["nollZernike"],
                self.sensorless_data["sensorless_correct_coef"],
                self.sensorless_data["actuator_offset"],
                self.sensorless_data["metric_stack"],
                self.sensorless_data["z_steps"]
            )

            logger.log.debug(
                "Actuator positions applied: %s", self.sensorless_data["actuator_offset"]
            )

            # Set final correction
            self.send(self.sensorless_data["actuator_offset"])

            # Record AO correction
            ao_correction = self.sensorless_data["sensorless_correct_coef"] * -1.0
            self.set_correction("sensorless", modes=ao_correction)

        # Add current correction to stack
        ao_correction = self.sensorless_data["sensorless_correct_coef"] * -1.0
        self.sensorless_data["correction_stack"].append(np.ndarray.tolist(ao_correction))

        # Update/create metric plot
        sensorless_data = {
            'image_stack': self.sensorless_data["image_stack"],
            'metric_stack': self.sensorless_data["metric_stack"],
            'correction_stack': self.sensorless_data["correction_stack"],
            'nollZernike': self.sensorless_params["nollZernike"],
            'z_steps': self.sensorless_data["z_steps"],
            'iterations': self.sensorless_params["num_it"] 
        }

        # Publish data
        events.publish(PUBSUB_SENSORLESS_RESULTS, sensorless_data)


        # Take image, but ensure it's called after the phase is applied
        time.sleep(0.1)
        wx.CallAfter(wx.GetApp().Imager.takeImage)

    def getCamera(self):
        cameras = depot.getActiveCameras()
        
        camera = None
        if not cameras:
            wx.MessageBox(
                "There are no cameras enabled.", caption="No cameras active"
            )
        elif len(cameras) == 1:
            camera = cameras[0]
        else:
            cameras_dict = dict([(camera.descriptiveName, camera) for camera in cameras])

            dlg = wx.SingleChoiceDialog(
                None, "Select camera", 'Camera', list(cameras_dict.keys()),
            wx.CHOICEDLG_STYLE
                )
            if dlg.ShowModal() == wx.ID_OK:
                camera = cameras_dict[dlg.GetStringSelection()]

        return camera

    def getImager(self):
        imagers = depot.getHandlersOfType(depot.IMAGER)

        imager = None
        if not imagers:
            wx.MessageBox(
                "There are no available imagers.", caption="No imagers"
            )
        elif len(imagers) == 1:
            imager = imagers[0]
        else:
            imagers_dict = dict([(imager.name, imager) for imager in imagers])

            dlg = wx.SingleChoiceDialog(
                None, "Select imager", 'Imager', list(imagers_dict.keys()),
            wx.CHOICEDLG_STYLE
                )
            if dlg.ShowModal() == wx.ID_OK:
                imager = imagers_dict[dlg.GetStringSelection()]
        
        return imager

    def getStage(self, axis=2):
        stages = depot.getSortedStageMovers()

        stage = None

        if axis not in stages.keys():
            wx.MessageBox(
                "There are no stages for axis {} enabled.".format(axis), caption="No stages with axis {} active".formaT(axis)
            )

            return None

        if len(stages[axis]) == 1:
            stage = stages[axis][0]
        else:
            stages_dict = dict((stage.name, stage) for stage in stages[axis])

            dlg = wx.SingleChoiceDialog(
                None, "Select stage", 'Stage', list(stages_dict.keys()),
            wx.CHOICEDLG_STYLE
                )
            if dlg.ShowModal() == wx.ID_OK:
                stage = stages_dict[dlg.GetStringSelection()]

        return stage

    def captureImage(self, camera, imager, timeout=5.0):
        # Set capture method
        capture_method = imager.takeImage

        # Add camera timeout
        camera_timeout = camera.getExposureTime()+ camera.getTimeBetweenExposures()/1000
        timeout = timeout + camera_timeout

        # Fire capture action with timeout
        result = events.executeAndWaitForOrTimeout(
            events.NEW_IMAGE % camera.name,
            capture_method,
            timeout,
        )

        if result is not None:
            return result[0]
        else:
            raise TimeoutError("Camera capture timed out")

    def get_system_flat(self):
        sys_flat = np.asarray(userConfig.getValue("dm_sys_flat"))

        # Check system flat is defined
        if not np.any(sys_flat):
            sys_flat = None

        logger.log.warn("System flat is not defined")

        return sys_flat
    
    def set_system_flat(self, values):
        # Set in cockpit user config
        userConfig.setValue("dm_sys_flat", np.ndarray.tolist(values))

        # Set in device
        self.proxy.set_system_flat(values)

    def add_correction(self, name, modes=None, actuator_values=None):
        self.proxy.add_correction(name, modes=modes, actuator_values=actuator_values)

    def remove_correction(self, name):
        self.proxy.remove_correction(name)

    def set_correction(self, name, modes=None, actuator_values=None):
        self.proxy.set_correction(name, modes=modes, actuator_values=actuator_values)

    def apply_corrections(self, corrections):
        actuator_values, corrections_applied = self.proxy.apply_corrections(corrections)

        # Publish events
        events.publish(PUBSUB_SET_ACTUATORS, actuator_values)
        events.publish(PUBSUB_APPLY_CORRECTIONS, corrections_applied)

    def refresh_corrections(self, corrections=None):
        actuator_pos, corrections_applied = self.proxy.refresh_corrections(corrections=corrections)

        # Publish events
        events.publish(PUBSUB_SET_ACTUATORS, actuator_pos)
        events.publish(PUBSUB_APPLY_CORRECTIONS, corrections_applied)

        return actuator_pos, corrections_applied

    def send(self, actuator_values):
        # Send values to device
        self.proxy.send(actuator_values)

        # Publish events
        events.publish(PUBSUB_SET_ACTUATORS, actuator_values)

    def set_phase(self, applied_z_modes, offset=None, corrections=[]):
        # Eagerly publish phase value update
        events.publish(PUBSUB_SET_PHASE, applied_z_modes)

        # Send values to device
        actuator_values = self.proxy.set_phase(applied_z_modes, offset=offset, corrections=corrections)
        
        # Publish actuator change
        events.publish(PUBSUB_SET_ACTUATORS, actuator_values)

        # Get last corrections and publish
        corrections_applied  = self.proxy.get_last_corrections()
        events.publish(PUBSUB_APPLY_CORRECTIONS, corrections_applied)

        return actuator_values, corrections_applied

    def set_phase_map(self, phase_map):
        # Convert the phase map to zernike coefficients
        zernike_coeff = aoAlg.get_zernike_modes(
            phase_map,
            self.no_actuators
        )
        # Apply the zernike coefficients
        return self.set_phase(zernike_coeff)
    
    def update_control_matrix(self, control_matrix):
        self.proxy.set_controlMatrix(control_matrix)
        self.remotez.set_control_matrix(control_matrix)

    def _rf_get_movement_time(self):
        return (self.RF_DURATION_TRAVEL, self.RF_DURATION_STABILISATION)

    def _rf_move_absolute(self, position):
        self.remotez.set_z(position)
        time.sleep(sum(self._rf_get_movement_time()))
        events.publish(events.STAGE_MOVER, 2)
        events.publish(events.STAGE_STOPPED, self.RF_POSHAN_NAME)

    def _rf_move_relative(self, delta):
        self.remotez.set_z(self.remotez.get_z() + delta)
        time.sleep(sum(self._rf_get_movement_time()))
        events.publish(events.STAGE_MOVER, 2)
        events.publish(events.STAGE_STOPPED, self.RF_POSHAN_NAME)

    def _rf_setup_exp_zstack(self, start, step_size, steps, repeats=1):
        ttype, tmode = self.proxy.get_trigger()
        if (
            ttype
            not in (
                microscope.devices.TriggerType.RISING_EDGE,
                microscope.devices.TriggerType.FALLING_EDGE,
            )
            or tmode != microscope.devices.TriggerMode.ONCE
        ):
            raise Exception(
                "Wrong trigger configuration for adaptive element. In order to"
                " run experiments, please ensure that the device's trigger "
                "type is set to RISING_EDGE/HIGH or FALLING_EDGE/LOW and that "
                "its trigger mode is set to ONCE."
            )
        # Calculate patterns
        patterns = np.zeros((steps * repeats, self.no_actuators))
        for i in range(patterns.shape[0]):
            actuators = self.remotez.calc_shape(start + (step_size * i))
            patterns[i] = actuators
        # Queue patterns
        self.proxy.queue_patterns(patterns)
