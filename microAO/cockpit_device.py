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

import json
import decimal
import dataclasses

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
import microAO.dm_layouts

from microAO.events import *
from microAO.gui.main import MicroscopeAOCompositeDevicePanel
from microAO.gui.sensorlessViewer import ConventionalResults
from microAO.aoAlg import AdaptiveOpticsFunctions
from microAO.aoRoutines import routines
from microAO.remotez import RemoteZ

aoAlg = AdaptiveOpticsFunctions()

class MicroscopeAOCompositeDevice(cockpit.devices.device.Device):
    RF_DEFAULT_LIMITS = (-1, 1)  # micrometres
    RF_POSHAN_NAME = "2 remote focus"
    RF_POSHAN_GNAME = "2 stage motion"
    RF_DURATION_TRAVEL = decimal.Decimal(1)  # miliseconds
    RF_DURATION_STABILISATION = decimal.Decimal(1)  # miliseconds

    def __init__(self, name: str, config={}) -> None:
        super().__init__(name, config)
        self.proxy = None

    def set_sensorless_routine(self, routine):
        # Set up sensorless parameters dict        
        self.sensorless_params = {
            "routine": routine
        }

        # Merge default sensorless parameters for the selected routine
        default_params = routines[self.sensorless_params['routine']].defaults()
        if default_params:
            self.sensorless_params.update(default_params)

    def initialize(self):
        self.proxy = Pyro4.Proxy(self.uri)
        # self.proxy.set_trigger(cp_ttype="FALLING_EDGE", cp_tmode="ONCE")
        self.no_actuators = self.proxy.get_n_actuators()

        # Need initial values for system flat calculations
        self.sys_flat_parameters = {
            "iterations" : 10,
            "error_threshold" : np.inf,
            "modes_to_ignore" : np.array([0, 1, 2])
        }

        # Default dict for sensorless parameters
        self.sensorless_params = {}
        self.set_sensorless_routine('conventional')

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
            "calib_calc": False,
            "sensorless": False
        }
        events.subscribe(events.USER_ABORT, self._on_abort)

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
            # Get flat modes
            sys_flat_modes = userConfig.getValue("dm_sys_flat_modes", None)
            if sys_flat_modes is not None:
                sys_flat_modes = np.asarray(sys_flat_modes)

            # Get flat actuators
            sys_flat_actuators = userConfig.getValue("dm_sys_flat_actuators", None)
            if sys_flat_actuators is not None:
                sys_flat_actuators = np.asarray(sys_flat_actuators)

            # Set flat, if found
            if sys_flat_modes is not None or sys_flat_actuators is not None:
                self.set_system_flat(sys_flat_modes, sys_flat_actuators)
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
                    "setupDigitalStack": lambda *args: self._rf_setup_exp_zstack(*args),
                    "flushDigitalStack": lambda *_: self.proxy.flush_patterns()
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

    def calibrationGetData(self, camera_name):
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
            events.NEW_IMAGE % camera_name,
            lambda image, _: self._calibrationOnImage(camera_name, image)
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
            self.send(
                np.zeros_like(
                    self._calibration_data["actuator_patterns"][0]
                ) + 0.5
            )

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

    def calc_error_RMS(self, unwrapped_phase, modes_to_subtract=(0, 1, 2)):
        return aoAlg.calc_phase_error_RMS(unwrapped_phase, modes_to_subtract)

    def _mask_circular(self, dims, radius=None, centre=None):
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

    def _get_no_discontinuities(self, phase_unwrapped):
        phase_unwrapped_diff = (
            abs(np.diff(np.diff(phase_unwrapped, axis=1), axis=0))
            * self._mask_circular(
                np.array(phase_unwrapped.shape),
                radius=min(phase_unwrapped.shape) / 2 - 3
            )[:-1, :-1]
        )
        return np.shape(np.where(phase_unwrapped_diff > 2 * np.pi))[1]

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
                phase = page.asarray()
                # Phase unwrap
                phase_unwrapped = self.unwrap_phase(phase)
                # Check for discontinuities
                no_discontinuities = self._get_no_discontinuities(
                    phase_unwrapped
                )
                if no_discontinuities > np.prod(phase_unwrapped.shape) / 1000.0:
                    logger.error(
                        f"Unwrapped phase for image {image_index} contained "
                        "discontinuites. Aborting calibration..."
                    )
                else:
                    # Calculate Zernike coefficients
                    zernike_coefficients[image_index] = aoAlg.get_zernike_modes(
                        phase_unwrapped,
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

        # Ensure all corrections are disabled
        original_corrections = self.get_corrections()
        self.reset()

        assay = self.proxy.assess_character()

        # The default system corrections should be for the zernike
        # modes we can accurately recreate.
        self.sys_flat_parameters["ignoreZernike"] = (np.abs(np.diag(assay)) > 0.25).nonzero()[0]

        # Restore original corrections
        for key, value in original_corrections.items():
            self.toggle_correction(key, value["enabled"])
        self.refresh_corrections()

        return assay

    @cockpit.util.threads.callInNewThread
    def sysFlatCalc(self, camera, imager):
        params = self.sys_flat_parameters
        # Ensure phase unwrapping works and the system is calibrated
        self.updateROI()
        self.checkFourierFilter()
        self.checkIfCalibrated()
        # Ensure all corrections, except the system flat one, are disabled
        original_corrections = self.get_corrections()
        self.reset()
        self.toggle_correction("system_flat", True)
        # Perform flattening
        iteration = 0
        error = np.inf
        modes = np.zeros(self.no_actuators)
        itr_max_str = (
            "Inf"
            if params["iterations"] == np.inf
            else int(params["iterations"])
        )
        err_max_str = (
            "Inf"
            if params["error_threshold"] == np.inf
            else f"{params['error_threshold']:.05f}"
        )
        while True:
            # Update status light
            events.publish(
                events.UPDATE_STATUS_LIGHT,
                "image count",
                f"Flattening phase: iter. {iteration + 1}/{itr_max_str}, "
                f"error {error:.05f}/{err_max_str}"
            )
            # Send modes and wait for them to take effect
            self.set_correction("system_flat", modes)
            self.refresh_corrections()
            time.sleep(0.1)
            # Get interferogram
            phase = self.captureImage(camera, imager)
            # Unwrap
            phase_unwrapped = self.unwrap_phase(phase)
            # Check for discontinuities
            no_discontinuities = self._get_no_discontinuities(phase_unwrapped)
            if no_discontinuities > np.prod(phase_unwrapped.shape) / 1000.0:
                print(
                    "Too many discontinuites in unwrapped phase. Aborting..."
                )
                break
            # Calculate RMS error
            error_current = self.calc_error_RMS(phase_unwrapped)
            # Get Zernike modes and filter out modes that should be ignored
            modes_measured = aoAlg.get_zernike_modes(
                phase_unwrapped,
                self.no_actuators
            )
            for mode_index in range(modes_measured.shape[0]):
                if mode_index in params["modes_to_ignore"]:
                    modes_measured[mode_index] = 0
            # Update state variables
            iteration += 1
            if error_current < error:
                modes += -modes_measured
                error = error_current
            # Check if exit conditions have been met
            if (iteration >= params["iterations"]) or (
                params["error_threshold"] < np.inf
                and error <= params["error_threshold"]
            ):
                self.set_system_flat(modes)
                break
        # Restore original corrections
        for key, value in original_corrections.items():
            self.toggle_correction(key, value["enabled"])
        self.refresh_corrections()
        # Clear the status light
        events.publish(events.UPDATE_STATUS_LIGHT, "image count", "")

    def reset(self):
        for key in self.get_corrections().keys():
            self.toggle_correction(key, False)
        self.refresh_corrections()

    def applySysFlat(self):
        self.toggle_correction("system_flat", True)
        self.refresh_corrections()

    def correctSensorlessSetup(self, camera):
        logger.log.info("Performing sensorless AO setup")

        # Check for calibration
        self.checkIfCalibrated()

        # Get control matrix details
        control_matrix = self.proxy.get_controlMatrix()
        n_modes = control_matrix.shape[1]

        # Set additional sensorless parameters
        pixel_size = wx.GetApp().Objectives.GetPixelSize() * 10 ** -6
        self.sensorless_params['pixel_size'] = pixel_size

        # Get the initial sensorless correction modes
        init_sensorless = np.zeros(n_modes)
        corrections = self.get_corrections(include_default=True)
        if (
            "sensorless" in corrections
            and corrections["sensorless"]["enabled"]
        ):
            init_sensorless = corrections["sensorless"]["modes"]

        # Shared state for the new image callbacks during sensorless
        self.sensorless_data = {
            "camera_name": camera.name,
            "image_stack": [],
            "metrics_stack": [],
            "corrections": init_sensorless
        }

        # Perform sensorless setup
        self._sensorless_routine = routines[self.sensorless_params['routine']](self.sensorless_params)
        results = self._sensorless_routine.setup(self.sensorless_data)

        # Set the first correction
        self.set_correction("sensorless", results.new_modes)
        self.toggle_correction("sensorless", True)
        self.refresh_corrections()

        # Subscribe to camera events
        events.subscribe(
            events.NEW_IMAGE % self.sensorless_data["camera_name"],
            self.correctSensorlessImage,
        )

        # Signal start of sensorless AO routine
        events.publish(PUBSUB_SENSORLESS_START, self.sensorless_params, self.sensorless_data)

        # Take image. This will trigger the iterative sensorless AO correction
        wx.CallAfter(wx.GetApp().Imager.takeImage)

    def correctSensorlessImage(self, image, _):
        # Check for abort flag and abort if set
        if self._abort["sensorless"]:
            self.correctSensorlessAbort()
            return

        # Add the image to the stack and request its eventual processing
        self.sensorless_data["image_stack"].append(image)
        wx.CallAfter(self.correctSensorlessProcessing)

    def correctSensorlessProcessing(self):
        # Perform processing of image through AO routine
        results = self._sensorless_routine.process(self.sensorless_data)
        # Check for error in processing and abort if so
        if results.error is not None:
            wx.MessageBox(
                results.error, caption="Sensorless error"
            )
            self.correctSensorlessAbort()

            return

        # Update status light
        if results.status is not None:
            events.publish(
                events.UPDATE_STATUS_LIGHT,
                "image count",
                results.status
            )

        # Publish result from processing, if specified
        if results.result is not None:
            events.publish(
                PUBSUB_SENSORLESS_RESULTS,
                results.result
            )

        # Check for process completion, otherwise set new modes
        if results.done == False:
            if results.new_modes is not None:
                new_modes = results.new_modes
                self.set_correction("sensorless", modes=new_modes)
            self.refresh_corrections()

        elif results.done == True:
            # Once all images have been obtained, unsubscribe
            logger.log.debug(
                "Unsubscribing to camera %s events"
                % self.sensorless_data["camera_name"]
            )
            events.unsubscribe(
                events.NEW_IMAGE % self.sensorless_data["camera_name"],
                self.correctSensorlessImage,
            )
            events.publish(events.UPDATE_STATUS_LIGHT, "image count", "")

            # Log data if filepath provided
            filepath = self.sensorless_params.get('log_path', None)
            if filepath:
                self._log_sensorless(
                    self.sensorless_params,
                    self.sensorless_data,
                    filepath=filepath
                )

            # Set the sensorless AO correction and enable it
            self.set_correction(
                "sensorless", self.sensorless_data["corrections"]
            )
            self.refresh_corrections()

            # Signal end of sensorless AO routine
            events.publish(PUBSUB_SENSORLESS_FINISH)

        # Take image, but ensure it's called after the phase is applied
        time.sleep(0.1)
        wx.CallAfter(wx.GetApp().Imager.takeImage)

    def correctSensorlessAbort(self):
        logger.log.debug(
            "Unsubscribing to camera %s events"
            % self.sensorless_data["camera_name"]
        )
        events.unsubscribe(
            events.NEW_IMAGE % self.sensorless_data["camera_name"],
            self.correctSensorlessImage,
        )
        events.publish(events.UPDATE_STATUS_LIGHT, "image count", "")

        self._abort["sensorless"] = False

    def _log_sensorless(
        self,
        params,
        data,
        filepath = None
    ):
        # Create timestamp
        ts = time.strftime("%Y%m%d_%H%M", time.gmtime())

        # Derive file path
        if not filepath:
            filepath = os.path.join(
                wx.GetApp().Config["log"].getpath("dir"),
                "sensorless_AO_" + ts + ".h5",
            )

        # Write data to file
        with h5py.File(filepath, "w") as f:
            # Write params and data
            data = [('params', params), ('data', data)]
            for group_name, group_data in data:
                group = f.create_group(group_name)
                for key, val in group_data.items():
                    try:
                        group.create_dataset(key, data=val)
                    except Exception as e:
                        logger.log.error(
                            "Failed to write sensorless data: {}".format(key), e
                        )

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

    def set_system_flat(self, modes=None, actuator_values=None):
        # Set in cockpit user config
        modes_config = np.ndarray.tolist(modes) if modes is not None else None
        actuators_config = np.ndarray.tolist(actuator_values) if actuator_values is not None else None
    
        userConfig.setValue("dm_sys_flat_modes", modes_config)
        userConfig.setValue("dm_sys_flat_actuators", actuators_config)

        # Set correction
        self.set_correction("system_flat", modes, actuator_values)

    def get_corrections(self, include_default=False):
        return self.proxy.get_corrections(include_default)

    def set_correction(self, name, modes=None, actuator_values=None):
        self.proxy.set_correction(name, modes=modes, actuator_values=actuator_values)

        # Publish event
        events.publish(
            PUBUSB_CHANGED_CORRECTION,
            name,
            self.get_corrections()[name]["enabled"]
        )

    def toggle_correction(self, name, enable):
        self.proxy.toggle_correction(name, enable)

        # Publish event
        events.publish(
            PUBUSB_CHANGED_CORRECTION,
            name,
            self.get_corrections()[name]["enabled"]
        )

    def refresh_corrections(self):
        """Clear the default correction and apply all other enabled ones."""
        return self.set_phase()

    def sum_corrections(self, corrections=None, only_enabled=True):
        return self.proxy.sum_corrections(corrections, only_enabled)

    def send(self, actuator_values):
        # Send values to device
        self.proxy.send(actuator_values)

        # Publish events
        events.publish(PUBSUB_SET_ACTUATORS, actuator_values)

    def set_phase(self, applied_z_modes=None, offset=None):
        actuator_values = self.proxy.set_phase(applied_z_modes, offset=offset)
        
        # Publish events
        events.publish(PUBSUB_SET_PHASE)
        events.publish(PUBSUB_SET_ACTUATORS, actuator_values)

        return actuator_values
    
    def update_control_matrix(self, control_matrix):
        self.proxy.set_controlMatrix(control_matrix)

    def _rf_get_movement_time(self):
        return (self.RF_DURATION_TRAVEL, self.RF_DURATION_STABILISATION)

    def _rf_move_absolute(self, position):
        self.remotez.set_z(position)
        time.sleep(
            float(sum(self._rf_get_movement_time()) * decimal.Decimal(1e-3))
        )
        events.publish(events.STAGE_MOVER, 2)
        events.publish(events.STAGE_STOPPED, self.RF_POSHAN_NAME)

    def _rf_move_relative(self, delta):
        self.remotez.set_z(self.remotez.get_z() + delta)
        time.sleep(
            float(sum(self._rf_get_movement_time()) * decimal.Decimal(1e-3))
        )
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
        patterns = np.zeros((steps, self.no_actuators))
        for i in range(steps):
            actuators = self.remotez.calc_shape(start + (step_size * i))
            patterns[i] = actuators
        # Repeat as necessary
        patterns = np.tile(patterns, (repeats, 1))
        # Skip the first pattern because experiments already place the remote
        # focusing stage at the starting position
        patterns = patterns[1:, :]
        # Queue patterns
        self.proxy.queue_patterns(patterns)

    def _generate_exercise_pattern(
        self,
        gain=0.0,
        layout_name="alpao69",
        pattern_name="checker"
    ):
        if gain < -1.0 or gain > 1.0:
            raise Exception(
                f"Invalid gain {gain}. Must be in the range [-1;1]."
            )
        # Get the layout and the pattern
        layout = microAO.dm_layouts.get_layout(layout_name)
        if pattern_name not in layout["presets"]:
            raise Exception(
                f"A pattern with name '{pattern_name}' does not exist."
            )
        pattern = np.array(layout["presets"][pattern_name])
        # Convert pattern to the range [-1, 1]
        pattern = np.interp(pattern, (0, 1), (-1, 1))
        # Apply gain
        pattern = pattern * gain
        # Convert back to the range [0, 1]
        pattern = np.interp(pattern, (-1, 1), (0, 1))
        return pattern

    @cockpit.util.threads.callInNewThread
    def exercise(self, gain, pattern_hold_time, repeats):
        pattern = self._generate_exercise_pattern(gain)
        pattern_inverted = self._generate_exercise_pattern(-gain)
        for _ in range(repeats):
            self.send(pattern)
            time.sleep(pattern_hold_time * 1e-3)
            self.send(pattern_inverted)
            time.sleep(pattern_hold_time * 1e-3)
        self.reset()
