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

import cockpit.depot
import cockpit.devices
import cockpit.devices.device
import cockpit.experiment.experiment
import cockpit.interfaces.imager
import cockpit.interfaces.stageMover
import cockpit.handlers.deviceHandler
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
from microAO.gui.sensorlessViewer import MetricPlotData
import microAO.aoAlg

@dataclasses.dataclass(frozen=True)
class SensorlessParamsMode:
    # Noll index
    index_noll: int
    # The amplitude offsets used for scanning the mode
    offsets: np.ndarray


class AOHandler(cockpit.handlers.deviceHandler.DeviceHandler):
    def __init__(
        self, name, groupName, callbacks, trigHandler=None, trigLine=None
    ):
        super().__init__(
            name, groupName, False, callbacks, cockpit.depot.AO_DEVICE
        )
        # Register with executor
        if trigHandler and trigLine:
            trigHandler.registerDigital(self, trigLine)
        # Initialise cleanup flag
        self._needs_cleanup = False
        # Listen for some experiment events
        events.subscribe(events.PREPARE_FOR_EXPERIMENT, self.experiment_setup)
        events.subscribe(
            events.CLEANUP_AFTER_EXPERIMENT, self.experiment_cleanup
        )

    def is_eligible_for_experiments(self) -> bool:
        return self.callbacks["is_eligible_for_experiments"]()

    def is_RF_enabled(self) -> bool:
        return self.callbacks["is_RF_enabled"]()

    def get_RF_position(self) -> float:
        return self.callbacks["get_RF_position"]()

    def get_RF_limits(self):
        return self.callbacks["get_RF_limits"]()

    def get_transition_time_ms(self) -> decimal.Decimal:
        return self.callbacks["get_transition_time_ms"]()

    def experiment_setup(
        self, experiment: cockpit.experiment.experiment.Experiment
    ):
        if self.is_eligible_for_experiments():
            self._needs_cleanup = True
            return self.callbacks["experiment_setup"](experiment)

    def experiment_cleanup(self):
        if self._needs_cleanup:
            self._needs_cleanup = False
            return self.callbacks["experiment_cleanup"]()


class MicroscopeAOCompositeDevice(cockpit.devices.device.Device):
    RF_DEFAULT_LIMITS = (-1, 1)  # micrometres
    RF_POSHAN_NAME = "2 remote focus"
    RF_POSHAN_GNAME = "2 stage motion"
    RF_DURATION_TRAVEL = decimal.Decimal(1)  # miliseconds
    RF_DURATION_STABILISATION = decimal.Decimal(1)  # miliseconds

    def __init__(self, name: str, config={}) -> None:
        super().__init__(name, config)
        self.proxy = None

    def initialize(self):
        self.proxy = Pyro4.Proxy(self.uri)
        # self.proxy.set_trigger(cp_ttype="FALLING_EDGE", cp_tmode="ONCE")
        self.no_actuators = self.proxy.get_n_actuators()

        self.aoAlg = microAO.aoAlg.AdaptiveOpticsFunctions()

        # Need initial values for system flat calculations
        self.sys_flat_parameters = {
            "iterations" : 10,
            "error_threshold" : np.inf,
            "modes_to_ignore" : np.array([0, 1, 2])
        }

        # Need intial values for sensorless AO
        self.sensorless_params = {
            "num_reps": 1,
            "modes": (
                SensorlessParamsMode(11, np.linspace(-1.5, 1.5, 7)),
                SensorlessParamsMode(22, np.linspace(-1.5, 1.5, 7)),
                SensorlessParamsMode(5, np.linspace(-1.5, 1.5, 7)),
                SensorlessParamsMode(6, np.linspace(-1.5, 1.5, 7)),
                SensorlessParamsMode(7, np.linspace(-1.5, 1.5, 7)),
                SensorlessParamsMode(8, np.linspace(-1.5, 1.5, 7)),
                SensorlessParamsMode(9, np.linspace(-1.5, 1.5, 7)),
                SensorlessParamsMode(10, np.linspace(-1.5, 1.5, 7)),
            ),
            "NA": 1.1,
            "wavelength": 560,
            "save_as_datapoint": False,
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

        # Reset the DM
        self.reset()

        # Load values from depot config
        transition_time_ms = self.config.get("transition_time_ms")
        if transition_time_ms is None:
            raise Exception(
                f"Invalid depot config for device '{self.name}'. Please ensure"
                " that the 'transition_time_ms' key is set."
            )
        self.transition_time_ms = decimal.Decimal(transition_time_ms)

        # Load values from user config
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
            sys_flat = userConfig.getValue("dm_sys_flat")
            if sys_flat is not None:
                self.set_system_flat(np.asarray(sys_flat))
        except Exception:
            pass
        # Initialise the sensorless and remote focus corrections, as well as
        # their fitting data
        cnames = ("sensorless", "remote focus")
        for cname in cnames:
            self.set_correction(cname)
        self._corrfit_dpts = {cname: {} for cname in cnames}
        self._corrfit_polys = {cname: [] for cname in cnames}
        self._rf_pos = 0

        # Subscribe to stage stopping events
        events.subscribe(events.STAGE_STOPPED, self._on_stage_stopped)

    def makeInitialPublications(self):
        # Load datapoints stored in the user config; this needs to happen after
        # the initialisation phase, so that the stage mover interface is ready
        datapoints_init = userConfig.getValue("ao_corrfit_dpts")
        if datapoints_init:
            for cname in datapoints_init:
                for z in datapoints_init[cname]:
                    self.corrfit_dp_add(
                        cname,
                        z,
                        np.array(datapoints_init[cname][z])
                    )

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
        t_handler = None
        t_source = self.config.get("triggersource", None)
        t_line = self.config.get("triggerline", None)
        if t_source:
            t_handler = depot.getHandler(t_source, depot.EXECUTOR)
        # Return the handlers
        return [
            cockpit.handlers.stagePositioner.PositionerHandler(
                self.RF_POSHAN_NAME,
                self.RF_POSHAN_GNAME,
                False,
                {
                    "getPosition": lambda _: self._rf_pos,
                    "moveAbsolute": lambda _, position: self._rf_move_absolute(
                        position
                    ),
                    "moveRelative": lambda _, delta: self._rf_move_relative(
                        delta
                    ),
                },
                2,
                limits,
            ),
            AOHandler(
                self.name,
                "AO",
                {
                    "is_eligible_for_experiments": self._is_eligible_for_experiments,
                    "is_RF_enabled": self._rf_is_enabled,
                    "get_RF_position": lambda: self._rf_pos,
                    "get_RF_limits": lambda: limits,
                    "get_transition_time_ms": lambda: self.transition_time_ms,
                    "experiment_setup": self._experiment_setup,
                    "experiment_cleanup": self._experiment_cleanup,
                },
                t_handler,
                t_line,
            ),
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
        self.aoAlg.make_mask(int(np.round(circle_parameters[2])))

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
                self.aoAlg.make_fft_filter(
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
        if np.any(self.aoAlg.mask) is None:
            self.aoAlg.make_mask(self.roi[2])
            image = image_cropped
        else:
            image = image_cropped * self.aoAlg.mask
        # Unwrap the phase
        return self.aoAlg.unwrap_interferometry(image)

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
                    logger.log.error(
                        f"Unwrapped phase for image {image_index} contained "
                        "discontinuites. Aborting calibration..."
                    )
                else:
                    # Calculate Zernike coefficients
                    zernike_coefficients[image_index] = self.aoAlg.get_zernike_modes(
                        phase_unwrapped,
                        self.no_actuators
                    )
                # Increment the image index
                image_index += 1
        if not self._abort["calib_calc"]:
            # Calculate the control matrix
            control_matrix = self.aoAlg.create_control_matrix(
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
            error_current = self.aoAlg.calc_phase_error_RMS(phase_unwrapped)
            # Get Zernike modes and filter out modes that should be ignored
            modes_measured = self.aoAlg.get_zernike_modes(
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

    def correctSensorlessSetup(self, camera, datapoint_z=None):
        logger.log.info("Performing sensorless AO setup")

        # Check for calibration
        self.checkIfCalibrated()

        # Get control matrix details
        control_matrix = self.proxy.get_controlMatrix()
        n_modes = control_matrix.shape[1]

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
            "corrections": init_sensorless,
            "total_measurements": sum(
                [len(mode.offsets) for mode in self.sensorless_params["modes"]]
            ) * self.sensorless_params["num_reps"],
            "metric_params": {
                "wavelength": self.sensorless_params["wavelength"] * 10 ** -9,
                "NA": self.sensorless_params["NA"],
                "pixel_size": wx.GetApp().Objectives.GetPixelSize() * 10 ** -6,
            },
            "mode_index": 0,
            "offset_index": 0,
            "datapoint_z": datapoint_z
        }

        # Signal start of sensorless AO routine
        max_scan_range = max(
            [
                mode.offsets.max() - mode.offsets.min()
                for mode in self.sensorless_params["modes"]
            ]
        )
        events.publish(
            PUBSUB_SENSORLESS_START,
            max_scan_range,
            self.aoAlg.get_metric(),
            self.sensorless_data["metric_params"]
        )

        # Apply the first set of modes
        new_modes = self.sensorless_data["corrections"].copy()
        new_modes[
            self.sensorless_params["modes"][
                self.sensorless_data["mode_index"]
            ].index_noll
            - 1
        ] += self.sensorless_params["modes"][
            self.sensorless_data["mode_index"]
        ].offsets[
            self.sensorless_data["offset_index"]
        ]
        self.set_correction("sensorless", new_modes)
        self.toggle_correction("sensorless", True)
        self.refresh_corrections()

        # Subscribe to camera events
        events.subscribe(
            events.NEW_IMAGE % self.sensorless_data["camera_name"],
            self.correctSensorlessImage,
        )

        # Take image. This will trigger the iterative sensorless AO correction
        wx.CallAfter(wx.GetApp().Imager.takeImage)

    def correctSensorlessImage(self, image, _):
        # Log the progress
        logger.log.info(
            "Correction image %i/%i"
            % (
                len(self.sensorless_data["image_stack"]) + 1,
                self.sensorless_data["total_measurements"],
            )
        )
        # Update status light
        events.publish(
            events.UPDATE_STATUS_LIGHT,
            "image count",
            "Sensorless AO: image %s/%s, mode %s, meas. %s"
            % (
                len(self.sensorless_data["image_stack"]) + 1,
                self.sensorless_data["total_measurements"],
                self.sensorless_params["modes"][
                    self.sensorless_data["mode_index"]
                ].index_noll,
                self.sensorless_data["offset_index"] + 1,
            ),
        )
        # Add the image to the stack and request its eventual processing
        self.sensorless_data["image_stack"].append(image)
        wx.CallAfter(self.correctSensorlessProcessing)

    def correctSensorlessAberation(self):
        # Calculate required parameters
        mode_index_noll_0 = (
            self.sensorless_params["modes"][
                self.sensorless_data["mode_index"]
            ].index_noll
            - 1
        )
        modes = (
            self.sensorless_data["corrections"][mode_index_noll_0]
            + self.sensorless_params["modes"][
                self.sensorless_data["mode_index"]
            ].offsets
        )
        image_stack = self.sensorless_data["image_stack"][-modes.shape[0] :]
        # Find aberration amplitudes and correct
        peak, metrics, metric_diagnostics = self.aoAlg.find_zernike_amp_sensorless(
            image_stack=image_stack,
            modes=modes,
            **self.sensorless_data["metric_params"]
        )
        self.sensorless_data["corrections"][mode_index_noll_0] = peak[0]
        # Signal sensorless results
        events.publish(
            PUBSUB_SENSORLESS_RESULTS,
            image_stack,
            MetricPlotData(
                peak=peak,
                metrics=metrics,
                modes=modes,
                mode_label=f"Z{mode_index_noll_0 + 1}"
            ),
            metric_diagnostics
        )
        # Update indices
        self.sensorless_data["offset_index"] = 0
        self.sensorless_data["mode_index"] += 1
        if self.sensorless_data["mode_index"] == len(
            self.sensorless_params["modes"]
        ):
            self.sensorless_data["mode_index"] = 0

    def correctSensorlessProcessing(self):
        if (
            len(self.sensorless_data["image_stack"])
            < self.sensorless_data["total_measurements"]
        ):
            # Increment offset index
            self.sensorless_data["offset_index"] += 1
            # Correct mode if enough measurements have been taken
            if (
                self.sensorless_data["offset_index"]
                == self.sensorless_params["modes"][
                    self.sensorless_data["mode_index"]
                ].offsets.shape[0]
            ):
                self.correctSensorlessAberation()
            # Apply next set of modes
            new_modes = self.sensorless_data["corrections"].copy()
            new_modes[
                self.sensorless_params["modes"][
                    self.sensorless_data["mode_index"]
                ].index_noll
                - 1
            ] += self.sensorless_params["modes"][
                self.sensorless_data["mode_index"]
            ].offsets[
                self.sensorless_data["offset_index"]
            ]
            self.set_correction("sensorless", new_modes)
            self.refresh_corrections()
        else:
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

            self.correctSensorlessAberation()

            if self.sensorless_params["save_as_datapoint"]:
                # Save the result as a datapoint
                self.corrfit_dp_add(
                    "sensorless",
                    self.sensorless_data["datapoint_z"],
                    self.sensorless_data["corrections"]
                )
            else:
                # Update the correction's values directly
                self.set_correction(
                    "sensorless", self.sensorless_data["corrections"]
                )
                self.refresh_corrections()

            # Signal end of sensorless AO routine
            events.publish(PUBSUB_SENSORLESS_FINISH)

        # Take image, but ensure it's called after the phase is applied
        time.sleep(0.1)
        wx.CallAfter(wx.GetApp().Imager.takeImage)

    def _log_correction_applied(
        self,
        image_stack,
        corrections,
        modes,
        NA,
        wavelength,
    ):
        # Derive file path
        ao_log_filepath = os.path.join(
            wx.GetApp().Config["log"].getpath("dir"),
            "sensorless_AO_" + time.strftime("%Y%m%d_%H%M", time.gmtime()) + ".h5",
        )

        # Write data to file
        with h5py.File(ao_log_filepath, "w") as f:
            # Assemble data to write
            data = [
                ("timestamp", time.strftime("%Y%m%d_%H%M", time.gmtime())),
                ("image_stack", image_stack),
                ("corrections", corrections),
                ("modes", modes),
                ("NA", NA),
                ("wavelength", wavelength),
            ]

            # Write assembled data
            for datum in data:
                try:
                    f.create_dataset(datum[0], data=datum[1])
                except Exception as e:
                    print("Failed to write: {}".format(datum[0]), e)

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

    def set_system_flat(self, modes):
        # Set in cockpit user config
        userConfig.setValue("dm_sys_flat", np.ndarray.tolist(modes))

        # Set correction
        self.set_correction("system_flat", modes)

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

    def rf_get_position(self):
        return self._rf_pos

    def _rf_set_position(self, position):
        # Re-evaluate the regression model and update the correction
        modes = self._corrfit_eval("remote focus", position)
        if len(modes) == 0:
            raise Exception("Remote focusing requires at least 2 data points.")
        self.set_correction("remote focus", modes=modes)
        self.refresh_corrections()

        # Update internal position
        self._rf_pos = position

    def _rf_move_absolute(self, position):
        self._rf_set_position(position)
        events.publish(events.STAGE_MOVER, 2)
        time.sleep(
            float(self.transition_time_ms * decimal.Decimal(1e-3))
        )
        events.publish(events.STAGE_STOPPED, self.RF_POSHAN_NAME)

    def _rf_move_relative(self, delta):
        self._rf_set_position(self._rf_pos + delta)
        events.publish(events.STAGE_MOVER, 2)
        time.sleep(
            float(self.transition_time_ms * decimal.Decimal(1e-3))
        )
        events.publish(events.STAGE_STOPPED, self.RF_POSHAN_NAME)

    def _rf_is_enabled(self):
        corrections = self.get_corrections()
        return corrections["remote focus"]["enabled"]

    def _on_stage_stopped(self, _):
        # Get new Z position
        new_z = cockpit.interfaces.stageMover.getPosition()[2]
        # Update correction if necessary
        modes = self._corrfit_eval("sensorless", new_z)
        if len(modes) > 0:
            self.set_correction("sensorless", modes=modes)
            self.refresh_corrections()

    def _is_eligible_for_experiments(self):
        # Determine if the AO device needs to be triggered during experiments
        corrections = self.get_corrections()
        if corrections["remote focus"]["enabled"] or (
            corrections["sensorless"]["enabled"]
            and len(self._corrfit_dpts["sensorless"]) > 1
        ):
            return True
        return False

    def _experiment_setup(
        self, experiment: cockpit.experiment.experiment.Experiment
    ):
        # Validate the triggering configuration of the device
        ttype, tmode = self.proxy.get_trigger()
        if (
            ttype
            not in (
                microscope.devices.TriggerType.RISING_EDGE,
                microscope.devices.TriggerType.FALLING_EDGE,
            )
            or tmode != microscope.devices.TriggerMode.ONCE
        ):
            events.publish(events.USER_ABORT)
            raise Exception(
                "Wrong trigger configuration for adaptive element. In order to"
                " run experiments, please ensure that the device's trigger "
                "type is set to RISING_EDGE/HIGH or FALLING_EDGE/LOW and that "
                "its trigger mode is set to ONCE."
            )
        # Take snapshot of current corrections
        corrections = self.get_corrections()
        corrections_to_restore = set()
        # Ensure that the remote focusing correction has enough datapoints
        if (
            corrections["remote focus"]["enabled"]
            and len(self._corrfit_dpts["remote focus"]) < 2
        ):
            events.publish(events.USER_ABORT)
            raise Exception(
                "Need at least 2 data points for 'remote focus' correction."
            )
        # Calculate patterns
        patterns = np.zeros((experiment.numZSlices, self.no_actuators))
        for i in range(patterns.shape[0]):
            if corrections["remote focus"]["enabled"]:
                z_rf = experiment.aoRFBottom + (experiment.sliceHeight * i)
                z_abs = cockpit.interfaces.stageMover.getPosition()[2] + z_rf
            else:
                z_rf = None
                z_abs = experiment.zStart + (experiment.sliceHeight * i)
            for cname, z in (
                ("remote focus", z_rf),
                ("sensorless", z_abs),
            ):
                if corrections[cname]["enabled"]:
                    corrections_to_restore.add(cname)
                    modes = self._corrfit_eval(cname, z)
                    if len(modes) > 0:
                        self.set_correction(cname, modes=modes)
            patterns[i] = self.proxy.calc_shape()
        # Repeat as necessary
        patterns = np.tile(patterns, (experiment.numReps, 1))
        # Apply the first pattern and omit it from the queue
        self.proxy.send(patterns[0])
        patterns = patterns[1:, :]
        # Queue patterns
        self.proxy.queue_patterns(patterns)
        # Restore original corrections
        for cname in corrections_to_restore:
            self.set_correction(
                cname,
                modes=corrections[cname]["modes"],
                actuator_values=corrections[cname]["actuator_values"]
            )

    def _experiment_cleanup(self):
        # Flush the patterns, in case the experiment ended prematurely
        self.proxy.flush_patterns()
        # Refresh the corrections to apply the shape from before the experiment
        self.refresh_corrections()

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

    def corrfit_dp_get(self):
        return self._corrfit_dpts

    def corrfit_dp_add(self, cname, z, modes):
        self._corrfit_dpts[cname][z] = modes
        self._corrfit_update(cname)
        # Update corrections if necessary
        z_current = self._rf_pos
        if cname == "sensorless":
            z_current = cockpit.interfaces.stageMover.getPosition()[2]
        modes = self._corrfit_eval(cname, z_current)
        if len(modes) > 0:
            self.set_correction(cname, modes=modes)
            self.refresh_corrections()

    def corrfit_dp_rem(self, cname, z):
        # Delete the data point and update the regression model
        del self._corrfit_dpts[cname][z]
        self._corrfit_update(cname)
        # Get the current Z position
        z_current = self._rf_pos
        if cname == "sensorless":
            z_current = cockpit.interfaces.stageMover.getPosition()[2]
        # Re-evaluate the regression model
        modes = self._corrfit_eval(cname, z_current)
        if len(modes) == 0:
            # Not enough data points for evaluation => disable and clear the
            # correction
            self.toggle_correction(cname, False)
            modes = None
        self.set_correction(cname, modes=modes)
        self.refresh_corrections()
        # Reset the remote focus position if there were not enough data points
        if modes is None and cname == "remote focus":
            self._rf_pos = 0
            events.publish(events.STAGE_MOVER, 2)
            time.sleep(
                float(self.transition_time_ms * decimal.Decimal(1e-3))
            )
            events.publish(events.STAGE_STOPPED, self.RF_POSHAN_NAME)

    def _corrfit_update(self, cname):
        # Clear the list of polynomials
        self._corrfit_polys[cname] = []
        # Check if there are enough datapoints
        zs = sorted(self._corrfit_dpts[cname].keys())
        if len(zs) < 2:
            # Not enough datapoints for fitting a line => do nothing
            return
        # Arrange all the modes into a matrix of shape Z x M, where Z is
        # the number of datapoints and M is the number of modes
        modes = np.array(
            [self._corrfit_dpts[cname][z] for z in zs]
        )
        # Fit a line to the set of (z, mode) points for each mode
        for mode_index in range(modes.shape[1]):
            self._corrfit_polys[cname].append(
                np.polynomial.Polynomial.fit(zs, modes[:, mode_index], 1)
            )

    def _corrfit_eval(self, cname, z):
        return np.array([poly(z) for poly in self._corrfit_polys[cname]])