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

import cockpit.devices
import cockpit.devices.device
import cockpit.interfaces.imager
from cockpit import depot
import numpy as np
import Pyro4
import wx
from cockpit import events
from cockpit.util import logger, userConfig
import h5py

from microAO.events import *
from microAO.gui.main import MicroscopeAOCompositeDevicePanel

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
        


class MicroscopeAOCompositeDevice(cockpit.devices.device.Device):
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
            self.proxy.set_controlMatrix(controlMatrix)
        except Exception:
            controlMatrix = None

        try:
            sys_flat = np.asarray(userConfig.getValue("dm_sys_flat"))
            self.set_system_flat(sys_flat)
        except Exception:
            pass

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
                self.proxy.set_controlMatrix(controlMatrix)
            except Exception:
                raise e

    def calibrate(self):
        self.updateROI()
        self.checkFourierFilter()

        controlMatrix = self.proxy.calibrate(numPokeSteps=5)
        userConfig.setValue(
            "dm_controlMatrix", np.ndarray.tolist(controlMatrix)
        )

    def characterise(self):
        self.updateROI()
        self.checkFourierFilter()
        self.checkIfCalibrated()
        assay = self.proxy.assess_character()

        if np.mean(assay[1:, 1:]) < 0:
            controlMatrix = self.proxy.get_controlMatrix()
            self.proxy.set_controlMatrix((-1 * controlMatrix))
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
            imagers = imagers[0]
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
