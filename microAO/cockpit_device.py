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
    correction_stack,
    zernike_applied,
    nollZernike,
    sensorless_correct_coef,
    actuator_offset,
    metric_stack,
    z_steps
):
    # Save full stack of images used
    # _np_save_with_timestamp(
    #     np.asarray(correction_stack),
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
                ('correction_stack',np.asarray(correction_stack)),
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
        if self.config.get('exercise_on_startup', 'true').lower() == 'true':
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
            error_thresh=self.sys_flat_parameters["sys_error_thresh"],
            z_modes_ignore=z_ignore,
        )

        userConfig.setValue("dm_sys_flat", np.ndarray.tolist(sys_flat_values))
        self.proxy.set_system_flat(sys_flat_values)
        return sys_flat_values, best_z_amps_corrected

    def reset(self):
        self.proxy.reset()

    def applySysFlat(self):
        sys_flat_values = np.asarray(userConfig.getValue("dm_sys_flat"))
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
            "correction_stack" : [],
            "metric_stack" : [],
            "sensorless_correct_coef" : np.zeros(n_modes),          # Measured aberrations
            "z_steps" : np.linspace(self.sensorless_params["z_min"], self.sensorless_params["z_max"], self.sensorless_params["numMes"]),
            "zernike_applied" : np.zeros((0, n_modes)), # Array of all z aberrations to apply during experiment
            "metric_calculated" : np.array(())
        }

        if self.sensorless_params.get("start_from_flat", False):
            self.sensorless_data["actuator_offset"] = userConfig.getValue("dm_sys_flat")
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
            print(it_zernike_applied)
            self.sensorless_data["zernike_applied"] = np.concatenate(
                [self.sensorless_data["zernike_applied"], it_zernike_applied]
            )

        logger.log.info("Applying the first Zernike mode")
        # Apply the first Zernike mode
        logger.log.debug(self.sensorless_data["zernike_applied"][len(self.sensorless_data["correction_stack"]), :])
        self.set_phase(
            self.sensorless_data["zernike_applied"][len(self.sensorless_data["correction_stack"]), :],
            offset=self.sensorless_data["actuator_offset"],
        )

        # Take image. This will trigger the iterative sensorless AO correction
        wx.CallAfter(wx.GetApp().Imager.takeImage)

    def correctSensorlessImage(self, image, timestamp):
        del timestamp
        if len(self.sensorless_data["correction_stack"]) < self.sensorless_data["zernike_applied"].shape[0]:
            logger.log.info(
                "Correction image %i/%i"
                % (
                    len(self.sensorless_data["correction_stack"]) + 1,
                    self.sensorless_data["zernike_applied"].shape[0],
                )
            )
            events.publish(
                events.UPDATE_STATUS_LIGHT,
                "image count",
                "Sensorless AO: image %s/%s, mode %s, meas. %s"
                % (
                    len(self.sensorless_data["correction_stack"]) + 1,
                    self.sensorless_data["zernike_applied"].shape[0],
                    self.sensorless_params["nollZernike"][
                        len(self.sensorless_data["correction_stack"])
                        // self.sensorless_params["numMes"]
                        % len(self.sensorless_params["nollZernike"])
                    ],
                    (len(self.sensorless_data["correction_stack"]) + 1) % self.sensorless_params["numMes"] + 1,
                ),
            )
            # Store image for current applied phase
            self.sensorless_data["correction_stack"].append(np.ndarray.tolist(image))
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
        ind = int(len(self.sensorless_data["correction_stack"]) / self.sensorless_params["numMes"])
        nollInd = (
            np.where(self.sensorless_data["zernike_applied"][len(self.sensorless_data["correction_stack"]) - 1, :])[
                0
            ][0]
            + 1
        )
        logger.log.debug("Current Noll index being corrected: %i" % nollInd)
        current_stack = np.asarray(self.sensorless_data["correction_stack"])[
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
        if len(self.sensorless_data["correction_stack"]) < self.sensorless_data["zernike_applied"].shape[0]:
            if len(self.sensorless_data["correction_stack"]) % self.sensorless_params["numMes"] == 0:
                self.correctSensorlessAberation()

            # Advance counter by 1 and apply next phase
            self.set_phase(
                self.sensorless_data["zernike_applied"][len(self.sensorless_data["correction_stack"]), :],
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
                self.sensorless_data["correction_stack"],
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
            self.send(self.sensorless_data["actuator_offset"])

        # Update/create metric plot
        sensorless_data = {
            'correction_stack': self.sensorless_data["correction_stack"],
            'metric_stack': self.sensorless_data["metric_stack"],
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

    def send(self, actuator_values):
        # Send values to device
        self.proxy.send(actuator_values)

        # Publish events
        events.publish(PUBSUB_SET_ACTUATORS, actuator_values)

    def set_phase(self, applied_z_modes, offset=None):
        # Send values to device
        actuator_values = self.proxy.set_phase(applied_z_modes, offset)

        # Publish events
        events.publish(PUBSUB_SET_ACTUATORS, actuator_values)
        events.publish(PUBSUB_SET_PHASE, applied_z_modes, offset)


# Start a timer to update modes.
# self._timer = wx.Timer(self)
# self._timer.Start(500)
# self.Bind(wx.EVT_TIMER, self.RefreshModes, self._timer)