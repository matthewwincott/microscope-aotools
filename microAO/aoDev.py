#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2018 Nicholas Hall <nicholas.hall@dtc.ox.ac.uk>, Josh Edwards
## <Josh.Edwards222@gmail.com> & Jacopo Antonello
## <jacopo.antonello@dpag.ox.ac.uk>
##
## microAO is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## microAO is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with microAO.  If not, see <http://www.gnu.org/licenses/>.

# Import required packs
import functools

import numpy as np
import Pyro4
import time
import logging
import copy

from microAO.aoAlg import AdaptiveOpticsFunctions

# Should fix this with multiple inheritance for this class!
aoAlg = AdaptiveOpticsFunctions()

from microscope import TriggerType, TriggerMode
from microscope.abc import Device


unwrap_method = {
    "interferometry": aoAlg.unwrap_interferometry,
}

_logger = logging.getLogger(__name__)

wavefront_error_modes = ["RMS", "Strehl"]


def _with_wavefront_camera_ttype_software(func):
    """Method decorator to set camera with software trigger type."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        ttype = self.wavefront_camera.trigger_type
        tmode = self.wavefront_camera.trigger_mode

        ttype_needs_change = ttype is not TriggerType.SOFTWARE
        try:
            if ttype_needs_change:
                self.wavefront_camera.set_trigger(TriggerType.SOFTWARE, tmode)
            return_value = func(self, *args, **kwargs)
        finally:
            if ttype_needs_change:
                self.wavefront_camera.set_trigger(ttype, tmode)
        return return_value

    return wrapper


class AdaptiveOpticsDevice(Device):
    """Class for the adaptive optics device
    This class requires an adaptive element and a camera.
    Everything else is generated on or after __init__"""


    def __init__(
        self, ao_element_uri, wavefront_uri=None, slm_uri=None, dm_layout=None, control_matrix=None, system_flat=None, saturation_limits=(0.1, 0.9), **kwargs
    ):
        # Init will fail if devices it depends on aren't already running, but
        # deviceserver should retry automatically.
        super(AdaptiveOpticsDevice, self).__init__(**kwargs)
        # Adaptive optic element device.
        self.ao_element = Pyro4.Proxy(
            "PYRO:%s@%s:%d"
            % (
                ao_element_uri[0].__name__,
                ao_element_uri[1],
                ao_element_uri[2],
            )
        )
        # Wavefront sensor. Must support soft_trigger for now.
        if wavefront_uri is not None:
            self.wavefront_camera = Pyro4.Proxy(
                "PYRO:%s@%s:%d"
                % (
                    wavefront_uri[0].__name__,
                    wavefront_uri[1],
                    wavefront_uri[2],
                )
            )
        # SLM device
        if slm_uri is not None:
            self.slm = Pyro4.Proxy(
                "PYRO:%s@%s:%d" % (slm_uri[0], slm_uri[1], slm_uri[2])
            )

        self.numActuators = self.ao_element.n_actuators
        # Region of interest (i.e. pupil offset and radius) on camera.
        self.roi = None
        # Mask for the interferometric data
        self.mask = None
        # Mask to select phase information
        self.fft_filter = None
        # Phase acquisition method
        self.phase_method = "interferometry"
        # Control Matrix
        if control_matrix is not None:
            if isinstance(control_matrix, str):
                control_matrix = np.loadtxt(control_matrix)
            self.set_controlMatrix(control_matrix)
        else:
            self.controlMatrix = None
        # Store DM layout
        self.dm_layout = dm_layout
        # Store corrections as zernike modes/actuator values.
        # The sum of these are applied optionally to device.
        self.corrections = {}
        # System correction
        if system_flat is not None:
            if isinstance(system_flat, str):
                system_flat = np.loadtxt(system_flat)
            self.set_correction("system_flat", system_flat)
        else:
            self.set_correction("system_flat", np.zeros(self.numActuators))
        # Last applied zenrike  modes
        self.last_zernike_modes = None
        # Last applied actuators values
        self.last_actuator_values = None
        # Last applied phase image
        self.last_phase_pattern = None

        # We might not use all the actuators. Create a mask for the actuators outside
        # the pupil so we can selectively calibrate them. 0 denotes actuators at
        # the edge, i.e. outside the pupil, and 1 denotes actuators in the pupil

        # Preliminary mask for DeepSIM
        self.pupil_ac = np.ones(self.numActuators)

        try:
            assert np.shape(self.pupil_ac)[0] == self.numActuators
        except Exception:
            raise Exception(
                "Length mismatch between pupil mask (%i) and "
                "number of actuators (%i). Please provide a mask "
                "of the correct length"
                % (np.shape(self.pupil_ac)[0], self.numActuators)
            )

        self._wavefront_error_mode = self.wavefront_rms_error

        if (
            saturation_limits[0] < 0
            or saturation_limits[1] > 1
            or saturation_limits[0] > saturation_limits[1]
        ):
            raise Exception(
                "Wrong value for parameter 'saturation_limits'. The limits "
                "should be in the range [0;1] and the lower limit (first "
                "element) should be the lesser of the two."
            )
        self._saturation_limits = saturation_limits

    def _do_shutdown(self):
        pass

    def initialize(self, *args, **kwargs):
        pass

    @Pyro4.expose
    def enable_camera(self):
        self.wavefront_camera.enable()

    @Pyro4.expose
    def disable_camera(self):
        self.wavefront_camera.disable()

    def generate_isosense_pattern_image(
        self, shape, dist, wavelength, NA, pixel_size
    ):
        try:
            assert type(shape) is tuple
        except:
            raise Exception(
                "Expected %s instead recieved %s"
                % (type((512, 512)), type(shape))
            )

        try:
            assert len(shape) == 2
        except:
            raise Exception(
                "Expected tuple of length 2, instead recieved length %i"
                % len(shape)
            )

        ray_crit_dist = (1.22 * wavelength) / (2 * NA)
        ray_crit_freq = 1 / ray_crit_dist
        max_freq = 1 / (2 * pixel_size)
        freq_ratio = ray_crit_freq / max_freq
        OTF_outer_radx = freq_ratio * (shape[1] / 2)
        OTF_outer_rady = freq_ratio * (shape[0] / 2)

        pattern_ft = np.zeros(shape)

        f1x = shape[1] // 2
        f1y = shape[0] // 2
        f2x = f1x - int(np.round(0.5 * OTF_outer_radx * dist))
        f2y = f1y - int(np.round(0.5 * OTF_outer_rady * dist))
        f3x = f1x + int(np.round(0.5 * OTF_outer_radx * dist))
        f3y = f1y + int(np.round(0.5 * OTF_outer_rady * dist))
        f4x = f1x - int(np.round(OTF_outer_radx * dist))
        f4y = f1y - int(np.round(OTF_outer_rady * dist))
        f5x = f1x + int(np.round(OTF_outer_radx * dist))
        f5y = f1y + int(np.round(OTF_outer_rady * dist))
        freq_loc_half = (
            np.asarray([f2y, f2y, f3y, f3y], dtype="int64"),
            np.asarray([f2x, f3x, f2x, f3x], dtype="int64"),
        )
        freq_loc_quart = (
            np.asarray([f1y, f1y, f4y, f5y], dtype="int64"),
            np.asarray([f4x, f5x, f1x, f1x], dtype="int64"),
        )
        pattern_ft[f1y, f1x] = 1
        pattern_ft[freq_loc_half] = 1 / 2
        pattern_ft[freq_loc_quart] = 1 / 4

        pattern_unscaled = abs(np.fft.fft2(np.fft.ifftshift(pattern_ft)))
        pattern = (pattern_unscaled / np.max(pattern_unscaled)) * (
            (2 ** 16) - 1
        )
        pattern = pattern.astype("uint16")
        return pattern

    @Pyro4.expose
    def apply_isosense_pattern(self, fill_frac, wavelength, NA, pixel_size):

        if fill_frac < 0:
            raise ValueError("Fill fraction must be greater than 0")
        elif fill_frac > 100:
            raise ValueError("Fill fraction must be less than 100")
        else:
            pass
        ## Tell the SLM to prepare the pattern sequence.
        dist = fill_frac / 100
        shape = self.slm.get_shape()
        pattern = self.generate_isosense_pattern_image(
            shape=shape,
            wavelength=wavelength,
            dist=dist,
            NA=NA,
            pixel_size=pixel_size,
        )
        self.slm.set_custom_sequence(wavelength, [pattern, pattern])

    @Pyro4.expose
    def set_trigger(self, ttype, tmode):
        self.ao_element.set_trigger(ttype, tmode)

    @Pyro4.expose
    def get_trigger(self):
        return (self.ao_element.trigger_type, self.ao_element.trigger_mode)

    @Pyro4.expose
    def get_pattern_index(self):
        return self.ao_element.get_pattern_index()

    @Pyro4.expose
    def get_n_actuators(self):
        return self.numActuators

    @Pyro4.expose
    def get_dm_layout(self):
        return self.dm_layout

    @Pyro4.expose
    def set_pupil_ac(self, pupil_ac):
        try:
            assert np.shape(pupil_ac)[0] == self.numActuators
        except Exception:
            raise Exception(
                "Length mismatch between pupil mask (%i) and "
                "number of actuators (%i). Please provide a mask "
                "of the correct length"
                % (np.shape(pupil_ac)[0], self.numActuators)
            )

        self.pupil_ac = pupil_ac

    @Pyro4.expose
    def get_pupil_ac(self):
        return self.pupil_ac

    @Pyro4.expose
    def get_all_unwrap_methods(self):
        return unwrap_method.keys()

    @Pyro4.expose
    def get_unwrap_method(self):
        return self.phase_method

    @Pyro4.expose
    def set_unwrap_method(self, phase_method):
        if not phase_method in unwrap_method:
            raise Exception(
                "TypeError: Not a phase unwrapping method. Check available unwrap methods."
            )
        else:
            self.phase_method = phase_method

    @Pyro4.expose
    def get_all_wavefront_error_modes(self):
        return wavefront_error_modes

    @Pyro4.expose
    def get_wavefront_error_mode(self):
        if self._wavefront_error_mode is self.wavefront_rms_error:
            mode = wavefront_error_modes[0]
        elif self._wavefront_error_mode is self.wavefront_strehl_ratio:
            mode = wavefront_error_modes[1]
        return mode

    @Pyro4.expose
    def set_wavefront_error_mode(self, mode):
        if not mode in wavefront_error_modes:
            raise Exception("TypeError: Not a valid wavefront error mode")
        else:
            if mode == wavefront_error_modes[0]:
                self._wavefront_error_mode = self.wavefront_rms_error
            elif mode == wavefront_error_modes[1]:
                self._wavefront_error_mode = self.wavefront_strehl_ratio

    @Pyro4.expose
    def get_system_flat(self):
        flat_correction = self.corrections.get('system_flat', None)
        return flat_correction

    @Pyro4.expose
    def set_system_flat(self, modes=None, actuator_values=None):
        # Set correction
        self.set_correction("system_flat", modes, actuator_values)

    # This method is used for AO elements such as DMs which have actuators which require direct signal values to be set.
    @Pyro4.expose
    def send(self, values):
        _logger.info("Sending pattern to AO element")

        ttype, tmode = self.get_trigger()
        if ttype != TriggerType.SOFTWARE:
            self.set_trigger(TriggerType.SOFTWARE, TriggerMode.ONCE)

        # Saturate values
        for limit, candidates in (
            (
                self._saturation_limits[0],
                values < self._saturation_limits[0]
            ),
            (
                self._saturation_limits[1],
                values > self._saturation_limits[1]
            )
        ):
            if candidates.any():
                indices = candidates.nonzero()[0]
                _logger.info(
                    f"Saturating the voltages of actuators {list(indices)} to "
                    f"{limit}."
                )
                values[indices] = limit

        _logger.debug("Sending values: {}".format(values))

        try:
            self.ao_element.apply_pattern(values)
        except Exception as e:
            raise e

        self.last_actuator_values = values
        if (ttype, tmode) != self.get_trigger():
            self.set_trigger(ttype, tmode)

    # This method is for AO elements such as SLMs where the phase shape can be applied directly by sending an image of
    # the desired phase.
    @Pyro4.expose
    def apply_phase_pattern(self, wavelength, pattern):
        ao_shape = self.ao_element.get_shape()
        try:
            assert ao_shape == pattern.shape
        except:
            raise Exception(
                "AO element shape is (%i,%i), recieved pattern of shape (%i,%i)"
                % (
                    ao_shape[0],
                    ao_shape[1],
                    pattern.shape[0],
                    pattern.shape[1],
                )
            )

        self.ao_element.set_custom_sequence(wavelength, [pattern, pattern])
        self.last_phase_pattern = pattern

    @Pyro4.expose
    def get_last_actuator_values(self):
        return self.last_actuator_values

    @Pyro4.expose
    def get_last_phase_pattern(self):
        return self.last_phase_pattern

    @Pyro4.expose
    def get_last_modes(self):
        return self.last_zernike_modes

    @Pyro4.expose
    def queue_patterns(self, patterns):
        _logger.info("Queuing patterns on DM")

        ttype, tmode = self.get_trigger()
        if ttype != "RISING_EDGE":
            self.set_trigger(TriggerType.RISING_EDGE, TriggerMode.ONCE)

        # Need to normalise patterns because general DM class expects 0-1 values
        patterns[patterns > 1.0] = 1.0
        patterns[patterns < 0.0] = 0.0

        try:
            self.ao_element.queue_patterns(patterns)
        except Exception as e:
            raise e

        if (ttype, tmode) is not self.get_trigger():
            self.set_trigger(ttype, tmode)

    @Pyro4.expose
    def flush_patterns(self):
        return self.ao_element.flush_patterns()

    @Pyro4.expose
    def set_roi(self, y0, x0, radius):
        self.roi = (
            int(np.round(y0)),
            int(np.round(x0)),
            int(np.round(radius)),
        )
        try:
            assert self.roi is not None
        except:
            raise Exception("ROI assignment failed")

        # Mask will need to be reconstructed as radius has changed
        self.mask = aoAlg.make_mask(self.roi[2])
        try:
            assert self.mask is not None
        except:
            raise Exception("Mask construction failed")

        # Fourier filter should be erased, as it's probably wrong.
        ##Might be unnecessary
        self.fft_filter = None
        return

    @Pyro4.expose
    def get_roi(self):
        if np.any(self.roi) is None:
            raise Exception(
                "No region of interest selected. Please select a region of interest"
            )
        else:
            return self.roi

    @Pyro4.expose
    def get_fourierfilter(self):
        if np.any(self.fft_filter) is None:
            raise Exception(
                "Fourier filter is None. Please construct Fourier filter"
            )
        else:
            return self.fft_filter

    @Pyro4.expose
    def get_controlMatrix(self):
        if np.any(self.controlMatrix) is None:
            raise Exception(
                "No control matrix available. Please load a control matrix or calibrate the AO element"
            )
        else:
            return self.controlMatrix

    @Pyro4.expose
    def set_controlMatrix(self, controlMatrix):
        self.controlMatrix = controlMatrix
        aoAlg.set_controlMatrix(controlMatrix)
        _logger.info("Set control matrix")
        return

    @Pyro4.expose
    def make_mask(self, radius):
        self.mask = aoAlg.make_mask(radius)
        return self.mask

    @Pyro4.expose
    @_with_wavefront_camera_ttype_software
    def acquire_raw(self):
        """This method changes trigger type to software.  If something is
        planning on calling this method multiple times in a row it
        should ensure that it sets software trigger type itself
        otherwise the enable/disable cycle that it involves will take
        a lot of time.
        """
        # FIXME: this can loop forever if the camera keeps timing out.
        # It's unlikely that this is the right thing to do.
        while True:
            try:
                data_raw, _ = self.wavefront_camera.grab_next_data()
                break
            except Exception as e:
                # FIXME: this only catches the error from Ximea
                # cameras (I'm not sure it still does).  We should not
                # be trying to handle hardware specific exceptions.
                if str(e) == "ERROR 10: Timeout":
                    _logger.info(
                        "Received Timeout error from camera. Waiting to try again..."
                    )
                    time.sleep(1)
                else:
                    _logger.info(type(e))
                    _logger.info("Error is: %s" % (e))
                    raise e
        return data_raw

    @Pyro4.expose
    def acquire(self):
        data_raw = self.acquire_raw()
        if np.any(self.roi) is None:
            data = data_raw
        else:
            data_cropped = np.zeros(
                (self.roi[2] * 2, self.roi[2] * 2), dtype=float
            )
            data_cropped[:, :] = data_raw[
                self.roi[0] - self.roi[2] : self.roi[0] + self.roi[2],
                self.roi[1] - self.roi[2] : self.roi[1] + self.roi[2],
            ]
            if np.any(self.mask) is None:
                self.mask = self.make_mask(self.roi[2])
                data = data_cropped
            else:
                data = data_cropped * self.mask
        return data

    @Pyro4.expose
    def set_fourierfilter(
        self, test_image, region=None, window_dim=None, mask_di=None
    ):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception(
                "No region of interest selected. Please select a region of interest"
            )

        try:
            self.fft_filter = aoAlg.make_fft_filter(
                test_image,
                region=region,
                window_dim=window_dim,
                mask_di=mask_di,
            )
        except Exception as e:
            _logger.info(e)
        return self.fft_filter

    @Pyro4.expose
    def check_unwrap_conditions(self, image=None):
        if self.phase_method == "interferometry":
            if np.any(self.mask) is None:
                raise Exception("Mask is None. Please construct mask.")
            else:
                pass
            if np.any(self.fft_filter) is None:
                if image is not None:
                    self.set_fourierfilter(image)
                else:
                    raise Exception(
                        "Fourier filter is None. Please construct Fourier filter"
                    )
            else:
                pass

    @Pyro4.expose
    def phaseunwrap(self, image=None):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception(
                "No region of interest selected. Please select a region of interest"
            )

        if np.any(image) is None:
            image = self.acquire()

        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions()

        out = unwrap_method[self.phase_method](image)
        return out

    @Pyro4.expose
    def getzernikemodes(self, image_unwrap, noZernikeModes, resize_dim=128):
        coef = aoAlg.get_zernike_modes(
            image_unwrap, noZernikeModes, resize_dim=resize_dim
        )
        return coef

    @Pyro4.expose
    def createcontrolmatrix(
        self,
        imageStack,
        noZernikeModes,
        pokeSteps,
        pupil_ac=None,
        threshold=0.005,
    ):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception(
                "No region of interest selected. Please select a region of interest"
            )

        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions(imageStack[0, :, :])

        if np.any(pupil_ac == None):
            pupil_ac = self.pupil_ac
        else:
            pass

        noImages, y, x = np.shape(imageStack)
        numPokeSteps = len(pokeSteps)

        assert x == y
        edge_mask = np.sqrt(
            (np.arange(-x / 2.0, x / 2.0) ** 2).reshape((x, 1))
            + (np.arange(-x / 2.0, x / 2.0) ** 2)
        ) < ((x / 2.0) - 3)
        all_zernikeModeAmp = []
        all_pokeAmps = []

        curr_calc = 0
        for ac in range(self.numActuators):
            image_stack_cropped = np.zeros((numPokeSteps, y, x))
            unwrapped_stack_cropped = np.zeros((numPokeSteps, y, x))

            # Determine if the current actuator is in the pupil
            if pupil_ac[ac] == 1:
                pokeAc = np.zeros(self.numActuators)
                zernikeModeAmp_list = []

                for im in range(numPokeSteps):
                    curr_calc += 1
                    # Acquire the current poke image
                    poke_image = imageStack[curr_calc - 1, :, :]
                    image_stack_cropped[im, :, :] = poke_image

                    # Unwrap the current image
                    image_unwrap = unwrap_method[self.phase_method](poke_image)
                    unwrapped_stack_cropped[im, :, :] = image_unwrap

                    # Check the current phase map for discontinuities which can interfere with the Zernike mode measurements
                    diff_image = (
                        abs(np.diff(np.diff(image_unwrap, axis=1), axis=0))
                        * edge_mask[:-1, :-1]
                    )
                    no_discontinuities = np.shape(
                        np.where(diff_image > 2 * np.pi)
                    )[1]
                    if no_discontinuities > (x * y) / 1000.0:
                        _logger.info(
                            "Unwrap image %d/%d contained discontinuites"
                            % (curr_calc, noImages)
                        )
                        _logger.info(
                            "Zernike modes %d/%d not calculated"
                            % (curr_calc, noImages)
                        )
                    else:
                        pokeAc[ac] = pokeSteps[im]
                        all_pokeAmps.append(pokeAc.tolist())
                        _logger.info(
                            "Calculating Zernike modes %d/%d..."
                            % (curr_calc, noImages)
                        )

                    curr_amps = aoAlg.get_zernike_modes(
                        image_unwrap, noZernikeModes
                    )
                    zernikeModeAmp_list.append(curr_amps)
                    all_zernikeModeAmp.append(curr_amps)
            np.save("image_stack_cropped_ac_%i" % ac, image_stack_cropped)
            np.save("unwrap_stack_cropped_ac_%i" % ac, unwrapped_stack_cropped)

        all_zernikeModeAmp = np.asarray(all_zernikeModeAmp)
        all_pokeAmps = np.asarray(all_pokeAmps)

        _logger.info("Computing Control Matrix")
        self.controlMatrix = aoAlg.create_control_matrix(
            zernikeAmps=all_zernikeModeAmp,
            pokeSteps=all_pokeAmps,
            numActuators=self.numActuators,
            pupil_ac=self.pupil_ac,
            threshold=threshold,
        )
        _logger.info("Control Matrix computed")
        return self.controlMatrix

    @Pyro4.expose
    def acquire_unwrapped_phase(self):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception(
                "No region of interest selected. Please select a region of interest"
            )

        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions()

        interferogram = self.acquire()
        interferogram_unwrap = self.phaseunwrap(interferogram)
        _logger.info("Phase unwrapped ")
        return interferogram, interferogram_unwrap

    @Pyro4.expose
    def measure_zernike(self, noZernikeModes):
        interferogram, unwrapped_phase = self.acquire_unwrapped_phase()
        zernike_amps = self.getzernikemodes(unwrapped_phase, noZernikeModes)
        return zernike_amps

    @Pyro4.expose
    def wavefront_rms_error(self, phase_map=None):
        if phase_map is None:
            phase_map = self.acquire_unwrapped_phase()

        if self.mask is None:
            self.make_mask(phase_map.shape // 2)

        true_flat = np.zeros(np.shape(phase_map))
        rms_error = np.sqrt(
            np.mean((true_flat[self.mask] - phase_map[self.mask]) ** 2)
        )
        return rms_error

    @Pyro4.expose
    def wavefront_strehl_ratio(self, phase_map=None):
        if phase_map is None:
            phase_map = self.acquire_unwrapped_phase()

        if self.mask is None:
            self.make_mask(phase_map.shape // 2)

        strehl_ratio = np.exp(
            -np.mean(
                (phase_map[self.mask] - np.mean(phase_map[self.mask])) ** 2
            )
        )
        return strehl_ratio

    @Pyro4.expose
    def get_actuator_pos_from_modes(self, applied_z_modes, offset):
        actuator_pos = np.zeros(self.numActuators)
        if np.any(applied_z_modes != 0.0):
            try:
                actuator_pos = aoAlg.ac_pos_from_zernike(
                    applied_z_modes, self.numActuators
                )
            except Exception as err:
                _logger.info(err)
                raise err

        actuator_pos += offset

        return actuator_pos

    @Pyro4.expose
    def get_corrections(self, include_default=False):
        corrections = copy.deepcopy(self.corrections)
        if "default" in corrections and not include_default:
            # Remove default correction if not explicitly requested
            del corrections["default"]
        return corrections

    @Pyro4.expose
    def set_correction(self, name, modes=None, actuator_values=None):
        if name not in self.corrections:
            # Initialise new correction
            self.corrections[name] = {}
            self.corrections[name]["enabled"] = False
        self.corrections[name]["modes"] = modes
        self.corrections[name]["actuator_values"] = actuator_values

    @Pyro4.expose
    def toggle_correction(self, name, enable):
        self.corrections[name]["enabled"] = enable

    @Pyro4.expose
    def sum_corrections(self, corrections=None, only_enabled=True):
        # If no subset of correction is specified => default to all corrections
        if corrections is None:
            corrections = self.corrections

        # Filter out disabled corrections if necessary
        if only_enabled:
            corrections = {
                key: value
                for key, value in corrections.items()
                if value["enabled"]
            }

        # Sum corrections (modes and actuator values)
        corrections_modes = [
            value["modes"]
            for value in corrections.values()
            if value["modes"] is not None
        ]
        if len(corrections_modes) > 0:
            total_corrections_phase = sum(corrections_modes)
        else:
            total_corrections_phase = np.zeros(self.numActuators)

        corrections_offset = [
            value["actuator_values"] - 0.5
            for value in corrections.values()
            if value["actuator_values"] is not None
        ]
        if len(corrections_offset) > 0:
            total_corrections_offset = sum(corrections_offset) + 0.5
        else:
            total_corrections_offset = np.zeros(self.numActuators) + 0.5

        return total_corrections_phase, total_corrections_offset

    @Pyro4.expose
    def calc_shape(self):
        # Get phase and offset from corrections
        zernikes, offset = self.sum_corrections()

        # Get actuator values from corrections
        return self.get_actuator_pos_from_modes(zernikes, offset)

    @Pyro4.expose
    def apply_corrections(self):
        # Get actuator positions and corrections
        actuator_pos = self.calc_shape()

        # Send to device
        self.send(actuator_pos)

        return actuator_pos

    @Pyro4.expose
    def set_phase(self, applied_z_modes=None, offset=None):
        # Set applied modes and offset as default correction
        self.set_correction(
            "default",
            modes=applied_z_modes,
            actuator_values=offset
        )
        self.toggle_correction("default", True)

        # Apply corrections
        actuator_pos = self.apply_corrections()

        # Record last applied modes
        self.last_zernike_modes = applied_z_modes

        return actuator_pos

    @Pyro4.expose
    @_with_wavefront_camera_ttype_software
    def assess_character(self, modes_tba=None, step=1.0):
        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions()

        if modes_tba is None:
            modes_tba = self.get_controlMatrix().shape[1]
        assay = np.zeros((modes_tba, modes_tba))
        applied_z_modes = np.zeros(modes_tba)
        self.set_phase(np.zeros(modes_tba))
        z_modes_ac0 = self.measure_zernike(modes_tba)
        for ii in range(modes_tba):
            applied_z_modes[ii] = step
            self.set_phase(applied_z_modes)
            _logger.info("Appling Zernike mode %i/%i" % (ii + 1, modes_tba))
            acquired_z_modes = self.measure_zernike(modes_tba)
            _logger.info("Measured phase")
            # Calculate the error terms of the assay. The step is zero for all
            # modes besides the one currently being probed.
            assay[:, ii] = acquired_z_modes - z_modes_ac0
            assay[ii, ii] -= step
            applied_z_modes[ii] = 0.0
        self.set_phase(np.zeros(modes_tba))
        return assay

    @Pyro4.expose
    def measure_metric(self, metric_name, image, **kwargs):
        metric = aoAlg.measure_metric(metric_name, image, **kwargs)
        return metric
