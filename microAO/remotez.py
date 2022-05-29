import os
from functools import partial
import glob
import dataclasses
import copy

import numpy as np
import scipy
import h5py
import tifffile
import skimage.color
import skimage.draw
import skimage.filters
import skimage.measure
import skimage.transform

from cockpit import depot, events
from cockpit.util import logger, userConfig, threads
from microAO.gui.remoteFocus import RF_DATATYPES
from microAO.gui.sensorlessViewer import SensorlessResultsViewer
from microAO.events import *

RF_DATATYPES = ["zernike", "actuator"]

@dataclasses.dataclass(frozen=True)
class _RemoteFocusStack:
    stage_position_abs: float
    stage_position_rel: float
    rf_offsets: np.ndarray
    images: list[np.ndarray]

@dataclasses.dataclass(frozen=True)
class _OrthoProjCalib:
    image: np.ndarray
    image_labelled: np.ndarray
    stage_position_rel: float
    stage_position_rel_corrected: float
    centroid: tuple[float, float]
    centroid_padded: tuple[float, float]

def _orthogonal_projection(image, point = None, xy2z_ratio=1.0):
    """Both image and point have dimensions (Z, Y, X)."""
    # Create new images
    projection_zx = np.zeros((image.shape[0], image.shape[2]))
    projection_zy = np.zeros((image.shape[0], image.shape[1]))
    # Determine projection point
    if point is None:
        point = [dim // 2 for dim in image.shape]
    # Project and return
    for zi in range(image.shape[0]):
        projection_zx[zi] = image[zi, point[1], :]
        projection_zy[zi] = image[zi, :, point[2]]
    # Scale projections if necessary
    if xy2z_ratio != 1.0:
        new_z = image.shape[0] // xy2z_ratio
        projection_zx = np.around(
            skimage.transform.resize(
                projection_zx,
                (new_z, projection_zx.shape[1])
            )
        ).astype(image.dtype)
        projection_zy = np.around(
            skimage.transform.resize(
                projection_zy,
                (new_z, projection_zy.shape[1])
            )
        ).astype(image.dtype)
    return (image[point[0]], projection_zx, projection_zy)

def _find_bead_centre(bead_image):
    # Get the centre point an expand its dimensions to 1 above the image
    centre = np.array(bead_image.shape) / 2
    for _ in range(bead_image.ndim):
        centre = centre[:, np.newaxis]
    # Calculate distances from the centre
    distances = np.linalg.norm(
        np.indices(bead_image.shape) - centre + 0.5, axis=0
    )
    # Build a list of tuples
    pixels = []
    for i in range(0, np.prod(bead_image.shape)):
        indices = np.unravel_index(i, bead_image.shape)
        pixels.append((indices, bead_image[indices], distances[indices]))
    # Sort by distance first because it is less important
    pixels.sort(key=lambda x: x[-1])
    pixels.sort(key=lambda x: x[-2], reverse=True)
    # Return the indices of the top item
    return pixels[0][0]

def _find_psf_centroid(image_projection):
    # Threshold the image
    image_binary = image_projection > skimage.filters.threshold_yen(
        image_projection
    )
    # Label and get region properties
    image_labelled = skimage.measure.label(image_binary)
    region_properties = skimage.measure.regionprops(image_labelled)
    # Find a suitable region
    for region in region_properties:
        if region["area"] >= 100:
            return region["centroid"]

class RemoteZ():
    def __init__(self, device):
        # Store reference to cockpit device
        self._device = device

        # Store state
        self.datapoints = {}
        self.z_lookup = {key:[] for key in RF_DATATYPES}
        self._position = 0
        self._compensation_poly = None

        self._n_actuators = 0
        self._n_modes = 0
        control_matrix = self._device.proxy.get_controlMatrix()
        if control_matrix is not None:
            self._n_actuators = control_matrix.shape[0]
            self._n_modes = control_matrix.shape[1]

        datapoints_init = userConfig.getValue("rf_datapoints")
        if datapoints_init:
            self.datapoints = copy.deepcopy(datapoints_init)
            self.update_calibration()
            self.set_z(0)

    def set_control_matrix(self, control_matrix):
        self._n_actuators = control_matrix.shape[0]
        self._n_modes = control_matrix.shape[1]

    def calibrate_z_pos(self, zstage, zpos, defocus_modes=[3,10], other_modes=[21, 4, 5, 6, 7, 8, 9]):
        if self._n_actuators == 0:
            raise Exception(
                "Remote focusing calibration failed because the adaptive "
                "element has not been calibrated."
            )

        mover = depot.getHandlerWithName("{}".format(zstage.name))

        for z in zpos:
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
            self.sensorless_correct(defocus_modes)

            # Correct other aberrations
            self.sensorless_correct(other_modes)

            # Save datapoint
            values = self._device.proxy.get_last_actuator_values()
            datapoint = {
                'datatype': 'actuator',
                'z': z,
                'values': values
            }

            self.add_datapoint(datapoint)

    @threads.callInNewThread
    def calibrate_counteraction_get_data(
        self,
        handlers_zstage,
        handlers_camera,
        handlers_imager,
        calib_params,
        xy_pixelsize,
        output_dir_path
    ):
        # Obtain the image stacks
        stage_offsets = np.linspace(
            calib_params["stage_min"],
            calib_params["stage_max"],
            int((
                (calib_params["stage_max"] - calib_params["stage_min"]) /
                calib_params["stage_step"]
            )) + 1
        )
        rf_stacks = [None] * stage_offsets.shape[0]
        stage_original_position = handlers_zstage.getPosition() # um
        for index, zpos_stage in enumerate(
            stage_original_position + stage_offsets
        ):
            # Update status bar
            events.publish(
                events.UPDATE_STATUS_LIGHT,
                "image count",
                f"Remote focus calibration | Obtaining Z stack {index + 1} / "
                f"{stage_offsets.shape[0]}..."
            )
            # Move the stage to the right position
            handlers_zstage.moveAbsolute(zpos_stage)
            # Do a remote focus Z stack and store the images
            stage_current_position = handlers_zstage.getPosition()
            rf_stacks[index] = _RemoteFocusStack(
                stage_position_abs=stage_current_position,
                stage_position_rel=stage_current_position - stage_original_position,
                rf_offsets=np.linspace(
                    calib_params["defocus_min"],
                    calib_params["defocus_max"],
                    int((calib_params["defocus_max"] - calib_params["defocus_min"]) / calib_params["defocus_step"]) + 1
                ),
                images=np.array(
                    self.zstack(
                        calib_params["defocus_min"],
                        calib_params["defocus_max"],
                        calib_params["defocus_step"],
                        camera=handlers_camera,
                        imager=handlers_imager
                    )
                )
            )
            # For convenience, ensure the stack is always ascending in terms of
            # Z positions
            if (
                rf_stacks[index].rf_offsets[0] >
                rf_stacks[index].rf_offsets[-1]
            ):
                # Descending offsets => reverse both offsets and images
                rf_stacks[index].rf_offsets = np.flipud(
                    rf_stacks[index].rf_offsets
                )
                rf_stacks[index].images = np.flipud(rf_stacks[index].images)
        # Move the stage to its original position
        handlers_zstage.moveAbsolute(stage_original_position)
        # Ensure stacks are sorted by stage position
        rf_stacks.sort(key=lambda item: item.stage_position_abs)
        # Save the collected data
        events.publish(
            events.UPDATE_STATUS_LIGHT,
            "image count",
            "Remote focus calibration | Saving data..."
        )
        for rf_stack in rf_stacks:
            tifffile.imwrite(
                output_dir_path.joinpath(
                    f"remote-focus-zstack_z{rf_stack.stage_position_rel:.03f}"
                    "um.ome.tif"
                ),
                rf_stack.images,
                metadata = {
                    "axes": "ZYX",
                    "PhysicalSizeX": xy_pixelsize,                  # um
                    "PhysicalSizeY": xy_pixelsize,                  # um
                    "PhysicalSizeZ": calib_params["defocus_step"],  # um
                }
            )
        # Inform the GUI thread that a bead needs to be selected
        events.publish(
            PUBSUB_RF_CALIB_CACT_DATA,
            rf_stacks,
            output_dir_path,
            calib_params["defocus_step"]
        )

    @threads.callInNewThread
    def calibrate_counteraction_get_projections(
        self,
        rf_stacks,
        bead_roi,
        xy_pixelsize,
        defocus_step,
        output_dir_path
    ):
        events.publish(
            events.UPDATE_STATUS_LIGHT,
            "image count",
            "Remote focus calibration | Calculating orthogonal views..."
        )
        # Find minimum and maximum Z values, and total Z range in px
        z_min_um = rf_stacks[0].stage_position_abs + rf_stacks[0].rf_offsets[0]
        z_max_um = rf_stacks[-1].stage_position_abs + rf_stacks[-1].rf_offsets[-1]
        z_range_px = int(np.ceil((z_max_um - z_min_um) / xy_pixelsize))
        # Process stacks
        projection_data = [[], []]  # list of lists; [[zx0, zx1, ...], [zy0, zy1, ...]]
        for rf_stack in rf_stacks:
            # Crop bead
            bead = rf_stack.images[
                :,
                bead_roi[1] : bead_roi[1] + bead_roi[3],
                bead_roi[0] : bead_roi[0] + bead_roi[2]
            ]
            # Derive orthogonal views
            projections = _orthogonal_projection(
                bead,
                _find_bead_centre(bead),
                xy_pixelsize / defocus_step
            )[1:]
            # Calculate the necessary padding (identical for both zx and zy)
            offset_top_um = (
                z_max_um -
                (rf_stack.stage_position_abs + rf_stack.rf_offsets[-1])
            )
            offset_top_px = int(np.around(offset_top_um / xy_pixelsize))
            offset_bottom_px = (
                z_range_px - offset_top_px - projections[0].shape[0]
            )
            if offset_bottom_px < 0:
                # The image has been offset too much => clamp it to the bottom
                offset_top_px += offset_bottom_px
                offset_bottom_px = 0
            # Process the two projections
            for index, projection in enumerate(projections):
                # Create a padded image
                padded_stack = []
                if offset_top_px > 0:
                    padded_stack.append(
                        np.zeros(
                            (offset_top_px, projection.shape[1]),
                            dtype=projection.dtype
                        )
                    )
                padded_stack.append(projection)
                if offset_bottom_px > 0:
                    padded_stack.append(
                        np.zeros(
                            (offset_bottom_px, projection.shape[1]),
                            dtype=projection.dtype
                        )
                    )
                projection_padded = np.vstack(padded_stack)
                # Find centroid
                centroid = _find_psf_centroid(projection)
                # Derive error, padded_centroid, and labelled projection, if necessary
                if not centroid:
                    print(
                        "!!! Warning !!! Failed to find centroid for RF "
                        f"calibration stack at Z = {rf_stack.stage_position_rel} "
                        f"and projection {index}."
                    )
                    position_corrected = None
                    centroid_padded = None
                    projection_padded_labelled = None
                else:
                    # Calculate error term
                    image_offset_um = ((projections.shape[0] / 2) - centroid[0]) * xy_pixelsize
                    error = rf_stack.stage_position_rel - image_offset_um
                    position_corrected = rf_stack.stage_position_rel + error
                    # Calculate padded centroid
                    centroid_padded = (
                        centroid[0] + abs(offset_top_px - offset_bottom_px) / 2,
                        centroid[1]
                    )
                    # Label projection image
                    projection_padded_labelled = skimage.color.gray2rgb(projection_padded)
                    rr, cc = skimage.draw.disk(
                        centroid_padded,
                        2.5,
                        shape=projection_padded_labelled.shape
                    )
                    projection_padded_labelled[rr, cc, :] = (1, 0, 1)
                # Store the data
                projection_data[index].append(
                    _OrthoProjCalib(
                        projection_padded,
                        projection_padded_labelled,
                        rf_stack.stage_position_rel,
                        position_corrected,
                        centroid,
                        centroid_padded
                    )
                )
        # Fit a line to the corrected Z offsets and save the coefficients
        polynomials = []
        for index in range(len(projection_data)):
            polynomials.append(
                np.polynomial.polynomial.Polynomial.fit(
                    [
                        datum.stage_position_rel_corrected
                        for datum in projection_data[index]
                        if datum.stage_position_rel_corrected
                    ],
                    [
                        datum.stage_position_rel
                        for datum in projection_data[index]
                        if datum.stage_position_rel_corrected
                    ],
                    1
                )
            )
        self._compensation_poly = sum(polynomials) / 2
        np.savetxt(
            output_dir_path.joinpath(
                f"remote-focus_compensation-polynomial-coefficients.txt",
            ),
            self._compensation_poly.convert().coef
        )
        # Concatenate projections and save them
        for index, proj_type in enumerate(("zx", "zy")):
            for suffix in ("", "_labelled"):
                concat = np.concatenate(
                    [
                        getattr(datum, "image" + suffix)
                        for datum in projection_data[index]
                    ],
                    axis=1
                )
                tifffile.imwrite(
                    output_dir_path.joinpath(
                        f"orthogonal-views_{proj_type}{suffix}.ome.tif"
                    ),
                    concat,
                    metadata = {
                        "PhysicalSizeX": xy_pixelsize,  # um
                        "PhysicalSizeY": xy_pixelsize,  # um
                    }
                )
        # Ask the GUI thread to update the status light
        events.publish(PUBSUB_RF_CALIB_CACT_PROJ)

    def sensorless_correct(self, modes):
        # Get current sensorless params
        params_prev = copy.deepcopy(self._device.sensorless_params.copy)

        try:
            self._device.sensorless_params["modes_subset"] = modes


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
                PUBSUB_SENSORLESS_FINISH,
                action,
                60
            )
            logger.log.debug("Done sensorless")
            window.Close()

        except Exception as e:
            print('error', e)

        finally:
            # Return sensorless params to previous
            self._device.sensorless_params = params_prev

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

    def add_datapoint(self, z, datatype, value):
        if datatype not in RF_DATATYPES:
            raise Exception(f"Unrecognised datatype '{datatype}'.")
        if z not in self.datapoints:
            self.datapoints[z] = {key: None for key in RF_DATATYPES}
        self.datapoints[z][datatype] = value
        self.update_calibration()

    def remove_datapoint(self, z, datatype):
        if datatype not in RF_DATATYPES:
            raise Exception(f"Unrecognised datatype '{datatype}'.")
        if z in self.datapoints:
            self.datapoints[z][datatype] = None
            if not any([self.datapoints[z][dtype] for dtype in RF_DATATYPES]):
                del self.datapoints[z]
        self.update_calibration()

    def update_calibration(self, datatypes=None):
        # Get data
        if datatypes is None:
            datatypes = RF_DATATYPES

        if type(datatypes) is not list:
            datatypes = [datatypes]

        for datatype in datatypes:
            z = [z for z in self.datapoints.keys() if self.datapoints[z][datatype] is not None]
            z = np.array(z)
            values = np.array([self.datapoints[zz][datatype] for zz in z])

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

    def calc_shape(self, z, datatype="actuator"):
        if self._n_actuators == 0:
            raise Exception(
                "Failed to calculate wavefront shape for remote focusing "
                "because the adaptive element has not been calibrated."
            )
        if len(self.z_lookup[datatype]) < 2:
            raise Exception(
                "Failed to calculate wavefront shape for remote focusing "
                "because the remote focusing has not been calibrated."
            )

        # Get current remotez correction
        original_corrections = self._device.get_corrections()

        # Compensate Z
        if self._compensation_poly:
            z = self._compensation_poly(z)

        try:
            if datatype == "zernike":
                values = np.array([self.z_lookup[datatype][i](z) for i in range(0,self._n_modes)])
                self._device.set_correction("remotez", modes=values)
            elif datatype == "actuator":
                values = np.array([self.z_lookup[datatype][i](z) for i in range(0,self._n_actuators)])
                self._device.set_correction("remotez", actuator_values=values)

        except IndexError:
            # No lookup data
            pass

        # Get shape
        self._device.toggle_correction("remotez", True)
        actuator_pos = self._device.proxy.calc_shape()

        # Restore original remotez correction
        if "remotez" in original_corrections:
            self._device.set_correction(
                "remotez",
                modes=original_corrections["remotez"]["modes"],
                actuator_values=original_corrections["remotez"]["actuator_values"]
            )
            self._device.toggle_correction("remotez", original_corrections["remotez"]["enabled"])
        else:
            self._device.toggle_correction("remotez", False)

        return actuator_pos

    def set_z(self, z, datatype="actuator"):
        if self._n_actuators == 0:
            raise Exception(
                "Failed to change the remote focus plane because the adaptive "
                "element has not been calibrated."
            )
        if len(self.z_lookup[datatype]) < 2:
            raise Exception(
                "Failed to change the remote focus plane because the remote "
                "focusing has not been calibrated."
            )

        try:
            if datatype == "zernike":
                values = np.array([self.z_lookup[datatype][i](z) for i in range(0,self._n_modes)])
                self._device.set_correction("remotez", modes=values)
            elif datatype == "actuator":
                values = np.array([self.z_lookup[datatype][i](z) for i in range(0,self._n_actuators)])
                self._device.set_correction("remotez", actuator_values=values)

            actuator_pos = self._device.refresh_corrections()

            self._position = z

        except IndexError:
            # No lookup data
            pass

        return actuator_pos

    def get_z(self):
        return self._position

    def save_datapoints(self, output_dir):
        for z in self.datapoints.keys():
            for datatype in self.datapoints[z].keys():
                value = self.datapoints[z][datatype]
                if value is None:
                    continue
                fname = "{}-{}.h5".format(datatype, str(z).replace('.', '_'))
                fpath = os.path.join(output_dir, fname)

                with h5py.File(fpath, 'w') as f:
                    f.create_dataset("z", data=z)
                    f.create_dataset("datatype", data=datatype)
                    f.create_dataset("value", data=value)

    def load_datapoints(self, input_dir):
        search_string = str(os.path.join(input_dir, '*.h5'))

        for fpath in glob.glob(search_string):
            with h5py.File(fpath, 'r') as f:
                z = f["z"][()]
                datatype = f["datatype"][()].decode("utf-8")
                value = f["value"][()]
                self.add_datapoint(z, datatype, value)
