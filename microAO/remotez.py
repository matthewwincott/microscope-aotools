import dataclasses
import copy

import numpy as np
import tifffile
import skimage.color
import skimage.draw
import skimage.filters
import skimage.measure
import skimage.transform

from cockpit import events
from cockpit.util import userConfig, threads
from microAO.events import *

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
    # Project
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
        self.z_lookup = []
        self._position = 0
        self._compensation_poly = None

        # Load datapoints stored in the user config
        datapoints_init = userConfig.getValue("rf_datapoints")
        if datapoints_init:
            for z in datapoints_init:
                self.datapoints[z] = np.array(datapoints_init[z])
            self.update_calibration()

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
                    "remote-focus-zstack_"
                    f"zr={rf_stack.stage_position_rel:+.05f}um_"
                    f"za={rf_stack.stage_position_abs:+.05f}um.ome.tif"
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
            # Crop bead and save it
            bead = rf_stack.images[
                :,
                bead_roi[1] : bead_roi[1] + bead_roi[3],
                bead_roi[0] : bead_roi[0] + bead_roi[2]
            ]
            tifffile.imwrite(
                output_dir_path.joinpath(
                    f"bead_zr={rf_stack.stage_position_rel:+.05f}um_"
                    f"za={rf_stack.stage_position_abs:+.05f}um.ome.tif"
                ),
                bead,
                metadata = {
                    "axes": "ZYX",
                    "PhysicalSizeX": xy_pixelsize,  # um
                    "PhysicalSizeY": xy_pixelsize,  # um
                    "PhysicalSizeZ": defocus_step,  # um
                }
            )
            # Derive orthogonal views and save them
            projections = _orthogonal_projection(
                bead,
                _find_bead_centre(bead),
                xy_pixelsize / defocus_step
            )[1:]
            for projection_index, projection_type in enumerate(("zx", "zy")):
                tifffile.imwrite(
                    output_dir_path.joinpath(
                        f"proj-{projection_type}_"
                        f"zr={rf_stack.stage_position_rel:+.05f}um_"
                        f"za={rf_stack.stage_position_abs:+.05f}um.ome.tif"
                    ),
                    projections[projection_index],
                    metadata = {
                        "PhysicalSizeX": xy_pixelsize,  # um
                        "PhysicalSizeY": xy_pixelsize,  # um
                    }
                )
            # ----------------------------------------------------------------
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
                    projection_padded_labelled = skimage.color.gray2rgb(projection_padded)
                else:
                    # Calculate error term
                    image_offset_um = ((projection.shape[0] / 2) - centroid[0]) * xy_pixelsize
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
        # Fit a line to the corrected Z offsets and save the coefficients
        polynomials = []
        for index in range(len(projection_data)):
            x_data = [
                datum.stage_position_rel_corrected
                for datum in projection_data[index]
                if datum.stage_position_rel_corrected
            ]
            y_data = [
                datum.stage_position_rel
                for datum in projection_data[index]
                if datum.stage_position_rel_corrected
            ]
            if len(x_data) > 0:
                polynomials.append(
                    np.polynomial.polynomial.Polynomial.fit(x_data, y_data, 1)
                )
        if len(polynomials) > 0:
            self._compensation_poly = np.mean(polynomials)
            np.savetxt(
                output_dir_path.joinpath(
                    f"remote-focus_compensation-polynomial-coefficients.txt",
                ),
                self._compensation_poly.convert().coef
            )
        # Ask the GUI thread to update the status light
        events.publish(PUBSUB_RF_CALIB_CACT_PROJ)

    def zstack(self, zmin, zmax, zstepsize, camera, imager):
        zpositions = np.linspace(
            zmin,
            zmax,
            int((zmax - zmin) / zstepsize) + 1
        )

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

    def add_datapoint(self, z, modes):
        self.datapoints[z] = modes
        self.update_calibration()

    def remove_datapoint(self, z):
        if z not in self.datapoints:
            return
        del self.datapoints[z]
        if len(self.datapoints) < 2:
            # Not enough datapoints for calibration => reset position to 0 and
            # clear the remotez correction
            self._position = 0
            self._device.set_correction("remotez")
            self._device.toggle_correction("remotez", False)
            self._device.refresh_corrections()
            # Signal cockpit that the stage has moved
            events.publish(events.STAGE_MOVER, 2)
            events.publish(events.STAGE_STOPPED, self._device.RF_POSHAN_NAME)
        # Update calibration
        self.update_calibration()

    def update_calibration(self):
        # Clear the lookup
        self.z_lookup = []
        # Check if there are enough datapoints
        zs = sorted(self.datapoints.keys())
        if len(zs) < 2:
            return
        # Fit a line to the Zs for each mode
        modes = np.array([self.datapoints[z] for z in zs])
        for mode_index in range(modes.shape[1]):
            self.z_lookup.append(
                np.polynomial.Polynomial.fit(zs, modes[:, mode_index], 1)
            )
        # Apply new lookup
        self.set_z(self._position)

    def calc_shape(self, z):
        if len(self.z_lookup) < 2:
            raise Exception(
                "Failed to calculate wavefront shape for remote focusing "
                "because it has not been calibrated."
            )

        # Get current remotez correction
        original_corrections = self._device.get_corrections()

        # Compensate Z
        if self._compensation_poly:
            z = self._compensation_poly(z)

        modes = np.array([poly(z) for poly in self.z_lookup])
        self._device.set_correction("remotez", modes=modes)

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

    def set_z(self, z):
        if len(self.z_lookup) < 2:
            raise Exception(
                "Failed to change the remote focus plane because the remote "
                "focusing has not been calibrated."
            )

        corrections = self._device.get_corrections()

        modes = np.array([poly(z) for poly in self.z_lookup])
        self._device.set_correction("remotez", modes=modes)
        if "remotez" in corrections and corrections["remotez"]["enabled"]:
            self._device.refresh_corrections()

        self._position = z

    def get_z(self):
        return self._position
