from microAO.gui import *
from microAO.gui.modeControl import ModesControl
from microAO.gui.correctionFitting import CorrectionFittingFrame
from microAO.gui.sensorlessViewer import SensorlessResultsViewer
from microAO.gui.DMViewer import DMViewer
from microAO import cockpit_device
import microAO.events
import microAO.aoAlg

import cockpit.events
import cockpit.gui.device
import cockpit.gui.camera.window
from cockpit.util import logger, userConfig
from cockpit import depot
import cockpit.interfaces.stageMover

import microscope.devices

import wx
from wx.lib.floatcanvas.FloatCanvas import FloatCanvas
import wx.lib.floatcanvas.FCObjects as FCObjects

import matplotlib.pyplot
import matplotlib.ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import numpy as np

import typing
import pathlib
import json

import tifffile


def _np_grey_img_to_wx_image(np_img: np.ndarray) -> wx.Image:
    img_min = np.min(np_img)
    img_max = np.max(np_img)
    scaled_img = (np_img - img_min) / (img_max - img_min)

    uint8_img = (scaled_img * 255).astype("uint8")
    scaled_img_rgb = np.require(
        np.stack((uint8_img,) * 3, axis=-1), requirements="C"
    )

    wx_img = wx.Image(
        scaled_img_rgb.shape[0],
        scaled_img_rgb.shape[1],
        scaled_img_rgb,
    )
    return wx_img

def _computePowerSpectrum(interferogram):
    interferogram_ft = np.fft.fftshift(np.fft.fft2(interferogram))
    power_spectrum = np.log(abs(interferogram_ft))
    return power_spectrum

class _ROISelect(wx.Dialog):
    """Display a window that allows the user to select a circular area.

    This is a window for selecting the ROI for interferometry.
    """

    _INITIAL_IMAGE_HEIGHT = 512
    _ROI_MIN_RADIUS = 32

    def __init__(self, parent, image_or_path) -> None:
        super().__init__(
            parent,
            title="ROI selector",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
        )

        # What, if anything, is being dragged.
        # XXX: When we require Python 3.8, annotate better with
        # `typing.Literal[None, "xy", "r"]`
        self._dragging: typing.Optional[str] = None

        # Create a wx image instance
        if isinstance(image_or_path, np.ndarray):
            # Argument is image => use it directly
            self._img = _np_grey_img_to_wx_image(image_or_path)
        else:
            # Argument is path => load the first frame of the image
            frame = tifffile.imread(image_or_path, key=0)
            self._img = _np_grey_img_to_wx_image(frame)

        # Calculate image scale based on size of the canvas
        self._scale = self._INITIAL_IMAGE_HEIGHT / self._img.GetHeight()

        # Derive the initial ROI, in pixel units
        init_roi = userConfig.getValue("dm_circleParams")
        if init_roi is None:
            init_roi = (
                *[d // 2 for d in self._img.GetSize()],
                min(self._img.GetSize()) // 4,
            )
        init_roi = (
            init_roi[1] * self._scale,
            init_roi[0] * self._scale,
            init_roi[2] * self._scale,
        )

        # Create the canvas and draw the required objects
        # NOTE: World and screen coordinates have the same unit (pixel) but
        # different origins. The world origin is centre of the canvas, whereas
        # the screen origin is in the top left corner.
        self._canvas = FloatCanvas(
            self,
            size=wx.Size(
                self._INITIAL_IMAGE_HEIGHT,
                self._INITIAL_IMAGE_HEIGHT
            )
        )
        self._canvas_bitmap = self._canvas.AddObject(
            FCObjects.ScaledBitmap(
                self._img,
                (0, 0),
                self._INITIAL_IMAGE_HEIGHT,
                Position="cc"
            )
        )
        self._canvas_circle = self._canvas.AddObject(
            FCObjects.Circle(
                self._canvas.PixelToWorld(init_roi[:2]),
                init_roi[2] * 2,
                LineColor="cyan",
                LineWidth=2,
                InForeground=True
            )
        )
        self._canvas.Bind(wx.EVT_MOUSE_EVENTS, self._OnMouse)
        self._canvas.Bind(wx.EVT_SIZE, self._OnSize)

        # Create the standard buttons
        sizer_stdbuttons = wx.StdDialogButtonSizer()
        for button_id in (wx.ID_OK, wx.ID_CANCEL):
            button = wx.Button(self, button_id)
            sizer_stdbuttons.Add(button)
        sizer_stdbuttons.Realize()

        # Finalise layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self._canvas, 1, wx.SHAPED)
        sizer.Add(sizer_stdbuttons, 0, wx.ALL, 5)
        self.SetSizerAndFit(sizer)

    def GetROI(self):
        return [x / self._scale for x in self._circle_roi()]

    def _circle_roi(self):
        """Convert circle world parameters to ROI in screen coordinates."""
        roi_x, roi_y = self._canvas.WorldToPixel(self._canvas_circle.XY)
        roi_r = max(self._canvas_circle.WH)
        return (roi_x, roi_y, roi_r)

    def _MoveCircle(self, pos: wx.Point, r) -> None:
        """Set position and radius of circle with bounds checks."""
        x, y = pos
        _, _, _r = self._circle_roi()
        img_dims = [d * self._scale for d in self._img.GetSize()]
        # Calculate the radius
        r_bounded = r
        if r != _r:
            r_bounded = max(
                min(
                    img_dims[0] - pos[0],  # clip to right
                    img_dims[1] - pos[1],  # clip to bottom
                    pos[0],  # clip to left
                    pos[1],  # clip to top
                    r  # necessary for the outer max() to work
                ),
                self._ROI_MIN_RADIUS,
            )
        # Calculate XY
        xy_bounded = [None, None]
        for i in range(2):
            xy_bounded[i] = min(
                max(r_bounded, pos[i]),  # clip to left and top
                img_dims[i] - r_bounded  # clip to right and bottom
            )
        # Set circle parameters
        self._canvas_circle.SetPoint(
            self._canvas.PixelToWorld(xy_bounded)
        )
        self._canvas_circle.SetDiameter(2 * r_bounded)
        if any((xy_bounded[0] != pos[0], xy_bounded[1] != pos[1], r_bounded != r)):
            self._canvas_circle.SetColor("magenta")
        else:
            self._canvas_circle.SetColor("cyan")

    def _OnMouse(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        x, y, r = self._circle_roi()
        if event.LeftDClick():
            # Set circle centre
            self._MoveCircle(pos, r)
        elif event.Dragging():
            # Drag circle centre or radius
            drag_r = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
            if self._dragging is None:
                # determine what to drag
                if drag_r < 0.5 * r:
                    # closer to center
                    self._dragging = "xy"
                else:
                    # closer to edge
                    self._dragging = "r"
            elif self._dragging == "r":
                # Drag circle radius
                self._MoveCircle((x, y), drag_r)
            elif self._dragging == "xy":
                # Drag circle centre
                self._MoveCircle(pos, r)

        if not event.Dragging():
            # Stop dragging
            self._dragging = None
            self._canvas_circle.SetColor("cyan")

        self._canvas.Draw(Force=True)

    def _OnSize(self, event: wx.SizeEvent):
        size_canvas_new = event.GetSize()
        size_canvas_old = (
            self._canvas_bitmap.Width,
            self._canvas_bitmap.Height
        )
        # Calculate new scales
        self._scale = size_canvas_new[1] / self._img.GetHeight()
        circle_scale = size_canvas_new[1] / size_canvas_old[1]
        # Re-add the bitmap
        self._canvas.RemoveObject(self._canvas_bitmap)
        self._canvas_bitmap = self._canvas.AddObject(
            FCObjects.ScaledBitmap(
                self._img,
                (0, 0),
                size_canvas_new[1],
                Position="cc"
            )
        )
        # Scale the circle
        self._canvas_circle.SetPoint(
            [d * circle_scale for d in self._canvas_circle.XY]
        )
        self._canvas_circle.SetDiameter(
            self._canvas_circle.WH[0] * 2 * circle_scale
        )
        # Let the FloatCanvas handle the event too
        event.Skip()

class _PhaseViewer(wx.Frame):
    """This is a window for visualising a phase map."""

    _INITIAL_IMAGE_HEIGHT = 512
    _DEFAULT_CMAP = "rainbow"

    def __init__(
        self,
        parent,
        phase,
        phase_roi,
        phase_unwrapped,
        phase_power_spectrum,
        phase_unwrapped_MPTT_RMS_error,
        *args,
        **kwargs
    ):
        super().__init__(parent, title="Phase View")
        self.Bind(wx.EVT_CLOSE, self._OnClose)
        self._panel = wx.Panel(self, *args, **kwargs)

        # Store the important data
        self._data = {
            "phase": phase,
            "phase_roi": phase_roi,
            "phase_unwrapped": phase_unwrapped,
            "phase_power_spectrum": phase_power_spectrum,
            "phase_unwrapped_MPTT_RMS_error": phase_unwrapped_MPTT_RMS_error,
        }

        fig, self._axes = matplotlib.pyplot.subplots()
        self._axes_image = self._axes.imshow(
            phase_unwrapped, cmap=self._DEFAULT_CMAP
        )
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._axes.set_frame_on(False)
        self._cbar = matplotlib.pyplot.colorbar(
            self._axes_image, ax=self._axes
        )
        fig.tight_layout()
        self._canvas = FigureCanvas(self._panel, wx.ID_ANY, fig)

        # Create a choice widget for selection of colormap
        self._cmap_choice = wx.Choice(
            self._panel, choices=matplotlib.pyplot.colormaps()
        )
        self._cmap_choice.SetSelection(
            matplotlib.pyplot.colormaps().index(self._DEFAULT_CMAP)
        )
        self._cmap_choice.Bind(wx.EVT_CHOICE, self._OnCmapChoice)

        button_fourier = wx.ToggleButton(self._panel, label="Show Fourier")
        button_fourier.Bind(wx.EVT_TOGGLEBUTTON, self._OnToggleFourier)

        button_save = wx.Button(self._panel, label="Save data")
        button_save.Bind(wx.EVT_BUTTON, self._OnButtonSave)

        self._rms_txt = wx.StaticText(
            self._panel,
            label="RMS error without piston, tip, and tilt modes: %.05f"
            % (phase_unwrapped_MPTT_RMS_error),
        )

        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(self._canvas, 1, wx.SHAPED)

        bottom_sizer_1 = wx.BoxSizer(wx.HORIZONTAL)
        bottom_sizer_1.Add(button_fourier, wx.SizerFlags().Center().Border())
        bottom_sizer_1.Add(button_save, wx.SizerFlags().Center().Border())
        bottom_sizer_1.Add(
            self._cmap_choice, wx.SizerFlags().Center().Border()
        )
        panel_sizer.Add(bottom_sizer_1, 0, wx.EXPAND)

        bottom_sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        bottom_sizer_2.Add(self._rms_txt, wx.SizerFlags().Center().Border())
        panel_sizer.Add(bottom_sizer_2, 0, wx.EXPAND)

        self._panel.SetSizer(panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self._panel, 1, wx.EXPAND)
        self.SetSizerAndFit(frame_sizer)

    def _OnClose(self, event: wx.CloseEvent) -> None:
        # Make sure the figure has been closed
        matplotlib.pyplot.close(self._axes.get_figure())
        # Allow other handlers to process the event
        event.Skip(True)

    def _OnToggleFourier(self, event: wx.CommandEvent) -> None:
        cmap = self._cmap_choice.GetString(self._cmap_choice.GetSelection())
        if event.GetEventObject().GetValue():
            # Show fourier spectrum
            self._axes_image = self._axes.imshow(
                self._data["phase_power_spectrum"], cmap=cmap
            )
        else:
            # Show phase map
            self._axes_image = self._axes.imshow(
                self._data["phase_unwrapped"], cmap=cmap
            )
        self._cbar.update_normal(self._axes_image)
        self._cbar.update_ticks()
        self._canvas.draw()

    def _OnButtonSave(self, event: wx.CommandEvent) -> None:
        file_path = ""
        with wx.FileDialog(
            self,
            "Save phase data",
            wildcard="Numpy binary file (*.npy)|*.npy",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as file_dialog:
            if file_dialog.ShowModal() != wx.ID_OK:
                return
            file_path = file_dialog.GetPath()
        np.save(file_path, self._data)

    def _OnCmapChoice(self, event: wx.CommandEvent):
        self._axes_image.set_cmap(event.GetString())
        self._cbar.update_normal(self._axes_image)
        self._canvas.draw()

class _CharacterisationAssayViewer(wx.Frame):

    _ASSAY_COLORMAP = "seismic"

    def __init__(self, parent, characterisation_assay):
        super().__init__(parent, title="Characterisation Asssay")

        self._assay = characterisation_assay

        root_panel = wx.Panel(self)

        figure = Figure(constrained_layout=True)

        assay_max = np.max(np.abs(characterisation_assay))
        img_ax = figure.add_subplot(1, 2, 1)
        axes_image = img_ax.matshow(
            characterisation_assay,
            cmap=self._ASSAY_COLORMAP,
            vmin=-assay_max,
            vmax=assay_max
        )
        figure.colorbar(axes_image)
        img_ax.set(xticks=[], yticks=[])
        img_ax.set_xlabel("Modes assessed")
        img_ax.set_ylabel("Modes measured")

        diag_ax = figure.add_subplot(1, 2, 2)
        assay_diag = np.diag(characterisation_assay)
        diag_ax.axhline()
        diag_ax.plot(np.arange(assay_diag.shape[0]) + 2, assay_diag, "C0.")
        diag_ax.axhspan(-0.25, 0.25, color="C0", alpha=0.2)
        diag_ax.set_ylim(-assay_max, assay_max)
        diag_ax.set_xlabel("Mode")
        diag_ax.set_ylabel("Error")
        diag_ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        diag_ax.grid(which="major", axis="x", color="#666666")
        diag_ax.grid(which="minor", axis="x", color="#444444", linestyle=":")

        canvas = FigureCanvas(root_panel, wx.ID_ANY, figure)

        info_txt = wx.StaticText(
            root_panel,
            label=(
                "Diagonal mean = %.5f and RMS = %.5f"
                % (np.mean(assay_diag), np.sqrt(np.mean(assay_diag ** 2)))
            ),
        )

        button_save = wx.Button(root_panel, label="Save assay")
        button_save.Bind(wx.EVT_BUTTON, self._OnButtonSave)

        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.Add(button_save, wx.SizerFlags().Center().Border())
        top_sizer.Add(info_txt, wx.SizerFlags().Center().Border())
        panel_sizer.Add(top_sizer)
        panel_sizer.Add(canvas, wx.SizerFlags(1).Expand())
        root_panel.SetSizer(panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root_panel, wx.SizerFlags(1).Expand())
        self.SetSizerAndFit(frame_sizer)

    def _OnButtonSave(self, event: wx.CommandEvent) -> None:
        file_path = ""
        with wx.FileDialog(
            self,
            "Save characterisation assay",
            wildcard="Numpy binary file (*.npy)|*.npy",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as file_dialog:
            if file_dialog.ShowModal() != wx.ID_OK:
                return
            file_path = file_dialog.GetPath()
        np.save(file_path, self._assay)

class _SensorlessParametersDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(
            parent,
            title="Sensorless parameters selection",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
        )

        self._device = parent._device

        panel = wx.Panel(self)

        # Create the configurable controls
        params = self._device.sensorless_params
        self._textctrl_reps = wx.TextCtrl(panel, value=str(params["num_reps"]))
        self._textctrl_na = wx.TextCtrl(panel, value=str(params["NA"]))
        self._textctrl_wavelength = wx.TextCtrl(
            panel,
            value=str(params["wavelength"])
        )
        self._textctrl_ranges = wx.TextCtrl(
            panel,
            value=self._params2text(params),
            size=wx.Size(400, 200),
            style=wx.TE_MULTILINE,
        )
        self._checkbox_dp_save = wx.CheckBox(panel)
        self._checkbox_dp_save.SetValue(params["save_as_datapoint"])

        # Configure the font of the scan ranges' text control
        self._textctrl_ranges.SetFont(
            wx.Font(
                14,
                wx.FONTFAMILY_MODERN,
                wx.FONTSTYLE_NORMAL,
                wx.FONTWEIGHT_NORMAL
            )
        )

        # Define the grid and the text widgets
        widgets_data = (
            (
                wx.StaticText(panel, label="Number of repeats:"),
                wx.GBPosition(0, 0),
                wx.GBSpan(1, 1),
                wx.ALL,
                5
            ),
            (
                self._textctrl_reps,
                wx.GBPosition(0, 1),
                wx.GBSpan(1, 1),
                wx.ALL,
                5
            ),
            (
                wx.StaticText(panel, label="Numerical aperture:"),
                wx.GBPosition(1, 0),
                wx.GBSpan(1, 1),
                wx.ALL,
                5
            ),
            (
                self._textctrl_na,
                wx.GBPosition(1, 1),
                wx.GBSpan(1, 1),
                wx.ALL,
                5
            ),
            (
                wx.StaticText(panel, label="Excitation wavelength:"),
                wx.GBPosition(2, 0),
                wx.GBSpan(1, 1),
                wx.ALL,
                5
            ),
            (
                self._textctrl_wavelength,
                wx.GBPosition(2, 1),
                wx.GBSpan(1, 1),
                wx.ALL,
                5
            ),
            (
                wx.StaticText(panel, label="Save results as datapoint?"),
                wx.GBPosition(3, 0),
                wx.GBSpan(1, 1),
                wx.ALL,
                5
            ),
            (
                self._checkbox_dp_save,
                wx.GBPosition(3, 1),
                wx.GBSpan(1, 1),
                wx.ALL,
                5
            ),
            (
                wx.StaticText(
                    panel,
                    label=(
                        "Define the scanning ranges in the text field below.\n"
                        "Columns should be separated by four or more SPACE "
                        "characters.\nThe columns are:\n\tModes (Noll indices)"
                        "\n\tScan range min amplitude\n\tScan range max "
                        "amplitude\n\tScan range steps"
                    )
                ),
                wx.GBPosition(4, 0),
                wx.GBSpan(1, 2),
                wx.ALL,
                5
            ),
            (
                self._textctrl_ranges,
                wx.GBPosition(5, 0),
                wx.GBSpan(1, 2),
                wx.ALL | wx.EXPAND,
                5
            ),
        )

        # Construct the grid
        panel_sizer = wx.GridBagSizer(vgap=0, hgap=0)
        panel_sizer.SetCols(2)
        panel_sizer.AddGrowableCol(1)
        for widget_data in widgets_data:
            panel_sizer.Add(*widget_data)
        panel.SetSizer(panel_sizer)

        # Create the standard buttons
        sizer_stdbuttons = wx.StdDialogButtonSizer()
        button_ok = wx.Button(self, wx.ID_OK)
        button_ok.Bind(wx.EVT_BUTTON, self._on_ok)
        sizer_stdbuttons.Add(button_ok)
        button_cancel = wx.Button(self, wx.ID_CANCEL)
        sizer_stdbuttons.Add(button_cancel)
        sizer_stdbuttons.Realize()

        # Finalise layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(panel, 1, wx.EXPAND)
        sizer.Add(sizer_stdbuttons, 0, wx.ALL, 5)
        self.SetSizerAndFit(sizer)

    def _params2text(self, params):
        mode_sets = []
        # Split into sets of sequential mode entries with same offsets
        for index in range(0, len(params["modes"])):
            if len(mode_sets) > 0 and np.array_equal(
                params["modes"][index - 1].offsets,
                params["modes"][index].offsets,
            ):
                mode_sets[-1][0].append(params["modes"][index].index_noll)
            else:
                mode_sets.append(
                    [
                        # Modes in the set
                        [params["modes"][index].index_noll],
                        # Scanning parameters of the set
                        (
                            params["modes"][index].offsets.min(),
                            params["modes"][index].offsets.max(),
                            params["modes"][index].offsets.shape[0],
                        ),
                    ]
                )
        # Split modes further into range sequences, i.e. list of lists
        for mode_set in mode_sets:
            mode_ranges = [[mode_set[0][0]]]
            for mode in mode_set[0][1:]:
                if mode == mode_ranges[-1][-1] + 1:
                    mode_ranges[-1].append(mode)
                else:
                    mode_ranges.append([mode])
            mode_set[0] = mode_ranges
        # Convert the modes to a string
        for mode_set in mode_sets:
            # Build the mode string
            mode_string = ", ".join(
                [
                    f"{r[0]}" if len(r) == 1 else f"{r[0]}-{r[-1]}"
                    for r in mode_set[0]
                ]
            )
            mode_set[0] = mode_string
        # Find the longest mode string
        max_mod_string = max([len(mode_set[0]) for mode_set in mode_sets])
        # Convert all sets into a multiline formatted string
        sets_string = ""
        col_sep = "    "
        for mode_set in mode_sets:
            range_string = col_sep.join([str(x) for x in mode_set[1]])
            # Append a new row
            sets_string += (
                f"{mode_set[0]: <{max_mod_string}}{col_sep}{range_string}\n"
            )
        return sets_string

    def _on_ok(self, event: wx.CommandEvent):
        # Parse the simple single-line widgets first
        widgets_data = [
            # widget, label, value, parsing function
            [self._textctrl_reps, "repeats", 0, int],
            [self._textctrl_na, "numerical aperture", 0, float],
            [self._textctrl_wavelength, "wavelength", 0, int],
        ]
        for widget_data in widgets_data:
            try:
                widget_data[2] = widget_data[3](widget_data[0].GetValue())
            except ValueError:
                with wx.MessageDialog(
                    self,
                    f"Error! Cannot convert {widget_data[1]} to a number of "
                    f"type {widget_data[3].__name__}!",
                    "Parsing error",
                    wx.OK | wx.ICON_ERROR,
                ) as dlg:
                    dlg.ShowModal()
                return
        # Do widget-specific parsing for each of the single-line widgets
        if widgets_data[0][2] < 1:
            with wx.MessageDialog(
                self,
                f"Error! Repeats must be 1 or greater!",
                "Parsing error",
                wx.OK | wx.ICON_ERROR,
            ) as dlg:
                dlg.ShowModal()
            return
        if widgets_data[1][2] <= 0.0:
            with wx.MessageDialog(
                self,
                f"Error! Numerical aperture must be greater than 0.0!",
                "Parsing error",
                wx.OK | wx.ICON_ERROR,
            ) as dlg:
                dlg.ShowModal()
            return
        if widgets_data[2][2] <= 0:
            with wx.MessageDialog(
                self,
                f"Error! Wavelength must be greater than 0!",
                "Parsing error",
                wx.OK | wx.ICON_ERROR,
            ) as dlg:
                dlg.ShowModal()
            return
        # Parse the multi-line widget
        mode_params = []
        lines = [
            line.strip()
            for line in self._textctrl_ranges.GetValue().splitlines()
        ]
        lines = [line for line in lines if line]
        if len(lines) == 0:
            with wx.MessageDialog(
                self,
                f"Error! At least one scanning range needs to be defined!",
                "Parsing error",
                wx.OK | wx.ICON_ERROR,
            ) as dlg:
                dlg.ShowModal()
            return
        for line_index, line in enumerate(lines):
            columns = [column.strip() for column in line.split("    ")]
            columns = [column for column in columns if column]
            # Parse number of columns
            if len(columns) != 4:
                with wx.MessageDialog(
                    self,
                    f"Error! Improper formatting on line {line_index + 1} of "
                    f"scan ranges! Expected 4 column but got {len(columns)} "
                    "instead.",
                    "Parsing error",
                    wx.OK | wx.ICON_ERROR,
                ) as dlg:
                    dlg.ShowModal()
                return
            # Parse modes
            modes = []
            mode_ranges = [x.strip() for x in columns[0].split(",")]
            mode_ranges = [x for x in mode_ranges if x]
            for mode_range in mode_ranges:
                if "-" in mode_range:
                    bounds = [x.strip() for x in mode_range.split("-")]
                    if len(bounds) != 2:
                        with wx.MessageDialog(
                            self,
                            "Error! Improper formatting of modes on line "
                            f"{line_index + 1} of scan ranges!",
                            "Parsing error",
                            wx.OK | wx.ICON_ERROR,
                        ) as dlg:
                            dlg.ShowModal()
                        return
                    try:
                        range_start = int(bounds[0])
                        range_end = int(bounds[1]) + 1
                        modes.extend(list(range(range_start, range_end)))
                    except TypeError:
                        with wx.MessageDialog(
                            self,
                            "Error! Improper formatting of modes on line "
                            f"{line_index + 1} of scan ranges! Modes need to "
                            "be integers.",
                            "Parsing error",
                            wx.OK | wx.ICON_ERROR,
                        ) as dlg:
                            dlg.ShowModal()
                        return
                else:
                    try:
                        modes.append(int(mode_range))
                    except ValueError:
                        with wx.MessageDialog(
                            self,
                            "Error! Improper formatting of modes on line "
                            f"{line_index + 1} of scan ranges! Modes need to "
                            "be integers.",
                            "Parsing error",
                            wx.OK | wx.ICON_ERROR,
                        ) as dlg:
                            dlg.ShowModal()
                        return
            if min(modes) <= 0:
                with wx.MessageDialog(
                    self,
                    "Error! Improper formatting of modes on line "
                    f"{line_index + 1} of scan ranges! Modes need to be "
                    "specified as Noll indices, therefore integers greater "
                    "than 0.",
                    "Parsing error",
                    wx.OK | wx.ICON_ERROR,
                ) as dlg:
                    dlg.ShowModal()
                return
            # Parse range bounds
            for index, label in ((1, "range min"), (2, "range max")):
                try:
                    columns[index] = float(columns[index])
                except ValueError:
                    with wx.MessageDialog(
                        self,
                        f"Error! Cannot convert {label} on line "
                        f"{line_index + 1} to a floating-point number!",
                        "Parsing error",
                        wx.OK | wx.ICON_ERROR,
                    ) as dlg:
                        dlg.ShowModal()
                    return
            # Parse steps
            try:
                columns[3] = int(columns[3])
            except ValueError:
                with wx.MessageDialog(
                    self,
                    f"Error! Cannot convert steps on line {line_index + 1} to "
                    "an integer number!",
                    "Parsing error",
                    wx.OK | wx.ICON_ERROR,
                ) as dlg:
                    dlg.ShowModal()
                return
            # Create the mode parameters
            for mode in modes:
                mode_params.append(
                    cockpit_device.SensorlessParamsMode(
                        mode, np.linspace(columns[1], columns[2], columns[3])
                    )
                )
        # Update the sensorless AO parameters
        self._device.sensorless_params["num_reps"] = widgets_data[0][2]
        self._device.sensorless_params["modes"] = mode_params
        self._device.sensorless_params["NA"] = widgets_data[1][2]
        self._device.sensorless_params["wavelength"] = widgets_data[2][2]
        self._device.sensorless_params["save_as_datapoint"] = (
            self._checkbox_dp_save.GetValue()
        )
        # Propagate event
        event.Skip()

class MicroscopeAOCompositeDevicePanel(wx.Panel):
    def __init__(self, parent, device):
        super().__init__(parent)
        self.SetDoubleBuffered(True)

        # Store reference to AO device
        self._device = device

        # Dict to store reference to child component ids
        self._components = {
            "modes_control": None,
            "sensorless_results": None
        }

        # Store previous trigger choice values
        self.ao_trigger = self._device.proxy.get_trigger()

        # Create tabbed interface
        tabs = wx.Notebook(self, size=(350,-1))
        panel_calibration = wx.Panel(tabs)
        panel_AO = wx.Panel(tabs)
        panel_control = wx.Panel(tabs)
        panel_setup = wx.Panel(tabs)

        # Button to load control matrix
        loadControlMatrixButton = wx.Button(panel_setup, label="Load control matrix")
        loadControlMatrixButton.Bind(wx.EVT_BUTTON, self.OnLoadControlMatrix)
        
        # Button to save control matrix
        saveControlMatrixButton = wx.Button(panel_setup, label="Save control matrix")
        saveControlMatrixButton.Bind(wx.EVT_BUTTON, self.OnSaveControlMatrix)

        # Button to load flat
        loadFlatButton = wx.Button(panel_setup, label="Load flat")
        loadFlatButton.Bind(wx.EVT_BUTTON, self.OnLoadFlat)
        
        # Button to save flat
        saveFlatButton = wx.Button(panel_setup, label="Save flat")
        saveFlatButton.Bind(wx.EVT_BUTTON, self.OnSaveFlat)

        # Choices to select adaptive element's trigger type and mode
        trigger_choices = [
            [member.name for member in list(enumeration)]
            for enumeration in (
                microscope.devices.TriggerType,
                microscope.devices.TriggerMode,
            )
        ]
        triggerTypeSizer = wx.BoxSizer(wx.HORIZONTAL)
        triggerModeSizer = wx.BoxSizer(wx.HORIZONTAL)
        triggerTypeLabel = wx.StaticText(panel_setup, label="AE trigger type:")
        triggerTypeChoice = wx.Choice(panel_setup, choices=trigger_choices[0])
        triggerTypeChoice.SetSelection(
            trigger_choices[0].index(self.ao_trigger[0].name)
        )
        triggerTypeChoice.Bind(wx.EVT_CHOICE, self.OnTriggerTypeChoice)
        triggerTypeSizer.Add(triggerTypeLabel, 1, wx.EXPAND | wx.RIGHT, 5)
        triggerTypeSizer.Add(triggerTypeChoice, 1, wx.EXPAND)
        triggerModeLabel = wx.StaticText(panel_setup, label="AE trigger mode:")
        triggerModeChoice = wx.Choice(panel_setup, choices=trigger_choices[1])
        triggerModeChoice.SetSelection(
            trigger_choices[1].index(self.ao_trigger[1].name)
        )
        triggerModeChoice.Bind(wx.EVT_CHOICE, self.OnTriggerModeChoice)
        triggerModeSizer.Add(triggerModeLabel, 1, wx.EXPAND | wx.RIGHT, 5)
        triggerModeSizer.Add(triggerModeChoice, 1, wx.EXPAND)

        # Button to load applied aberration
        loadModesButton = wx.Button(panel_control, label="Load modes")
        loadModesButton.Bind(wx.EVT_BUTTON, self.OnLoadModes)
        
        # Button to save applied aberration
        saveModesButton = wx.Button(panel_control, label="Save modes")
        saveModesButton.Bind(wx.EVT_BUTTON, self.OnSaveModes)

        # Button to load actuator values
        loadActuatorsButton = wx.Button(panel_control, label="Load actuators")
        loadActuatorsButton.Bind(wx.EVT_BUTTON, self.OnLoadActuatorValues)
        
        # Button to save actuator values
        saveActuatorsButton = wx.Button(panel_control, label="Save actuators")
        saveActuatorsButton.Bind(wx.EVT_BUTTON, self.OnSaveActuatorValues)

        # Button to set current actuator values as system flat
        setCurrentAsFlatButton = wx.Button(panel_control, label="Set current as flat")
        setCurrentAsFlatButton.Bind(wx.EVT_BUTTON, self.OnSetCurrentAsFlat)

        # Visualise current interferometric phase
        visPhaseButton = wx.Button(panel_calibration, label="Visualise Phase")
        visPhaseButton.Bind(wx.EVT_BUTTON, self.OnVisualisePhase)

        # Exercise the adaptive element
        exerciseAdaptiveElementButton = wx.Button(panel_calibration, label="Exercise adaptive element")
        exerciseAdaptiveElementButton.Bind(wx.EVT_BUTTON, self.OnExercise)

        # Button to set calibration parameters
        calibrationParametersButton = wx.Button(panel_calibration, label="Calibration parameters")
        calibrationParametersButton.Bind(wx.EVT_BUTTON, self.OnCalibrationParameters)

        # Button to get calibration data
        calibrationGetDataButton = wx.Button(panel_calibration, label="Get calibration data")
        calibrationGetDataButton.Bind(wx.EVT_BUTTON, self.OnCalibrationData)

        # Button to calculate control matrix
        calibrationCalcButton = wx.Button(panel_calibration, label="Calculate control matrix")
        calibrationCalcButton.Bind(wx.EVT_BUTTON, self.OnCalibrationCalc)

        # Button to characterise DM
        characteriseButton = wx.Button(panel_calibration, label="Characterise")
        characteriseButton.Bind(wx.EVT_BUTTON, self.OnCharacterise)

        # Button to set system flat calculation parameters
        sysFlatParametersButton = wx.Button(panel_calibration, label="Set system flat parameters")
        sysFlatParametersButton.Bind(wx.EVT_BUTTON, self.OnSetSystemFlatCalculationParameters)
        
        # Button to flatten mirror and apply as system flat
        sysFlatCalcButton = wx.Button(panel_calibration, label="Generate system flat")
        sysFlatCalcButton.Bind(wx.EVT_BUTTON, self.OnCalcSystemFlat)

        # Reset the DM actuators
        resetButton = wx.Button(panel_AO, label="Reset DM")
        resetButton.Bind(wx.EVT_BUTTON, self.OnResetDM)

        # Apply the actuator values correcting the system aberrations
        applySysFlat = wx.Button(panel_AO, label="Apply system flat")
        applySysFlat.Bind(wx.EVT_BUTTON, self.OnSystemFlat)

        # Button to set metric
        metricSelectButton = wx.Button(panel_AO, label="Set sensorless metric")
        metricSelectButton.Bind(wx.EVT_BUTTON, self.OnSetMetric)

        # Button to set sensorless correction parameters
        sensorlessParametersButton = wx.Button(panel_AO, label="Set sensorless parameters")
        sensorlessParametersButton.Bind(wx.EVT_BUTTON, self.OnSetSensorlessParameters)

        # Button to perform sensorless correction
        sensorlessAOButton = wx.Button(panel_AO, label="Sensorless AO")
        sensorlessAOButton.Bind(wx.EVT_BUTTON, self.OnSensorlessAO)

        # Button to manually apply aberration
        manualAberrationButton = wx.Button(panel_control, label="Manual")
        manualAberrationButton.Bind(wx.EVT_BUTTON, self.OnManualAberration)

        # Button to view DM pattern
        DMViewButton = wx.Button(panel_control, label="DM view")
        DMViewButton.Bind(wx.EVT_BUTTON, self.OnDMViewer)

        # Button to show remote focus dialog
        correctionFittingButton = wx.Button(panel_AO, label="Correction fitting")
        correctionFittingButton.Bind(wx.EVT_BUTTON, self.OnCorrectionFitting)

        panel_flags = wx.SizerFlags(0).Expand().Border(wx.LEFT|wx.RIGHT, 50)

        sizer_panel_setup = wx.BoxSizer(wx.VERTICAL)
        for btn in [
            loadControlMatrixButton,
            saveControlMatrixButton,
            loadFlatButton,
            saveFlatButton,
            triggerTypeSizer,
            triggerModeSizer
        ]:
            sizer_panel_setup.Add(btn, panel_flags)

        sizer_calibration = wx.BoxSizer(wx.VERTICAL)
        for btn in [
            visPhaseButton,
            exerciseAdaptiveElementButton,
            calibrationParametersButton,
            calibrationGetDataButton,
            calibrationCalcButton,
            characteriseButton,
            sysFlatParametersButton,
            sysFlatCalcButton
        ]:
            sizer_calibration.Add(btn, panel_flags)

        sizer_AO = wx.BoxSizer(wx.VERTICAL)
        for btn in [
            resetButton,
            applySysFlat,
            metricSelectButton,
            sensorlessParametersButton,
            sensorlessAOButton,
            correctionFittingButton
        ]:
            sizer_AO.Add(btn, panel_flags)

        sizer_control = wx.BoxSizer(wx.VERTICAL)
        for widget in [
            manualAberrationButton,
            DMViewButton,
            loadModesButton,
            saveModesButton,
            loadActuatorsButton,
            saveActuatorsButton,
            setCurrentAsFlatButton
        ]:

            sizer_control.Add(widget, panel_flags)

        panel_calibration.SetSizer(sizer_calibration)
        panel_AO.SetSizer(sizer_AO)
        panel_control.SetSizer(sizer_control)
        panel_setup.SetSizer(sizer_panel_setup)

        # Add pages to tabs
        tabs.AddPage(panel_setup,"Setup") 
        tabs.AddPage(panel_calibration,"Calibration") 
        tabs.AddPage(panel_AO,"AO")
        tabs.AddPage(panel_control,"Control") 

        tabs.Layout()

        # Corrections checkbox list
        self.checklist_corrections = wx.CheckListBox(
            self,
            id=wx.ID_ANY,
            size=wx.Size(150,-1),
            choices=sorted(self._device.get_corrections().keys())
        )
        self.checklist_corrections.Bind(
            wx.EVT_CHECKLISTBOX,
            self.OnCorrectionState
        )
        cockpit.events.subscribe(
            microAO.events.PUBUSB_CHANGED_CORRECTION,
            self.OnCorrectionChange
        )
        checklist_sizer = wx.BoxSizer(wx.VERTICAL)
        checklist_sizer.Add(self.checklist_corrections, 1, wx.EXPAND)

        sizer_main = wx.BoxSizer(wx.HORIZONTAL)
        sizer_main.Add(tabs, wx.SizerFlags(1).Expand())
        sizer_main.Add(checklist_sizer, 0, wx.EXPAND)
        self.SetSizer(sizer_main)

    def OnVisualisePhase(self, _: wx.CommandEvent) -> None:
        # Select the camera whose window contains the interferogram
        camera = self.getCamera()
        if camera is None:
            raise Exception(
                "Failed to visualise phase because no active cameras were "
                "found."
            )
        # Get the image data
        phase = cockpit.gui.camera.window.getImageForCamera(camera)
        if phase is None:
            raise Exception(
                f"The camera view for camera '{camera.name}' contained no "
                "image. Please capture an interferogram first."
            )
        # Select the ROI
        with _ROISelect(self, phase) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                roi = dlg.GetROI()
                userConfig.setValue(
                    "dm_circleParams",
                    (roi[1], roi[0], roi[2])
                )
            else:
                raise Exception(
                    "ROI selection cancelled. Aborting phase visualisation..."
                )
        # Update device ROI and filter
        self._device.updateROI()
        self._device.checkFourierFilter()
        # Unwrap the phase
        phase_unwrapped = self._device.unwrap_phase(phase)
        # Get the RMS error of the unwrapped phase without the Piston, Tip, and
        # Tilt modes
        phase_unwrapped_MPTT_RMS_error = (
            self._device.aoAlg.calc_phase_error_RMS(phase_unwrapped)
        )
        # Calculate the power spectrum
        phase_power_spectrum = _computePowerSpectrum(phase)
        # View the phase
        frame = _PhaseViewer(
            self,
            phase,
            (roi[1], roi[0], roi[2]),
            phase_unwrapped,
            phase_power_spectrum,
            phase_unwrapped_MPTT_RMS_error
        )
        frame.Show()

    def OnCalibrationParameters(self, event: wx.CommandEvent) -> None:
        params = self._device.calibration_params

        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Set calibration parameters",
            [
                "Minimum poking magnitude",
                "Maximum poking magnitude",
                "Number of poking steps per actuator"
            ],
            (
                params["poke_min"],
                params["poke_max"],
                params["poke_steps"],
            ),
        )
        params["poke_min"] = float(inputs[0])
        params["poke_max"] = float(inputs[1])
        params["poke_steps"] = int(inputs[2])

    def OnCalibrationData(self, event: wx.CommandEvent) -> None:
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
        if isinstance(self._device._calibration_data["output_filename"], pathlib.Path):
            default_directory = self._device._calibration_data["output_filename"].parent
        with wx.FileDialog(
            self,
            message="Save calibration image stack",
            defaultDir=default_directory,
            wildcard="TIFF images (*.tif; *.tiff)|*.tif;*.tiff",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as file_dialog:
            if file_dialog.ShowModal() != wx.ID_OK:
                return
            self._device._calibration_data["output_filename"] = pathlib.Path(
                file_dialog.GetPath()
            )
        # Start the calibration process
        self._device.calibrationGetData(camera.name)

    def OnCalibrationCalc(self, event: wx.CommandEvent) -> None:
        # Navigate to the image file
        file_path_image = None
        with wx.FileDialog(
            self,
            message="Load calibration image stack",
            wildcard="TIFF images (*.tif; *.tiff)|*.tif;*.tiff",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as file_dialog:
            if file_dialog.ShowModal() != wx.ID_OK:
                return
            file_path_image = pathlib.Path(file_dialog.GetPath())

        # Check if an accompanying JSON file exists
        file_path_json = file_path_image.with_name(
            file_path_image.stem + ".json"
        )
        if not file_path_json.exists():
            logger.log.error(
                "Couldn't find accompanying JSON file for image file "
                f"{str(file_path_image)}."
            )

        # Load actuator values
        actuator_values = []
        with open(file_path_json, "r", encoding="utf-8") as fi:
            actuator_values = np.array(json.load(fi))

        # Define ROI
        with _ROISelect(self, file_path_image) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                roi = dlg.GetROI()
                userConfig.setValue(
                    "dm_circleParams",
                    (roi[1], roi[0], roi[2])
                )
            else:
                logger.log.error(
                    "ROI selection cancelled. Aborting calibration process..."
                )
                return

        # Calculate control matrix
        self._device.calculateControlMatrix(
            actuator_values,
            file_path_image
        )

    def OnCharacterise(self, event: wx.CommandEvent) -> None:
        del event
        assay = self._device.characterise()
        # Show characterisation assay, excluding piston.
        frame = _CharacterisationAssayViewer(self, assay[1:, 1:])
        frame.Show()

    def OnCalcSystemFlat(self, event: wx.CommandEvent) -> None:
        del event

        # Select the interferometer camera and the imager
        camera = self.getCamera()
        if camera is None:
            logger.log.error(
                "Failed to select active cameras for flattening the phase."
            )
            return
        imager = self.getImager()
        if imager is None:
            logger.log.error(
                "Failed to select an imager for flattening the phase."
            )
            return

        # Obtain the system flat
        self._device.sysFlatCalc(camera, imager)

    def OnResetDM(self, event: wx.CommandEvent) -> None:
        del event
        self._device.reset()

    def OnSystemFlat(self, event: wx.CommandEvent) -> None:
        del event
        self._device.applySysFlat()

    def OnSensorlessAO(self, event: wx.CommandEvent) -> None:
        # Perform sensorless AO but if there is more than one camera
        # available display a menu letting the user choose a camera.
        del event

        action = self._device.correctSensorlessSetup

        camera = self.getCamera()

        if camera is None:
            return

        # Create results viewer
        try:
            window = self.FindWindowById(self._components["sensorless_results"])
        except:
            window = None

        if window is None:
            window = SensorlessResultsViewer(None)
            self._components["sensorless_results"] = window.GetId()

        window.Show()

        # Check if a datapoint exists for this position
        datapoint_z = None
        if self._device.sensorless_params["save_as_datapoint"]:
            datapoint_z = cockpit.interfaces.stageMover.getPosition()[2]
            if datapoint_z in self._device.corrfit_dp_get()["sensorless"]:
                # Wrong selection => warn but do nothing
                with wx.MessageDialog(
                    self,
                    f"Datapoint already exists for position {datapoint_z}. "
                    "Overwrite?",
                    "Warning",
                    wx.YES_NO | wx.ICON_WARNING
                ) as dlg:
                    if dlg.ShowModal() != wx.OK:
                        window.Show(False)
                        raise Exception(
                            "Aborting sensorless AO because datapoint already "
                            f"exists for position {datapoint_z}..."
                        )

        # Start sensorless AO
        action(camera, datapoint_z)

    def OnManualAberration(self, event: wx.CommandEvent) -> None:
        # Try to find modes window
        try:
            window = self.FindWindowById(self._components["modes_control"])
        except:
            window = None

        # If not found, create new window and save reference to its id
        if window is None:
            window = ModesControl(self, self._device)    
            self._components["modes_control"] = window.GetId()

        # Show window and bring to front
        window.Show()
        window.Raise()

    def HandleSensorlessResults(self, e):
        print('e', e)      


    def OnLoadControlMatrix(self, event: wx.CommandEvent) -> None:
        del event

        # Prompt for file to load
        with wx.FileDialog(self, "Load calibration", wildcard="Data (*.txt)|*.txt", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()

        # Load calibration matrix from file, check format, and set on device
        try:
            control_matrix = np.loadtxt(fpath)
            assert (control_matrix.ndim == 2 and control_matrix.shape[0] == self._device.no_actuators)
            self._device.proxy.set_controlMatrix(control_matrix)

            # Set control matrix in cockpit config
            userConfig.setValue(
                "dm_controlMatrix", np.ndarray.tolist(control_matrix)
            )

            # Log
            logger.log.info("Control matrix loaded from file")

        except Exception as e:
            message = ('Error loading calibration file.')
            logger.log.error(message)
            wx.MessageBox(message, caption='Error')
        

    def OnSaveControlMatrix(self, event: wx.CommandEvent) -> None:
        del event

        # Prompt for file to load
        with wx.FileDialog(self, "Save calibration", wildcard="Data (*.txt)|*.txt",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            
            fpath = fileDialog.GetPath()

        # Get calibration matrix and save to file
        try:
            cmatrix = self._device.proxy.get_controlMatrix()
            np.savetxt(fpath, cmatrix)
        except:
            logger.log.error("Failed to save calibration data")

        logger.log.info("Saved DM calibration data to file {}".format(fpath))

    def OnLoadFlat(self, event: wx.CommandEvent) -> None:
        del event

        # Prompt for file to load
        with wx.FileDialog(self, "Load DM flat values", wildcard="Flat data (*.txt)|*.txt", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()

        try:
            # Load flat values from file and check format
            new_flat = np.loadtxt(fpath)
            assert new_flat.ndim == 1

            # Set new flat and refresh corrections
            self._device.set_system_flat(new_flat)
            self._device.refresh_corrections()

            # Log
            logger.log.info("System flat loaded from file")

        except Exception as e:
            message = ('Error loading flat file.')
            logger.log.error(message)
            wx.MessageBox(message, caption='Error')

    def OnSaveFlat(self, event: wx.CommandEvent) -> None:
        # Prompt for file to load
        with wx.FileDialog(self, "Save DM flat values", wildcard="Flat data (*.txt)|*.txt",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            
            fpath = fileDialog.GetPath()

        # Get flat values from device and save to file
        try:
            values = self._device.proxy.get_system_flat()
            np.savetxt(fpath, values)
        except:
            logger.log.error("Failed to save DM flat value data")

        logger.log.info("Saved DM flat data to file {}".format(fpath))

    def OnLoadActuatorValues(self, event: wx.CommandEvent) -> None:
        del event
        # Prompt for file to load
        with wx.FileDialog(self, "Load flat", wildcard="Flat data (*.txt)|*.txt", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()

        # Get data from file, check format, and send to DM
        try:
            values = np.loadtxt(fpath)
            assert (values.ndim == 1 and values.size <= self._device.no_actuators)
            self._device.set_phase(offset=values)
        except Exception as e:
            message = ('Error loading acuator values.')
            logger.log.error(message)
            wx.MessageBox(message, caption='Error')

    def OnSaveActuatorValues(self, event: wx.CommandEvent) -> None:
        del event

        # Prompt for file to load
        with wx.FileDialog(self, "Save DM actuator values", wildcard="Data (*.txt)|*.txt",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            
            fpath = fileDialog.GetPath()

        # Get actuator values and save to file
        try:
            values = self._device.proxy.get_last_actuator_values()
            np.savetxt(fpath, values)
        except:
            logger.log.error("Failed to save DM actuator value data")

        logger.log.info("Saved DM actuator values to file {}".format(fpath))

    def OnLoadModes(self, event: wx.CommandEvent) -> None:
        del event
        # Prompt for file to load
        with wx.FileDialog(self, "Load modes", wildcard="Flat data (*.txt)|*.txt", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()

        # Get data from file, check format, and send to DM
        try:
            values = np.loadtxt(fpath)
            assert (values.ndim == 1)
            self._device.set_phase(values)
        except Exception as e:
            message = ('Error loading modes. ({})'.format(e))
            logger.log.error(message)
            wx.MessageBox(message, caption='Error')

    def OnSaveModes(self, event: wx.CommandEvent) -> None:
        del event

        # Prompt for file to load
        with wx.FileDialog(self, "Save modes", wildcard="Data (*.txt)|*.txt",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            
            fpath = fileDialog.GetPath()

        # Get actuator values and save to file
        try:
            modes, _ = self._device.sum_corrections()
            np.savetxt(fpath, modes)
            logger.log.info("Saved modes to file {}".format(fpath))
        except:
            logger.log.error("Failed to save modes")


    def OnSetSystemFlatCalculationParameters(self, event: wx.CommandEvent) -> None:
        del event

        params = self._device.sys_flat_parameters

        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Set system flat parameters",
            [
                "Number of iterations",
                "Error threshold",
                "Noll indices to ignore",
                "Gain",
            ],
            (
                int(params["iterations"]),
                params["error_threshold"],
                (params["modes_to_ignore"] + 1).tolist(),
                params["gain"],
            ),
        )
        iterations = float(inputs[0])
        error_threshold = float(inputs[1])
        if iterations == float("inf") and error_threshold == float("inf"):
            wx.MessageBox(
                "Cannot have both the iterations and the error threshold set "
                "to infinity, because the flattening algorithm will not "
                "converge.",
                caption="Error"
            )
            return
        params["iterations"]= iterations
        params["error_threshold"] = error_threshold

        # FIXME: we should probably do some input checking here and
        # maybe not include a space in `split(", ")`
        if inputs[2] == "":
            params["modes_to_ignore"] = np.array([])
        else:
            params["modes_to_ignore"] = np.asarray(
                [int(z_ind) - 1 for z_ind in inputs[2][1:-1].split(", ")]
            )

        params["gain"] = float(inputs[3])

    def OnSetMetric(self, event: wx.CommandEvent) -> None:
        del event

        metrics = sorted(list(microAO.aoAlg.metric_function.keys()))

        with wx.SingleChoiceDialog(
            self,
            "Select metric",
            "Metric",
            metrics,
            wx.CHOICEDLG_STYLE
        ) as dlg:
            dlg.SetSelection(metrics.index(self._device.aoAlg.get_metric()))
            if dlg.ShowModal() == wx.ID_OK:
                metric = metrics[dlg.GetSelection()]
                self._device.aoAlg.set_metric(metric)
                logger.log.info(f"Set sensorless AO metric to: {metric}")

    def OnSetSensorlessParameters(self, _) -> None:
        with _SensorlessParametersDialog(self) as dlg:
            dlg.ShowModal()

    def OnDMViewer(self, event: wx.CommandEvent) -> None:
        # Try to find DM viewer window
        try:
            window = self.FindWindowById(self._components["dm_view"])
        except:
            window = None

        # If not found, create new window and save reference to its id
        if window is None:
            window = DMViewer(self, self._device)
            self._components["dm_view"] = window.GetId()

            actuator_values = self._device.proxy.get_last_actuator_values()
            if actuator_values is not None:
                window.SetActuators(actuator_values)

        # Show window and bring to front
        window.Show()
        window.Raise()           

    def OnCorrectionFitting(self, _: wx.CommandEvent) -> None:
        # Try to find remote focus window
        try:
            window = self.FindWindowById(
                self._components["correction fitting"]
            )
        except:
            window = None

        # If not found, create new window and save reference to its id
        if window is None:
            window = CorrectionFittingFrame(self, self._device)
            self._components["correction fitting"] = window.GetId()

        # Show window and bring to front
        window.Show()
        window.Raise()

    def OnSetCurrentAsFlat(self, event: wx.CommandEvent) -> None:
        """ Sets current actuator values as the new flat """
        modes, _ = self._device.sum_corrections()
        self._device.set_system_flat(modes)

    def OnTriggerTypeChoice(self, event: wx.CommandEvent):
        try:
            _, tmode = self._device.proxy.get_trigger()
            ttype = microscope.devices.TriggerType[event.GetString()]
            self._device.proxy.set_trigger(
                ttype,
                tmode
            )
            self.ao_trigger = (ttype, tmode)
        except Exception as e:
            # Changing the trigger failed => restore previous choice value
            choice = event.GetEventObject()
            choice.SetSelection(choice.FindString(self.ao_trigger[0].name))
            raise e

    def OnTriggerModeChoice(self, event: wx.CommandEvent):
        try:
            ttype, _ = self._device.proxy.get_trigger()
            tmode = microscope.devices.TriggerMode[event.GetString()]
            self._device.proxy.set_trigger(
                ttype,
                tmode
            )
            self.ao_trigger = (ttype, tmode)
        except Exception as e:
            # Changing the trigger failed => restore previous choice value
            choice = event.GetEventObject()
            choice.SetSelection(choice.FindString(self.ao_trigger[1].name))
            raise e

    def OnExercise(self, event: wx.CommandEvent):
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Set exercise parameters",
            [
                "Gain",
                "Pattern hold time [ms]",
                "Repeats"
            ],
            (
                0.7,
                3000,
                50,
            ),
        )
        self._device.exercise(
            float(inputs[0]),
            float(inputs[1]),
            int(inputs[2])
        )

    def OnCorrectionState(self, event: wx.CommandEvent):
        index = event.GetInt()
        name = self.checklist_corrections.GetString(index)
        state = self.checklist_corrections.IsChecked(index)
        self._device.toggle_correction(name, state)
        self._device.refresh_corrections()

    def OnCorrectionChange(self, name, state):
        items = {
            self.checklist_corrections.GetString(i):self.checklist_corrections.IsChecked(i)
            for i in range(self.checklist_corrections.GetCount())
        }
        # Add/update the correction
        items[name] = state
        # Update widget
        self.checklist_corrections.Set(sorted(items.keys()))
        for i in range(self.checklist_corrections.GetCount()):
            self.checklist_corrections.Check(
                i,
                items[self.checklist_corrections.GetString(i)]
            )

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

            with wx.SingleChoiceDialog(
                None,
                "Select camera",
                "Camera",
                list(cameras_dict.keys()),
                wx.CHOICEDLG_STYLE
            ) as dlg:
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

            with wx.SingleChoiceDialog(
                None,
                "Select imager",
                "Imager",
                list(imagers_dict.keys()),
                wx.CHOICEDLG_STYLE
            ) as dlg:
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

            with wx.SingleChoiceDialog(
                None,
                "Select stage",
                "Stage",
                list(stages_dict.keys()),
                wx.CHOICEDLG_STYLE
            ) as dlg:
                if dlg.ShowModal() == wx.ID_OK:
                    stage = stages_dict[dlg.GetStringSelection()]

        return stage