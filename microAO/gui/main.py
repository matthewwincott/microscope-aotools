from microAO.aoDev import AdaptiveOpticsDevice
from microAO.gui import *
from microAO.gui.modeControl import ModesControl
from microAO.gui.remoteFocus import RemoteFocusControl
from microAO.gui.sensorlessViewer import SensorlessResultsViewer
from microAO.gui.DMViewer import DMViewer
from microAO import cockpit_device
import microAO.events

import cockpit.events
import cockpit.gui.device
import cockpit.gui.camera.window
from cockpit.util import logger, userConfig

import microscope.devices

import wx
from wx.lib.floatcanvas.FloatCanvas import FloatCanvas
import wx.lib.floatcanvas.FCObjects as FCObjects

import matplotlib.pyplot
import matplotlib.ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import numpy as np

import aotools
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

def _subtractModesFromUnwrappedPhase(unwrapped_phase, modes=(0, 1, 2)):
    # XXX: AdaptiveOpticsDevice.getzernikemodes method does not
    # actually make use of its instance.  It should have been a free
    # function or at least a class method.  Using it like this means
    # we can compute it client-side instead of having send the data.
    # This should be changed in microscope-aotools.
    #
    # Get the Zernike modes of the phase map, up to the largest requested
    z_amps = AdaptiveOpticsDevice.getzernikemodes(
        None,
        unwrapped_phase,
        max(modes) + 1
    )
    # Suppress the modes which are not required
    for i in range(len(z_amps)):
        if i not in modes:
            z_amps[i] = 0
    # Convert the modes to a phase map
    phase_modes = aotools.phaseFromZernikes(z_amps, unwrapped_phase.shape[0])
    # Subtract the modes' phase from the unwrapped phase
    return unwrapped_phase - phase_modes

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

        # Button to apply offset between phases
        phaseOffsetButton = wx.Button(panel_calibration, label="Apply phase offset")
        phaseOffsetButton.Bind(wx.EVT_BUTTON, self.OnPhaseOffset)

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
        RemoteFocusButton = wx.Button(panel_AO, label="Remote focus")
        RemoteFocusButton.Bind(wx.EVT_BUTTON, self.OnRemoteFocus)

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
            sysFlatCalcButton,
            phaseOffsetButton
        ]:
            sizer_calibration.Add(btn, panel_flags)

        sizer_AO = wx.BoxSizer(wx.VERTICAL)
        for btn in [
            resetButton,
            applySysFlat,
            metricSelectButton,
            sensorlessParametersButton,
            sensorlessAOButton,
            RemoteFocusButton
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

    def _get_image_and_unwrap(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
        # Select the camera whose window contains the interferogram
        camera = self._device.getCamera()
        if camera is None:
            logger.log.error(
                "Failed to visualise phase because no active cameras were "
                "found."
            )
            return
        # Get the image data
        phase = cockpit.gui.camera.window.getImageForCamera(camera)
        if phase is None:
            logger.log.error(
                f"The camera view for camera '{camera.name}' contained no "
                "image. Please capture an interferogram first."
            )
            return
        # Select the ROI
        with _ROISelect(self, phase) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                roi = dlg.GetROI()
                userConfig.setValue(
                    "dm_circleParams",
                    (roi[1], roi[0], roi[2])
                )
            else:
                logger.log.error(
                    "ROI selection cancelled. Aborting phase visualisation..."
                )
                return
        # Update device ROI and filter
        self._device.updateROI()
        self._device.checkFourierFilter()
        # Unwrap the phase image
        return (phase, self._device.unwrap_phase(phase), (roi[1], roi[0], roi[2]))

    def OnVisualisePhase(self, event: wx.CommandEvent) -> None:
        phase, phase_unwrapped, roi = self._get_image_and_unwrap()
        if phase_unwrapped is None:
            # All the logging has already been done in the
            # _get_image_and_unwrap() method
            return
        # Subtract piston, tip, and tilt modes
        phase_unwrapped_mptt = _subtractModesFromUnwrappedPhase(phase_unwrapped)
        # Compute the RMS error of the adjusted unwrapped phase
        true_flat = np.zeros(np.shape(phase_unwrapped_mptt))
        mask = cockpit_device.aoAlg.mask
        phase_unwrapped_MPTT_RMS_error = np.sqrt(
            np.mean((true_flat[mask] - phase_unwrapped_mptt[mask]) ** 2)
        )
        # Calculate the power spectrum
        phase_power_spectrum = _computePowerSpectrum(phase)

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
        self._device.calibrationGetData(self)

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
        sys_flat_values, best_z_amps_corrected = self._device.sysFlatCalc()
        logger.log.debug(
            "Zernike modes amplitudes corrected:\n %s", best_z_amps_corrected
        )
        logger.log.debug("System flat actuator values:\n%s", sys_flat_values)

    def OnPhaseOffset(self, event: wx.CommandEvent) -> None:
        # Naviage to the phase data file used as reference
        file_path_input = None
        with wx.FileDialog(
            self,
            message="Load phase data file",
            wildcard="Numpy file (*.npy)|*.npy",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as file_dialog:
            if file_dialog.ShowModal() != wx.ID_OK:
                return
            file_path_input = pathlib.Path(file_dialog.GetPath())
        # Specify the location and base filename for the outputs
        file_path_outputs = None
        with wx.FileDialog(
            self,
            message="Save output files",
            wildcard="Numpy file (*.npy)|*.npy",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as file_dialog:
            if file_dialog.ShowModal() != wx.ID_OK:
                return
            file_path_outputs = pathlib.Path(file_dialog.GetPath())
        # Get the current unwrapped phase
        _, phase_unwrapped, _ = self._get_image_and_unwrap()
        # Load the reference unwrapped phase
        phasedata = np.load(file_path_input, allow_pickle=True)
        phase_unwrapped_ref = phasedata.item()["phase_unwrapped"]
        # Subtract piston modes
        phase_unwrapped = _subtractModesFromUnwrappedPhase(
            phase_unwrapped,
            (0,)
        )
        phase_unwrapped_ref = _subtractModesFromUnwrappedPhase(
            phase_unwrapped_ref,
            (0,)
        )
        # Calculate the difference
        phase_unwrapped_difference = (
            phase_unwrapped - phase_unwrapped_ref
        )
        # Set the phase difference map
        actuators = self._device.set_phase_map(
            -1.0 * phase_unwrapped_difference
        )
        # Check what the new phase looks like
        _, phase_unwrapped_result, _ = self._get_image_and_unwrap()
        phase_unwrapped_result = _subtractModesFromUnwrappedPhase(
            phase_unwrapped_result,
            (0,)
        )
        # Derive names for all the output files
        file_path_phase = file_path_outputs.with_name(
            file_path_outputs.stem + "_phase.npy"
        )
        file_path_phase_ref = file_path_outputs.with_name(
            file_path_outputs.stem + "_phase-reference.npy"
        )
        file_path_phase_diff = file_path_outputs.with_name(
            file_path_outputs.stem + "_phase-difference.npy"
        )
        file_path_phase_result = file_path_outputs.with_name(
            file_path_outputs.stem + "_phase-result.npy"
        )
        file_path_actuators = file_path_outputs.with_name(
            file_path_outputs.stem + "_actuators.txt"
        )
        # Write the output files
        np.save(file_path_phase, phase_unwrapped)
        np.save(file_path_phase_ref, phase_unwrapped_ref)
        np.save(file_path_phase_diff, phase_unwrapped_difference)
        np.save(file_path_phase_result, phase_unwrapped_result)
        np.savetxt(file_path_actuators, actuators)
        # Plot
        fig, axes = matplotlib.pyplot.subplots(2, 2, figsize=(8.5, 7.3))
        axes = axes.ravel()
        for i, (img, title) in enumerate(
            (
                (
                    phase_unwrapped_ref,
                    "Reference phase without piston"
                ),
                (
                    phase_unwrapped,
                    "Current phase without piston"
                ),
                (
                    phase_unwrapped_difference,
                    "Difference between current\nand reference phases"
                ),
                (
                    phase_unwrapped_result,
                    "Current phase without piston,\n"
                    "after applying the inverse of the difference"
                )
            )
        ):
            axes_image = axes[i].imshow(img)
            axes[i].set_title(title)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_frame_on(False)
            matplotlib.pyplot.colorbar(axes_image, ax=axes[i])
        fig.tight_layout()
        fig.show()

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

        camera = self._device.getCamera()

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

        # Start sensorless AO
        action(camera)

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
            assert (control_matrix.ndim == 2 and control_matrix.shape[1] == self._device.no_actuators)
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
            assert (new_flat.ndim == 1 and new_flat.size <= self._device.no_actuators)

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
            corrections = self._device.get_corrections()
            modes = np.zeros(self._device.no_actuators) + sum(
                [
                    np.array(correction["modes"])
                    for correction in corrections.values()
                    if correction["enabled"] and correction["modes"] is not None
                ]
            )
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
                "System Flat Noll indeces",
            ],
            (
                params["num_it"],
                params["error_thresh"],
                params["nollZernike"].tolist(),
            ),
        )
        params["num_it"]= int(inputs[0])
        params["error_thresh"] = np.float(inputs[1])

        # FIXME: we should probably do some input checking here and
        # maybe not include a space in `split(", ")`
        if inputs[2] == "":
            params["nollZernike"] = None
        else:
            params = np.asarray(
                [int(z_ind) for z_ind in inputs[-1][1:-1].split(", ")]
            )

    def OnSetMetric(self, event: wx.CommandEvent) -> None:
        del event

        metrics = dict([
            ("Fourier metric", "fourier"),
            ("Contrast metric", "contrast"),
            ("Fourier Power metric", "fourier_power"),
            ("Gradient metric", "gradient"),
            ("Second Moment metric", "second_moment"),
        ])

        dlg = wx.SingleChoiceDialog(
            self, "Select metric", "Metric", list(metrics.keys()),
        wx.CHOICEDLG_STYLE
            )
        if dlg.ShowModal() == wx.ID_OK:
            metric = metrics[dlg.GetStringSelection()]
            self._device.proxy.set_metric(metric)

            logger.log.info("Set sensorless AO metric to: {}".format(metric))
        
        dlg.Destroy()

    def OnSetSensorlessParameters(self, event: wx.CommandEvent) -> None:

        params = self._device.sensorless_params

        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Set sensorless AO parameters",
            [
                "Aberration range minima",
                "Aberration range maxima",
                "Number of measurements",
                "Number of repeats",
                "Noll indices",
                "NA",
                "wavelength"
            ],
            (
                params["range_min"],
                params["range_max"],
                params["num_meas"],
                params["num_reps"],
                [mode_index + 1 for mode_index in params["modes_subset"]],
                params["NA"],
                params["wavelength"],
            ),
        )
        params["range_min"] = float(inputs[0])
        params["range_max"] = float(inputs[1])
        params["num_meas"] = int(inputs[2])
        params["num_reps"] = int(inputs[3])
        params["modes_subset"] = [
            int(z_ind) - 1 for z_ind in inputs[4][1:-1].split(", ")
        ]
        params["NA"] = float(inputs[5])
        params["wavelength"] = int(inputs[6])

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

    def OnRemoteFocus(self, event: wx.CommandEvent) -> None:
        # Try to find remote focus window
        try:
            window = self.FindWindowById(self._components["remote_focus"])
        except:
            window = None

        # If not found, create new window and save reference to its id
        if window is None:
            window = RemoteFocusControl(self, self._device)    
            self._components["remote_focus"] = window.GetId()

        # Show window and bring to front
        window.Show()
        window.Raise()           
    
    def OnSetCurrentAsFlat(self, event: wx.CommandEvent) -> None:
        """ Sets current actuator values as the new flat """
        current_actuator_values = self._device.proxy.get_last_actuator_values()
        self._device.set_system_flat(current_actuator_values)

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