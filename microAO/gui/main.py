from microAO.aoDev import AdaptiveOpticsDevice
from microAO.remotez import RemoteZ
from microAO.gui import *

import cockpit.gui.device
from cockpit import depot
from cockpit.util import logger, userConfig

import wx
from wx.lib.floatcanvas.FloatCanvas import FloatCanvas

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import numpy as np

import aotools
import typing

from microAO.gui.modeControl import ModesControl
from microAO.gui.remoteFocus import RemoteFocusControl
from microAO.gui.sensorlessViewer import SensorlessResultsViewer
from microAO.gui.DMViewer import DMViewer

_ROI_MIN_RADIUS = 8


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


def _bin_ndarray(ndarray, new_shape):
    """Bins an ndarray in all axes based on the target shape by averaging.


    Number of output dimensions must match number of input dimensions
    and new axes must divide old ones.

    Example
    -------

    m = np.arange(0,100,1).reshape((10,10))
    n = bin_ndarray(m, new_shape=(5,5))
    print(n)
    [[ 5.5  7.5  9.5 11.5 13.5]
     [25.5 27.5 29.5 31.5 33.5]
     [45.5 47.5 49.5 51.5 53.5]
     [65.5 67.5 69.5 71.5 73.5]
     [85.5 87.5 89.5 91.5 93.5]]

    Function acquired from Stack Overflow at
    https://stackoverflow.com/a/29042041. Stack Overflow or other
    Stack Exchange sites is cc-wiki (aka cc-by-sa) licensed and
    requires attribution.

    """
    if ndarray.ndim != len(new_shape):
        raise ValueError(
            "Shape mismatch: {} -> {}".format(ndarray.shape, new_shape)
        )
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        ndarray = ndarray.mean(-1 * (i + 1))
    return ndarray


def _computeUnwrappedPhaseMPTT(unwrapped_phase):
    # XXX: AdaptiveOpticsDevice.getzernikemodes method does not
    # actually make use of its instance.  It should have been a free
    # function or at least a class method.  Using it like this means
    # we can compute it client-side instead of having send the data.
    # This should be changed in microscope-aotools.
    z_amps = AdaptiveOpticsDevice.getzernikemodes(None, unwrapped_phase, 3)
    phase = aotools.phaseFromZernikes(z_amps[0:3], unwrapped_phase.shape[0])
    return unwrapped_phase - phase


def _computePowerSpectrum(interferogram):
    interferogram_ft = np.fft.fftshift(np.fft.fft2(interferogram))
    power_spectrum = np.log(abs(interferogram_ft))
    return power_spectrum

class _ROISelect(wx.Frame):
    """Display a window that allows the user to select a circular area.

    This is a window for selecting the ROI for interferometry.
    """

    def __init__(
        self, parent, input_image: np.ndarray, initial_roi, scale_factor=1
    ) -> None:
        super().__init__(parent, title="ROI selector")
        self._panel = wx.Panel(self)
        self._img = _np_grey_img_to_wx_image(input_image)
        self._range_factor = scale_factor

        # What, if anything, is being dragged.
        # XXX: When we require Python 3.8, annotate better with
        # `typing.Literal[None, "xy", "r"]`
        self._dragging: typing.Optional[str] = None

        # Canvas
        self.canvas = FloatCanvas(self._panel, size=self._img.GetSize())
        self.canvas.Bind(wx.EVT_MOUSE_EVENTS, self.OnMouse)
        self.bitmap = self.canvas.AddBitmap(self._img, (0, 0), Position="cc")

        self.circle = self.canvas.AddCircle(
            self.canvas.PixelToWorld(initial_roi[:2]),
            initial_roi[2] * 2,
            LineColor="cyan",
            LineWidth=2,
        )

        # Save button
        saveBtn = wx.Button(self._panel, label="Save ROI")
        saveBtn.Bind(wx.EVT_BUTTON, self.OnSave)

        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(self.canvas)
        panel_sizer.Add(saveBtn, wx.SizerFlags().Border())
        self._panel.SetSizer(panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self._panel)
        self.SetSizerAndFit(frame_sizer)

    @property
    def ROI(self):
        """Convert circle parameters to ROI x, y and radius"""
        roi_x, roi_y = self.canvas.WorldToPixel(self.circle.XY)
        roi_r = max(self.circle.WH)
        return (roi_x, roi_y, roi_r)

    def OnSave(self, event: wx.CommandEvent) -> None:
        del event
        roi = [x * self._range_factor for x in self.ROI]
        userConfig.setValue("dm_circleParams", (roi[1], roi[0], roi[2]))

    def MoveCircle(self, pos: wx.Point, r) -> None:
        """Set position and radius of circle with bounds checks."""
        x, y = pos
        _x, _y, _r = self.ROI
        xmax, ymax = self._img.GetSize()
        if r == _r:
            x_bounded = min(max(r, x), xmax - r)
            y_bounded = min(max(r, y), ymax - r)
            r_bounded = r
        else:
            r_bounded = max(_ROI_MIN_RADIUS, min(xmax - x, x, ymax - y, y, r))
            x_bounded = min(max(r_bounded, x), xmax - r_bounded)
            y_bounded = min(max(r_bounded, y), ymax - r_bounded)
        self.circle.SetPoint(self.canvas.PixelToWorld((x_bounded, y_bounded)))
        self.circle.SetDiameter(2 * r_bounded)
        if any((x_bounded != x, y_bounded != y, r_bounded != r)):
            self.circle.SetColor("magenta")
        else:
            self.circle.SetColor("cyan")

    def OnMouse(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        x, y, r = self.ROI
        if event.LeftDClick():
            # Set circle centre
            self.MoveCircle(pos, r)
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
                self.MoveCircle((x, y), drag_r)
            elif self._dragging == "xy":
                # Drag circle centre
                self.MoveCircle(pos, r)

        if not event.Dragging():
            # Stop dragging
            self._dragging = None
            self.circle.SetColor("cyan")

        self.canvas.Draw(Force=True)


class _PhaseViewer(wx.Frame):
    """This is a window for selecting the ROI for interferometry."""

    def __init__(self, parent, input_image, image_ft, RMS_error, *args, **kwargs):
        super().__init__(parent, title="Phase View")
        self._panel = wx.Panel(self, *args, **kwargs)

        _wx_img_real = _np_grey_img_to_wx_image(input_image)
        _wx_img_fourier = _np_grey_img_to_wx_image(image_ft)

        self._canvas = FloatCanvas(self._panel, size=_wx_img_real.GetSize())
        self._real_bmp = self._canvas.AddBitmap(
            _wx_img_real, (0, 0), Position="cc"
        )
        self._fourier_bmp = self._canvas.AddBitmap(
            _wx_img_fourier, (0, 0), Position="cc"
        )

        # By default, show real and hide the fourier transform.
        self._fourier_bmp.Hide()

        save_btn = wx.ToggleButton(self._panel, label="Show Fourier")
        save_btn.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleFourier)

        self._rms_txt = wx.StaticText(
            self._panel, label="RMS difference: %.05f" % (RMS_error)
        )

        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(self._canvas)

        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        bottom_sizer.Add(save_btn, wx.SizerFlags().Center().Border())
        bottom_sizer.Add(self._rms_txt, wx.SizerFlags().Center().Border())
        panel_sizer.Add(bottom_sizer)

        self._panel.SetSizer(panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self._panel)
        self.SetSizerAndFit(frame_sizer)

    def OnToggleFourier(self, event: wx.CommandEvent) -> None:
        show_fourier = event.IsChecked()
        # These bmp are wx.lib.floatcanvas.FCObjects.Bitmap and not
        # wx.Bitmap.  Their Show method does not take show argument
        # and therefore we can't do `Show(show_fourier)`.
        if show_fourier:
            self._fourier_bmp.Show()
            self._real_bmp.Hide()
        else:
            self._real_bmp.Show()
            self._fourier_bmp.Hide()
        self._canvas.Draw(Force=True)

    def SetData(self, input_image, image_ft=None, RMS_error=None):
        _wx_img_real = _np_grey_img_to_wx_image(input_image)
        _wx_img_fourier = _np_grey_img_to_wx_image(image_ft)

        self._real_bmp.Bitmap.CopyFromBuffer(_wx_img_real.GetData())
        self._fourier_bmp.Bitmap.CopyFromBuffer(_wx_img_fourier.GetData())
        
        self._canvas.Draw(Force=True)

        self._rms_txt.SetLabel("RMS difference: %.05f" % (RMS_error))

class _CharacterisationAssayViewer(wx.Frame):
    def __init__(self, parent, characterisation_assay):
        super().__init__(parent, title="Characterisation Asssay")
        root_panel = wx.Panel(self)

        figure = Figure()

        img_ax = figure.add_subplot(1, 2, 1)
        img_ax.imshow(characterisation_assay)

        diag_ax = figure.add_subplot(1, 2, 2)
        assay_diag = np.diag(characterisation_assay)
        diag_ax.plot(assay_diag)

        canvas = FigureCanvas(root_panel, wx.ID_ANY, figure)

        info_txt = wx.StaticText(
            root_panel,
            label=(
                "Mean Zernike reconstruction accuracy: %0.5f"
                % np.mean(assay_diag)
            ),
        )

        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(info_txt, wx.SizerFlags().Centre().Border())
        panel_sizer.Add(canvas, wx.SizerFlags(1).Expand())
        root_panel.SetSizer(panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root_panel, wx.SizerFlags().Expand())
        self.SetSizerAndFit(frame_sizer)

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

        # Button to select the interferometer ROI
        selectCircleButton = wx.Button(panel_calibration, label="Select ROI")
        selectCircleButton.Bind(wx.EVT_BUTTON, self.OnSelectROI)

        # Visualise current interferometric phase
        visPhaseButton = wx.Button(panel_calibration, label="Visualise Phase")
        visPhaseButton.Bind(wx.EVT_BUTTON, self.OnVisualisePhase)

        # Button to calibrate the DM
        calibrateButton = wx.Button(panel_calibration, label="Calibrate")
        calibrateButton.Bind(wx.EVT_BUTTON, self.OnCalibrate)

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

        # Apply last actuator values
        applyLastPatternButton = wx.Button(panel_AO, label="Apply last pattern")
        applyLastPatternButton.Bind(wx.EVT_BUTTON, self.OnApplyLastPattern)

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
        RemoteFocusButton = wx.Button(panel_control, label="Remote focus")
        RemoteFocusButton.Bind(wx.EVT_BUTTON, self.OnRemoteFocus)

        panel_flags = wx.SizerFlags(0).Expand().Border(wx.LEFT|wx.RIGHT, 50)

        sizer_panel_setup = wx.BoxSizer(wx.VERTICAL)
        for btn in [
            loadControlMatrixButton,
            saveControlMatrixButton,
            loadFlatButton,
            saveFlatButton
        ]:
            sizer_panel_setup.Add(btn, panel_flags)

        sizer_calibration = wx.BoxSizer(wx.VERTICAL)
        for btn in [
            selectCircleButton,
            visPhaseButton,
            calibrateButton,
            characteriseButton,
            sysFlatParametersButton,
            sysFlatCalcButton
        ]:
            sizer_calibration.Add(btn, panel_flags)

        sizer_AO = wx.BoxSizer(wx.VERTICAL)
        for btn in [
            resetButton,
            applySysFlat,
            applyLastPatternButton,
            metricSelectButton,
            sensorlessParametersButton,
            sensorlessAOButton
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
            setCurrentAsFlatButton,
            RemoteFocusButton
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

        sizer_main = wx.BoxSizer(wx.VERTICAL)
        sizer_main.Add(tabs, wx.SizerFlags(1).Expand())
        self.SetSizer(sizer_main)

    def OnSelectROI(self, event: wx.CommandEvent) -> None:
        del event
        image_raw = self._device.acquireRaw()
        if np.max(image_raw) > 10:
            original_dim = np.shape(image_raw)[0]
            resize_dim = 512

            while original_dim % resize_dim != 0:
                resize_dim -= 1

            if resize_dim < original_dim / resize_dim:
                resize_dim = int(np.round(original_dim / resize_dim))

            scale_factor = original_dim / resize_dim
            img = _bin_ndarray(image_raw, new_shape=(resize_dim, resize_dim))
            img = np.require(img, requirements="C")

            last_roi = userConfig.getValue(
                "dm_circleParams",
            )
            # We need to check if getValue() returns None, instead of
            # passing a default value to getValue().  The reason is
            # that if there is no ROI at the start, by the time we get
            # here the device initialize has called updateROI which
            # also called getValue() which has have the side effect of
            # setting its value to None.  And we can't set a sensible
            # default at that time because we have no method to get
            # the wavefront camera sensor size.
            if last_roi is None:
                last_roi = (
                    *[d // 2 for d in image_raw.shape],
                    min(image_raw.shape) // 4,
                )

            last_roi = (
                last_roi[1] / scale_factor,
                last_roi[0] / scale_factor,
                last_roi[2] / scale_factor,
            )

            frame = _ROISelect(self, img, last_roi, scale_factor)
            frame.Show()
        else:
            wx.MessageBox(
                "Detected nothing but background noise.",
                caption="No good image acquired",
                style=wx.ICON_ERROR | wx.OK | wx.CENTRE,
            )

    def OnVisualisePhase(self, event: wx.CommandEvent) -> None:
        del event
        self._device.updateROI()
        self._device.checkFourierFilter()

        interferogram, unwrapped_phase = self._device.acquireUnwrappedPhase()
        power_spectrum = _computePowerSpectrum(interferogram)
        unwrapped_phase_mptt = _computeUnwrappedPhaseMPTT(unwrapped_phase)

        unwrapped_RMS_error = self._device.wavefrontRMSError(
            unwrapped_phase_mptt
        )

        frame = _PhaseViewer(
            self, unwrapped_phase, power_spectrum, unwrapped_RMS_error
        )
        frame.Show()

    def OnCalibrate(self, event: wx.CommandEvent) -> None:
        del event
        self._device.calibrate()

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

    def OnResetDM(self, event: wx.CommandEvent) -> None:
        del event
        self._device.reset()

    def OnSystemFlat(self, event: wx.CommandEvent) -> None:
        del event
        self._device.applySysFlat()

    def OnApplyLastPattern(self, event: wx.CommandEvent) -> None:
        del event
        self._device.applyLastPattern()

    def OnSensorlessAO(self, event: wx.CommandEvent) -> None:
        # Perform sensorless AO but if there is more than one camera
        # available display a menu letting the user choose a camera.
        del event

        action = self._device.correctSensorlessSetup

        camera = self._device.getCamera()

        if camera is None:
            return

        # Start sensorless AO
        action(camera)

        # Create results viewer
        try:
            window = self.FindWindowById(self._components["sensorless_results"])
        except:
            window = None

        if window is None:
            window = SensorlessResultsViewer(None, None)
            self._components["sensorless_results"] = window.GetId()

            window.Show()

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

            # Calculate updated actuator values using new flat
            curr_flat = self._device.proxy.get_system_flat()
            curr_actuator_values = self._device.proxy.get_last_actuator_values()
            new_actuator_values = curr_actuator_values - curr_flat + new_flat

            # Set new flat and update actuator values
            self._device.set_system_flat(new_flat)
            self._device.send(new_actuator_values)

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
            self._device.send(values)
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
            self._device.set_phase(values, self._device.proxy.get_system_flat())
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
            values = self._device.proxy.get_last_modes()
            np.savetxt(fpath, values)
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
                "Start from flat"
            ],
            (
                params["z_min"],
                params["z_max"],
                params["numMes"],
                params["num_it"],
                params["nollZernike"].tolist(),
                params["start_from_flat"]
            ),
        )
        params["z_min"] = float(inputs[0])
        params["z_max"] = float(inputs[1])
        params["numMes"] = int(inputs[2])
        params["num_it"] = int(inputs[3])
        params["nollZernike"] = np.asarray(
            [int(z_ind) for z_ind in inputs[4][1:-1].split(", ")]
        )
        if inputs[5].lower() in ["true", "t", "yes", "y", "1"]:
            params["start_from_flat"] = True
        elif inputs[5].lower() in ["false", "f", "no", "n", "0"]:
            params["start_from_flat"] = False

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
            self._device.remotez = RemoteZ(self._device)
            window = RemoteFocusControl(self, self._device)    
            self._components["remote_focus"] = window.GetId()

        # Show window and bring to front
        window.Show()
        window.Raise()           
    
    def OnSetCurrentAsFlat(self, event: wx.CommandEvent) -> None:
        """ Sets current actuator values as the new flat """
        current_actuator_values = self._device.proxy.get_last_actuator_values()
        userConfig.setValue("dm_sys_flat", np.ndarray.tolist(current_actuator_values))
        self._device.proxy.set_system_flat(current_actuator_values)