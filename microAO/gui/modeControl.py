
from cockpit import events

import wx
import wx.lib.newevent
import wx.lib.scrolledpanel

import numpy as np

from microAO.events import *
from microAO.gui.common import FloatCtrl, FilterModesCtrl


_DEFAULT_ZERNIKE_MODE_NAMES = {
    1: "Piston",
    2: "Tip",
    3: "Tilt",
    4: "Defocus",
    5: "Astig (O)",
    6: "Astig (V)",
    7: "Coma (V)",
    8: "Coma (H)",
    9: "Trefoil (V)",
    10: "Trefoil (O)",
    11: "Spherical",
    12: "Astig 2 (V)",
    13: "Astig 2 (O)",
    14: "Quadrafoil (V)",
    15: "Quadrafoil (O)",
}

ModeChangeEvent, EVT_MODE_CHANGED = wx.lib.newevent.NewEvent()

class _Mode(wx.Panel):
    """Manual mode selection GUI."""

    def __init__(self, parent, id, value=0):
        super().__init__(parent)

        # id to identify mode
        self.id = id

        # Mode value
        self.value = value

        # Store focus state
        self.focus = False

        row_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Label for mode
        mode_index = self.id+1
        try:
            mode_label = _DEFAULT_ZERNIKE_MODE_NAMES[mode_index]
        except KeyError:
            mode_label = ""

        self._mode_index = wx.StaticText(self, label=str(mode_index), size=wx.Size(30,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
        self._mode_label = wx.StaticText(self, label=mode_label, size=wx.Size(120,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))

        # Mode slider: drag to set mode
        default_range = 1.5   # range of slider
        self._slider = wx.Slider(self, value=0, minValue=-100, maxValue=100, size=wx.Size(200,-1))
        self._slider.Bind(wx.EVT_SCROLL, self.OnSlider)

        # Adjust mode adjustment range. Influences range of slider.
        self._slider_min = FloatCtrl(self, wx.ID_ANY, value="{}".format(-default_range), size=wx.Size(50,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
        self._slider_max = FloatCtrl(self, wx.ID_ANY, value="{}".format(default_range), size=wx.Size(50,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
        
        self._slider_min.Bind(wx.EVT_TEXT, self.UpdateValueRanges)
        self._slider_max.Bind(wx.EVT_TEXT, self.UpdateValueRanges)

        # Current mode value
        self._val = wx.SpinCtrlDouble(self, initial=0, inc=0.001, size=wx.Size(160,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
        self._val.SetDigits(4)
        self.UpdateValueRanges()
        self._val.SetValue(self.value)
        self._val.Bind(wx.EVT_SPINCTRLDOUBLE, self.OnModeValueChange)
        self._val.Bind(wx.EVT_SET_FOCUS, self.OnModeGetFocus)
        self._val.Bind(wx.EVT_KILL_FOCUS, self.OnModeLoseFocus)

        # Layout
        row_sizer.Add(self._mode_index, wx.SizerFlags().CentreVertical())
        row_sizer.Add(self._mode_label, wx.SizerFlags().CentreVertical())
        row_sizer.Add(self._slider_min, wx.SizerFlags().Expand())
        row_sizer.Add(self._slider, wx.SizerFlags().Expand())
        row_sizer.Add(self._slider_max, wx.SizerFlags().Expand())
        row_sizer.Add(self._val, wx.SizerFlags().Expand())

        # Set widget sizer
        self.SetSizerAndFit(row_sizer)

    def OnSlider(self, evt):
        # Assign focus to control if sliding
        self.focus = True

        # Set value
        try:
            val = (self._slider.GetValue()+100)/200 * (self._slider_max.value - self._slider_min.value) + self._slider_min.value
            self.SetValue(val)
        except TypeError:
            pass

        # Reset slider and ranges when slide end (mouse released)
        if not wx.GetMouseState().LeftIsDown():
            # Lose focus as released
            self.focus = False

    def OnModeValueChange(self, evt):
        new_val = self._val.GetValue()
        self.UpdateValueRanges(new_val)
        self.SetValue(new_val)

    def OnRangeChange(self, evt):
        self.UpdateValueRanges()
        evt.Skip()
    
    def OnModeGetFocus(self, evt):
        self.focus = True

    def OnModeLoseFocus(self, evt):
        self.focus = False

    def UpdateValueRanges(self, middle=None, range=None):
        min_val =  self._slider_min.value
        if min_val is not None:
            self._val.SetMin(min_val)

        max_val = self._slider_max.value
        if max_val is not None:
            self._val.SetMax(max_val)

        self.SetValue(self.value, quiet=True)

    def GetValue(self):
        return self._val.GetValue()

    def SetValue(self, val, quiet=False):
        """ Set control value 

            Sets control value. Emits mode change event by default, a quiet flag can be 
            used to override this behaviour.        
        """
        # Set value property
        self.value = val

        # Set value control
        self._val.SetValue(val)
        
        # Set slider control value
        slider_val = 200 * (val - self._slider_min.value)/(self._slider_max.value - self._slider_min.value) - 100 
        self._slider.SetValue(slider_val)

        # Emit mode change event, if required
        if not quiet:
            evt = ModeChangeEvent(mode=self.id, value= self.value)
            wx.PostEvent(self, evt)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

class _ModesPanel(wx.lib.scrolledpanel.ScrolledPanel):
    def __init__(self, parent, device):
        super().__init__(parent)

        # Set attributes
        self._device = device
        control_matrix = self._device.proxy.get_controlMatrix()
        self._n_modes = control_matrix.shape[1]

        # Create root panel and sizer
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Reset button
        reset_btn = wx.Button(self, label="Reset")
        reset_btn.Bind(wx.EVT_BUTTON, self.OnReset)
        sizer.Add(reset_btn, 0, wx.ALL, 5)

        # Filter modes control
        filter_modes_lbl = wx.StaticText(self, label="Mode filter")
        self.filter_modes = FilterModesCtrl(self)
        self.filter_modes.ChangeValue("{}-{}".format(1, self._n_modes))
        self.filter_modes.Bind(wx.EVT_TEXT, self.OnFilterModes)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(filter_modes_lbl, wx.SizerFlags().Centre().Border(wx.RIGHT, 8))
        hbox.Add(self.filter_modes, wx.SizerFlags().Centre())
        sizer.Add(hbox, wx.SizerFlags().Border(wx.BOTTOM, 8))

        # Set headings
        heading_sizer = wx.BoxSizer(wx.HORIZONTAL)

        headings = [
            ("", 150), 
            ("Min", 50), 
            ("Control", 200), 
            ("Max", 50), 
            ("Value", 160)
        ]

        font = wx.Font( wx.FontInfo(10).Bold())
        flags = wx.SizerFlags().Centre()

        for heading in headings:
            heading_widget = wx.StaticText(self, label=heading[0], size=wx.Size(heading[1],-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
            heading_widget.SetFont(font)
            heading_sizer.Add(heading_widget, 0, wx.CENTER | wx.BOTTOM, 8)
        sizer.Add(heading_sizer, 0, wx.BOTTOM, 8)

        # Add control per mode
        modes = np.zeros(self._n_modes)
        last_modes = self._device.proxy.get_last_modes()
        if last_modes is not None:
            modes += last_modes

        self._mode_controls = []
        for i, mode in enumerate(modes):
            mode_control = _Mode(self, id=i, value=mode)
            mode_control.Bind(EVT_MODE_CHANGED, self.OnMode)
            self._mode_controls.append(mode_control)
            sizer.Add(mode_control, 0)

        # Show only filtered modes
        self.FilterModes()
        
        # Set sizer and finalise layout
        self.SetSizerAndFit(sizer)
        self.SetupScrolling()
        self.Layout()

        # Initialise the mode control correction
        self._device.set_correction("mode control")

        # Subscribe to pubsub events
        events.subscribe(PUBSUB_SET_PHASE, self.HandleSetPhase)

        # Bind close event
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def FilterModes(self):
        # Show only filtered modes
        modes_filtered = self.filter_modes.GetValue()
        for control in self._mode_controls:
            if control.id+1 in modes_filtered:
                control.Show()
            else:
                control.Hide()

    def OnFilterModes(self, evt):
        self.FilterModes()
        self.SetupScrolling()

    def OnMode(self, evt):
        self.RefreshModes()

    def OnReset(self, evt):
        self.Reset()

    def OnClose(self):
        # Unsubscribe from pubsub events
        events.unsubscribe(PUBSUB_SET_PHASE, self.HandleSetPhase)
        
    def GetModes(self):
        modes = []
        for mode_control in self._mode_controls:
            modes.append(mode_control.value)
        
        return modes

    def SetModes(self, modes):
        # Update each mode
        for i, value in enumerate(modes):
            mode_control = self._mode_controls[i]
            if value != mode_control.value and not mode_control.focus:
                mode_control.SetValue(value, quiet=True)
                mode_control.UpdateValueRanges()
    
    def RefreshModes(self):
        modes = self.GetModes()
        self._device.set_correction("mode control", modes)
        self._device.refresh_corrections()

    def Reset(self, quiet=False):
        for mode_control in self._mode_controls:
            mode_control.SetValue(0, quiet=quiet)

    def HandleSetPhase(self):
        corrections = self._device.get_corrections(include_default=True)
        if corrections["mode control"]["enabled"]:
            modes = np.zeros(self._n_modes) + sum([
                np.array(correction["modes"])
                for correction in corrections.values()
                if correction["enabled"] and correction["modes"] is not None
            ])
            self.SetModes(modes)


class ModesControl(wx.Frame):
    def __init__(self, parent, device):
        super().__init__(parent)
        self._panel = _ModesPanel(self, device)
        self._sizer = wx.BoxSizer(wx.VERTICAL)
        self._sizer.Add(self._panel, 1, wx.EXPAND)
        self.SetSizer(self._sizer)
        self.SetMinSize(wx.Size(650, 300))
        self.SetSize(wx.Size(650, 650))
        self.SetTitle('DM mode control')
        self.Bind(wx.EVT_CLOSE, self.OnClose)
    
    def OnClose(self, evt):
        self._panel.OnClose()
        evt.Skip()