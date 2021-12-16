import wx

from cockpit.gui.guiUtils import FLOATVALIDATOR

class FloatCtrl(wx.TextCtrl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, validator=FLOATVALIDATOR, **kwargs)

    @property
    def value(self):
        try:
            val = float(self.GetValue())
        except Exception as e:
            val = None

        return val
ValueChangeEvent, EVT_VALUE = wx.lib.newevent.NewEvent()
class MinMaxSliderCtrl(wx.Panel):
    """Slider control component."""

    def __init__(self, parent, id=None, value=0):
        super().__init__(parent)

        # Store attributes
        self.id = id
        self.value = value

        # Store focus state
        self.focus = False

        row_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Main slider 
        default_range = 1   # default range of slider
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
        self._val.Bind(wx.EVT_SPINCTRLDOUBLE, self.OnValueChange)
        self._val.Bind(wx.EVT_SET_FOCUS, self.OnModeGetFocus)
        self._val.Bind(wx.EVT_KILL_FOCUS, self.OnModeLoseFocus)

        # Layout
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

    def OnValueChange(self, evt):
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
            evt = ValueChangeEvent(mode=self.id, value= self.value)
            wx.PostEvent(self, evt)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
