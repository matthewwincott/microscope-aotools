import re
import wx
import numpy
import matplotlib.cm

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

class FilterModesCtrl(wx.TextCtrl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def GetValue(self):
        # Get values as list
        modes_str = super().GetValue()
        modes_list = re.split(',|;',modes_str)

        # Set up patterns
        regex_range = re.compile("^(\d+)-(\d+)$")
        regex_number = re.compile("^\d+$")

        # Check string against pattern and add to modes list if a match
        modes = []
        for mode in modes_list:
            if regex_range.match(mode):
                parts = mode.split("-")
                start = int(parts[0])
                end = int(parts[1]) + 1
                modes = modes + list(range(start, end))
            
            if regex_number.match(mode):
                modes = modes + [int(mode)]

        return modes

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
        self._slider.SetValue(int(slider_val))

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

class ModeIndicator(wx.Panel):
    """A simple widget used to indicate the value of a mode.

    Args:
        parent: parent widget.
        size_hints: (width, height) tuple that specifies the preferred size of
            the widget. Zero and negative values lead to using the widget's
            client size.
        norm_range: normalisation range of the mode value; (-1, 1) for values
            which negative numbers and (0, 1) otherwise.
        cmap_name: name of the colour map; check matplotlib for valid names

    """

    def __init__(
        self,
        parent,
        size_hints=(-1, -1),
        norm_range=(-1, 1),
        cmap_name="coolwarm",
        *args,
        **kwargs
    ):
        super().__init__(parent, *args, **kwargs)
        self._size_hints = size_hints
        self._norm_range = norm_range
        self._value_norm = 0
        self._cmap = matplotlib.cm.get_cmap(cmap_name)
        self._colour = wx.Colour(
            (numpy.array(self._cmap(0.5)) * 255).astype(int)
        )
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, self._on_size)
        self.SetMinSize(self._size_hints)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

    def _on_paint(self, _):
        dc = wx.BufferedPaintDC(self)
        dc.Clear()
        # Get parameters
        width, height = self.GetClientSize()
        x_offset = 0
        y_offset = 0
        if self._size_hints[0] > 0:
            x_offset = width - self._size_hints[0]
            width = self._size_hints[0]
        if self._size_hints[1] > 0:
            y_offset = (height - self._size_hints[1]) // 2
            height = self._size_hints[1]
        rect_width = round(0.025 * width)  # 2.5% of total width
        # Draw middle line
        dc.SetBrush(wx.LIGHT_GREY_BRUSH)
        dc.SetPen(wx.LIGHT_GREY_PEN)
        dc.DrawLine(
            x_offset,
            y_offset + height // 2,
            x_offset + width,
            y_offset + height // 2
        )
        # Draw rectangle
        dc.SetBrush(wx.Brush(self._colour))
        dc.SetPen(wx.Pen(self._colour))
        dc.DrawRectangle(
            x_offset + round(
                numpy.interp(self._value_norm, self._norm_range, (0, width)) -
                rect_width / 2.0
            ),
            y_offset,
            rect_width,
            height
        )
        return

    def _on_size(self, _):
        self.Refresh()

    def Update(self, value, amplitude):
        # Normalise the value
        self._value_norm = numpy.interp(value, (-amplitude, amplitude), self._norm_range)
        # Interpolate the value to the colormap range of [0; 1]
        self._colour = wx.Colour(
            (numpy.array(self._cmap(numpy.interp(value, (-amplitude, amplitude), (0, 1)))) * 255).astype(int)
        )
        self.Refresh()