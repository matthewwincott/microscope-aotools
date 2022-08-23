import os

import wx
import numpy as np
from matplotlib import cm
from matplotlib.colors import rgb2hex, Normalize
import json

from cockpit import events
import microAO
import microAO.dm_layouts
from microAO.events import *

class _DMView(wx.Panel):
    def __init__(self, parent, actuators=[], actuator_shape=None, actuator_scale=None, *args, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Attributes
        self.actuators = actuators
        self.actuator_shape = actuator_shape
        self.actuator_scale = actuator_scale
        self.values = np.zeros(len(actuators))
        self.min = -1
        self.max = 1

        self.cmap = cm.get_cmap('jet')
        self.autoscale = True

        # Set rectangular actuator shape, if not set
        if actuator_shape is None:
            self.actuator_shape = [(-1,-1), (-1,1), (1,1), (1,-1)]

        # Bind paint event
        self.Bind(wx.EVT_PAINT, self.OnPaint)

        self.Centre()

    def SetActuator(self, i, val):
        """ Set single actuator value """
        self.values[i] = val

    def SetActuators(self, vals):
        """ Set all actuator values """
        assert len(vals) == len(self.values)

        # Set values on instance
        self.values = vals * 2 - 1
        
        # Refresh
        self.Refresh()

    def SetAutoscale(self, autoscale):
        self.autoscale = autoscale
        
        self.Refresh()

    def SetScale(self, range):
        self.min = range[0]
        self.max = range[1]

        self.Refresh()

    def OnPaint(self, e):
        # Create graphics canvas and dc
        dc = wx.PaintDC(self)
        gcdc = wx.GCDC(dc)
        gcdc.Clear()
        gc = gcdc.GetGraphicsContext()

        # Set scaling (from 0-1)
        size = gcdc.GetSize()
        scale = min(size)

        # Normalise actuators
        if self.autoscale:
            vmin = None
            vmax = None

            # values_clip = self.values
        else:
            vmin = self.min
            vmax = self.max

            # values_clip = np.clip(self.values)

        values_norm = Normalize(vmin=vmin, vmax=vmax)(self.values)

        # Draw actuators
        for i, actuator in enumerate(self.actuators):
            # Create filled path
            path = gc.CreatePath()

            # Set colour
            c = rgb2hex(self.cmap(values_norm[i]))
            gc.SetPen(wx.Pen(c))
            gc.SetBrush(wx.Brush(c))

            # Define actuator shape coordinates
            offset = ((actuator[0]+1)/2*scale, (actuator[1]+1)/2*scale)
            coords=[(s[0]*self.actuator_scale/2*scale, s[1]*self.actuator_scale/2*scale) for s in self.actuator_shape]

            # Draw closed shape
            path.MoveToPoint(offset[0]+coords[0][0],offset[1]+coords[0][1])
            path_start = path.GetCurrentPoint()
            for p in coords:
                path.AddLineToPoint(p[0]+offset[0], p[1]+offset[1])
            path.AddLineToPoint(path_start)
            path.CloseSubpath()

            gc.DrawPath(path)

class _ColourBar(wx.Panel):
    def __init__(self, parent, min=0, max=1, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Attributes
        self.min = min
        self.max = max
        self.cmap = cm.get_cmap('jet')

        # Bind paint event
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)

        self.Centre()
    
    def SetMin(self, val):
        self.min = val

    def SetMax(self, val):
        self.max = val

    def SetScale(self, scale):
        self.min = scale[0]
        self.max = scale[1]
        self.Refresh()

    def OnPaint(self, e):
            # Create graphics canvas and dc
            dc = wx.PaintDC(self)
            gcdc = wx.GCDC(dc)
            gcdc.Clear()
            gc = gcdc.GetGraphicsContext()

            # Get size of canvas for calculations
            canvas_size = gcdc.GetSize()

            # Draw text labels
            text_min = '{:.02f}'.format(self.min)
            text_min_extent = self.GetFullTextExtent(text_min)
            
            gc.DrawText(text_min, canvas_size[0]/2 - text_min_extent[0]/2, canvas_size[1]-text_min_extent[1])

            text_max = '{:.02f}'.format(self.max)
            text_max_extent = self.GetFullTextExtent(text_max)
            gc.DrawText(text_max, canvas_size[0]/2 - text_max_extent[0]/2, 0)

            # Get scale values
            colourbar_height = canvas_size[1] - (text_min_extent[1] + text_max_extent[1])
            values = np.linspace(self.max, self.min, colourbar_height)
            values_norm = Normalize(vmin=self.min, vmax=self.max)(values)

            # Draw colourbar
            for i,val in enumerate(values_norm):
                # Set colour
                c = rgb2hex(self.cmap(val))
                gc.SetPen(wx.Pen(c))
                gc.SetBrush(wx.Brush(c))

                # Draw bar
                gc.DrawRectangle(0,text_max_extent[1] + i,canvas_size[0],1)

    def OnSize(self, evt):
        self.Refresh()


class DMViewer(wx.Frame):
    def __init__(self, parent, device, dm_layout_name=None, actuator_scale=None, *args, **kwargs):
        super().__init__(parent, title="DM viewer")
        self._panel = wx.Panel(self, *args, **kwargs)

        # Store reference to the DM
        self._device = device

        # Set up DM layout
        dm_layout = microAO.dm_layouts.get_layout(dm_layout_name)
        actuators = dm_layout['locations']
        actuator_scale = dm_layout['scale_shapes']

        # Set other attributes
        self.actuator_values = np.zeros(len(actuators))

        # Create widgets
        self._dm_view = _DMView(self._panel, actuators, actuator_scale=actuator_scale, size=(200,200))
        self._dm_colourbar = _ColourBar(self._panel, size=(50,-1))

        self._autoscale_btn = wx.ToggleButton(self._panel, label="Autoscale", size=wx.Size(200,-1))

        self._middle_label = wx.StaticText(self._panel, label="centre", size=wx.Size(80,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
        self._middle = wx.SpinCtrlDouble(self._panel, initial=0, inc=0.01, min=-1, max=1, size=wx.Size(120,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))

        self._range_label = wx.StaticText(self._panel, label="range", size=wx.Size(80,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))
        self._range = wx.SpinCtrlDouble(self._panel, initial=1, inc=0.01, min=0, max=1, size=wx.Size(120,-1), style=(wx.ALIGN_CENTRE_HORIZONTAL|wx.ST_NO_AUTORESIZE))

        # Layout widgets
        panel_sizer = wx.BoxSizer(wx.VERTICAL)

        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self._dm_view, wx.SizerFlags(1).Expand())
        row.Add(self._dm_colourbar, wx.SizerFlags().Expand().Border(wx.ALL, 10))
        panel_sizer.Add(row, wx.SizerFlags(1).Expand())

        block = wx.BoxSizer(wx.VERTICAL)

        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self._autoscale_btn, wx.SizerFlags().Centre())
        block.Add(row, wx.SizerFlags().Centre())
        
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self._middle_label, wx.SizerFlags().Centre())
        row.Add(self._middle, wx.SizerFlags().Centre())
        block.Add(row, wx.SizerFlags().Centre())
        
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self._range_label, wx.SizerFlags().Centre())
        row.Add(self._range, wx.SizerFlags().Centre())
        block.Add(row, wx.SizerFlags().Centre())

        panel_sizer.Add(block, wx.SizerFlags().Centre().Border(wx.ALL, 10))

        self._panel.SetSizer(panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self._panel, wx.SizerFlags(1).Expand())
        self.SetSizerAndFit(frame_sizer)

        # Bind handlers
        self._autoscale_btn.Bind(wx.EVT_TOGGLEBUTTON, self.OnAutoscale)
        self._middle.Bind(wx.EVT_SPINCTRLDOUBLE, self.OnScale)
        self._range.Bind(wx.EVT_SPINCTRLDOUBLE, self.OnScale)

        # Subscribe to pubsub events
        events.subscribe(PUBSUB_SET_ACTUATORS, self.HandleActuators)

        # Update viewer
        self._timer = wx.Timer(self._panel)
        self._timer.Start(1000)
        self._panel.Bind(wx.EVT_TIMER, self.RefreshValuesFromDevice, self._timer)

        # Force necessary updates
        self.OnScale(None)

        # Bind close event
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def SetActuators(self, actuator_values):
        if actuator_values is None:
            return

        self.actuator_values = actuator_values

        self._dm_view.SetActuators(actuator_values)

        if self._autoscale_btn.GetValue():
            colourbar_min = min(actuator_values) * 2 - 1
            colourbar_max = max(actuator_values) * 2 - 1
            self._dm_colourbar.SetScale((colourbar_min, colourbar_max))

    def HandleActuators(self, actuator_values):
        self.SetActuators(actuator_values)

    def RefreshValuesFromDevice(self, evt):
        values = self._device.proxy.get_last_actuator_values()
        self.SetActuators(values)

    def OnAutoscale(self, e):
        autoscale = self._autoscale_btn.GetValue()
        self._dm_view.SetAutoscale(autoscale)

        if autoscale:
            colourbar_min = min(self.actuator_values) * 2 - 1
            colourbar_max = max(self.actuator_values) * 2 - 1
            self._dm_colourbar.SetScale((colourbar_min, colourbar_max))

            self._range.Disable()
            self._middle.Disable()
        else:
            self._range.Enable()
            self._middle.Enable()

            # Force scale update
            self.OnScale(None)

    def OnScale(self, e):
        middle = self._middle.GetValue()
        range = self._range.GetValue()
        scale = (middle-range, middle+range)
        
        self._dm_view.SetScale(scale)
        self._dm_colourbar.SetScale(scale)
    
    def OnClose(self, e):
        # Unsubscribe from events
        events.unsubscribe(PUBSUB_SET_ACTUATORS, self.HandleActuators)
        
        # Continue + destroy frame
        e.Skip()