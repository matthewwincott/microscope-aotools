from cockpit import events

import wx
import wx.lib.newevent

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import numpy as np

from microAO.events import *


class SensorlessResultsViewer(wx.Frame):
    def __init__(self, parent, data, **kwargs):
        super().__init__(parent, title="Metric viewer")
        root_panel = wx.Panel(self)


        figure = Figure()

        self.ax = figure.add_subplot(1, 1, 1)

        self.data = data
        
        if self.data is not None:
            self.update()

        self.canvas = FigureCanvas(root_panel, wx.ID_ANY, figure)

        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(self.canvas, wx.SizerFlags(1).Expand())
        root_panel.SetSizer(panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root_panel, wx.SizerFlags().Expand())
        self.SetSizerAndFit(frame_sizer)

        # Subscribe to pubsub events
        events.subscribe(PUBSUB_SENSORLESS_RESULTS, self.HandleSensorlessData)
        

    def update(self):
        # Calclate required data
        n_images = len(self.data['metric_stack'])
        n_z_steps = len(self.data['z_steps'])
        nollZernike = self.data['nollZernike']
        n_modes = n_images // n_z_steps

        if n_images < 1:
            return

        # Compute mode boundaries and labels
        mode_boundaries = np.arange(0, n_images, n_z_steps)
        vlines = list(mode_boundaries[1:] - 0.5)
        xticks = mode_boundaries + (n_z_steps - 1) / 2 
        xticklabels = ['z-{}'.format(z) for z in nollZernike[0:n_modes] ]

        # Set up x,y data
        x = np.arange(0,n_images)
        y = self.data['metric_stack']

        # Plot data
        self.ax.clear()
        self.ax.plot(x,y, '-x')

        for x_pos in vlines:
            self.ax.axvline(x_pos, color='gray')

        self.ax.xaxis.set_ticks(xticks)
        
        self.ax.xaxis.set_ticklabels(xticklabels)

        self.ax.set_xlim(min(x)-0.5, max(x)+0.5)

        self.ax.set_xlabel('Mode')
        self.ax.set_ylabel('Metric value')
        self.ax.set_title('Metric vs iteration (grouped by mode)')

        self.canvas.draw()

    def set_data(self, data):
        self.data = data
        self.update()
    
    def HandleSensorlessData(self, data):
        self.set_data(data)