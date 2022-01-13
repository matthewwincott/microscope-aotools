from cockpit import events

import wx
import wx.grid
import wx.lib.newevent

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import numpy as np

from microAO.events import *


class SensorlessResultsViewer(wx.Frame):
    def __init__(self, parent, data, **kwargs):
        super().__init__(parent, title="Metric viewer")
        root_panel = wx.Panel(self)

        self.data = data
        
        if self.data is not None:
            self.update()

        tabs = wx.Notebook(root_panel, size=(-1,-1))

        metric_panel = wx.Panel(tabs)
        figure = Figure()
        self.ax = figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(metric_panel, wx.ID_ANY, figure)

        metric_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        metric_panel_sizer.Add(self.canvas, wx.SizerFlags(1).Expand())
        metric_panel.SetSizer(metric_panel_sizer)

        # Correction panel
        corrections_panel = wx.Panel(tabs)    
        self.corrections_grid = wx.grid.Grid(corrections_panel, -1)
        self.corrections_grid.CreateGrid(1, 1)

        correction_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        correction_panel_sizer.Add(self.corrections_grid, wx.SizerFlags(1).Expand())
        corrections_panel.SetSizer(correction_panel_sizer)

        tabs.AddPage(metric_panel, "Metric plot")
        tabs.AddPage(corrections_panel, "Corrections")
        tabs.Layout()

        root_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        root_panel_sizer.Add(tabs)
        root_panel.SetSizerAndFit(root_panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root_panel, wx.SizerFlags().Expand())
        self.SetSizerAndFit(frame_sizer)

        # Subscribe to pubsub events
        events.subscribe(PUBSUB_SENSORLESS_RESULTS, self.HandleSensorlessData)

        # Bind to close event
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        

    def update(self):
        # Calclate required data
        n_images = len(self.data['metric_stack'])
        n_z_steps = len(self.data['z_steps'])
        nollZernike = self.data['nollZernike']
        iterations = self.data['iterations']
        n_modes = len(nollZernike)

        if n_images < 1:
            return

        # Compute mode boundaries and labels
        mode_boundaries = np.arange(0, n_images, n_z_steps)
        vlines = list(mode_boundaries[1:] - 0.5)
        xticks = mode_boundaries + (n_z_steps - 1) / 2 
        xticklabels = ['z-{}'.format(z) for z in nollZernike] * iterations
        xticklabels = xticklabels[0:len(xticks)]

        # Set up x,y data
        x = np.arange(0,n_images)
        y = self.data['metric_stack']

        # Plot metric data
        self.ax.clear()
        self.ax.plot(x,y, '-x')

        # Plot mode boundaries, ticks and labels
        for x_pos in vlines:
            self.ax.axvline(x_pos, color='gray')

        self.ax.xaxis.set_ticks(xticks)        
        self.ax.xaxis.set_ticklabels(xticklabels)

        # Display computed corrections
        correction_stack = self.data["correction_stack"]
        n_corrections = len(correction_stack)

        rows = self.corrections_grid.GetNumberRows()
        cols = self.corrections_grid.GetNumberCols()
        
        # Calculate length of one AO iteration
        len_iteration = n_modes * n_z_steps

        # Populate values if AO iteration is complete
        if n_corrections % len_iteration == 0:
            # Get current correction from stack
            curr_correction = self.data["correction_stack"][-1]

            # Calculate number of complete AO iterations completed
            complete_iterations = n_corrections // len_iteration
            
            # Set column label
            self.corrections_grid.SetColLabelValue(complete_iterations - 1, "Iteration{}".format(complete_iterations))
            
            # Add rows/columns as required
            if (complete_iterations - cols) > 0:
                self.corrections_grid.AppendCols(complete_iterations - cols)
            if (n_modes - rows) > 0:
                self.corrections_grid.AppendRows(n_modes - rows)

            # Set row labels
            for i, z in enumerate(nollZernike):
                self.corrections_grid.SetRowLabelValue(i, "z{}".format(z))

            # Populate values
            values = ["{:.2f}".format(curr_correction[ind - 1]) for ind in nollZernike]
            for i, value in enumerate(values):
                self.corrections_grid.SetCellValue(i, complete_iterations - 1, value)

        # Set x-axis limits
        self.ax.set_xlim(min(x)-0.5, max(x)+0.5)

        # Set labels
        self.ax.set_xlabel('Mode')
        self.ax.set_ylabel('Metric value')
        self.ax.set_title('Metric vs iteration (grouped by mode)')

        # Refresh canvas
        self.canvas.draw()

    def set_data(self, data):
        self.data = data
        self.update()
    
    def HandleSensorlessData(self, data):
        self.set_data(data)
    
    def OnClose(self, evt):
        # Unsubscribe from pubsub events
        events.unsubscribe(PUBSUB_SENSORLESS_RESULTS, self.HandleSensorlessData)

        # Continue + destroy frame
        evt.Skip()