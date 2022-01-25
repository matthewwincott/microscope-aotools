#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2021 Matthew Wincott <matthew.wincott@eng.ox.ac.uk>
##
## microAO is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## microAO is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with microAO.  If not, see <http://www.gnu.org/licenses/>.

import os

from cockpit import events
import cockpit.gui.dialogs

import wx
from wx.core import DirDialog
import wx.lib.newevent

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import numpy as np
import scipy
import imageio

from microAO.events import *
from microAO.gui.common import EVT_VALUE, FloatCtrl, FilterModesCtrl, MinMaxSliderCtrl

RF_DATATYPES = ["zernike", "actuator"]

class RFDatatypeChoice(wx.Choice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, choices=RF_DATATYPES, **kwargs)
        
        # Select first item by default
        self.SetSelection(0)
    
class RFAddDatapointFromFile(wx.Dialog):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, title="Add position from file")
        
        self.data = None

        sizer_dialog = wx.BoxSizer(wx.VERTICAL)

        panel = wx.Panel(self)
        sizer_panel = wx.BoxSizer(wx.VERTICAL)

        self.datatype_label = wx.StaticText(panel, label="type")
        self.datatype = RFDatatypeChoice(panel)
        self.zpos_label = wx.StaticText(panel, label="z (μm)")
        self.zpos = FloatCtrl(panel, value="0")
        self.fileCtrl = wx.FilePickerCtrl(panel, size=(400, 50))

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.datatype_label, wx.SizerFlags().Centre())
        hbox.Add(self.datatype, wx.SizerFlags().Centre())
        sizer_panel.Add(hbox)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.zpos_label, wx.SizerFlags().Centre())
        hbox.Add(self.zpos, wx.SizerFlags().Centre())
        sizer_panel.Add(hbox)
        sizer_panel.Add(self.fileCtrl)

        panel.SetSizer(sizer_panel)

        sizer_buttons = wx.BoxSizer(wx.HORIZONTAL)
        confirmBtn = wx.Button(self, label='Ok')
        cancelBtn = wx.Button(self, label='Close')
        sizer_buttons.Add(confirmBtn)
        sizer_buttons.Add(cancelBtn)
        
        # Bind events
        confirmBtn.Bind(wx.EVT_BUTTON, self.OnConfirm)
        cancelBtn.Bind(wx.EVT_BUTTON, self.OnCancel)

        sizer_dialog.Add(panel, flags=wx.SizerFlags().Border(wx.ALL, 10))
        sizer_dialog.Add(sizer_buttons, wx.SizerFlags().Centre().Border(wx.ALL, 10))

        self.SetSizerAndFit(sizer_dialog)

        self.Centre()

    def OnConfirm(self, e):
        zpos = self.zpos.value
        fpath = self.fileCtrl.GetPath()
        values = np.loadtxt(fpath)
        datatype = self.datatype.GetString(self.datatype.GetSelection())

        self.data = {
            'z': zpos,
            'datatype': datatype.lower(),
            'values': values
        }

        self.EndModal(wx.ID_OK)

    def OnCancel(self, e):
        self.EndModal(wx.ID_CANCEL)
    
    def GetData(self):
        return self.data

class RFAddDatapointFromCurrent(wx.Dialog):
    def __init__(self, parent, device, **kwargs):
        super().__init__(parent, title="Add position from file")
        
        self.data = None

        self._device = device

        sizer_dialog = wx.BoxSizer(wx.VERTICAL)

        panel = wx.Panel(self)
        sizer_panel = wx.BoxSizer(wx.VERTICAL)

        self.datatype_label = wx.StaticText(panel, label="type")
        self.datatype = RFDatatypeChoice(panel)
        self.zpos_label = wx.StaticText(panel, label="z (μm)")
        self.zpos = FloatCtrl(panel, value="0")

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.zpos_label, wx.SizerFlags().Centre())
        hbox.Add(self.zpos, wx.SizerFlags().Centre())
        sizer_panel.Add(hbox)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.datatype_label, wx.SizerFlags().Centre())
        hbox.Add(self.datatype, wx.SizerFlags().Centre())
        sizer_panel.Add(hbox)

        panel.SetSizer(sizer_panel)

        sizer_buttons = wx.BoxSizer(wx.HORIZONTAL)
        confirmBtn = wx.Button(self, label='Ok')
        cancelBtn = wx.Button(self, label='Close')
        sizer_buttons.Add(confirmBtn)
        sizer_buttons.Add(cancelBtn)
        
        # Bind events
        confirmBtn.Bind(wx.EVT_BUTTON, self.OnConfirm)
        cancelBtn.Bind(wx.EVT_BUTTON, self.OnCancel)

        sizer_dialog.Add(panel, flags=wx.SizerFlags().Border(wx.ALL, 10))
        sizer_dialog.Add(sizer_buttons, wx.SizerFlags().Centre().Border(wx.ALL, 10))

        self.SetSizerAndFit(sizer_dialog)

        self.Centre()

    def OnConfirm(self, e):
        zpos = self.zpos.value
        datatype = self.datatype.GetString(self.datatype.GetSelection()).lower()        

        if datatype == "zernike":
            values = self._device.proxy.get_last_modes()
        elif datatype == "actuator":
            values = self._device.proxy.get_last_actuator_values()
        else:
            values = None

        assert values is not None

        self.data = {
            'z': zpos,
            'datatype': datatype,
            'values': values
        }

        self.EndModal(wx.ID_OK)

    def OnCancel(self, e):
        self.EndModal(wx.ID_CANCEL)
    
    def GetData(self):
        return self.data


class RemoteFocusControl(wx.Frame):
    def __init__(self, parent, device, **kwargs):
        super().__init__(parent, title="Remote focus")

        # Set attributes
        self._device = device
        control_matrix = self._device.proxy.get_controlMatrix()
        self._n_modes = control_matrix.shape[1]
        self._n_actuators = control_matrix.shape[0]

        self.z_target = 0        
        
        # Subscribe to pubsub events
      
        root_panel = wx.Panel(self)

        # Create tabbed control interface
        tabs = wx.Notebook(root_panel, size=(-1,-1))

        # Main data panel
        data_panel = wx.Panel(tabs, size=(-1,-1))
        self.listbox = wx.ListBox(data_panel, size=(300,200))

        # Button side panel
        data_panel_btns = wx.Panel(data_panel)
        addFromCurrentBtn = wx.Button(data_panel_btns, wx.ID_ANY, 'Add from current', size=(140, 30))
        addFromFileBtn = wx.Button(data_panel_btns, wx.ID_ANY, 'Add from file', size=(140, 30))
        removeBtn = wx.Button(data_panel_btns, wx.ID_ANY, 'Remove selected', size=(140, 30))
        saveBtn = wx.Button(data_panel_btns, wx.ID_ANY, 'Save data', size=(140, 30))
        loadBtn = wx.Button(data_panel_btns, wx.ID_ANY, 'Load data', size=(140, 30))
        calibrateBtn = wx.Button(data_panel_btns, wx.ID_ANY, 'Calibrate', size=(140, 30))

        addFromCurrentBtn.Bind(wx.EVT_BUTTON, self.OnAddDatapointFromCurrent)
        addFromFileBtn.Bind(wx.EVT_BUTTON, self.OnAddDatapointFromFile)
        removeBtn.Bind(wx.EVT_BUTTON, self.OnRemoveDatapoint)
        saveBtn.Bind(wx.EVT_BUTTON, self.OnSaveDatapoints)
        loadBtn.Bind(wx.EVT_BUTTON, self.OnLoadDatapoints)
        calibrateBtn.Bind(wx.EVT_BUTTON, self.OnCalibrate)
        
        data_panel_btns_sizer = wx.BoxSizer(wx.VERTICAL)
        data_panel_btns_sizer.Add(addFromFileBtn)
        data_panel_btns_sizer.Add(addFromCurrentBtn)
        data_panel_btns_sizer.Add(removeBtn)
        data_panel_btns_sizer.Add(-1, 10)
        data_panel_btns_sizer.Add(saveBtn)
        data_panel_btns_sizer.Add(loadBtn)
        data_panel_btns_sizer.Add(-1, 10)
        data_panel_btns_sizer.Add(calibrateBtn)
        data_panel_btns.SetSizerAndFit(data_panel_btns_sizer)

        # Layout data panel
        data_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(data_panel_btns, wx.SizerFlags())
        hbox.Add(self.listbox, wx.SizerFlags())
        data_panel_sizer.Add(hbox, wx.SizerFlags().Centre())
        data_panel.SetSizer(data_panel_sizer)

        # Control panel
        control_panel = wx.Panel(tabs)

        # Mode slider: drag to set mode
        row_sizer = wx.BoxSizer(wx.HORIZONTAL)
        # default_range = 1.5   # range of slider
        self.remotezSlider = MinMaxSliderCtrl(control_panel, value=0)
        self.remotezSlider.Bind(EVT_VALUE, self.OnRemoteZ)
        row_sizer.Add(self.remotezSlider)
        
        # Layout
        self.datatype_control = RFDatatypeChoice(control_panel)
        self.datatype_control.Bind(wx.EVT_CHOICE, self.OnDatatypeChange)

        control_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        control_panel_sizer.Add(self.datatype_control)
        control_panel_sizer.Add(row_sizer)

        control_panel.SetSizerAndFit(control_panel_sizer)

        # Experiment panel
        experiment_panel = wx.Panel(tabs)

        btn_remotez_stack = wx.Button(experiment_panel, label="Remote z-stack")
        btn_remotez_stack.Bind(wx.EVT_BUTTON, self.OnRemoteZStack)
        
        # Layout

        experiment_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        experiment_panel_sizer.Add(btn_remotez_stack)

        experiment_panel.SetSizerAndFit(experiment_panel_sizer)

        # Visualisation panel
        vis_panel = wx.Panel(root_panel)

        self.modes = FilterModesCtrl(vis_panel, value="1-12")
        self.datatype_vis = RFDatatypeChoice(vis_panel)
        figure = Figure()
        self.ax = figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(vis_panel, wx.ID_ANY, figure)

        self.datatype_vis.Bind(wx.EVT_CHOICE, self.OnDatatypeChange)
        self.modes.Bind(wx.EVT_TEXT, self.OnDatatypeChange)

        vis_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        vis_panel_sizer.Add(self.datatype_vis)
        vis_panel_sizer.Add(self.modes)
        vis_panel_sizer.Add(self.canvas)

        vis_panel.SetSizerAndFit(vis_panel_sizer)
        
        # Add pages to tabs
        tabs.AddPage(data_panel,"Data") 
        tabs.AddPage(control_panel,"Control") 
        tabs.AddPage(experiment_panel,"Experiments") 

        tabs.Layout()

        # Layout root panel
        root_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        root_panel_sizer.Add(tabs, wx.SizerFlags(1).Centre().Border(wx.BOTTOM, 12))
        # root_panel_sizer.Add(data_panel, wx.SizerFlags(1).Centre().Border(wx.BOTTOM, 12))
        root_panel_sizer.Add(vis_panel, wx.SizerFlags(1).Centre())
        root_panel.SetSizer(root_panel_sizer)

        # Main frame sizer
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root_panel, wx.SizerFlags().Centre().Border(wx.ALL, 20))
        self.SetSizerAndFit(frame_sizer)


    def addDatapont(self, datapoint):
        self._device.remotez.add_datapoint(datapoint)
        
        self.updateDatapointList()
        self.update()

    def removeDatapoint(self, datapoint):
        self._device.remotez.remove_datapoint(datapoint)
        
        self.updateDatapointList()
        self.update()

    def updateDatapointList(self):
        self.listbox.Clear()
        for d in self._device.remotez.datapoints:
            self.listbox.AppendItems("{} ({})".format(d["z"],d["datatype"]))

    def OnRemoteZ(self, e):
        self.z_target = self.remotezSlider.GetValue()

        mode = self.datatype_control.GetStringSelection().lower()
        self._device.remotez.set_z(self.z_target, mode)

        self.update_zpos()

    def OnRemoteZStack(self, e):
        # Get parameters
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Get remote Z stack parameters",
            [
                "z min",
                "z max",
                "z step size",
            ],
            (
                0,
                0,
                1,

            ),
        )

        zmin = float(inputs[0])
        zmax = float(inputs[1])
        zstepssize = float(inputs[2])

        # Select output folder
        dlg = DirDialog(None, "Select data output directory")

        if dlg.ShowModal() == wx.ID_OK:
            output_dir = dlg.GetPath()
            print(output_dir)
        else:
            return

        # Perform z stack
        images = self._device.remotez.zstack(zmin, zmax, zstepssize)

        # Save data
        for i,image in enumerate(images):
            fname = "{}to{}-{:03}".format(zmin, zmax, i).replace('.','_')+ ".tif"
            fpath = os.path.join(output_dir, fname)
            imageio.imwrite(fpath, image, format='tif')


    def OnCalibrate(self, e):
        # Get parameters
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Get remote Z stack parameters",
            [
                "z min",
                "z max",
                "z step",
            ],
            (
                0,
                5,
                2,

            ),
        )

        zmin = float(inputs[0])
        zmax = float(inputs[1])
        zsteps = int(inputs[2])

        zpos = np.linspace(zmin, zmax, zsteps)

        zstage = self._device.getStage()

        self._device.remotez.calibrate(zstage, zpos)

    def OnAddDatapointFromFile(self, e):
        dlg = RFAddDatapointFromFile(self)
        
        if dlg.ShowModal() == wx.ID_OK:
            datapoint = dlg.GetData()
            self.addDatapont(datapoint)
        else:
            pass
        
        dlg.Destroy()
    
        self.update()

    def OnAddDatapointFromCurrent(self, e):
        dlg = RFAddDatapointFromCurrent(self, self._device)
        
        if dlg.ShowModal() == wx.ID_OK:
            datapoint = dlg.GetData()
            self.addDatapont(datapoint)
        else:
            pass

        dlg.Destroy()

        self.update()

    def OnRemoveDatapoint(self, e):
        # Get selected datapoints and remove
        selected = self.listbox.GetSelections()

        for i in selected:
            datapoint = self._device.remotez.datapoints[i]
            self.removeDatapoint(datapoint)
        
        # Update GUI
        self.update()

    def OnSaveDatapoints(self, e):
        # Select output folder
        dlg = wx.DirDialog(self, "Select data output directory")

        if dlg.ShowModal() == wx.ID_OK:
            output_dir = dlg.GetPath()
        else:
            return None
        
        # Save datapoints
        self._device.remotez.save_datapoints(output_dir)

    def OnLoadDatapoints(self, e):
        # Select input directory
        dlg = wx.DirDialog(None, "Select data output directory")
        
        if dlg.ShowModal() == wx.ID_OK:
            input_dir = dlg.GetPath()
        else:
            return None

        # Load datapoints
        self._device.remotez.load_datapoints(input_dir)

        # Update gui
        self.updateDatapointList()
        self.update()

    def OnDatatypeChange(self, e):
        for control in [self.datatype_vis, self.datatype_control]:
            evt_emitter = e.GetEventObject()

            # Change other datatype controls
            if control is not evt_emitter:
                control.SetStringSelection(evt_emitter.GetStringSelection())
        
        # Update GUI
        self.update()

    def update(self):
        # Plot data
        self.ax.clear()

        # Get data
        current_datatype = self.datatype_vis.GetStringSelection().lower()
        points = [a for a in self._device.remotez.datapoints if a["datatype"].lower() == current_datatype]
        z = np.array([point["z"] for point in points])
        values = np.array([point["values"] for point in points])
        z_lookup = self._device.remotez.z_lookup[current_datatype]
        
        try:
            n_measurements = values.shape[0]
        except IndexError:
            n_measurements = 0

        # Continue of more than one value
        if n_measurements > 1:
            # Plot modes and regression
            # - filter visible modes
            filter_modes = [mode-1 for mode in self.modes.GetValue() if (mode-1) < len(z_lookup)]
            
            for mode in filter_modes:
                self.ax.scatter(z, values[:,mode])
                
                x1=np.linspace(np.min(z),np.max(z),500)
                self.ax.plot(z, z_lookup[mode](z))

            # Plot current z_target
            self.ax_z_target = self.ax.axvline(self.z_target)

        self.canvas.draw()

    def update_zpos(self):
        try:
            self.ax_z_target.set_xdata([self.z_target, self.z_target])
            self.canvas.draw()
        except AttributeError:
            # No vline yet, skip
            pass
