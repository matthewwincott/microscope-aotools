from cockpit import events

import wx
import wx.lib.newevent

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import numpy as np
import scipy

from microAO.events import *
from microAO.gui.common import EVT_VALUE, FloatCtrl, FilterModesCtrl, MinMaxSliderCtrl

RF_DATATYPES = ["Zernike", "actuator"]

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
            'datatype': datatype,
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
        datatype = self.datatype.GetString(self.datatype.GetSelection())        

        if datatype.lower() == "zernike":
            values = self._device.proxy.get_last_modes()
        elif datatype.lower() == "actuator":
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

        self.datapoints = []
        self.value = 0        # Subscribe to pubsub events
        # events.subscribe(PUBSUB_SENSORLESS_RESULTS, self.HandleSensorlessData)
        # events.oneShotSubscribe        
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

        addFromCurrentBtn.Bind(wx.EVT_BUTTON, self.OnAddDatapointFromCurrent)
        addFromFileBtn.Bind(wx.EVT_BUTTON, self.OnAddDatapointFromFile)
        removeBtn.Bind(wx.EVT_BUTTON, self.OnRemoveDatapoint)
        
        data_panel_btns_sizer = wx.BoxSizer(wx.VERTICAL)
        data_panel_btns_sizer.Add(addFromFileBtn)
        data_panel_btns_sizer.Add(addFromCurrentBtn)
        data_panel_btns_sizer.Add(removeBtn)
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
        self.remotez = MinMaxSliderCtrl(control_panel, value=0)
        self.remotez.Bind(EVT_VALUE, self.OnRemoteZ)
        row_sizer.Add(self.remotez)
        
        # Layout
        self.datatype = RFDatatypeChoice(control_panel)
        self.datatype.Bind(wx.EVT_CHOICE, self.OnModesChange)

        control_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        control_panel_sizer.Add(self.datatype)
        control_panel_sizer.Add(row_sizer)

        control_panel.SetSizerAndFit(control_panel_sizer)

        # Visualisation panel
        vis_panel = wx.Panel(root_panel)

        self.modes = FilterModesCtrl(vis_panel, value="5,6,7,8")
        self.datatype = RFDatatypeChoice(vis_panel)
        figure = Figure()
        self.ax = figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(vis_panel, wx.ID_ANY, figure)

        self.datatype.Bind(wx.EVT_CHOICE, self.OnModesChange)
        self.modes.Bind(wx.EVT_TEXT, self.OnModesChange)

        vis_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        vis_panel_sizer.Add(self.datatype)
        vis_panel_sizer.Add(self.modes)
        vis_panel_sizer.Add(self.canvas)

        vis_panel.SetSizerAndFit(vis_panel_sizer)
        
        # Add pages to tabs
        tabs.AddPage(data_panel,"Data") 
        tabs.AddPage(control_panel,"Control") 

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
        datapoint[id] = len(self.datapoints) + 1
        self.datapoints.append(datapoint)

        self.datapoints.sort(key=lambda d: d["z"])
        
        self.listbox.Clear()
        for d in self.datapoints:
            self.listbox.AppendItems("{} ({})".format(d["z"],d["datatype"]))

        self.update()

    def OnRemoteZ(self, e):
        z_target = self.remotez.GetValue()

        if self.datatype.GetStringSelection().lower() == "zernike":
            values = [self.z_lookup[i](z_target) for i in range(0,self._n_modes)]
            self._device.set_phase(values, offset=self._device.proxy.get_system_flat())
        elif self.datatype.GetStringSelection().lower() == "actuator":
            values = [self.z_lookup[i](z_target) for i in range(0,self._n_actuators)]
            self._device.send(values)

    def UpdateValueRanges(self, middle=None, range=None):
        min_val =  self._slider_min.value
        if min_val is not None:
            self._val.SetMin(min_val)

        max_val = self._slider_max.value
        if max_val is not None:
            self._val.SetMax(max_val)

        self.SetZValue(self.value, quiet=True)

    def GetZValue(self):
        return self._val.GetValue()

    def SetZValue(self, val, quiet=False):
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
        self.self.remotez.SetValue(slider_val)

        # Emit mode change event, if required
        if not quiet:
            pass
            # evt = ModeChangeEvent(mode=self.id, value= self.value)
            # wx.PostEvent(self, evt)

    def OnAddDatapointFromFile(self, e):
        dlg = RFAddDatapointFromFile(self)
        
        if dlg.ShowModal() == wx.ID_OK:
            datapoint = dlg.GetData()
            self.addDatapont(datapoint)
        else:
            pass
        
        dlg.Destroy()

    def OnAddDatapointFromCurrent(self, e):
        dlg = RFAddDatapointFromCurrent(self, self._device)
        
        if dlg.ShowModal() == wx.ID_OK:
            datapoint = dlg.GetData()
            self.addDatapont(datapoint)
        else:
            pass

        dlg.Destroy()

    def OnRemoveDatapoint(self, e):
        selected = self.listbox.GetSelections()

        for i in selected:
            self.datapoints.pop(i)
            self.listbox.Delete(i)

    def OnModesChange(self, e):
        self.update()

    def update(self):
        # Plot data
        self.ax.clear()

        # Get data
        current_datatype = self.datatype.GetStringSelection().lower()
        points = [a for a in self.datapoints if a["datatype"].lower() == current_datatype]
        z = np.array([point["z"] for point in points])
        values = np.array([point["values"] for point in points])
        
        # Calculate regression
        try:
            n_measurements = values.shape[0]
            n_values = values.shape[1]
        except IndexError:
            n_measurements = 0
            n_values = 0
        slopes = np.zeros(n_values)
        intercepts = np.zeros(n_values)

        self.z_lookup = []

        # Continue of more than one value
        if n_measurements > 1:
            for i in range(n_values):
                slope, intercept, r, p, se = scipy.stats.linregress(z, values[:,i])
                slopes[i] = slope
                intercepts[i] = intercept
                coef = [slope, intercept]
                # coef = np.polyfit(z,values[:,i],1)

                self.z_lookup.append(np.poly1d(coef)) 

            # Plot modes and regression
            # - filter visible modes
            filter_modes = self.modes.GetValue()
            
            for mode in filter_modes:
                self.ax.scatter(z, values[:,mode])
                
                x1=np.linspace(np.min(z),np.max(z),500)
                y1=slopes[mode]*x1+intercepts[mode]
                # self.ax.plot(x1,y1)
                self.ax.plot(z, self.z_lookup[mode](z))

        self.canvas.draw()

    def set_data(self, data):
        self.datapoints = data
        self.update()
    