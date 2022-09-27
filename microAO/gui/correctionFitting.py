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

import cockpit.gui.dialogs
import cockpit.interfaces.stageMover
import cockpit.events
import cockpit.handlers.stagePositioner
import cockpit.util.userConfig

import wx
import matplotlib.figure
import matplotlib.backends.backend_wxagg
import matplotlib.backends.backend_wx
import matplotlib.cm
import matplotlib.lines
import numpy as np
import json

from microAO.events import *
from microAO.gui.common import FilterModesCtrl


class _MultiSelectionChoiceDialog(wx.Dialog):
    def __init__(self, parent, title, message, choices):
        super().__init__(parent, title=title)

        # Create a message label
        stxt = wx.StaticText(self, label=message)

        # Create a list box of conflicts
        self._listbox = wx.ListBox(self, choices=choices, style=wx.LB_EXTENDED)

        # Create the standard buttons
        sizer_stdbuttons = wx.StdDialogButtonSizer()
        for button_id in (wx.ID_OK, wx.ID_CANCEL):
            button = wx.Button(self, button_id)
            sizer_stdbuttons.Add(button)
        sizer_stdbuttons.Realize()

        # Finalise layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(stxt, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5)
        sizer.Add(self._listbox, 1, wx.LEFT | wx.RIGHT, 5)
        sizer.AddSpacer(10)
        sizer.Add(sizer_stdbuttons, 0, wx.ALL, 5)
        self.SetSizerAndFit(sizer)

    def GetSelections(self):
        return self._listbox.GetSelections()


class CorrectionFittingFrame(wx.Frame):
    _RANGE_MULTIPLIER_Z = 1.1

    def __init__(self, parent, device):
        super().__init__(parent, title="Remote focus")

        # Set attributes
        self._device = device

        # Get correction names
        correction_names = self._device.corrfit_dp_get().keys()

        # Create tabbed control interface
        notebook = wx.Notebook(self)

        # Data page
        data_panel = wx.Panel(notebook)
        self._data_tree = wx.TreeCtrl(data_panel)

        root = self._data_tree.AddRoot("Corrections")
        self._correction_nodes = []
        for correction_name in correction_names:
            self._correction_nodes.append(
                self._data_tree.AppendItem(root, correction_name)
            )
        self._data_tree.Expand(root)

        data_add_corr_btn = wx.Button(
            data_panel, wx.ID_ANY, "Add datapoint from active corrections"
        )
        data_add_file_btn = wx.Button(
            data_panel, wx.ID_ANY, "Add datapoint from file"
        )
        data_rem_btn = wx.Button(
            data_panel, wx.ID_ANY, "Remove selected datapoint"
        )
        data_save_cfg_btn = wx.Button(
            data_panel, wx.ID_ANY, "Save datapoints to config"
        )
        data_save_file_btn = wx.Button(
            data_panel, wx.ID_ANY, "Save datapoints to file"
        )
        data_load_file_btn = wx.Button(
            data_panel, wx.ID_ANY, "Load datapoints from file"
        )

        data_add_corr_btn.Bind(
            wx.EVT_BUTTON, lambda _: self._on_add_datapoint(from_file=False)
        )
        data_add_file_btn.Bind(
            wx.EVT_BUTTON, lambda _: self._on_add_datapoint(from_file=True)
        )
        data_rem_btn.Bind(wx.EVT_BUTTON, self._on_remove_datapoint)
        data_save_cfg_btn.Bind(wx.EVT_BUTTON, self._on_save_to_config)
        data_save_file_btn.Bind(wx.EVT_BUTTON, self._on_save_to_file)
        data_load_file_btn.Bind(wx.EVT_BUTTON, self._on_load_from_file)

        data_btns_sizer = wx.BoxSizer(wx.VERTICAL)
        data_btns_sizer.Add(data_add_corr_btn, 0, wx.EXPAND)
        data_btns_sizer.Add(data_add_file_btn, 0, wx.EXPAND)
        data_btns_sizer.Add(data_rem_btn, 0, wx.EXPAND)
        data_btns_sizer.Add(data_save_cfg_btn, 0, wx.EXPAND)
        data_btns_sizer.AddSpacer(10)
        data_btns_sizer.Add(data_save_file_btn, 0, wx.EXPAND)
        data_btns_sizer.Add(data_load_file_btn, 0, wx.EXPAND)

        data_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        data_panel_sizer.Add(data_btns_sizer, 0, wx.EXPAND | wx.ALL, 5)
        data_panel_sizer.Add(self._data_tree, 1, wx.EXPAND | wx.ALL, 5)
        data_panel.SetSizerAndFit(data_panel_sizer)

        # Fitting plot page
        fitting_panel = wx.Panel(notebook)

        self._fit_filter = FilterModesCtrl(fitting_panel, value="1-12")
        self._correction_choice = wx.Choice(
            fitting_panel, choices=list(correction_names)
        )
        figure = matplotlib.figure.Figure(constrained_layout=True)
        self._fit_axes = figure.add_subplot()
        self._fit_canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
            fitting_panel, wx.ID_ANY, figure
        )
        self.ax_z_vline = matplotlib.lines.Line2D((0, 0), (0, 0))

        self._correction_choice.SetSelection(0)
        self._correction_choice.Bind(
            wx.EVT_CHOICE, lambda _: self._update_fitting_plot()
        )

        self._fit_filter.Bind(
            wx.EVT_TEXT, lambda _: self._update_fitting_plot()
        )

        row_sizer = wx.BoxSizer(wx.HORIZONTAL)
        row_sizer.Add(self._fit_filter, 0, wx.LEFT, 5)
        row_sizer.Add(self._correction_choice, 0, wx.LEFT, 5)
        fitting_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        fitting_panel_sizer.Add(row_sizer, 0, wx.ALL, 5)
        fitting_panel_sizer.Add(self._fit_canvas, 1, wx.EXPAND | wx.ALL, 5)
        fitting_panel.SetSizerAndFit(fitting_panel_sizer)

        # Add pages to notebook
        notebook.AddPage(data_panel, "Data")
        notebook.AddPage(fitting_panel, "Fitting plot")

        # Frame layout
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(notebook, 1, wx.EXPAND)
        self.SetSizerAndFit(frame_sizer)

        # Subscribe to stage events
        cockpit.events.subscribe(
            cockpit.events.STAGE_STOPPED, self._on_stage_stopped
        )

        # Handle close events
        self.Bind(wx.EVT_CLOSE, self._on_close)

        # Update GUI
        self._update_data_list()
        self._update_fitting_plot()

    def _on_close(self, event: wx.CloseEvent) -> None:
        # Unsubscribe from stage events
        cockpit.events.unsubscribe(
            cockpit.events.STAGE_STOPPED, self._on_stage_stopped
        )
        # Let the default event handler destroy the frame
        event.Skip()

    def _on_stage_stopped(self, _) -> None:
        self._update_fitting_plot()

    def _on_add_datapoint(self, from_file: bool) -> None:
        # Parse the selection
        focused_item = self._data_tree.GetFocusedItem()
        if focused_item not in self._correction_nodes:
            # Wrong selection => warn but do nothing
            with wx.MessageDialog(
                self,
                "Invalid selection! Please select the correction to which the "
                "datapoint should be added.",
                "Warning",
                wx.OK | wx.ICON_WARNING,
            ) as dlg:
                dlg.ShowModal()
            return
        correction_name = self._data_tree.GetItemText(focused_item)
        # Get the modes
        if from_file:
            fpath = ""
            with wx.FileDialog(
                self,
                "Add datapoint from file",
                wildcard="Modes file (*.txt)|*.txt",
                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
            ) as fileDialog:
                if fileDialog.ShowModal() != wx.ID_OK:
                    return
                fpath = fileDialog.GetPath()
            modes = np.loadtxt(fpath)
        else:
            modes = self._device.sum_corrections()[0]
        # Ask for Z position
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Datapoint's Z position",
            ["Datapoint's Z position [um]:"],
            (0.0,),
        )
        # Add the datapoint and update the list
        self._device.corrfit_dp_add(correction_name, float(inputs[0]), modes)
        self._update_data_list()
        self._update_fitting_plot()

    def _on_remove_datapoint(self, _: wx.CommandEvent) -> None:
        # Parse the focused item
        focused_item = self._data_tree.GetFocusedItem()
        focused_item_parent = self._data_tree.GetItemParent(focused_item)
        if focused_item_parent not in self._correction_nodes:
            # This is not a datapoint => do nothing
            with wx.MessageDialog(
                self,
                "Invalid selection! Please select the datapoint which should "
                "be removed.",
                "Warning",
                wx.OK | wx.ICON_WARNING,
            ) as dlg:
                dlg.ShowModal()
            return
        # Remove the datapoint and update
        self._device.corrfit_dp_rem(
            self._data_tree.GetItemText(focused_item_parent),
            float(self._data_tree.GetItemText(focused_item)[4:-3]),
        )
        self._update_data_list()
        self._update_fitting_plot()

    def _dp_ndarray_to_list(self) -> dict[str, dict[float, list[float]]]:
        datapoints_ndarrays = self._device.corrfit_dp_get()
        datapoints_lists = {}
        for cname in datapoints_ndarrays:
            datapoints_lists[cname] = {}
            for z in datapoints_ndarrays[cname]:
                datapoints_lists[cname][z] = np.ndarray.tolist(
                    datapoints_ndarrays[cname][z]
                )
        return datapoints_lists

    def _on_save_to_config(self, _: wx.CommandEvent) -> None:
        datapoints = self._dp_ndarray_to_list()
        cockpit.util.userConfig.setValue("ao_corrfit_dpts", datapoints)

    def _on_save_to_file(self, _: wx.CommandEvent) -> None:
        # Ask for a file path
        fpath = None
        with wx.FileDialog(
            self,
            "Save correction fitting data points",
            wildcard="JSON file (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()
        # Get the datapoints
        datapoints = self._dp_ndarray_to_list()
        # Write to file
        with open(fpath, "w", encoding="utf-8") as fo:
            json.dump(datapoints, fo, sort_keys=True, indent=4)

    def _on_load_from_file(self, _: wx.CommandEvent) -> None:
        # Ask for a file path
        fpath = None
        with wx.FileDialog(
            self,
            "Load correction fitting data points",
            wildcard="JSON file (*.json)|*.json",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()
        # Get the datapoints
        datapoints = self._device.corrfit_dp_get()
        datapoints_loaded = None
        with open(fpath, "r", encoding="utf-8") as fi:
            datapoints_loaded = json.load(fi)
        # Check for conflicts
        z_conflicts = []
        for cname in datapoints_loaded:
            for z in datapoints_loaded[cname]:
                if float(z) in datapoints[cname]:
                    z_conflicts.append(f"{cname}: {z}")
        z_conflicts.sort()
        with _MultiSelectionChoiceDialog(
            self,
            "Loaded data points conflict resolution",
            "The following data points already exist.\n"
            "Select which ones to overwrite.",
            z_conflicts,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                # Cancelled => abort the entire loading
                return
            # Get the selected datapoints
            datapoints_to_overwrite = [
                z_conflicts[i] for i in dlg.GetSelections()
            ]
        # Add datapoints
        for cname in datapoints_loaded:
            for z in datapoints_loaded[cname]:
                if float(z) in datapoints[cname]:
                    if f"{cname}: {z}" in datapoints_to_overwrite:
                        self._device.corrfit_dp_add(
                            cname,
                            float(z),
                            np.array(datapoints_loaded[cname][z]),
                        )
                else:
                    self._device.corrfit_dp_add(
                        cname, float(z), np.array(datapoints_loaded[cname][z])
                    )
        # Update list and plot
        self._update_data_list()
        self._update_fitting_plot()

    def _update_data_list(self) -> None:
        datapoints = self._device.corrfit_dp_get()
        for correction_node in self._correction_nodes:
            correction_name = self._data_tree.GetItemText(correction_node)
            # Clear datapoints
            self._data_tree.DeleteChildren(correction_node)
            # Add items
            for z in sorted(datapoints[correction_name].keys()):
                modes = datapoints[correction_name][z]
                # Add a datapoint node
                datapoint_node = self._data_tree.AppendItem(
                    correction_node, f"Z = {z:+.03f} um"
                )
                # Add mode nodes
                for mode_index, mode_value in enumerate(modes):
                    self._data_tree.AppendItem(
                        datapoint_node,
                        f"{mode_index + 1:02d}: {mode_value:+.05f}",
                    )
            # Show datapoints
            self._data_tree.Expand(correction_node)

    def _update_fitting_plot(self) -> None:
        # Get correction name
        correction_name = self._correction_choice.GetStringSelection()

        # Get datapoints
        datapoints = self._device.corrfit_dp_get()

        # Clear axes and update the canvas
        self._fit_axes.clear()
        self._fit_canvas.draw()

        # Get all values for the selected correction;
        # The modes are arranged in a Z x M matrix, where Z is the number of Zs
        # and M is the number of modes
        zs = np.array(sorted(datapoints[correction_name].keys()))
        modes_all = np.array([datapoints[correction_name][z] for z in zs])

        # Exit early in the absence of datapoints
        if zs.size == 0:
            return

        # Filter modes
        filter_modes = [
            mode - 1
            for mode in self._fit_filter.GetValue()
            if (mode - 1) < modes_all.shape[1]
        ]

        # Determine the current Z position
        current_z = self._device.rf_get_position()
        if correction_name == "sensorless":
            current_z = cockpit.interfaces.stageMover.getPosition()[2]

        # Determine the x-axis endpoints
        zs_middle = np.max(zs) - (np.max(zs) - np.min(zs)) / 2
        zs_max_span = max([abs(z - zs_middle) for z in zs])
        endpoints_z = np.array(
            (
                min(
                    round(current_z),
                    round(zs_middle - zs_max_span * self._RANGE_MULTIPLIER_Z),
                ),
                max(
                    round(current_z),
                    round(zs_middle + zs_max_span * self._RANGE_MULTIPLIER_Z),
                ),
            )
        )

        # Derive the modes at the endpoints
        endpoints_modes = np.array(
            (
                self._device._corrfit_eval(correction_name, endpoints_z[0]),
                self._device._corrfit_eval(correction_name, endpoints_z[1]),
            )
        )

        # Plot modes
        for mode_index in filter_modes:
            # Plot the datapoints
            self._fit_axes.scatter(
                zs,
                modes_all[:, mode_index],
                label=f"{mode_index + 1}",
            )
            # Plot the fitted line
            self._fit_axes.plot(
                endpoints_z,
                endpoints_modes[:, mode_index],
            )

        # Plot current z position
        self.ax_z_vline = self._fit_axes.axvline(current_z)

        # Set labels
        self._fit_axes.set_xlabel("Z position [um]")
        self._fit_axes.set_ylabel("Mode value [rad]")

        # Add legend
        ncol = np.ceil(len(filter_modes) / 10).astype(int)
        self._fit_axes.legend(
            title="Mode Noll indices",
            loc="upper center",
            ncol=ncol,
            fontsize="x-small",
        )

        # Update the canvas
        self._fit_canvas.draw()
