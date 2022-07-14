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

import pathlib
import json

from cockpit import events
import cockpit.gui.dialogs
import cockpit.util.userConfig
import cockpit.util.threads

import wx
from wx.core import DirDialog
import wx.lib.newevent

import matplotlib.figure
import matplotlib.backends.backend_wxagg
import matplotlib.backends.backend_wx
import matplotlib.backend_bases
import matplotlib.patches
import matplotlib.lines
import matplotlib.patheffects
import matplotlib.cm

import skimage.exposure

import numpy as np

from microAO.events import *
from microAO.gui.common import FilterModesCtrl

class _BeadPicker(wx.Dialog):
    """Window to select a rectangular 2D ROI in a 3D image.

    Used for picking beads.
    """

    _DEFAULT_CMAP = "plasma"

    def __init__(self, parent, image_stack: np.ndarray) -> None:
        super().__init__(
            parent,
            title="Bead picker",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
        )
        if len(image_stack.shape) != 3:
            print(
                "ERROR: Bead Picker instance received misshaped data! Expected"
                f" 3 dimensions (ZYX) but received {len(image_stack.shape)}."
            )
        self._imgs = image_stack
        self._imgs_idx = image_stack.shape[0] // 2
        self._selecting = False
        self._rect_origin = (0.0, 0.0)

        # Draw the figure and add a canvas
        fig = matplotlib.figure.Figure(constrained_layout=True)
        axes = fig.add_subplot()
        self._axes_image = axes.imshow(
            self._rescale_image(self._imgs[self._imgs_idx]),
            cmap=self._DEFAULT_CMAP
        )
        rect_edge = self._axes_image.get_cmap().get_over()
        rect_edge[3] = 0.3
        self._rect = matplotlib.patches.Rectangle(
            (0, 0), 0, 0, linewidth=2, ec=rect_edge, fc=(0, 0, 0, 0)
        )
        axes.add_patch(self._rect)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_frame_on(False)
        fig.colorbar(self._axes_image, ax=axes)
        self._canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
            self, wx.ID_ANY, fig
        )

        # Event handling configuration for the canvas
        self._canvas.mpl_connect('motion_notify_event', self._on_motion_notify)
        self._canvas.mpl_connect('button_press_event', self._on_button_press)
        self._canvas.mpl_connect('button_release_event', self._on_button_release)

        # Add toolbar
        self._toolbar = matplotlib.backends.backend_wx.NavigationToolbar2Wx(
            self._canvas
        )
        self._toolbar.Show()

        # Add a slider
        sizer_row0 = wx.BoxSizer(wx.VERTICAL)
        self._slider = wx.Slider(
            self,
            value=self._imgs_idx + 1,
            minValue=1,
            maxValue=image_stack.shape[0],
            style=wx.SL_LABELS
        )
        self._slider.Bind(wx.EVT_SLIDER, self._on_slider)
        sizer_row0.Add(self._slider, 0, wx.EXPAND)

        # Add buttons
        sizer_row1 = wx.StdDialogButtonSizer()
        for button_id in (wx.ID_OK, wx.ID_CANCEL):
            button = wx.Button(self, button_id)
            sizer_row1.Add(button)
        sizer_row1.Realize()

        # Finalise layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self._canvas, 1, wx.SHAPED)
        sizer.Add(self._toolbar, 0, wx.LEFT | wx.EXPAND)
        sizer.Add(sizer_row0, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(sizer_row1, 0, wx.ALL, 5)
        self.SetSizerAndFit(sizer)

    def _on_motion_notify(self, event: matplotlib.backend_bases.MouseEvent):
        if not event.inaxes or self._toolbar.mode:
            # Do not do anything if the cursor is not over the axes or if one
            # of the toolbar tools is being used
            return
        if self._selecting:
            side = max(
                event.xdata - self._rect_origin[0],
                event.ydata - self._rect_origin[1]
            )
            self._rect.set_width(side)
            self._rect.set_height(side)
            self._canvas.draw()

    def _on_button_press(self, event: matplotlib.backend_bases.MouseEvent):
        if not event.inaxes or self._toolbar.mode:
            # Do not do anything if the cursor is not over the axes or if one
            # of the toolbar tools is being used
            return
        self._selecting = True
        self._rect_origin = (event.xdata, event.ydata)
        self._rect.set_xy(self._rect_origin)
        self._rect.set_width(0)
        self._rect.set_height(0)

    def _on_button_release(self, event: matplotlib.backend_bases.MouseEvent):
        if not event.inaxes or self._toolbar.mode:
            # Do not do anything if the cursor is not over the axes or if one
            # of the toolbar tools is being used
            return
        self._selecting = False

    def _on_slider(self, event: wx.CommandEvent):
        self._axes_image.set_data(
            self._rescale_image(self._imgs[event.GetInt() - 1])
        )
        self._canvas.draw()

    def _rescale_image(self, image):
        p2, p98 = np.percentile(image, (2, 98))
        return skimage.exposure.rescale_intensity(image, in_range=(p2, p98))

    def get_roi(self):
        roi = (
            *self._rect.get_xy(),
            self._rect.get_width(),
            self._rect.get_height()
        )
        if roi[2] < 0.0:
            # The rectangle was drawn right to left => roi needs adjustment
            roi = (
                roi[0] + roi[2],
                roi[1] + roi[3],
                np.abs(roi[2]),
                np.abs(roi[3])
            )
        return tuple(map(round, roi))

class RemoteFocusControl(wx.Frame):
    _DEFAULT_CMAP = matplotlib.cm.get_cmap("viridis")

    def __init__(self, parent, device, **kwargs):
        super().__init__(parent, title="Remote focus")

        # Set attributes
        self._device = device

        # Create tabbed control interface
        notebook = wx.Notebook(self)

        # Data page
        data_panel = wx.Panel(notebook)
        self._data_tree = wx.TreeCtrl(data_panel)

        self._data_tree_root = self._data_tree.AddRoot("Datapoints")
        self._data_tree.SetItemData(self._data_tree_root, None)

        data_add_corr_btn = wx.Button(data_panel, wx.ID_ANY, "Add datapoint from active corrections")
        data_add_file_btn = wx.Button(data_panel, wx.ID_ANY, "Add datapoint from file")
        data_rem_btn = wx.Button(data_panel, wx.ID_ANY, "Remove selected datapoint")
        data_save_btn = wx.Button(data_panel, wx.ID_ANY, "Save datapoints to config")
        data_calib_btn = wx.Button(data_panel, wx.ID_ANY, "Calibrate counteraction")

        data_add_corr_btn.Bind(wx.EVT_BUTTON, self._on_add_datapoint_corr)
        data_add_file_btn.Bind(wx.EVT_BUTTON, self._on_add_datapoint_file)
        data_rem_btn.Bind(wx.EVT_BUTTON, self._on_remove_datapoint)
        data_save_btn.Bind(wx.EVT_BUTTON, self._on_save_datapoints)
        data_calib_btn.Bind(wx.EVT_BUTTON, self._on_calib_counteraction)

        data_btns_sizer = wx.BoxSizer(wx.VERTICAL)
        data_btns_sizer.Add(data_add_corr_btn, 0, wx.EXPAND)
        data_btns_sizer.Add(data_add_file_btn, 0, wx.EXPAND)
        data_btns_sizer.Add(data_rem_btn, 0, wx.EXPAND)
        data_btns_sizer.Add(data_save_btn, 0, wx.EXPAND)
        data_btns_sizer.Add(-1, 10)
        data_btns_sizer.Add(data_calib_btn, 0, wx.EXPAND)

        data_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        data_panel_sizer.Add(data_btns_sizer, 0, wx.EXPAND | wx.ALL, 5)
        data_panel_sizer.Add(self._data_tree, 1, wx.EXPAND | wx.ALL, 5)
        data_panel.SetSizerAndFit(data_panel_sizer)

        # Fitting plot page
        fitting_panel = wx.Panel(notebook)

        self._fit_filter = FilterModesCtrl(fitting_panel, value="1-12")
        figure = matplotlib.figure.Figure(constrained_layout=True)
        self._fit_axes = figure.add_subplot()
        self._fit_canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
            fitting_panel, wx.ID_ANY, figure
        )

        self._fit_filter.Bind(wx.EVT_TEXT, self._on_fitting_filter)

        fitting_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        fitting_panel_sizer.Add(self._fit_filter, 0, wx.ALL, 5)
        fitting_panel_sizer.Add(self._fit_canvas, 1, wx.EXPAND | wx.ALL, 5)
        fitting_panel.SetSizerAndFit(fitting_panel_sizer)

        # Add pages to notebook
        notebook.AddPage(data_panel,"Data") 
        notebook.AddPage(fitting_panel,"Fitting plot")

        # Frame layout
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(notebook, 1, wx.EXPAND)
        self.SetSizerAndFit(frame_sizer)

        # Update GUI
        self._update_data_list()
        self._update_fitting_plot()

        # Subscribe to calibration events
        events.subscribe(
            PUBSUB_RF_CALIB_CACT_DATA,
            self._on_calib_cact_data
        )
        events.subscribe(
            PUBSUB_RF_CALIB_CACT_PROJ,
            self._on_calib_cact_proj
        )

    def _on_add_datapoint_corr(self, _: wx.CommandEvent):
        # Ask for Z position
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Datapoint's Z position",
            ["Relative Z position [um]:"],
            (0.0,),
        )
        # Sum the modes and the actuators
        modes, _ = self._device.sum_corrections()
        # Add the datapoint and update the list
        self._device.remotez.add_datapoint(float(inputs[0]), modes)
        self._update_data_list()
        self._update_fitting_plot()

    def _on_add_datapoint_file(self, _: wx.CommandEvent):
        # Ask for file
        fpath = ""
        with wx.FileDialog(
            self,
            "Add datapoint from file",
            wildcard="Modes file (*.txt)|*.txt",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()
        modes = np.loadtxt(fpath)
        # Ask for Z position
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Datapoint's Z position",
            ["Relative Z position [um]:"],
            (0.0,),
        )
        # Add the datapoint and update the list
        self._device.remotez.add_datapoint(float(inputs[0]), modes)
        self._update_data_list()
        self._update_fitting_plot()

    def _on_remove_datapoint(self, _: wx.CommandEvent):
        focused_item = self._data_tree.GetFocusedItem()
        if focused_item.GetID() is None:
            # There is no valid selection => do nothing
            return
        parent = self._data_tree.GetItemParent(focused_item)
        if parent.GetID() is None:
            # This is the root node => do nothing
            return
        label = self._data_tree.GetItemText(focused_item)
        if label.startswith("Mode Noll"):
            # The selected item is a mode, rather than datapoint => do nothing
            return
        z = float(label[:-3])
        self._device.remotez.remove_datapoint(z)
        self._update_data_list()
        self._update_fitting_plot()

    def _on_save_datapoints(self, _: wx.CommandEvent):
        datapoints = {}
        for z in self._device.remotez.datapoints:
            datapoints[z] = np.ndarray.tolist(
                self._device.remotez.datapoints[z]
            )
        cockpit.util.userConfig.setValue("rf_datapoints", datapoints)

    def _on_calib_counteraction(self, _: wx.CommandEvent):
        # Update status bar
        events.publish(
            events.UPDATE_STATUS_LIGHT,
            "image count",
            "Remote focus calibration | Configuring..."
        )
        # Get all the necessary handlers
        handlers_zstage = self.GetParent().getStage()
        handlers_camera = self.GetParent().getCamera()
        handlers_imager = self.GetParent().getImager()
        # Select output directory
        output_dir_path = None
        with DirDialog(None, "Select data output directory") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                output_dir_path = pathlib.Path(dlg.GetPath())
            else:
                print(
                    "Output directory selection cancelled during remote focus "
                    "calibration. Aborting..."
                )
                return
        # Get the calibration parameters
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Get remote Z calibration parameters (all in micrometres)",
            [
                "Defocus min",
                "Defocus max",
                "Defocus step",
                "Stage offset min",
                "Stage offset max",
                "Stage step",
            ],
            (
                -5,
                5,
                0.5,
                -5,
                5,
                0.5
            ),
        )
        calib_params = {
            "defocus_min": float(inputs[0]),
            "defocus_max": float(inputs[1]),
            "defocus_step": float(inputs[2]),
            "stage_min": float(inputs[3]),
            "stage_max": float(inputs[4]),
            "stage_step": float(inputs[5]),
        }
        # Save the calibration parameters
        with open(output_dir_path.joinpath(f"calib_params.json"), "w") as fo:
            json.dump(calib_params, fo)
        # Launch the calibration process
        self._device.remotez.calibrate_counteraction_get_data(
            handlers_zstage,
            handlers_camera,
            handlers_imager,
            calib_params,
            wx.GetApp().Objectives.GetPixelSize(),
            output_dir_path
        )

    def _on_fitting_filter(self, _: wx.CommandEvent):
        self._update_fitting_plot()

    def _update_data_list(self):
        self._data_tree.DeleteChildren(self._data_tree_root)
        # Add items
        for z in sorted(self._device.remotez.datapoints.keys()):
            modes = self._device.remotez.datapoints[z]
            datapoint_node = self._data_tree.AppendItem(
                self._data_tree_root,
                f"{z:+.03f} um"
            )
            self._data_tree.SetItemData(datapoint_node, None)
            for mode_index, mode_value in enumerate(modes):
                mode_node = self._data_tree.AppendItem(
                    datapoint_node,
                    f"Mode Noll index: {mode_index + 1}. "
                    f"Mode value: {mode_value:+.03f}."
                )
                self._data_tree.SetItemData(mode_node, None)
        # Show datapoints
        self._data_tree.Expand(self._data_tree_root)

    def _update_fitting_plot(self):
        # Clear axes
        self._fit_axes.clear()

        # Get all values for the selected datatype
        zs = []
        modes_all = []  # NxM where N is number of Zs and M is number of modes
        for z, modes in sorted(
            self._device.remotez.datapoints.items(), key=lambda x: x[0]
        ):
            zs.append(z)
            modes_all.append(modes)
        zs = np.array(sorted(zs))
        modes_all = np.array(modes_all)

        # Filter modes
        filter_modes = [
            mode - 1
            for mode in self._fit_filter.GetValue()
            if (mode - 1) < len(self._device.remotez.z_lookup)
        ]

        # Plot modes
        for mode_index in filter_modes:
            # Plot the datapoints
            self._fit_axes.scatter(
                zs,
                modes_all[:, mode_index],
                label=f"{mode_index + 1}",
                color=self._DEFAULT_CMAP((mode_index + 1) / len(filter_modes))
            )
            # Plot the fitted line
            z_endpoints = np.array((zs[0], zs[-1]))
            self._fit_axes.plot(
                z_endpoints,
                self._device.remotez.z_lookup[mode_index](z_endpoints),
                color=self._DEFAULT_CMAP((mode_index + 1) / len(filter_modes))
            )

        # Plot current z position
        self.ax_z_position = self._fit_axes.axvline(
            self._device.remotez.get_z()
        )

        # Add legend
        ncol = np.ceil(len(filter_modes) / 10).astype(int)
        self._fit_axes.legend(
            title="Mode Noll indices",
            loc="upper center",
            ncol=ncol,
            fontsize="x-small"
        )

        self._fit_canvas.draw()

    @cockpit.util.threads.callInMainThread
    def _on_calib_cact_data(self, rf_stacks, output_dir_path, defocus_step):
        events.publish(
            events.UPDATE_STATUS_LIGHT,
            "image count",
            "Remote focus calibration | Selecting bead..."
        )
        # Ask the user to select a bead, using the middle stack
        bead_roi = None
        with _BeadPicker(self, rf_stacks[len(rf_stacks) // 2].images) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                bead_roi = dlg.get_roi()
            else:
                print(
                    "ERROR: Bead selection cancelled. Aborting remote focus "
                    "calibration process..."
                )
                return
        np.savetxt(
            output_dir_path.joinpath(
                f"remote-focus-zstack_bead-roi.txt",
            ),
            bead_roi
        )
        # Calculate orthogonal projects
        self._device.remotez.calibrate_counteraction_get_projections(
            rf_stacks,
            bead_roi,
            wx.GetApp().Objectives.GetPixelSize(),
            defocus_step,
            output_dir_path
        )

    @cockpit.util.threads.callInMainThread
    def _on_calib_cact_proj(self):
        # Clear status light
        events.publish(events.UPDATE_STATUS_LIGHT, "image count", "")
