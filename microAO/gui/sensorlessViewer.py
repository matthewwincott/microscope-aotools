import dataclasses
import typing
import json
import time
import numpy
import wx
import matplotlib.pyplot
import matplotlib.ticker
import matplotlib.figure
import matplotlib.backends.backend_wxagg
import matplotlib.backends.backend_wx
import matplotlib.patches
import tifffile
import skimage.exposure

import cockpit
import microAO.aoMetrics
import microAO.events
from microAO.aoRoutines import ConventionalResults


class _DiagnosticsPanelBase(wx.Panel):
    _DEFAULT_CMAP = "inferno"

    def __init__(self, parent):
        super().__init__(parent)

        # Create figure
        self._figure = matplotlib.figure.Figure(constrained_layout=True)
        self._axes = None
        self._canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
            self, wx.ID_ANY, self._figure
        )
        toolbar = matplotlib.backends.backend_wx.NavigationToolbar2Wx(
            self._canvas
        )
        toolbar.Show()

        # Create widgets
        stext_mode = wx.StaticText(self, label="Mode:")
        self._slider_mode = wx.Slider(self, style=wx.SL_LABELS)
        self._slider_mode.Bind(wx.EVT_SLIDER, self._on_slider_mode)
        stext_meas = wx.StaticText(self, label="Measurement:")
        self._slider_meas = wx.Slider(self, style=wx.SL_LABELS)
        self._slider_meas.Bind(wx.EVT_SLIDER, self._on_slider_meas)
        stext_noll_label = wx.StaticText(self, label="Noll index:")
        self._stext_noll = wx.StaticText(self, label="")
        stext_cmap = wx.StaticText(self, label="Colourmap:")
        self._cmap_choice = wx.Choice(
            self, choices=sorted(matplotlib.pyplot.colormaps())
        )
        self._cmap_choice.SetStringSelection(self._DEFAULT_CMAP)
        self._cmap_choice.Bind(wx.EVT_CHOICE, self._on_cmap_choice)

        # Lay out the widgets
        widgets_sizer = wx.GridBagSizer(vgap=0, hgap=10)
        widgets_sizer.SetCols(2)
        widgets_sizer.AddGrowableCol(1)
        widgets_sizer.Add(
            stext_mode,
            wx.GBPosition(0, 0),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALL,
            5,
        )
        widgets_sizer.Add(
            self._slider_mode, wx.GBPosition(0, 1), wx.GBSpan(1, 1), wx.EXPAND
        )
        widgets_sizer.Add(
            stext_meas,
            wx.GBPosition(1, 0),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALL,
            5,
        )
        widgets_sizer.Add(
            self._slider_meas, wx.GBPosition(1, 1), wx.GBSpan(1, 1), wx.EXPAND
        )
        widgets_sizer.Add(
            stext_noll_label,
            wx.GBPosition(2, 0),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALL,
            5,
        )
        widgets_sizer.Add(
            self._stext_noll,
            wx.GBPosition(2, 1),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_LEFT,
        )
        widgets_sizer.Add(
            stext_cmap,
            wx.GBPosition(3, 0),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALL,
            5,
        )
        widgets_sizer.Add(
            self._cmap_choice,
            wx.GBPosition(3, 1),
            wx.GBSpan(1, 1),
            wx.ALIGN_CENTRE_VERTICAL | wx.ALIGN_LEFT,
        )

        # Finalise layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self._canvas, 1, wx.EXPAND)
        sizer.Add(toolbar, 0, wx.EXPAND)
        sizer.Add(widgets_sizer, 0, wx.EXPAND)
        self.SetSizerAndFit(sizer)

    def _update_plot(self):
        raise NotImplementedError()

    def _on_slider_mode(self, event: wx.CommandEvent):
        mode_index = event.GetInt() - 1
        current_diagnostics = (
            self.GetParent().GetParent()._metric_diagnostics[mode_index]
        )
        current_data = self.GetParent().GetParent()._metric_data[mode_index]
        # Update the measurement slider if necessary
        new_meas_max = len(current_diagnostics)
        if new_meas_max < self._slider_meas.GetValue():
            # The new max value is less than the current value => clamp to max
            self._slider_meas.SetValue(new_meas_max)
        self._slider_meas.SetMax(new_meas_max)
        # Update the Noll index
        self._stext_noll.SetLabel(current_data.mode_label)
        # Update the plot
        self._update_plot()

    def _on_slider_meas(self, _: wx.CommandEvent):
        self._update_plot()

    def _on_cmap_choice(self, _: wx.CommandEvent):
        self._update_plot()

    def initialise(self):
        # Clear the figure and create axes
        self._figure.clear()
        self._axes = self._figure.add_subplot()
        # Reset sliders
        self._slider_mode.SetValue(1)
        self._slider_mode.SetMin(1)
        self._slider_mode.SetMax(1)
        self._slider_meas.SetValue(1)
        self._slider_meas.SetMin(1)
        self._slider_meas.SetMax(1)

    def update(self):
        diagnostics = self.GetParent().GetParent()._metric_diagnostics
        mode_index = self._slider_mode.GetValue() - 1
        data = self.GetParent().GetParent()._metric_data[mode_index]
        # Update the mode slider
        self._slider_mode.SetMax(len(diagnostics))
        # Update the measurement slider
        self._slider_meas.SetMax(len(diagnostics[mode_index]))
        # Update Noll index
        self._stext_noll.SetLabel(data.mode_label)
        # Update the plot
        self._update_plot()


class _DiagnosticsPanelFourier(_DiagnosticsPanelBase):
    def _update_plot(self):
        index_mode = self._slider_mode.GetValue() - 1
        index_meas = self._slider_meas.GetValue() - 1
        diagnostics = (
            self.GetParent()
            .GetParent()
            ._metric_diagnostics[index_mode][index_meas]
        )
        # Clear axes
        self._figure.clear()
        # Update images
        self._axes = self._figure.subplots(1, 2, sharex=True, sharey=True)
        self._axes[0].imshow(
            diagnostics.fft_sq_log, cmap=self._cmap_choice.GetStringSelection()
        )
        self._axes[1].imshow(
            diagnostics.freq_above_noise,
            cmap=self._cmap_choice.GetStringSelection(),
        )
        for a in self._axes:
            a.axis("off")
        # Update canvas
        self._canvas.draw()


class _DiagnosticsPanelContrast(_DiagnosticsPanelBase):
    def _update_plot(self):
        index_mode = self._slider_mode.GetValue() - 1
        index_meas = self._slider_meas.GetValue() - 1
        diagnostics = (
            self.GetParent()
            .GetParent()
            ._metric_diagnostics[index_mode][index_meas]
        )
        # Clear axes
        self._figure.clear()
        # Update image and table
        self._axes = self._figure.subplots(1, 2)
        self._axes[0].imshow(
            skimage.exposure.rescale_intensity(
                diagnostics.image_raw,
                in_range=tuple(
                    numpy.percentile(diagnostics.image_raw, (1, 99))
                ),
            ),
            cmap=self._cmap_choice.GetStringSelection(),
        )
        self._axes[1].table(
            [
                ["Mean top", "Mean bottom"],
                [diagnostics.mean_top, diagnostics.mean_bottom],
            ],
            cellLoc="center",
            loc="center",
        )
        for a in self._axes:
            a.axis("off")
        # Update canvas
        self._canvas.draw()


class _DiagnosticsPanelGradient(_DiagnosticsPanelBase):
    def _update_plot(self):
        index_mode = self._slider_mode.GetValue() - 1
        index_meas = self._slider_meas.GetValue() - 1
        diagnostics = (
            self.GetParent()
            .GetParent()
            ._metric_diagnostics[index_mode][index_meas]
        )
        # Clear axes
        self._figure.clear()
        # Update images
        self._axes = self._figure.subplots(1, 4, sharex=True, sharey=True)
        self._axes[0].imshow(
            skimage.exposure.rescale_intensity(
                diagnostics.image_raw,
                in_range=tuple(
                    numpy.percentile(diagnostics.image_raw, (1, 99))
                ),
            ),
            cmap=self._cmap_choice.GetStringSelection(),
        )
        self._axes[0].set_title("Raw image")
        self._axes[1].imshow(diagnostics.grad_mask_x)
        self._axes[1].set_title("Grad. mask X")
        self._axes[2].imshow(diagnostics.grad_mask_y)
        self._axes[2].set_title("Grad. mask Y")
        self._axes[3].imshow(diagnostics.correction_grad)
        self._axes[3].set_title("Gradient")
        for a in self._axes:
            a.axis("off")
        # Update canvas
        self._canvas.draw()


class _DiagnosticsPanelFourierPower(_DiagnosticsPanelBase):
    def _update_plot(self):
        index_mode = self._slider_mode.GetValue() - 1
        index_meas = self._slider_meas.GetValue() - 1
        diagnostics = (
            self.GetParent()
            .GetParent()
            ._metric_diagnostics[index_mode][index_meas]
        )
        # Clear axes
        self._figure.clear()
        # Update images
        self._axes = self._figure.subplots(1, 2, sharex=True, sharey=True)
        self._axes[0].imshow(
            diagnostics.fftarray_sq_log,
            cmap=self._cmap_choice.GetStringSelection(),
        )
        self._axes[1].imshow(
            diagnostics.freq_above_noise,
            cmap=self._cmap_choice.GetStringSelection(),
        )
        for a in self._axes:
            a.axis("off")
        # Update canvas
        self._canvas.draw()


class _DiagnosticsPanelSecondMoment(_DiagnosticsPanelBase):
    def _update_plot(self):
        index_mode = self._slider_mode.GetValue() - 1
        index_meas = self._slider_meas.GetValue() - 1
        diagnostics = (
            self.GetParent()
            .GetParent()
            ._metric_diagnostics[index_mode][index_meas]
        )
        # Clear axes
        self._figure.clear()
        # Update images
        self._axes = self._figure.subplots(1, 2, sharex=True, sharey=True)
        self._axes[0].imshow(
            diagnostics.fftarray_sq_log,
            cmap=self._cmap_choice.GetStringSelection(),
        )
        self._axes[1].imshow(
            diagnostics.fftarray_sq_log_masked,
            cmap=self._cmap_choice.GetStringSelection(),
        )
        for a in self._axes:
            a.axis("off")
        # Update canvas
        self._canvas.draw()


_DIAGNOSTICS_PANEL_MAP = {
    "fourier": _DiagnosticsPanelFourier,
    "contrast": _DiagnosticsPanelContrast,
    "fourier_power": _DiagnosticsPanelFourierPower,
    "gradient": _DiagnosticsPanelGradient,
    "second_moment": _DiagnosticsPanelSecondMoment,
}

@dataclasses.dataclass(frozen=True)
class MetricPlotData:
    peak: numpy.ndarray
    metrics: numpy.ndarray
    modes: numpy.ndarray
    mode_label: str

class MetricPlotPanel(wx.Panel):
    _MODE_SPACING_FRACTION = 0.5

    def __init__(self, parent):
        super().__init__(parent)

        self._x_position = 0
        self._x_tick_positions = []
        self._x_tick_labels = []
        self._max_scan_range = 0
        self._margin_x = 0

        self._figure = matplotlib.figure.Figure(constrained_layout=True)
        self._axes = None
        self._canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
            self, wx.ID_ANY, self._figure
        )
        toolbar = matplotlib.backends.backend_wx.NavigationToolbar2Wx(
            self._canvas
        )
        save_images_button = wx.Button(self, label="Save raw images")
        save_data_button = wx.Button(self, label="Save data")

        toolbar.Show()
        save_images_button.Bind(wx.EVT_BUTTON, self._on_save_images)
        save_data_button.Bind(wx.EVT_BUTTON, self._on_save_data)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(save_images_button, 0)
        button_sizer.Add(save_data_button, 0, wx.LEFT, 5)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self._canvas, 1, wx.EXPAND)
        sizer.Add(toolbar, 0, wx.EXPAND)
        sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.SetSizerAndFit(sizer)

    def _on_save_images(self, _: wx.CommandEvent):
        self.GetParent().GetParent().save_images()

    def _on_save_data(self, _: wx.CommandEvent):
        self.GetParent().GetParent().save_data()

    def initialise(self, max_scan_range):
        # Clear figure
        self._figure.clear()

        # Initialise attributes
        self._x_position = 0
        self._x_tick_positions = []
        self._x_tick_labels = []
        self._max_scan_range = max_scan_range
        self._margin_x = max_scan_range * self._MODE_SPACING_FRACTION

        # Create axes and set their labels
        self._axes = self._figure.add_subplot()
        self._axes.set_xlabel("Mode")
        self._axes.set_ylabel("Metric")

    def update(self):
        data = self.GetParent().GetParent()._metric_data[-1]
        # Calculate parameters
        x_range = (self._x_position, self._x_position + self._max_scan_range)

        # Draw vertical line
        if self._x_position > 0:
            # This is not the first iteration
            x_line = self._x_position - self._margin_x / 2
            spine = list(self._axes.spines.values())[0]
            self._axes.axvline(
                x=x_line,
                color=spine.get_edgecolor(),
                linewidth=spine.get_linewidth(),
            )

        # Plot
        self._axes.plot(
            numpy.interp(
                data.modes,
                (min(data.modes), max(data.modes)),
                x_range,
            ),
            data.metrics,
            marker="o",
            color="skyblue",
        )

        # Plot peak, if it has been found
        if data.peak is not None:
            self._axes.plot(
                numpy.interp(
                    data.peak[0],
                    (min(data.modes), max(data.modes)),
                    x_range,
                ),
                data.peak[1],
                marker="+",
                markersize=20,
                color="crimson",
            )

        # Configure ticks
        tick_position = x_range[0] + self._max_scan_range / 2
        self._x_tick_positions += [tick_position]
        self._x_tick_labels += [data.mode_label]
        self._axes.xaxis.set_major_locator(
            matplotlib.ticker.FixedLocator(self._x_tick_positions)
        )
        self._axes.xaxis.set_major_formatter(
            matplotlib.ticker.FixedFormatter(self._x_tick_labels)
        )

        # Update x position
        self._x_position = x_range[1] + self._margin_x

        # Set x-axis limits
        self._axes.set_xlim(
            left=-self._margin_x / 2,
            right=self._x_position - self._margin_x / 2,
        )

        # Refresh canvas
        self._canvas.draw()

class ConventionalResultsViewer(wx.Frame):
    def __init__(self, parent):
        super().__init__(parent, title="Metric viewer")

        # Instance attributes
        self._metric_images = []
        self._metric_data = []
        self._metric_diagnostics = []
        self._metric_name = ""
        self._metric_params = {}

        self._notebook = wx.Notebook(self)

        # Sizing
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self._notebook, 1, wx.EXPAND)
        self.SetSizerAndFit(frame_sizer)

        # Subscribe to pubsub events
        cockpit.events.subscribe(
            microAO.events.PUBSUB_SENSORLESS_START, self._on_start
        )
        cockpit.events.subscribe(
            microAO.events.PUBSUB_SENSORLESS_RESULTS, self._update
        )

        # Bind to close event
        self.Bind(wx.EVT_CLOSE, self._on_close)

    def save_images(self):
        # Ask the user to select file
        fpath = None
        with wx.FileDialog(
            self,
            "Save image stack",
            wildcard="TIFF file (*.tif; *.tiff)|*.tif;*.tiff",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()
        # Modes can have different scanning ranges and therefore it is not
        # possible to order the images in a proper 2D array => save them as a
        # linear sequence
        images = numpy.array(
            [
                image
                for image_stack in self._metric_images
                for image in image_stack
            ]
        )
        tifffile.imwrite(fpath, images)

    def save_data(self):
        # Ask the user to select file
        fpath = None
        with wx.FileDialog(
            self,
            "Save metric data",
            wildcard="JSON file (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()
        # Convert data to dicts and save them as JSON
        data_dicts = []
        for data in self._metric_data:
            data_dict = dataclasses.asdict(data)
            for key in data_dict:
                if isinstance(data_dict[key], numpy.ndarray):
                    data_dict[key] = data_dict[key].tolist()
            data_dicts.append(data_dict)
        json_dict = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metric name": self._metric_name,
            "metric parameters": self._metric_params,
            "correction results": data_dicts,
        }
        with open(fpath, "w", encoding="utf-8") as fo:
            json.dump(json_dict, fo, sort_keys=True, indent=4)

    def _on_start(self, sensorless_params, sensorless_data):
        # max_scan_range, metric_name, metric_params

        # Clear attributes
        self._metric_images = []
        self._metric_data = []
        self._metric_diagnostics = []
        self._metric_name = sensorless_params["metric"]
        metric_params = {
            "wavelength": sensorless_params["wavelength"],
            "NA": sensorless_params["NA"],
            "pixel_size": sensorless_params["pixel_size"],
        }
        self._metric_params = metric_params

        # Calculate required parameters
        max_scan_range = max(
            [
                mode.offsets.max() - mode.offsets.min()
                for mode in sensorless_params["modes"]
            ]
        )

        # Delete existing notebook pages
        self._notebook.DeleteAllPages()

        # Add new notebook pages and initialise them
        for panel_class, name, init_args in (
            (MetricPlotPanel, "Metric plot", (max_scan_range,)),
            (
                _DIAGNOSTICS_PANEL_MAP[self._metric_name],
                "Metric diagnostics",
                (),
            ),
        ):
            panel = panel_class(self._notebook)
            self._notebook.AddPage(panel, name)
            panel.initialise(*init_args)

        # Re-fit the frame after the notebook has been updated
        self.Fit()

    def _update(
        self,
        results: ConventionalResults,
    ):
        # Save data
        self._metric_images.append(results.image_stack)
        metric_plot_data = MetricPlotData(
            peak = results.peak,
            metrics = results.metrics,
            modes = results.modes,
            mode_label = results.mode_label
        )
        self._metric_data.append(metric_plot_data)
        self._metric_diagnostics.append(results.metric_diagnostics)

        # Update pages
        for page_id in range(self._notebook.GetPageCount()):
            self._notebook.GetPage(page_id).update()

    def _on_close(self, evt: wx.CloseEvent):
        # Unsubscribe from events
        cockpit.events.unsubscribe(
            microAO.events.PUBSUB_SENSORLESS_START, self._on_start
        )
        cockpit.events.unsubscribe(
            microAO.events.PUBSUB_SENSORLESS_RESULTS, self._update
        )

        # Continue + destroy frame
        evt.Skip()
