import numpy
import wx
import matplotlib.ticker
import matplotlib.figure
import matplotlib.backends.backend_wxagg
import matplotlib.backends.backend_wx

import cockpit
import microAO.events


class SensorlessResultsViewer(wx.Frame):
    _NUMBER_OF_DENSE_POINTS = 100
    _MODE_SPACING_FRACTION = 0.1

    def __init__(self, parent, data):
        super().__init__(parent, title="Metric viewer")

        root_panel = wx.Panel(self)

        # Instance attributes
        self._index = 0
        self._x_tick_positions = []
        self._x_tick_labels = []

        # Add figure and canvas
        self._figure = matplotlib.figure.Figure(constrained_layout=True)
        self._axes = None
        self._canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
            root_panel, wx.ID_ANY, self._figure
        )

        # Add a toolbar
        toolbar = matplotlib.backends.backend_wx.NavigationToolbar2Wx(
            self._canvas
        )
        toolbar.Show()

        # Sizing
        root_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        root_panel_sizer.Add(self._canvas, 1, wx.EXPAND)
        root_panel_sizer.Add(toolbar, 0, wx.EXPAND)
        root_panel.SetSizerAndFit(root_panel_sizer)
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root_panel, 1, wx.EXPAND)
        self.SetSizerAndFit(frame_sizer)

        # Subscribe to pubsub events
        cockpit.events.subscribe(
            microAO.events.PUBSUB_SENSORLESS_START, self._on_start
        )
        cockpit.events.subscribe(
            microAO.events.PUBSUB_SENSORLESS_RESULTS, self._on_results
        )

        # Bind to close event
        self.Bind(wx.EVT_CLOSE, self._on_close)

        # Initialise data
        if data is not None:
            self._update(data)

    def _update(self, data):
        # Calculate a more dense fit
        mode_values_dense = None
        metrics_dense = None
        if len(data["optimal_parameters"]) > 0:
            mode_values_dense = numpy.linspace(
                min(data["mode_values"]),
                max(data["mode_values"]),
                self._NUMBER_OF_DENSE_POINTS,
            )
            metrics_dense = data["fitted_function"](
                mode_values_dense, *data["optimal_parameters"]
            )

        # Calculate the position of the corrected amplitude
        correction_metric = None
        if len(data["optimal_parameters"]) > 0:
            correction_metric = data["fitted_function"](
                data["correction_amplitude"], *data["optimal_parameters"]
            )
        else:
            correction_metric = numpy.mean(data["metrics"])

        # Calculate parameters
        margin = data["measurement_range"] * self._MODE_SPACING_FRACTION
        x_start = (
            self._index * data["measurement_range"] + self._index * margin
        )
        x_range = (x_start, x_start + data["measurement_range"])

        # Plot
        self._axes.plot(
            numpy.linspace(*x_range, len(data["mode_values"])),
            data["metrics"],
            marker="o",
            color="blue",
        )
        if mode_values_dense is not None:
            self._axes.plot(
                numpy.linspace(*x_range, len(mode_values_dense)),
                metrics_dense,
                color="orange",
            )
        self._axes.plot(
            numpy.interp(
                data["correction_amplitude"],
                (min(data["mode_values"]), max(data["mode_values"])),
                x_range,
            ),
            correction_metric,
            "rx",
        )

        # Configure ticks
        tick_position = data["measurement_range"] / 2 + x_start
        self._x_tick_positions += [tick_position]
        self._x_tick_labels += [data["mode_label"]]
        self._axes.xaxis.set_major_locator(
            matplotlib.ticker.FixedLocator(self._x_tick_positions)
        )
        self._axes.xaxis.set_major_formatter(
            matplotlib.ticker.FixedFormatter(self._x_tick_labels)
        )

        # Set x-axis limits
        self._axes.set_xlim(left=0, right=x_range[1])

        # Increment plotting index
        self._index += 1

        # Refresh canvas
        self._canvas.draw()

    def _on_start(self):
        self._figure.clear()
        self._index = 0
        self._x_tick_positions = []
        self._x_tick_labels = []
        self._axes = self._figure.add_subplot()
        self._axes.set_xlabel("Mode")
        self._axes.set_ylabel("Metric")

    def _on_results(self, data):
        self._update(data)

    def _on_close(self, evt):
        # Unsubscribe from events
        cockpit.events.unsubscribe(
            microAO.events.PUBSUB_SENSORLESS_START, self._on_start
        )
        cockpit.events.unsubscribe(
            microAO.events.PUBSUB_SENSORLESS_RESULTS, self._on_results
        )

        # Continue + destroy frame
        evt.Skip()
