import dataclasses
import json
import numpy
import wx
import matplotlib.ticker
import matplotlib.figure
import matplotlib.backends.backend_wxagg
import matplotlib.backends.backend_wx

import cockpit
import microAO.events


@dataclasses.dataclass(frozen=True)
class SensorlessResultsData:
    metrics: numpy.ndarray
    modes: numpy.ndarray
    mode_label: str
    peak: numpy.ndarray

class SensorlessResultsViewer(wx.Frame):
    _MODE_SPACING_FRACTION = 0.1

    def __init__(self, parent):
        super().__init__(parent, title="Metric viewer")

        root_panel = wx.Panel(self)

        # Instance attributes
        self._data = []
        self._x_position = 0
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

        # Add a button for saving of data
        save_data_button = wx.Button(root_panel, label="Save data")
        save_data_button.Bind(wx.EVT_BUTTON, self._on_save_data)

        # Sizing
        root_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        root_panel_sizer.Add(self._canvas, 1, wx.EXPAND)
        root_panel_sizer.Add(toolbar, 0, wx.EXPAND)
        root_panel_sizer.Add(save_data_button, 0, wx.ALL, 5)
        root_panel.SetSizerAndFit(root_panel_sizer)
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root_panel, 1, wx.EXPAND)
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

    def _on_save_data(self, evt: wx.CommandEvent):
        # Ask the user to select file
        fpath = None
        with wx.FileDialog(
            self,
            "Save modes",
            wildcard="JSON file (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as fileDialog:
            if fileDialog.ShowModal() != wx.ID_OK:
                return
            fpath = fileDialog.GetPath()
        # Convert data to dicts and save them as JSON
        data_dicts = []
        for data in self._data:
            data_dict = dataclasses.asdict(data)
            for key in data_dict:
                if isinstance(data_dict[key], numpy.ndarray):
                    data_dict[key] = data_dict[key].tolist()
            data_dicts.append(data_dict)
        with open(fpath, "w", encoding="utf-8") as fo:
            json.dump(data_dicts, fo, sort_keys=True, indent=4)

    def _on_start(self):
        self._figure.clear()
        self._data = []
        self._x_position = 0
        self._x_tick_positions = []
        self._x_tick_labels = []
        self._axes = self._figure.add_subplot()
        self._axes.set_xlabel("Mode")
        self._axes.set_ylabel("Metric")

    def _update(self, data: SensorlessResultsData):
        # Calculate parameters
        all_modes = numpy.append(data.modes, data.peak[0])
        mode_range = data.modes.max() - data.modes.min()
        margin_x = mode_range * self._MODE_SPACING_FRACTION
        x_range = (
            self._x_position,
            self._x_position + mode_range
        )

        # Plot
        self._axes.plot(
            numpy.interp(
                data.modes,
                (min(all_modes), max(all_modes)),
                x_range,
            ),
            data.metrics,
            marker="o",
            color="skyblue"
        )
        self._axes.plot(
            numpy.interp(
                data.peak[0],
                (min(all_modes), max(all_modes)),
                x_range,
            ),
            data.peak[1],
            marker="+",
            markersize=20,
            color="crimson"
        )

        # Configure ticks
        tick_position = x_range[0] + mode_range / 2
        self._x_tick_positions += [tick_position]
        self._x_tick_labels += [data.mode_label]
        self._axes.xaxis.set_major_locator(
            matplotlib.ticker.FixedLocator(self._x_tick_positions)
        )
        self._axes.xaxis.set_major_formatter(
            matplotlib.ticker.FixedFormatter(self._x_tick_labels)
        )
        
        # Update x position
        self._x_position = x_range[1] + margin_x

        # Set x-axis limits
        self._axes.set_xlim(left=-margin_x, right=self._x_position)

        # Refresh canvas
        self._canvas.draw()

        # Save data
        self._data.append(data)

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
