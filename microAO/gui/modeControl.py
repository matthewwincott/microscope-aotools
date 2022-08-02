import wx
import wx.lib.scrolledpanel
import numpy as np
import cockpit
import cockpit.events
import microAO
import microAO.events
import microAO.gui.common


_DEFAULT_ZERNIKE_MODE_NAMES = {
    1: "Piston",
    2: "Tip",
    3: "Tilt",
    4: "Defocus",
    5: "Astig (O)",
    6: "Astig (V)",
    7: "Coma (V)",
    8: "Coma (H)",
    9: "Trefoil (V)",
    10: "Trefoil (O)",
    11: "Spherical",
    12: "Astig 2 (V)",
    13: "Astig 2 (O)",
    14: "Quadrafoil (V)",
    15: "Quadrafoil (O)",
}


class _ModesPanel(wx.lib.scrolledpanel.ScrolledPanel):
    _MIN_AMPLITUDE = 0.5

    def __init__(self, parent, device):
        super().__init__(parent)

        # Set attributes
        self._device = device
        self._n_modes = self._device.proxy.get_controlMatrix().shape[1]

        # Create root panel and sizer
        sizer = wx.GridBagSizer()
        sizer.SetCols(6)
        sizer.AddGrowableCol(3)
        row_counter = 0

        # Reset button
        reset_btn = wx.Button(self, label="Reset")
        reset_btn.Bind(wx.EVT_BUTTON, self._on_reset)
        sizer.Add(
            reset_btn,
            wx.GBPosition(row_counter, 0),
            wx.GBSpan(1, 6),
            wx.ALL,
            5,
        )
        row_counter += 1

        # Mode filter
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        filter_modes_lbl = wx.StaticText(self, label="Mode filter:")
        self.filter_modes = microAO.gui.common.FilterModesCtrl(
            self, value="{}-{}".format(1, self._n_modes)
        )
        self.filter_modes.Bind(wx.EVT_TEXT, self._on_filter_modes)
        hbox.Add(filter_modes_lbl, 0)
        hbox.Add(self.filter_modes, 0, wx.LEFT, 10)
        sizer.Add(
            hbox, wx.GBPosition(row_counter, 0), wx.GBSpan(1, 6), wx.ALL, 5
        )
        row_counter += 1

        # Amplitude field
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        amplitude_label = wx.StaticText(self, label="Amplitude:")
        self._amplitude = microAO.gui.common.FloatCtrl(
            self, value="1.5", style=wx.TE_PROCESS_ENTER
        )
        self._amplitude.Bind(wx.EVT_TEXT_ENTER, self._on_amplitude)
        self._amplitude.Bind(wx.EVT_KILL_FOCUS, self._on_amplitude)
        hbox.Add(amplitude_label, 0)
        hbox.Add(self._amplitude, 0, wx.LEFT, 10)
        sizer.Add(
            hbox, wx.GBPosition(row_counter, 0), wx.GBSpan(1, 6), wx.ALL, 5
        )
        row_counter += 1

        # Spacer
        sizer.Add(0, 10, wx.GBPosition(row_counter, 0), wx.GBSpan(1, 6))
        row_counter += 1

        # Set headings
        headings = (
            ("Mode", wx.GBPosition(row_counter, 0), wx.GBSpan(1, 2)),
            ("Min", wx.GBPosition(row_counter, 2), wx.GBSpan(1, 1)),
            ("Control", wx.GBPosition(row_counter, 3), wx.GBSpan(1, 1)),
            ("Max", wx.GBPosition(row_counter, 4), wx.GBSpan(1, 1)),
            ("Value", wx.GBPosition(row_counter, 5), wx.GBSpan(1, 1)),
        )
        row_counter += 1
        font = wx.Font(wx.FontInfo(10).Bold())
        for heading in headings:
            heading_widget = wx.StaticText(
                self, label=heading[0], style=wx.ALIGN_CENTRE_HORIZONTAL
            )
            heading_widget.SetFont(font)
            sizer.Add(
                heading_widget,
                heading[1],
                heading[2],
                wx.EXPAND | wx.ALIGN_CENTRE_VERTICAL | wx.BOTTOM,
                5,
            )

        # Add control per mode
        modes = np.zeros(self._n_modes)
        last_modes = self._device.proxy.get_last_modes()

        if last_modes is not None:
            modes += last_modes
        self._mode_controls = {}
        for mode_index, mode_value in enumerate(modes):
            mode_number = mode_index + 1
            # Create the mode controls
            (
                self._mode_controls[mode_number],
                new_rows,
            ) = self._create_mode_controls(
                mode_number, grid=sizer, row=row_counter
            )
            row_counter += new_rows
            # Initialise value
            self._synchronised_update(mode_number, mode_value)

        # Set sizer and finalise layout
        self.SetSizerAndFit(sizer)
        self.SetupScrolling()
        self.Layout()

        # Initialise the mode control correction
        self._device.set_correction("mode control")

        # Subscribe to pubsub events
        cockpit.events.subscribe(
            microAO.events.PUBSUB_SET_PHASE, self._on_new_modes
        )
        cockpit.events.subscribe(
            microAO.events.PUBUSB_CHANGED_CORRECTION, self._on_new_modes
        )

        # Bind close event
        parent.Bind(wx.EVT_CLOSE, self._on_close)

    def _create_mode_controls(self, mode_number, grid, row):
        # Create widget table with the following columns: widget, column, span,
        # flags, border
        widgets_data = (
            (
                wx.StaticText(
                    self,
                    label=str(mode_number),
                    style=wx.ALIGN_CENTRE_HORIZONTAL,
                ),
                wx.GBPosition(row, 0),
                wx.GBSpan(1, 1),
                wx.EXPAND | wx.ALIGN_CENTRE_VERTICAL | wx.LEFT | wx.RIGHT,
                4,
            ),
            (
                wx.StaticText(
                    self,
                    label=_DEFAULT_ZERNIKE_MODE_NAMES.get(mode_number, ""),
                    style=wx.ALIGN_CENTRE_HORIZONTAL,
                ),
                wx.GBPosition(row, 1),
                wx.GBSpan(1, 1),
                wx.EXPAND | wx.ALIGN_CENTRE_VERTICAL | wx.LEFT | wx.RIGHT,
                4,
            ),
            (
                wx.StaticText(
                    self,
                    label=str(-self._amplitude.value),
                    style=wx.ALIGN_CENTRE_HORIZONTAL,
                ),
                wx.GBPosition(row, 2),
                wx.GBSpan(1, 1),
                wx.EXPAND | wx.ALIGN_CENTRE_VERTICAL | wx.LEFT | wx.RIGHT,
                4,
            ),
            (
                wx.Slider(self, value=0, minValue=-100, maxValue=100),
                wx.GBPosition(row, 3),
                wx.GBSpan(1, 1),
                wx.EXPAND | wx.ALIGN_CENTRE_VERTICAL | wx.LEFT | wx.RIGHT,
                4,
            ),
            (
                wx.StaticText(
                    self,
                    label=str(self._amplitude.value),
                    style=wx.ALIGN_CENTRE_HORIZONTAL,
                ),
                wx.GBPosition(row, 4),
                wx.GBSpan(1, 1),
                wx.ALIGN_CENTRE_VERTICAL | wx.LEFT | wx.RIGHT,
                4,
            ),
            (
                wx.SpinCtrlDouble(
                    self,
                    initial=0.0,
                    min=-self._amplitude.value,
                    max=self._amplitude.value,
                    inc=0.01,
                    style=wx.TE_PROCESS_ENTER,
                ),
                wx.GBPosition(row, 5),
                wx.GBSpan(1, 1),
                wx.ALIGN_CENTRE_VERTICAL | wx.LEFT | wx.RIGHT,
                4,
            ),
            (
                wx.GBSizerItem(
                    0, 10, wx.GBPosition(row + 1, 0), wx.GBSpan(1, 6)
                ),
            ),
            (
                wx.StaticText(
                    self, label="-1.0", style=wx.ALIGN_CENTRE_HORIZONTAL
                ),
                wx.GBPosition(row + 2, 2),
                wx.GBSpan(1, 1),
                wx.ALIGN_CENTRE_VERTICAL | wx.LEFT | wx.RIGHT,
                4,
            ),
            (
                microAO.gui.common.ModeIndicator(self, size_hints=(-1, 7)),
                wx.GBPosition(row + 2, 3),
                wx.GBSpan(1, 1),
                wx.EXPAND | wx.ALIGN_CENTRE_VERTICAL | wx.LEFT | wx.RIGHT,
                8,
            ),
            (
                wx.StaticText(
                    self, label="1.0", style=wx.ALIGN_CENTRE_HORIZONTAL
                ),
                wx.GBPosition(row + 2, 4),
                wx.GBSpan(1, 1),
                wx.ALIGN_CENTRE_VERTICAL | wx.LEFT | wx.RIGHT,
                4,
            ),
            (
                wx.StaticText(
                    self, label="0.000", style=wx.ALIGN_CENTRE_HORIZONTAL
                ),
                wx.GBPosition(row + 2, 5),
                wx.GBSpan(1, 1),
                wx.ALIGN_CENTRE_VERTICAL | wx.LEFT | wx.RIGHT,
                4,
            ),
            (
                wx.GBSizerItem(
                    0, 10, wx.GBPosition(row + 3, 0), wx.GBSpan(1, 6)
                ),
            ),
        )
        # Style the mode indicator labels
        widgets_data[7][0].SetForegroundColour(wx.Colour(128, 128, 128))
        widgets_data[9][0].SetForegroundColour(wx.Colour(128, 128, 128))
        widgets_data[10][0].SetForegroundColour(wx.Colour(128, 128, 128))
        # Event binding
        widgets_data[3][0].Bind(
            wx.EVT_SCROLL,
            lambda event: self._on_slider(
                mode_number, event.GetEventObject().GetValue()
            ),
        )
        widgets_data[5][0].Bind(
            wx.EVT_SPINCTRLDOUBLE,
            lambda event: self._on_mode_value(
                mode_number, event.GetEventObject().GetValue()
            ),
        )
        widgets_data[5][0].Bind(
            wx.EVT_TEXT_ENTER,
            lambda event: self._on_mode_value(
                mode_number, float(event.GetString())
            ),
        )
        # Layout
        for widget_data in widgets_data:
            # Add the widget
            grid.Add(*widget_data)
        return [widget_data[0] for widget_data in widgets_data], 4

    def _on_reset(self, _):
        for mode_number in self._mode_controls.keys():
            self._synchronised_update(mode_number, 0.0)
        self._apply_modes()

    def _on_filter_modes(self, _):
        # Show only filtered modes
        modes_filtered = self.filter_modes.GetValue()
        for mode_number in self._mode_controls.keys():
            if mode_number in modes_filtered:
                for control in self._mode_controls[mode_number]:
                    control.Show(True)
            else:
                for control in self._mode_controls[mode_number]:
                    control.Show(False)
        self.SetupScrolling()

    def _on_amplitude(self, event):
        new_amplitude = max(abs(self._amplitude.value), self._MIN_AMPLITUDE)
        last_amplitude = float(
            next(iter(self._mode_controls.values()))[4].GetLabelText()
        )
        # Do nothing if the amplitude has not changed
        if new_amplitude == last_amplitude:
            event.Skip()
            return
        # Ensure text control shows a positive value
        self._amplitude.ChangeValue(str(new_amplitude))
        # Ensure amplitude is no less than the maximum mode value
        max_mode_value = 0.0
        max_mode_number = 1
        for mode_number in self._mode_controls.keys():
            mode_value = abs(self._mode_controls[mode_number][5].GetValue())
            if mode_value > max_mode_value:
                max_mode_number = mode_number
                max_mode_value = mode_value
        if new_amplitude < max_mode_value:
            # Restore the last amplitude, raise a warning message, and exit
            self._amplitude.ChangeValue(str(last_amplitude))
            with wx.MessageDialog(
                self,
                message=(
                    "Failed to set amplitude because the requested value "
                    f"({new_amplitude}) is less than the value for mode "
                    f"{max_mode_number}: {max_mode_value}."
                ),
                style=wx.ICON_EXCLAMATION | wx.STAY_ON_TOP | wx.OK,
            ) as dlg:
                dlg.ShowModal()
            event.Skip()
            return
        # Update all modes' controls
        for mode_number in self._mode_controls.keys():
            # Change min/max labels
            self._mode_controls[mode_number][2].SetLabel(str(-new_amplitude))
            self._mode_controls[mode_number][4].SetLabel(str(new_amplitude))
            # Change value's min/max
            mode_value = self._mode_controls[mode_number][5].GetValue()
            self._mode_controls[mode_number][5].SetMin(-new_amplitude)
            self._mode_controls[mode_number][5].SetMax(new_amplitude)
            # Change slider position
            self._mode_controls[mode_number][3].SetValue(
                round((mode_value / new_amplitude) * 100.0)
            )
        event.Skip()

    def _on_slider(self, mode_number, slider_value):
        self._synchronised_update(
            mode_number, (slider_value / 100) * self._amplitude.value
        )
        # Change value
        self._apply_modes()

    def _on_mode_value(self, mode_number, mode_value):
        self._synchronised_update(mode_number, mode_value)
        # Change slider
        self._apply_modes()

    def _on_new_modes(self, *_):
        corrections = self._device.get_corrections(include_default=True)
        del corrections["mode control"]
        modes, _ = self._device.sum_corrections(corrections)
        # Calculate new amplitude to the nearest 0.5
        new_amplitude = max(
            np.ceil(np.max(np.abs(modes)) * 2) / 2,
            self._MIN_AMPLITUDE
        )
        # Process each mode
        for index, value in enumerate(modes):
            mode_number = index + 1
            # Update min/max labels
            self._mode_controls[mode_number][7].SetLabel(str(-new_amplitude))
            self._mode_controls[mode_number][9].SetLabel(str(new_amplitude))
            # Update mode indicator
            self._mode_controls[mode_number][8].Update(value, new_amplitude)
            # Update text field
            self._mode_controls[mode_number][10].SetLabel(f"{value:.3f}")

    def _on_close(self, event):
        # Unsubscribe from pubsub events
        cockpit.events.unsubscribe(
            microAO.events.PUBSUB_SET_PHASE, self._on_new_modes
        )
        cockpit.events.unsubscribe(
            microAO.events.PUBUSB_CHANGED_CORRECTION, self._on_new_modes
        )
        # Continue + destroy frame
        event.Skip()

    def _synchronised_update(self, mode_number, mode_value):
        # Change value
        self._mode_controls[mode_number][5].SetValue(mode_value)
        # Update the slider position
        self._mode_controls[mode_number][3].SetValue(
            round((mode_value / self._amplitude.value) * 100.0)
        )

    def _apply_modes(self):
        modes = []
        for i in range(self._n_modes):
            modes.append(self._mode_controls[i + 1][5].GetValue())
        self._device.set_correction("mode control", np.array(modes))
        if self._device.get_corrections()["mode control"]["enabled"]:
            self._device.refresh_corrections()


class ModesControl(wx.Frame):
    def __init__(self, parent, device):
        super().__init__(parent)
        self._panel = _ModesPanel(self, device)
        self._sizer = wx.BoxSizer(wx.VERTICAL)
        self._sizer.Add(self._panel, 1, wx.EXPAND)
        self.SetSizer(self._sizer)
        self.SetMinSize(wx.Size(650, 300))
        self.SetSize(wx.Size(650, 650))
        self.SetTitle("DM mode control")
