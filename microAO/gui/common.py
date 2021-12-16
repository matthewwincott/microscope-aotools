import wx

from cockpit.gui.guiUtils import FLOATVALIDATOR

class FloatCtrl(wx.TextCtrl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, validator=FLOATVALIDATOR, **kwargs)

    @property
    def value(self):
        try:
            val = float(self.GetValue())
        except Exception as e:
            val = None

        return val