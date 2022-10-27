import abc
from dataclasses import dataclass
import typing
import numpy as np

from microAO.aoMetrics import metric_function
from microAO.aoAlg import AdaptiveOpticsFunctions


@dataclass
class RoutineOutput():
    sensorless_data: dict           # Data stored between correction rounds
    done: bool = False              # Flag to indicate routine completion
    new_modes: typing.List = None   # Modes to set before next image is taken (optional)
    result: typing.Any = None       # Results returned from the routine (optional)
    status: str = None              # Status message (optional)
    error: str = None               # Error message (optional)

class Routine(metaclass=abc.ABCMeta):
    """An abstract base class for an AO routine

        A setup and image processing method must be defined.

    """

    def __init__(self, sensorless_params, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sensorless_params = sensorless_params

    @staticmethod
    @abc.abstractmethod
    def name():
        """ Return a readable name for the routine.
            eg. return 'Conventional'
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def defaults():
        """ Return dict of default parameters for the routine.
        """

    @abc.abstractmethod
    def setup(self, sensorless_data) -> dict:
        """ Perform routine setup. Returns a dict"""
        pass

    @abc.abstractmethod
    def process(self, sensorless_data) -> dict:
        """Process each image."""
        pass

@dataclass(frozen=True)
class ConventionalResults:
    metrics: typing.List
    image_stack: typing.List
    metric_diagnostics: typing.List
    modes: np.ndarray
    mode_label: str
    peak: np.ndarray = None

@dataclass(frozen=True)
class ConventionalParamsMode:
    # Noll index
    index_noll: int
    # The amplitude offsets used for scanning the mode
    offsets: np.ndarray

class ConventionalRoutine(Routine):
    def name():
        return "Conventional"

    @staticmethod
    def defaults():
        parameters = {
            "num_reps": 1,
            "NA": 1.1,
            "wavelength": 560e-9,
            "metric": 'fourier',
            "modes": (
                ConventionalParamsMode(11, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(22, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(5, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(6, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(7, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(8, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(9, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(10, np.linspace(-1.5, 1.5, 7)),
            ),
            "datapoint_z": None,
            "save_as_datapoint": False,
            "log_path": None
        }

        return parameters

    def setup(self, sensorless_data):
        # Define additional data required for routine
        total_measurements = sum([len(mode.offsets) for mode in self.sensorless_params["modes"]]) * self.sensorless_params["num_reps"]
        additional_data = {
            "total_measurements": total_measurements,
            "mode_index": 0,
            "offset_index": 0,
        }

        # Merge additional data (note in-place merge of mutable dict)
        sensorless_data.update(additional_data)

        # Define the first correction to apply
        new_modes = sensorless_data["corrections"].copy()
        new_modes[
            self.sensorless_params["modes"][sensorless_data["mode_index"]].index_noll
            - 1
        ] += self.sensorless_params["modes"][sensorless_data["mode_index"]].offsets[
            sensorless_data["offset_index"]
        ]

        # Update status message
        status_message = self._get_status_message(sensorless_data)

        # Format return data
        return_data = RoutineOutput(
            sensorless_data = sensorless_data,
            status = status_message,
            new_modes = new_modes,
        )

        return return_data

    def process(self, sensorless_data):
        return_data = {}

        # Set default result
        result = None

        # Correct mode if enough measurements have been taken
        if sensorless_data["offset_index"] == (
            self.sensorless_params["modes"][sensorless_data["mode_index"]].offsets.shape[0]
            - 1
        ):
            # Calculate required parameters
            mode_index_noll_0 = (
                self.sensorless_params["modes"][sensorless_data["mode_index"]].index_noll - 1
            )
            modes = (
                sensorless_data["corrections"][mode_index_noll_0]
                + self.sensorless_params["modes"][sensorless_data["mode_index"]].offsets
            )
            image_stack = sensorless_data["image_stack"][-modes.shape[0] :]

            # Find aberration amplitudes and correct
            peak, metrics, metric_diagnostics = AdaptiveOpticsFunctions.find_zernike_amp_sensorless(
                image_stack=image_stack,
                modes=modes,
                metric_name=self.sensorless_params["metric"],
                wavelength=self.sensorless_params["wavelength"],
                NA=self.sensorless_params["NA"],
                pixel_size=self.sensorless_params["pixel_size"],
            )

            # If a peak isn't found, set abort flag
            if peak is not None:
                # Set correction (and label) in return data
                sensorless_data["corrections"][mode_index_noll_0] = peak[0]

            # Append metrics to stack
            sensorless_data["metrics_stack"].append(metrics.tolist())

            # Instantiate result
            result = ConventionalResults(
                metrics = metrics,
                image_stack = image_stack,
                metric_diagnostics = metric_diagnostics,
                modes = modes,
                mode_label = f"Z{mode_index_noll_0 + 1}",
                peak = peak,
            )

            # Update indices
            sensorless_data["offset_index"] = 0
            sensorless_data["mode_index"] += 1
            if sensorless_data["mode_index"] == len(self.sensorless_params["modes"]):
                sensorless_data["mode_index"] = 0

        else:
            # Increment offset index
            sensorless_data["offset_index"] += 1

        # Update status message
        status_message = self._get_status_message(sensorless_data)

        # Format return data
        return_data = RoutineOutput(
            sensorless_data = sensorless_data,
            status = status_message,
            result = result
        )

        # Set next mode and return data, unless all measurements acquired
        if len(sensorless_data["image_stack"]) < sensorless_data["total_measurements"]:
            # Apply next set of modes
            new_modes = sensorless_data["corrections"].copy()
            new_modes[
                self.sensorless_params["modes"][sensorless_data["mode_index"]].index_noll - 1
            ] += self.sensorless_params["modes"][sensorless_data["mode_index"]].offsets[
                sensorless_data["offset_index"]
            ]

            return_data.new_modes = new_modes

        # If all data acquired, set completion flag
        else:
            return_data.done = True

        return return_data

    def _get_status_message(self, sensorless_data):
        # Update status message
        status_message = "Sensorless AO: image {n}/{N}, mode {n_mode}, meas. {n_meas}".format(
            n = len(sensorless_data["image_stack"]) + 1,
            N = sensorless_data["total_measurements"],
            n_mode = self.sensorless_params["modes"][
                sensorless_data["mode_index"]
            ].index_noll,
            n_meas = sensorless_data["offset_index"] + 1,
        )

        return status_message

routines = {
    'conventional': ConventionalRoutine,
}