import casacore.tables
import numpy as np
from typing import Union

class DataHandler:
    """
    A class to read in measurement set data and provide methods to access the data.
    """

    def __init__(self, ms_path: str) -> None:
        """Initialize the DataHandler with the path to the measurement set.

        Parameters
        ----------
        ms_path : string
            path to the measurement set file.
        """
        self.ms_path = ms_path
        self.tbl = casacore.tables.table(ms_path)
        self.spec = casacore.tables.table(f"{ms_path}/SPECTRAL_WINDOW")
        self.time_all = self.tbl.getcol("TIME")
        self.uvw = self.tbl.getcol("UVW")
        self.data = self.tbl.getcol("DATA")

    def get_autocorr_filter(self) -> np.ndarray:
        """Return the autocorrelation filter.

        Returns
        -------
        numpy.ndarray
            A boolean array indicating autocorrelation visibilities.
        """
        return self.uvw[:, 0] == 0

    def get_visibilities(self) -> np.ndarray:
        """Return the visibilities summed over polarizations.

        Returns
        -------
        numpy.ndarray
            Visibility values summed over polarizations.
        """
        return np.sum(self.data, axis=2)

    def get_times(self) -> np.ndarray:
        """Return the unique time values and time step.

        Returns
        -------
        numpy.ndarray
            The unique time values from the measurement set.
        """
        return np.unique(self.time_all)

    def get_time_step(self) -> Union[float, int]:
        """Return the interval between successive time values.

        Returns
        -------
        numpy.float64
            The inteval between the first two unique time values.
        """
        times = self.get_times()
        dt = times[1] - times[0] if len(times) > 1 else 0
        return dt
    
if __name__ == '__main__':
    # Initialize the data handler.
    data_handler = DataHandler('../../pipeline_problem_data.ms')
    print("Data shape: ", data_handler.data.shape)
    print("Time shape: ", data_handler.time_all.shape)
    print("UVW shape: ", data_handler.uvw.shape)
    print("Freq shape: ", data_handler.spec.getcol('CHAN_FREQ').shape)
    print("Visibilities shape:", data_handler.get_visibilities().shape)
    print("Unique times:", data_handler.get_times())
    print("Time step:", data_handler.get_time_step())
    print("Autocorrelation filter:", data_handler.get_autocorr_filter())