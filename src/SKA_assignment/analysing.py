import scipy
import numpy as np

from SKA_assignment.data_handler import DataHandler
from SKA_assignment.plotting import plot_amplitude_vs_time

def monitor_data_quality(data_path: str, generate_plots: bool = False, first_t_frame: int = 0, nb_t_steps: int = 120, t_step: int = 1, first_freq_step: int = 0, n_freq_steps: int = 32, freq_step: int = 1, flag_multiplier: int = 10) -> None:
    """Monitor the data quality by checking the median and median absolute deviation of the visibility amplitude over time, for each frequency channel.

    Parameters
    ----------
    data_path : str
        Path to the measurement set file.
    generate_plots : bool, optional
        If true, generate a visibility amplitude vs time plot for each channel in the channel range, by default False.
    first_t_frame : int, optional
        Used to select a subset of time frames to analyse. 
        Time frame to start analysis from, by default 0.
    nb_t_steps : int, optional
        Used to select a subset of time frames to analyse.
        How many time frames to include, by default all 120.
    t_step : int, optional
        Used to select a subset of time frames to analyse.
        How many time frames to include in an averaged step, by default 1.
    first_freq_step : int, optional
        Used to select a subset of channels to analyse.
        Frequency channel to start analysis from, by default 0.
    n_freq_steps : int, optional
        Used to select a subset of channels to analyse.
        How many channels to include, by default all 32.
    freq_step : int, optional
        Used to select a subset of channels to analyse.
        How many channels to include in an averaged step, by default 1.
    flag_multiplier : int, optional
        How many MADs away from the median is a datapoint considered an outlier, by default 10.
    """
    # Load the data
    data_handler = DataHandler('../../pipeline_problem_data.ms')

    time_all = data_handler.time_all
    autocorr_filter = data_handler.get_autocorr_filter()
    vis = data_handler.get_visibilities()
    unique_times = data_handler.get_times()
    dt = data_handler.get_time_step()

    # Get the amplitude of the visibilities and group them by frequency instead
    amplitude_all_vis = np.abs(vis).T

    # Extract the requested frequency slices
    amplitude_vis = amplitude_all_vis[first_freq_step:first_freq_step+n_freq_steps*freq_step, :]

    # Create a mask to extract the requested time slices for each frequency
    combined_masks = []
    for i, start_frame in enumerate(range(first_t_frame, first_t_frame + nb_t_steps * t_step, t_step)):
        end_frame = start_frame + t_step
        t_start = unique_times[start_frame]
        try:
            t_end = unique_times[end_frame]
        except IndexError:
            t_end = unique_times[-1] + dt

        # Include the autocorrelation filter in the mask
        combined_masks.append([(~autocorr_filter) & (time_all >= t_start) & (time_all < t_end)])

    # Compute binned amplitudes for all frequencies and time bins in one go
    binned_amplitude = np.array([[
        np.mean(amplitude_vis[freq_index:freq_index+freq_step][combined_mask], axis=0) if np.any(combined_mask) else np.nan
        for combined_mask in combined_masks
        ]
        for freq_index in range(n_freq_steps)
    ])  # Shape: (nb_t_steps, n_freq_steps)

    # Compute statistics (median and median abs deviation) for each frequency
    amplitude_median = np.nanmedian(binned_amplitude, axis=1)  # Shape: (n_freq_steps,)
    amplitude_mad = scipy.stats.median_abs_deviation(binned_amplitude, axis=1, scale="normal")  # Shape: (n_freq_steps,)

    # Create a mask for outliers based on the median and MAD
    outlier_mask = np.array([
            (binned_amplitude[i, :] >= (amplitude_median[i] + flag_multiplier * amplitude_mad[i])) | \
            (binned_amplitude[i, :] <= amplitude_median[i] - flag_multiplier * amplitude_mad[i]) 
            for i in range(n_freq_steps)
            ])

    print(amplitude_median)
    print(amplitude_mad)

    # Plot results for each frequency
    if generate_plots:
        for i in range(n_freq_steps):
            # fig, ax = plt.subplots(figsize=(10, 5))
            title = f"Time Series of Visibility Amplitude for Channel {first_freq_step + i * freq_step}"
            filename = f"amplitude_vs_time_channel_{first_freq_step + i * freq_step}.png"
            plot_amplitude_vs_time(binned_amplitude[i, :], range(first_t_frame, first_t_frame + nb_t_steps), title, filename, outlier_mask[i, :], amplitude_median = amplitude_median[i], amplitude_mad = amplitude_mad[i], multiplier = flag_multiplier)

if __name__ == '__main__':
    monitor_data_quality("../../pipeline_problem_data.ms", generate_plots=True, flag_multiplier=10)