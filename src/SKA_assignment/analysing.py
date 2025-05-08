import scipy
import numpy as np
import logging
import math

from SKA_assignment.data_handler import DataHandler
from SKA_assignment.imaging import get_dirty_image, get_psf, get_beam, deconvolve_image
from SKA_assignment.plotting import plot_dirty_clean, plot_amplitude_vs_time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def monitor_data_quality(
    data_handler: DataHandler,
    generate_plots: bool = True,
    save_plots: bool = False,
    output_dir: str = None,
    first_t_frame: int = 0,
    nb_t_steps: int = 120,
    t_step: int = 1,
    first_freq_step: int = 0,
    n_freq_steps: int = 32,
    freq_step: int = 1,
    flag_multiplier: int = 10,
) -> None:
    """Monitor the data quality by checking the median and median absolute deviation
    of the visibility amplitude over time, for each frequency channel.

    Parameters
    ----------
    data_handler : DataHandler
        DataHandler object containing the measurement set data.
    generate_plots : bool, optional
        If true, generate a visibility amplitude vs time plot for each channel
        in the channel range, by default False.
    save_plots : bool, optional
        If true, save the plots to the output directory, by default False.
    ouput_dir : str, optional
        Directory to save the plots, by default None.
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
    # Unpack data from the DataHandler
    time_all = data_handler.time_all
    autocorr_filter = data_handler.get_autocorr_filter()
    vis = data_handler.get_visibilities()
    unique_times = data_handler.get_times()
    dt = data_handler.get_time_step()

    # Get the amplitude of the visibilities and group them by frequency instead
    amplitude_all_vis = np.abs(vis).T

    # Extract the requested frequency slices
    amplitude_vis = amplitude_all_vis[
        first_freq_step : first_freq_step + n_freq_steps * freq_step, :
    ]

    # Create a mask to extract the requested time slices for each frequency
    combined_masks = []
    for i, start_frame in enumerate(
        range(first_t_frame, first_t_frame + nb_t_steps * t_step, t_step)
    ):
        end_frame = start_frame + t_step
        t_start = unique_times[start_frame]
        try:
            t_end = unique_times[end_frame]
        except IndexError:
            t_end = unique_times[-1] + dt

        # Include the autocorrelation filter in the mask
        combined_masks.append(
            [(~autocorr_filter) & (time_all >= t_start) & (time_all < t_end)]
        )

    # Compute binned amplitudes for all frequencies and time bins in one go
    binned_amplitude = np.array(
        [
            [
                np.mean(
                    amplitude_vis[freq_index : freq_index + freq_step][combined_mask],
                    axis=0,
                )
                if np.any(combined_mask)
                else np.nan
                for combined_mask in combined_masks
            ]
            for freq_index in range(n_freq_steps)
        ]
    )

    # Compute statistics (median and median abs deviation) for each frequency
    amplitude_median = np.nanmedian(binned_amplitude, axis=1)  # Shape: (n_freq_steps,)
    amplitude_mad = scipy.stats.median_abs_deviation(
        binned_amplitude, axis=1, scale="normal"
    )  # Shape: (n_freq_steps,)

    # Create a mask for outliers based on the median and MAD
    outlier_mask = np.array(
        [
            (
                binned_amplitude[i, :]
                >= (amplitude_median[i] + flag_multiplier * amplitude_mad[i])
            )
            | (
                binned_amplitude[i, :]
                <= amplitude_median[i] - flag_multiplier * amplitude_mad[i]
            )
            for i in range(n_freq_steps)
        ]
    )

    # Log statistics per channel
    stats_str = "\n".join(
        f"Channel {first_freq_step + i * freq_step}: Median = {amplitude_median[i]}, MAD = {amplitude_mad[i]}"
        for i in range(n_freq_steps)
    )
    logging.info(stats_str)

    # Plot results for each frequency
    if generate_plots:
        for i in range(n_freq_steps):
            title = f"Time Series of Visibility Amplitude for Channel {first_freq_step + i * freq_step}"
            filename = (
                f"amplitude_vs_time_channel_{first_freq_step + i * freq_step}.png"
            )
            plot_amplitude_vs_time(
                binned_amplitude[i, :],
                range(first_t_frame, first_t_frame + nb_t_steps),
                title,
                filename,
                outlier_mask[i, :],
                amplitude_median=amplitude_median[i],
                amplitude_mad=amplitude_mad[i],
                multiplier=flag_multiplier,
                save=save_plots,
                output_dir=output_dir,
            )


def visualize_data(
    data_handler,
    apply_flags=False,
    save_plots: bool = False,
    output_dir: str = None,
    vmin: float = -20000,
    vmax: float = 200000,
    first_t_frame: int = 0,
    nb_t_steps: int = 120,
    t_step: int = 1,
    first_freq_step: int = 0,
    n_freq_steps: int = 32,
    freq_step: int = 1,
) -> None:
    """Generate dirty and cleaned images for a range of time and frequencies.

    Parameters
    ----------
    data_handler : DataHandler
        DataHandler object containing the measurement set data.
    apply_flags : bool, optional
        If true, remove flagged data from the final images, by default False
    save_plots : bool, optional
        If true, save the plots to the output directory, by default False.
    ouput_dir : str, optional
        Directory to save the plots, by default None.
    vmin : float
        Minimum intensity value for the color scale.
    vmax : float
        Maximum intensity value for the color scale.
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
    """
    # Unpack data from the DataHandler
    vis = data_handler.get_visibilities()
    uvw = data_handler.uvw
    freq = data_handler.spec.getcol("CHAN_FREQ")[0]
    time_all = data_handler.time_all
    unique_times = data_handler.get_times()
    dt = data_handler.get_time_step()
    autocorr_mask = ~(data_handler.get_autocorr_filter())

    fov_size = 2.2
    img_size = 1024
    pixsize = 2 * math.pi * fov_size / 360 / img_size

    logging.info(f"Imaging {nb_t_steps} time windows, each {dt * t_step} seconds long.")

    for start_freq in range(
        first_freq_step, first_freq_step + n_freq_steps * freq_step, freq_step
    ):
        for i in range(first_t_frame, first_t_frame + nb_t_steps * t_step, t_step):
            start_frame = i
            end_frame = i + t_step
            t_start = unique_times[start_frame]
            try:
                t_end = unique_times[end_frame]
            except IndexError:
                t_end = unique_times[-1] + dt

            # Create a time mask.
            time_mask = (time_all >= t_start) & (time_all < t_end)

            # Combine with autocorrelation filter.
            combined_mask = autocorr_mask & time_mask

            # Check if there are enough visibilities in this time window.
            if not np.any(combined_mask):
                logging.warning(
                    f"No data in time window {t_start} to {t_end}, skipping."
                )
                continue

            # Generate the dirty image.
            dirty_img = get_dirty_image(
                uvw[combined_mask],
                freq[start_freq : start_freq + freq_step],
                vis[combined_mask][:, start_freq : start_freq + freq_step],
                img_size,
                pixsize,
            )

            # Generate the PSF.
            psf = get_psf(
                uvw[combined_mask],
                freq[start_freq : start_freq + freq_step],
                vis[combined_mask][:, start_freq : start_freq + freq_step],
                img_size,
                pixsize,
            )

            # Generate the beam from the PSF.
            beam = get_beam(psf, beam_size=1)

            # Perform deconvolution / CLEAN-like loop to create the cleaned image.
            final_image = deconvolve_image(dirty_img, psf, beam, gain=0.1, niter=600)

            # Create extra label for the plot.
            time_label = f"Time window: {t_start:.2f} - {t_end:.2f} (units as in TIME)\nBin inteval: {start_frame} - {end_frame}"
            freq_label = (
                f"Freq window: Frequency: {freq[start_freq] / 1.0e9:.3f} to {freq[min(start_freq + freq_step, len(freq) - 1)] / 1.0e9:.3f} GHz\n"
                f"Channel {start_freq} to {start_freq + freq_step}"
            )
            extra_label = f"{time_label}\n{freq_label}"

            # Plot side-by-side comparisons.
            plot_dirty_clean(
                dirty_img.T,
                final_image.T,
                vmin,
                vmax,
                extra_label=extra_label,
                save=False,
            )
