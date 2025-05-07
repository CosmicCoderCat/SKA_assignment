import scipy
import numpy as np

from SKA_assignment.data_handler import DataHandler
from SKA_assignment.plotting import plot_amplitude_vs_time

def monitor_data_quality(data_path: str, generate_plots=False, first_t_frame=0, nb_t_steps=120, t_step=1, first_freq_step=0, n_freq_steps=32, freq_step=1) -> None:
    data_handler = DataHandler('../../pipeline_problem_data.ms')

    time_all = data_handler.time_all
    autocorr_filter = data_handler.get_autocorr_filter()
    vis = data_handler.get_visibilities()
    unique_times = data_handler.get_times()
    dt = data_handler.get_time_step()

    amplitude_all_vis = np.abs(vis).T

    amplitude_vis = amplitude_all_vis[first_freq_step:first_freq_step+n_freq_steps*freq_step, :]

    combined_masks = []
    for i, start_frame in enumerate(range(first_t_frame, first_t_frame + nb_t_steps * t_step, t_step)):
        end_frame = start_frame + t_step
        t_start = unique_times[start_frame]
        try:
            t_end = unique_times[end_frame]
        except IndexError:
            t_end = unique_times[-1] + dt

        combined_masks.append([(~autocorr_filter) & (time_all >= t_start) & (time_all < t_end)])

    # Compute binned amplitudes for all frequencies and time bins in one go
    binned_amplitude = np.array([[
        np.mean(amplitude_vis[freq_index:freq_index+freq_step][combined_mask], axis=0) if np.any(combined_mask) else np.nan
        for combined_mask in combined_masks
        ]
        for freq_index in range(n_freq_steps)
    ])  # Shape: (nb_t_steps, n_freq_steps)


    # # Plot results for each frequency
    if generate_plots:
        for i in range(n_freq_steps):
            # fig, ax = plt.subplots(figsize=(10, 5))
            title = f"Time Series of Visibility Amplitude for Channel {first_freq_step + i * freq_step}"
            filename = f"amplitude_vs_time_channel_{first_freq_step + i * freq_step}.png"
            plot_amplitude_vs_time(binned_amplitude[i, :], range(first_t_frame, first_t_frame + nb_t_steps), title, filename)

if __name__ == '__main__':
    monitor_data_quality("../../pipeline_problem_data.ms", generate_plots=True)