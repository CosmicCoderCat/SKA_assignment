import ducc0
import numpy as np
import scipy.signal.windows
import math
import matplotlib.pyplot as plt 

def get_dirty_image(uvw: np.ndarray, freq: np.ndarray, vis: np.ndarray, img_size: float, pixsize: int) -> np.ndarray:
    """Compute the dirty image for a given frequency slice.
    Returns the frequency array and the dirty image.

    Parameters
    ----------
    uvw : numpy.ndarray
        array of UVW coordinates for the visibilities
    freq : numpy.ndarray
        array of frequencies for the visibilities
    vis : numpy.ndarray
        visibility data
    img_size : float
        size of the image in pixels
    pixsize : int
        size of a pixel in radians

    Returns
    -------
    numpy.ndarray
        dirty image array
    """
    img = ducc0.wgridder.ms2dirty(
        uvw,
        freq,
        vis,
        npix_x=img_size,
        npix_y=img_size,
        pixsize_x=pixsize,
        pixsize_y=pixsize,
        epsilon=1e-3,
        do_wstacking=True,
    )
    return img

def get_psf(uvw: np.ndarray, freq: np.ndarray, vis: np.ndarray, img_size: float, pixsize: int) -> np.ndarray:
    """Compute the PSF for a given frequency slice using a unit signal.
    Returns the PSF array.

    Parameters
    ----------
    uvw : numpy.ndarray
        array of UVW coordinates for the visibilities
    freq : numpy.ndarray
        array of frequencies for the visibilities
    vis : numpy.ndarray
        visibility data
    img_size : float
        size of the image in pixels
    pixsize : int
        size of a pixel in radians

    Returns
    -------
    numpy.ndarray
        PSF array
    """
    psf = ducc0.wgridder.ms2dirty(
        uvw,
        freq,
        np.ones_like(vis),
        npix_x=img_size * 2,
        npix_y=img_size * 2,
        pixsize_x=pixsize,
        pixsize_y=pixsize,
        epsilon=1e-3,
        do_wstacking=True,
    )

    # Normalize the PSF
    center_y, center_x = psf.shape[0] // 2, psf.shape[1] // 2
    psf /= psf[center_y, center_x]
    
    return psf

def get_beam(psf: np.ndarray, beam_size: float = 1.) -> np.ndarray:
    """Construct a Gaussian beam matching the shape of the PSF.
    Returns the beam array.

    Parameters
    ----------
    psf : numpy.ndarray
        PSF array
    beam_size : float, optional
        Standard deviation used for the Gaussian, by default 1

    Returns
    -------
    numpy.ndarray
        Gaussian beam array
    """
    beam = scipy.signal.windows.gaussian(psf.shape[0], std=beam_size)[:, None] * \
           scipy.signal.windows.gaussian(psf.shape[0], std=beam_size)
    return beam

def deconvolve_image(img: np.ndarray, psf: np.ndarray, beam: np.ndarray, gain: float = 0.1, niter: int = 600) -> np.ndarray:
    """Deconvolve the dirty image using the provided PSF and beam via a CLEAN-like loop.
    Returns the final cleaned image.

    Parameters
    ----------
    img : numpy.ndarray
        Dirty image array
    psf : numpy.ndarray
        PSF array
    beam : numpy.ndarray
        Clean beam array
    gain : float, optional
        Clean gain for each iteration, by default 0.1
    niter : int, optional
        Number of clean iterations, by default 600

    Returns
    -------
    numpy.ndarray
        Cleaned image array
    """
    img_dec = np.array(img)
    out_image = np.zeros_like(img)
    half_size = img.shape[0] // 2

    for iteration in range(niter):
        # Find the maximum pixel in the dirty image
        y_max, x_max = np.unravel_index(np.argmax(img_dec), img_dec.shape)
        val = gain * img_dec[y_max, x_max]
        # Find its position relative to the center of the imag
        d_y = y_max - img.shape[0] // 2
        d_x = x_max - img.shape[1] // 2

        try:
            # Update the clean image and the dirty image 
            beam_slice_y = slice(half_size - d_y, -half_size - d_y)
            beam_slice_x = slice(half_size - d_x, -half_size - d_x)
            out_image += val * beam[beam_slice_y, beam_slice_x]
            img_dec -= val * psf[beam_slice_y, beam_slice_x]
        except Exception as e:
            print(f"Deconvolution terminated at iteration {iteration} due to error: {e}")
            break

    # Add residuals back to the clean image
    final_image = out_image + img_dec
    return final_image

if __name__ == '__main__':
    from SKA_assignment.data_handler import DataHandler
    data_handler = DataHandler('../../pipeline_problem_data.ms')

    vis = data_handler.get_visibilities()
    uvw = data_handler.uvw
    freq = data_handler.spec.getcol('CHAN_FREQ')[0]

    autocorr_filter = data_handler.get_autocorr_filter()
    mask = (~autocorr_filter)

    fov_size = 2.2
    img_size = 1024
    pixsize = 2 * math.pi * fov_size / 360 / img_size

    # Get dirty image
    dirty_img = get_dirty_image(uvw[mask], freq, vis[mask], img_size, pixsize)

    # Get PSF
    psf = get_psf(uvw[mask], freq, vis[mask], img_size, pixsize)

    # Get beam
    beam = get_beam(psf, beam_size=1)

    # Deconvolve image
    cleaned_img = deconvolve_image(dirty_img, psf, beam)

    # Plot images
    fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    vmin=-20000
    vmax=200000

    print(type(vis))
    print(type(uvw))
    print(type(freq))
    print(type(fov_size))
    print(type(img_size))
    print(type(pixsize))
    print(type(dirty_img))
    print(type(psf))
    print(type(beam))
    print(type(cleaned_img))

    # Dirty image subplot.
    im0 = axs[0].imshow(dirty_img.T, vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Dirty Image\nAll frequencies\nAll times")
    axs[0].set_xlabel("Pixel X")
    axs[0].set_ylabel("Pixel Y")
    cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Intensity")

    # Cleaned image subplot.
    im1 = axs[1].imshow(cleaned_img.T, vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Cleaned Image\nAll frequencies\nAll times")
    axs[1].set_xlabel("Pixel X")
    axs[1].set_ylabel("Pixel Y")
    cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Intensity")

    plt.tight_layout()
    plt.savefig('all_time_freq.png')
