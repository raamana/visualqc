"""
Module with algorithms to extract various features of interest for outlier detection methods.

"""

from visualqc import config as cfg
from visualqc.utils import read_image, scale_0to1
import numpy as np

def t1_histogram_whole_scan(in_mri_path, num_bins=cfg.num_bins_histogram_intensity_distribution):
    """
    Computes histogram over the intensity distribution over the entire scan, including brain, skull and background.

    Parameters
    ----------

    in_mri_path : str
        Path to an MRI scan readable by Nibabel

    Returns
    -------
    hist : ndarray
        Array of prob. densities for intensity

    """

    img = read_image(in_mri_path)
    # scaled, and reshaped
    arr_0to1 = scale_0to1(img).flatten()
    # compute prob. density
    hist = np.histogram(arr_0to1, bins=num_bins, density=True)

    return hist
