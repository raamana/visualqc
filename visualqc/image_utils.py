"""

Image processing utilities

"""

__all__ = ['background_mask', 'foreground_mask', 'overlay_edges', 'diff_image',
           'equalize_image_histogram']

from scipy import ndimage
from visualqc import config as cfg
from visualqc.utils import scale_0to1
import numpy as np
from functools import partial
from scipy.ndimage import sobel, binary_closing
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.filters import median_filter, minimum_filter, maximum_filter
from scipy.signal import medfilt2d

import matplotlib
matplotlib.interactive(True)

from matplotlib.cm import get_cmap

gray_cmap = get_cmap('gray')
hot_cmap = get_cmap('hot')

filter_params = dict(size=cfg.median_filter_size, mode='constant', cval=0)
min_filter = partial(minimum_filter, **filter_params)
max_filter = partial(maximum_filter, **filter_params)
med_filter = partial(median_filter , **filter_params)


def background_mask(mri, thresh_perc=1):
    """Creates the background mask from an MRI"""

    grad_magnitude = gradient_magnitude(mri)
    nonzero_grad_mag = grad_magnitude[grad_magnitude > 0]

    thresh_val = np.percentile(nonzero_grad_mag.flatten(), thresh_perc)
    background_mask = grad_magnitude < thresh_val

    se36 = ndimage.generate_binary_structure(3, 6)
    closed = ndimage.binary_closing(background_mask, se36, iterations=6)
    final_mask = ndimage.binary_erosion(closed, se36, iterations=5)

    return final_mask


def gradient_magnitude(mri):
    """Computes the gradient magnitude"""

    grad = np.asarray(np.gradient(mri))
    grad_magnitude = np.sqrt(np.sum(np.power(grad, 2.), axis=0))

    return grad_magnitude


def mask_image(input_img,
               update_factor=0.5,
               init_percentile=2,
               iterations_closing=5,
               return_inverse=False,
               out_dtype=None):
    """
    Estimates the foreground mask for a given image.
    Similar to 3dAutoMask from AFNI.


    iterations_closing : int
        Number of iterations of binary_closing to apply at the end.

    """

    prev_clip_level = np.percentile(input_img, init_percentile)
    while True:
        mask_img = input_img >= prev_clip_level
        cur_clip_level = update_factor * np.median(input_img[mask_img])
        if np.isclose(cur_clip_level, prev_clip_level, rtol=0.05):
            break
        else:
            prev_clip_level = cur_clip_level

    if len(input_img.shape) == 3:
        se = ndimage.generate_binary_structure(3, 6)
    elif len(input_img.shape) == 2:
        se = ndimage.generate_binary_structure(2, 4)
    else:
        raise ValueError('Image must be 2D or 3D')

    mask_img = binary_closing(mask_img, se, iterations=iterations_closing)
    mask_img = binary_fill_holes(mask_img, se)

    if return_inverse:
        mask_img = np.logical_not(mask_img)

    if out_dtype is not None:
        mask_img = mask_img.astype(out_dtype)

    return mask_img

# alias
foreground_mask = mask_image

def equalize_image_histogram(image_in, num_bins=cfg.num_bins_histogram_contrast_enhancement,
                             max_value=255):
    """Modifies the image to achieve an equalized histogram."""

    image_flat = image_in.flatten()
    hist_image, bin_edges = np.histogram(image_flat, bins=num_bins, normed=True)
    cdf = hist_image.cumsum()
    cdf = max_value * cdf / cdf[-1] # last element is total sum

    # linear interpolation
    array_equalized = np.interp(image_flat, bin_edges[:-1], cdf)

    return array_equalized.reshape(image_in.shape)


def overlay_edges(slice_one, slice_two, sharper=True):
    """
    Makes a composite image with edges from second image overlaid on first.

    It will be in colormapped (RGB format) already.
    """

    if slice_one.shape != slice_two.shape:
        raise ValueError("slices' dimensions do not match: "
                         " {} and {} ".format(slice_one.shape, slice_two.shape))

    # simple filtering to remove noise, while supposedly keeping edges
    slice_two = medfilt2d(slice_two, kernel_size=cfg.median_filter_size)
    # extracting edges
    edges = np.hypot(sobel(slice_two, axis=0, mode='constant'),
                     sobel(slice_two, axis=1, mode='constant'))

    # trying to remove weak edges
    if not sharper: # level of removal
        edges = med_filter(max_filter(min_filter(edges)))
    else:
        edges = min_filter(min_filter(max_filter(min_filter(edges))))
    edges_color_mapped = hot_cmap(edges, alpha=cfg.alpha_edge_overlay_alignment)
    composite = gray_cmap(slice_one, alpha=cfg.alpha_background_slice_alignment)

    composite[edges_color_mapped>0] = edges_color_mapped[edges_color_mapped>0]

    # mask_rgba = np.dstack([edges>0] * 4)
    # composite[mask_rgba] = edges_color_mapped[mask_rgba]

    return composite


def dwi_overlay_edges(slice_one, slice_two):
    """
    Makes a composite image with edges from second image overlaid on first.

    It will be in colormapped (RGB format) already.
    """

    if slice_one.shape != slice_two.shape:
        raise ValueError("slices' dimensions do not match: "
                         " {} and {} ".format(slice_one.shape, slice_two.shape))

    # simple filtering to remove noise, while supposedly keeping edges
    slice_two = medfilt2d(slice_two, kernel_size=cfg.median_filter_size)
    # extracting edges
    edges = med_filter(np.hypot(sobel(slice_two, axis=0, mode='constant'),
                                sobel(slice_two, axis=1, mode='constant')))

    edges_color_mapped = hot_cmap(edges, alpha=cfg.alpha_edge_overlay_alignment)
    composite = gray_cmap(slice_one, alpha=cfg.alpha_background_slice_alignment)

    composite[edges_color_mapped>0] = edges_color_mapped[edges_color_mapped>0]

    # mask_rgba = np.dstack([edges>0] * 4)
    # composite[mask_rgba] = edges_color_mapped[mask_rgba]

    return composite


def _get_checkers(slice_shape, patch_size):
    """Creates checkerboard of a given tile size, filling a given slice."""

    if patch_size is not None:
        patch_size = check_patch_size(patch_size)
    else:
        # 7 patches in each axis, min voxels/patch = 3
        patch_size = np.round(np.array(slice_shape) / 7).astype('int16')
        patch_size = np.maximum(patch_size, np.array([3, 3]))

    black = np.zeros(patch_size)
    white = np.ones(patch_size)
    tile = np.vstack((np.hstack([black, white]), np.hstack([white, black])))

    # using ceil so we can clip the extra portions
    num_tiles = np.ceil(np.divide(slice_shape, tile.shape)).astype(int)
    checkers = np.tile(tile, num_tiles)

    # clipping any extra columns or rows
    if any(np.greater(checkers.shape, slice_shape)):
        if checkers.shape[0] > slice_shape[0]:
            checkers = np.delete(checkers, np.s_[slice_shape[0]:], axis=0)
        if checkers.shape[1] > slice_shape[1]:
            checkers = np.delete(checkers, np.s_[slice_shape[1]:], axis=1)

    return checkers


def mix_color(slice1, slice2,
              alpha_channels=cfg.default_color_mix_alphas,
              color_space='rgb'):
    """Mixing them as red and green channels"""

    if slice1.shape != slice2.shape:
        raise ValueError('size mismatch between cropped slices and checkers!!!')

    alpha_channels = np.array(alpha_channels)
    if len(alpha_channels) != 2:
        raise ValueError('Alphas must be two value tuples.')

    slice1 = scale_0to1(slice1)
    slice2 = scale_0to1(slice2)

    # masking background
    combined_distr = np.concatenate((slice1.flatten(), slice2.flatten()))
    image_eps = np.percentile(combined_distr, 5)
    background = np.logical_or(slice1 <= image_eps, slice2 <= image_eps)

    if color_space.lower() in ['rgb']:

        red = alpha_channels[0] * slice1
        grn = alpha_channels[1] * slice2
        blu = np.zeros_like(slice1)

        # foreground = np.logical_not(background)
        # blu[foreground] = 1.0

        mixed = np.stack((red, grn, blu), axis=2)

    elif color_space.lower() in ['hsv']:

        raise NotImplementedError(
            'This method (color_space="hsv") is yet to fully conceptualized and implemented.')

    # ensuring all values are clipped to [0, 1]
    mixed[mixed <= 0.0] = 0.0
    mixed[mixed >= 1.0] = 1.0

    return mixed


def mix_slices_in_checkers(slice1, slice2,
                           checker_size=cfg.default_checkerboard_size):
    """Mixes the two slices in alternating areas specified by checkers"""

    checkers = _get_checkers(slice1.shape, checker_size)
    if slice1.shape != slice2.shape or slice2.shape != checkers.shape:
        raise ValueError('size mismatch between cropped slices and checkers!!!')

    mixed = slice1.copy()
    mixed[checkers > 0] = slice2[checkers > 0]

    return mixed


def diff_image(slice1, slice2, abs_value=True):
    """Computes the difference image"""

    diff = slice1 - slice2

    if abs_value:
        diff = np.abs(diff)

    return diff


def check_patch_size(patch_size):
    """Validation and typcasting"""

    patch_size = np.array(patch_size)
    if patch_size.size == 1:
        patch_size = np.repeat(patch_size, 2).astype('int16')

    return patch_size


def rescale_without_outliers(img, trim_percentile=1):
    """This utility trims the outliers first, and then rescales it [0, 1]"""

    return scale_0to1(img,
                      exclude_outliers_below=trim_percentile,
                      exclude_outliers_above=trim_percentile)
