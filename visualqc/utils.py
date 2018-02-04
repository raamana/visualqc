
__all__ = ['read_image', 'check_image_is_3d']

from genericpath import exists as pexists
from os.path import realpath
import numpy as np
import nibabel as nib
from matplotlib.colors import ListedColormap

def read_image(img_spec, error_msg='image'):
    """Image reader. Removes stray values close to zero (smaller than 5 %ile)."""

    if isinstance(img_spec, str):
        if pexists(realpath(img_spec)):
            hdr = nib.load(img_spec)
            # trying to stick to an orientation
            hdr = nib.as_closest_canonical(hdr)
            img = hdr.get_data()
        else:
            raise IOError('Given path to {} does not exist!\n\t{}'.format(error_msg, img_spec))
    elif isinstance(img_spec, np.ndarray):
        img = img_spec
    else:
        raise ValueError('Invalid input specified! '
                         'Input either a path to image data, or provide 3d Matrix directly.')

    img = check_image_is_3d(img)

    if not np.issubdtype(img.dtype, np.float):
        img = img.astype('float32')

    return img


def get_label_set(seg, label_set, background=0):
    """Extracts only the required labels"""

    if label_set is None:
        return seg

    # get the mask picking up all labels
    mask = np.full_like(seg, False)
    for label in label_set:
        mask = np.logical_or(mask, seg==label)

    out_seg = np.full_like(seg, background)
    out_seg[mask] = seg[mask]

    # remap labels from arbitrary range to 1:N
    # helps to facilitate distinguishable colors
    unique_labels = np.unique(out_seg.flatten())
    # removing background - 0 stays 0
    unique_labels = np.delete(unique_labels, background)
    for index, label in enumerate(unique_labels):
        out_seg[out_seg==label] = index+1 # index=0 would make it background

    return out_seg


def get_axis(array, axis, slice_num):
    """Returns a fixed axis"""

    slice_list = [slice(None)] * array.ndim
    slice_list[axis] = slice_num
    slice_data = array[slice_list].T  # no transpose

    return slice_data


def check_alpha_set(alpha_set):
    """Ensure given alphas are valid."""

    alpha_set = np.array(alpha_set).astype('float16')

    if any(alpha_set<0.0) or any(alpha_set> 1.0):
        raise ValueError("One of the alpha's is invalid - all alphas must be within [0.0, 1.0]")

    return alpha_set


def check_image_is_3d(img):
    """Ensures the image loaded is 3d and nothing else."""

    if len(img.shape) < 3:
        raise ValueError('Input volume must be atleast 3D!')
    elif len(img.shape) == 3:
        for dim_size in img.shape:
            if dim_size < 1:
                raise ValueError('Atleast one slice must exist in each dimension')
    elif len(img.shape) == 4:
        if img.shape[3] != 1:
            raise ValueError('Input volume is 4D with more than one volume!')
        else:
            img = np.squeeze(img, axis=3)
    elif len(img.shape) > 4:
        raise ValueError('Invalid shape of image : {}'.format(img.shape))

    return img

def void_subcortical_symmetrize_cortical(aseg, null_label=0):
    """Sets Freesurfer LUT labels for subcortical segmentations (<1000) to null,
        and sets the left and rights parts of the same structure to the same label
        (to make it appear as a single color in the final image).

    """

    aseg = check_image_is_3d(aseg)
    left_aseg  = np.full_like(aseg, null_label)
    right_aseg = np.full_like(aseg, null_label)
    symmetric_aseg = np.full_like(aseg, null_label)

    left_baseline = 1000
    right_baseline = 2000

    left_ctx  = np.logical_and(aseg >= left_baseline , aseg < 2000)
    right_ctx = np.logical_and(aseg >= right_baseline, aseg < 3000)

    symmetric_aseg[left_ctx]  = aseg[left_ctx]  - left_baseline
    symmetric_aseg[right_ctx] = aseg[right_ctx] - right_baseline

    return symmetric_aseg


def get_freesurfer_color_LUT():
    """
    Subset of Freesurfer ColorLUT for cortical labels

    Original at
    https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    """

    LUT = [ [25,5,25],
            [25,100,40],
            [125,100,160],
            [100,25,0],
            [120,70,50],
            [220,20,100],
            [220,20,10],
            [180,220,140],
            [220,60,220],
            [180,40,120],
            [140,20,140],
            [20,30,140],
            [35,75,50],
            [225,140,140],
            [200,35,75],
            [160,100,50],
            [20,220,60],
            [60,220,60],
            [220,180,140],
            [20,100,50],
            [220,60,20],
            [120,100,60],
            [220,20,20],
            [220,180,220],
            [60,20,220],
            [160,140,180],
            [80,20,140],
            [75,50,125],
            [20,220,160],
            [20,180,140],
            [140,220,220],
            [80,160,20],
            [100,0,100],
            [70,70,70],
            [150,150,200],
            [220,216,20]
           ]

    return LUT

