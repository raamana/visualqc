
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
            img = nib.load(img_spec).get_data()
        else:
            raise IOError('Given path to {} does not exist!'.format(error_msg))
    elif isinstance(img_spec, np.ndarray):
        img = img_spec
    else:
        raise ValueError('Invalid input specified! '
                         'Input either a path to image data, or provide 3d Matrix directly.')

    img = check_image_is_3d(img)

    if not np.issubdtype(img.dtype, np.float):
        img = img.astype('float32')

    return img


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

