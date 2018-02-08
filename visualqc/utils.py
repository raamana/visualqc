__all__ = ['read_image', 'check_image_is_3d']

import os
import warnings
from genericpath import exists as pexists
from os import makedirs
from os.path import realpath, join as pjoin
from shutil import copyfile, which
import nibabel as nib
import numpy as np
import visualqc.config as cfg
from visualqc.config import suffix_ratings_dir, file_name_ratings, file_name_ratings_backup, \
    visualization_combination_choices, default_out_dir_name, freesurfer_vis_types, freesurfer_vis_cmd


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
        mask = np.logical_or(mask, seg == label)

    out_seg = np.full_like(seg, background)
    out_seg[mask] = seg[mask]

    # remap labels from arbitrary range to 1:N
    # helps to facilitate distinguishable colors
    unique_labels = np.unique(out_seg.flatten())
    # removing background - 0 stays 0
    unique_labels = np.delete(unique_labels, background)
    for index, label in enumerate(unique_labels):
        out_seg[out_seg == label] = index + 1  # index=0 would make it background

    return out_seg


def get_axis(array, axis, slice_num):
    """Returns a fixed axis"""

    slice_list = [slice(None)] * array.ndim
    slice_list[axis] = slice_num
    slice_data = array[slice_list].T  # for proper appearance

    return slice_data


def pick_slices(img_shape, view_set, num_slices):
    """Picks the slices to display in each dimension"""

    num_views = len(view_set)
    skip_count = min(5 ,int(np.floor(num_slices / num_views)))

    slices = list()
    for view in view_set:
        dim_size = img_shape[view]
        slices_in_dim = np.around(np.linspace(0, dim_size, num_slices + 2 * skip_count)).astype('int64')
        # skipping not-so-important slices at boundaries
        slices_in_dim = slices_in_dim[skip_count: -skip_count]
        # ensure you do not overshoot
        slices_in_dim = [sn for sn in slices_in_dim if sn >= 0 or sn <= dim_size]
        # adding view and slice # at the same time.
        slices.extend([(view, slice) for slice in slices_in_dim])

    return slices


def check_layout(total_num_slices, num_views, num_rows_per_view, num_rows_for_surf_vis):
    """Ensures all odd cases are dealt with"""

    num_cols = int(np.ceil(total_num_slices / ((num_views * num_rows_per_view) + num_rows_for_surf_vis)))

    return num_cols


def check_finite_int(num_slices, num_rows):
    """Validates numbers."""

    num_slices = int(num_slices)
    num_rows = int(num_rows)

    if not all(np.isfinite((num_slices, num_rows))):
        raise ValueError('num_slices and num_rows must be finite.')

    if num_slices < 0 or num_rows < 0:
        raise ValueError('num_slices and num_rows must be positive (>=1).')

    return num_slices, num_rows


def check_alpha_set(alpha_set):
    """Ensure given alphas are valid."""

    alpha_set = np.array(alpha_set).astype('float16')

    if any(alpha_set < 0.0) or any(alpha_set > 1.0):
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
    symmetric_aseg = np.full_like(aseg, null_label)

    left_baseline = 1000
    right_baseline = 2000

    left_ctx = np.logical_and(aseg >= left_baseline, aseg < 2000)
    right_ctx = np.logical_and(aseg >= right_baseline, aseg < 3000)

    # labels 1000 and 2000 are unknown, so making them background is okay!
    # if not we need to make the baselines smaller by 1, to map 1000 and 2000 to 1
    symmetric_aseg[left_ctx] = aseg[left_ctx] - left_baseline
    symmetric_aseg[right_ctx] = aseg[right_ctx] - right_baseline

    return symmetric_aseg


def get_freesurfer_color_LUT():
    """
    Subset of Freesurfer ColorLUT for cortical labels

    Original at
    https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    """

    LUT = [[25, 5, 25],
           [25, 100, 40],
           [125, 100, 160],
           [100, 25, 0],
           [120, 70, 50],
           [220, 20, 100],
           [220, 20, 10],
           [180, 220, 140],
           [220, 60, 220],
           [180, 40, 120],
           [140, 20, 140],
           [20, 30, 140],
           [35, 75, 50],
           [225, 140, 140],
           [200, 35, 75],
           [160, 100, 50],
           [20, 220, 60],
           [60, 220, 60],
           [220, 180, 140],
           [20, 100, 50],
           [220, 60, 20],
           [120, 100, 60],
           [220, 20, 20],
           [220, 180, 220],
           [60, 20, 220],
           [160, 140, 180],
           [80, 20, 140],
           [75, 50, 125],
           [20, 220, 160],
           [20, 180, 140],
           [140, 220, 220],
           [80, 160, 20],
           [100, 0, 100],
           [70, 70, 70],
           [150, 150, 200],
           [220, 216, 20]
           ]

    return LUT


def get_ratings(out_dir, id_list):
    """Creates a separate folder for ratings, backing up any previous sessions."""

    # making a copy
    incomplete_list = list(id_list)
    prev_done = []  # empty list

    ratings_dir = pjoin(out_dir, suffix_ratings_dir)
    if pexists(ratings_dir):
        prev_ratings = pjoin(ratings_dir, file_name_ratings)
        prev_ratings_backup = pjoin(ratings_dir, file_name_ratings_backup)
        if pexists(prev_ratings):
            ratings = load_ratings_csv(prev_ratings)
            copyfile(prev_ratings, prev_ratings_backup)
            # finding the remaining
            prev_done = set(ratings.keys())
            incomplete_list = list(set(id_list) - prev_done)
        else:
            ratings = dict()
    else:
        makedirs(ratings_dir, exist_ok=True)
        ratings = dict()

    if len(prev_done) > 0:
        print('Ratings for {} subjects were restored from previous backup'.format(len(prev_done)))

    print('To be reviewed : {}'.format(len(incomplete_list)))

    return ratings, ratings_dir, incomplete_list, prev_done


def load_ratings_csv(prev_ratings):
    """read CSV into a dict"""

    if pexists(prev_ratings):
        info_dict = dict([line.strip().split(',') for line in open(prev_ratings).readlines()])
    else:
        info_dict = dict()

    return info_dict


def save_ratings(ratings, out_dir):
    """Save ratings before closing shop."""

    ratings_dir = pjoin(out_dir, suffix_ratings_dir)
    if not pexists(ratings_dir):
        makedirs(ratings_dir)

    ratings_file = pjoin(ratings_dir, file_name_ratings)
    prev_ratings_backup = pjoin(ratings_dir, file_name_ratings_backup)
    if pexists(ratings_file):
        copyfile(ratings_file, prev_ratings_backup)

    lines = '\n'.join(['{},{}'.format(sid, rating) for sid, rating in ratings.items()])
    try:
        with open(ratings_file, 'w') as cf:
            cf.write(lines)
    except:
        raise IOError(
            'Error in saving ratings to file!!\nBackup might be helpful at:\n\t{}'.format(prev_ratings_backup))

    return


def check_input_dir(fs_dir, user_dir, vis_type):
    """Ensures proper input is specified."""

    in_dir = fs_dir
    if fs_dir is None and user_dir is None:
        raise ValueError('At least one of --fs_dir or --user_dir must be specified.')

    if fs_dir is not None:
        if user_dir is not None:
            raise ValueError('Only one of --fs_dir or --user_dir can be specified.')

        if not freesurfer_installed():
            raise EnvironmentError(
                'Freesurfer functionality is requested(e.g. visualizing annotations), but is not installed!')

    if fs_dir is None and vis_type in freesurfer_vis_types:
        raise ValueError('vis_type depending on Freesurfer organization is specified, but --fs_dir is not provided.')

    if user_dir is None:
        if not pexists(fs_dir):
            raise IOError('Freesurfer directory specified does not exist!')
        else:
            in_dir = fs_dir
    elif fs_dir is None:
        if not pexists(user_dir):
            raise IOError('User-specified input directory does not exist!')
        else:
            in_dir = user_dir

    if not pexists(in_dir):
        raise IOError('Invalid specification - check proper combination of --fs_dir and --user_dir')

    return in_dir


def freesurfer_installed():
    """Checks whether Freesurfer is installed."""

    if os.getenv('FREESURFER_HOME') is None or which(freesurfer_vis_cmd) is None:
        return False

    return True


def check_out_dir(out_dir, fs_dir):
    """Creates the output folder."""

    if out_dir is None:
        out_dir = pjoin(fs_dir, default_out_dir_name)

    try:
        os.makedirs(out_dir, exist_ok=True)
    except:
        raise IOError('Unable to create the output directory as requested.')

    return out_dir


def check_id_list(id_list_in, in_dir, vis_type, mri_name, seg_name):
    """Checks to ensure each subject listed has the required files and returns only those that can be processed."""

    if id_list_in is not None:
        if not pexists(id_list_in):
            raise IOError('Given ID list does not exist!')

        try:
            # read all lines and strip them of newlines/spaces
            id_list = [line.strip('\n ') for line in open(id_list_in)]
        except:
            raise IOError('unable to read the ID list.')
    else:
        # get all IDs in the given folder
        id_list = [folder for folder in os.listdir(in_dir) if os.path.isdir(pjoin(in_dir, folder))]

    required_files = (mri_name, seg_name)

    id_list_out = list()
    id_list_err = list()
    invalid_list = list()

    for subject_id in id_list:
        path_list = [get_path_for_subject(in_dir, subject_id, req_file, vis_type) for req_file in required_files]
        invalid = [this_file for this_file in path_list if not pexists(this_file) or os.path.getsize(this_file) <= 0]
        if len(invalid) > 0:
            id_list_err.append(subject_id)
            invalid_list.extend(invalid)
        else:
            id_list_out.append(subject_id)

    if len(id_list_err) > 0:
        warnings.warn('The following subjects do NOT have all the required files or some are empty - skipping them!')
        print('\n'.join(id_list_err))
        print('\n\nThe following files do not exist or empty: \n {} \n\n'.format('\n'.join(invalid_list)))

    if len(id_list_out) < 1:
        raise ValueError('All the subject IDs do not have the required files - unable to proceed.')

    print('{} subjects are usable for review.'.format(len(id_list_out)))

    return id_list_out


def get_path_for_subject(in_dir, subject_id, req_file, vis_type):
    """Constructs the path for the image file based on chosen input and visualization type"""

    if vis_type in freesurfer_vis_types:
        out_path = realpath(pjoin(in_dir, subject_id, 'mri', req_file))
    else:
        out_path = realpath(pjoin(in_dir, subject_id, req_file))

    return out_path


def check_labels(vis_type, label_set):
    """Validates the selections."""

    vis_type = vis_type.lower()
    if vis_type not in visualization_combination_choices:
        raise ValueError('Selected visualization type not recognized! '
                         'Choose one of:\n{}'.format(visualization_combination_choices))

    if label_set is not None:
        if vis_type not in cfg.label_types:
            raise ValueError('Invalid selection of vis_type when labels are specifed. Choose --vis_type labels')

        label_set = np.array(label_set).astype('int16')

    return vis_type, label_set


def check_views(views):
    """Checks which views were selected."""

    if views is None:
        return range(3)

    views = [int(vw) for vw in views]
    out_views = list()
    for vw in views:
        if vw < 0 or vw > 2:
            print('one of the selected views is out of range - skipping it.')
        out_views.append(vw)

    if len(out_views) < 1:
        raise ValueError('Atleast one valid view must be selected. Choose one or more of 0, 1, 2.')

    return out_views
