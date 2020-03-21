__all__ = ['read_image', 'check_image_is_3d', 'check_bids_dir']

import os
import sys
import warnings
from genericpath import exists as pexists
from os.path import realpath
from os import makedirs
from shutil import copyfile, which

import nibabel as nib
import numpy as np
from os.path import basename, join as pjoin, realpath, splitext

import visualqc.config as cfg
from visualqc.config import default_out_dir_name, freesurfer_vis_cmd, \
    freesurfer_vis_types, visualization_combination_choices


def read_image(img_spec,
               error_msg='image',
               num_dims=3,
               reorient_canonical=True):
    """Image reader. Removes stray values close to zero (smaller than 5 %ile)."""

    if isinstance(img_spec, str):
        if pexists(realpath(img_spec)):
            hdr = nib.load(img_spec)
            # trying to stick to an orientation
            if reorient_canonical:
                hdr = nib.as_closest_canonical(hdr)
            img = hdr.get_data()
        else:
            raise IOError('Given path to {} does not exist!\n\t{}'
                          ''.format(error_msg, img_spec))
    elif isinstance(img_spec, np.ndarray):
        img = img_spec
    else:
        raise ValueError('Invalid input specified! '
                         'Input either a path to image data, '
                         'or provide 3d Matrix directly.')

    if num_dims == 3:
        img = check_image_is_3d(img)
    elif num_dims == 4:
        check_image_is_4d(img)
    else:
        raise ValueError('Requested check for {} dims - allowed: 3 or 4!')

    if not np.issubdtype(img.dtype, np.float64):
        img = img.astype('float32')

    return img


def scale_0to1(image_in,
               exclude_outliers_below=False,
               exclude_outliers_above=False
               , multiply_factor=1.0):
    """Scale the two images to [0, 1] based on min/max."""

    # making a copy to ensure no side-effects
    out_image = image_in.copy()

    min_value = np.percentile(out_image.flatten(),
                              float(exclude_outliers_below))
    max_value = np.percentile(out_image.flatten(),
                              100-float(exclude_outliers_above))

    if exclude_outliers_below:
        out_image[out_image < min_value] = min_value

    if exclude_outliers_above:
        out_image[out_image > max_value] = max_value

    out_image = (out_image - min_value) / (max_value - min_value)

    if not np.isclose(multiply_factor, 1.0):
        # makes it go from [0, 1] to [0, multiply_factor]
        # this may be unnecessary for plt.imshow commands,
        #   as everything gets normalized from 0 to 1 again.
        out_image = out_image * multiply_factor

    return out_image


def saturate_brighter_intensities(img,
                                  factor=0.1,
                                  percentile=None):
    """Sets all the intensities above max_intensity*factor to the max intensity.

    Helpful to detect ghosting.
    """

    saturated = img.copy()
    max_value = saturated.max()

    if percentile is None and factor is not None:
        saturated[saturated>(factor*max_value)] = max_value
    elif percentile:
        saturated[saturated > np.percentile(saturated, float(percentile))] = max_value
    else:
        raise ValueError('Invalid input specification')

    return saturated


def get_label_set(seg, label_set, background=cfg.background_value):
    """Extracts only the required labels, and remaps the labels from 1 to n"""

    if label_set is None:
        out_seg = seg
    else:
        # get the mask picking up all labels
        mask = np.full_like(seg, False)
        for label in label_set:
            mask = np.logical_or(mask, seg == label)

        masked_seg = np.full_like(seg, background)
        masked_seg[mask] = seg[mask]

        # remap labels from arbitrary range to 1:N
        out_seg = remap_labels_1toN(masked_seg, background)

    roi_set_empty = False
    if np.count_nonzero(out_seg) < 1:
        roi_set_empty = True

    return out_seg, roi_set_empty


def remap_labels_1toN(in_seg, background=cfg.background_value):
    """Remap [arbitrary] labels in a volumetric seg to range from 1 to N."""

    out_seg = np.full_like(in_seg, background)

    # remap labels from arbitrary range to 1:N
    # helps to facilitate distinguishable colors
    unique_labels = np.unique(in_seg.flatten())
    # removing background - 0 stays 0
    unique_labels = np.setdiff1d(unique_labels, background)
    for index, label in enumerate(unique_labels):
        # index=0 would make it background, so using index+1
        out_seg[in_seg == label] = index + 1

    return out_seg


def get_axis(array, axis, slice_num):
    """Returns a fixed axis"""

    slice_list = [slice(None)] * array.ndim
    slice_list[axis] = slice_num
    # handling the FutureWarning:
    # Using a non-tuple sequence for multidimensional indexing is deprecated;
    #   use `arr[tuple(seq)]` instead of `arr[seq]`.
    # In the future this will be interpreted as an array index,
    #   `arr[np.array(seq)]`, which will result either in an error or a different result.
    slice_data = array[tuple(slice_list)].T  # for proper appearance

    return slice_data


def pick_slices(img, view_set, num_slices):
    """
    Picks the slices to display in each dimension,
        skipping any empty slices (without any segmentation at all).

    """

    slices = list()
    for view in view_set:
        dim_size = img.shape[view]
        non_empty_slices = np.array([sl for sl in range(dim_size) if
                                     np.count_nonzero(get_axis(img, view, sl)) > 0])
        num_non_empty = len(non_empty_slices)

        # trying to 5% slices at the tails (bottom clipping at 0)
        skip_count = max(0, np.around(num_non_empty * 0.05).astype('int16'))
        # only when possible
        if skip_count > 0 and (num_non_empty - 2 * skip_count > num_slices):
            non_empty_slices = non_empty_slices[skip_count: -skip_count]
            num_non_empty = len(non_empty_slices)

        # sampling non-empty slices only
        sampled_indices = np.linspace(0, num_non_empty,
                                      num=min(num_non_empty, num_slices), endpoint=False)
        slices_in_dim = non_empty_slices[np.around(sampled_indices).astype('int64')]

        # ensure you do not overshoot
        slices_in_dim = [sn for sn in slices_in_dim if sn >= 0 or sn <= num_non_empty]
        # adding view and slice # at the same time.
        slices.extend([(view, slice) for slice in slices_in_dim])

    return slices


def check_layout(total_num_slices, num_views, num_rows_per_view, num_rows_for_surf_vis):
    """Ensures all odd cases are dealt with"""

    num_cols = int(np.ceil(
        total_num_slices / ((num_views * num_rows_per_view) + num_rows_for_surf_vis)))

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
        raise ValueError(
            "One of the alpha's is invalid - all alphas must be within [0.0, 1.0]")

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


def check_image_is_4d(img, min_num_volumes=1, name='4D image'):
    """Ensures the image loaded is 4d and nothing else."""

    if len(img.shape) <= 3:
        raise ValueError('Input {} must be atleast 4D!'.format(name))
    elif len(img.shape) == 4:
        for dim_size in img.shape:
            if dim_size < 1:
                raise ValueError('Atleast one slice must exist in each dimension')
        # atleast one volume existing is already checked in the above loop
        if min_num_volumes >1 and img.shape[-1]<min_num_volumes:
            raise ValueError('Given {} has fewer than {} volumes!'.format(name, min_num_volumes))
    elif len(img.shape) > 4:
        raise ValueError('{} has more than 4 dimensions : {}'.format(name, img.shape))

    if np.count_nonzero(img) == 0:
        raise ValueError('given image is empty!')

    return


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

    roi_set_empty = False
    if np.count_nonzero(symmetric_aseg) < 1:
        roi_set_empty = True

    return symmetric_aseg, roi_set_empty


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


def restore_previous_ratings(qcw):
    """Creates a separate folder for ratings, backing up any previous sessions."""

    # making a copy
    incomplete_list = list(qcw.id_list)
    prev_done = []  # empty list

    ratings_file, backup_name_ratings = get_ratings_path_info(qcw)

    if pexists(ratings_file):
        ratings, notes = load_ratings_csv(ratings_file)
        # finding the remaining
        prev_done = set(ratings.keys())
        incomplete_list = list(set(qcw.id_list) - prev_done)
    else:
        ratings = dict()
        notes = dict()

    if len(prev_done) > 0:
        print('\nRatings for {}/{} subjects were restored.'.format(len(prev_done),
                                                                   len(qcw.id_list)))

    if len(incomplete_list) < 1:
        print('No subjects to review/rate - exiting.')
        sys.exit(0)
    else:
        print('To be reviewed : {}\n'.format(len(incomplete_list)))

    return ratings, notes, incomplete_list


def load_ratings_csv(prev_ratings_file):
    """Reads ratings from a CSV file into a dict.

    Format expected in each line:
    subject_id,ratings,notes

    """

    if pexists(prev_ratings_file):
        csv_values = [line.strip().split(',') for line in
                      open(prev_ratings_file).readlines()]
        ratings = {item[0]: item[1] for item in csv_values}
        notes = {item[0]: item[2] for item in csv_values}
    else:
        ratings = dict()
        notes = dict()

    return ratings, notes


def save_ratings_to_disk(ratings, notes, qcw):
    """Save ratings before closing shop."""

    ratings_file, prev_ratings_backup = get_ratings_path_info(qcw)

    if pexists(ratings_file):
        copyfile(ratings_file, prev_ratings_backup)

    lines = '\n'.join(
        ['{},{},{}'.format(sid, rating, notes[sid]) for sid, rating in ratings.items()])
    try:
        with open(ratings_file, 'w') as cf:
            cf.write(lines)
    except:
        raise IOError(
            'Error in saving ratings to file!!\nBackup might be helpful at:\n\t{}'.format(
                prev_ratings_backup))

    return


def get_ratings_path_info(qcw):
    """Common routine to construct the same names"""

    ratings_dir = pjoin(qcw.out_dir, cfg.suffix_ratings_dir)
    if not pexists(ratings_dir):
        makedirs(ratings_dir)

    file_name_ratings = '{}_{}_{}'.format(qcw.vis_type, qcw.suffix, cfg.file_name_ratings)
    ratings_file = pjoin(ratings_dir, file_name_ratings)
    prev_ratings_backup = pjoin(ratings_dir,
                                '{}_{}'.format(cfg.prefix_backup, file_name_ratings))

    return ratings_file, prev_ratings_backup


def check_input_dir(fs_dir, user_dir, vis_type,
                    freesurfer_install_required=True):
    """Ensures proper input is specified."""

    in_dir = fs_dir
    if fs_dir is None and user_dir is None:
        raise ValueError('At least one of --fs_dir or --user_dir must be specified.')

    if fs_dir is not None:
        if user_dir is not None:
            raise ValueError('Only one of --fs_dir or --user_dir can be specified.')

        if freesurfer_install_required and not freesurfer_installed():
            raise EnvironmentError(
                'Freesurfer functionality is requested(e.g. visualizing annotations), but is not installed!')

    if fs_dir is None and vis_type in freesurfer_vis_types:
        raise ValueError(
            'vis_type depending on Freesurfer organization is specified, but --fs_dir is not provided.')

    if user_dir is None:
        if not pexists(fs_dir):
            raise IOError('Freesurfer directory specified does not exist!')
        else:
            in_dir = fs_dir
            type_of_features = 'freesurfer'
    elif fs_dir is None:
        if not pexists(user_dir):
            raise IOError('User-specified input directory does not exist!')
        else:
            in_dir = user_dir
            type_of_features = 'generic'

    if not pexists(in_dir):
        raise IOError(
            'Invalid specification - check proper combination of --fs_dir and --user_dir')

    return in_dir, type_of_features


def check_input_dir_T1(fs_dir, user_dir, bids_dir):
    """Ensures proper input is specified."""

    if fs_dir is None and user_dir is None and bids_dir is None:
        raise ValueError('At least one of --bids_dir or --fs_dir or --user_dir must '
                         'be specified, and only one can be specified.')

    if bids_dir is not None:
        if fs_dir is not None or user_dir is not None:
            raise ValueError('fs_dir and user_dir can NOT be specified when '
                             'specifying BIDS')
        in_dir = realpath(bids_dir)
        if not pexists(in_dir):
            raise IOError('BIDS directory specified does not exist!')

        type_of_features = 'BIDS'
        return in_dir, type_of_features

    if fs_dir is not None:
        if user_dir is not None:
            raise ValueError('--user_dir can not be specified when --fs_dir is ')

    if user_dir is None:
        if not pexists(fs_dir):
            raise IOError('Freesurfer directory specified does not exist!')
        else:
            in_dir = fs_dir
            type_of_features = 'freesurfer'
    elif fs_dir is None:
        if not pexists(user_dir):
            raise IOError('User-specified input directory does not exist!')
        else:
            in_dir = user_dir
            type_of_features = 'generic'

    if not pexists(in_dir):
        raise IOError('Invalid specification: check proper combination of '
                      '--bids_dir, --fs_dir,  --user_dir')

    return realpath(in_dir), type_of_features


def check_input_dir_alignment(in_dir):
    """Ensures proper input is specified."""

    if in_dir is None or not pexists(in_dir):
        raise IOError('Invalid dir is None or does not exist!')

    type_of_features = 'generic'

    return in_dir, type_of_features


def check_bids_dir(dir_path):
    """Checks if its a BIDS folder or not"""

    descr_file_name = 'dataset_description.json'
    descr_path = pjoin(dir_path, descr_file_name)
    if not pexists(descr_path):
        raise ValueError('There is no {} file at the root\n '
                         'Ensure folder is formatted according to BIDS spec.'
                         ''.format(descr_file_name))

    try:
        import json
        with open(descr_path) as df:
            descr = json.load(df)
    except:
        raise IOError('{} could not be read'.format(descr_path))

    ver_tag = 'BIDSVersion'
    if 'BIDSVersion' not in descr:
        raise IOError('There is no field {} in \n\t {}'.format(ver_tag, descr_path))

    in_dir = realpath(dir_path)
    dir_type = 'BIDSVersion:'+descr['BIDSVersion']

    return in_dir, dir_type


def freesurfer_installed():
    """Checks whether Freesurfer is installed."""

    if os.getenv('FREESURFER_HOME') is None or which(freesurfer_vis_cmd) is None:
        return False

    return True


def check_out_dir(out_dir, base_dir):
    """Creates the output folder."""

    if out_dir is None:
        out_dir = pjoin(base_dir, default_out_dir_name)

    try:
        os.makedirs(out_dir, exist_ok=True)
    except:
        raise IOError('Unable to create the output directory as requested.')

    return out_dir


def check_id_list(id_list_in, in_dir, vis_type,
                  mri_name, seg_name=None,
                  in_dir_type=None):
    """
    Checks to ensure each subject listed has the required files
    and returns only those that can be processed.
    """

    if id_list_in is not None:
        if not pexists(id_list_in):
            raise IOError('Given ID list does not exist!')

        try:
            id_list = read_id_list(id_list_in)
        except:
            raise IOError('unable to read the ID list.')
    else:
        # get all IDs in the given folder
        id_list = [folder for folder in os.listdir(in_dir) if
                   os.path.isdir(pjoin(in_dir, folder))]

    if seg_name is not None:
        required_files = {'mri': mri_name, 'seg': seg_name}
    else:
        required_files = {'mri': mri_name}

    id_list_out = list()
    id_list_err = list()
    invalid_list = list()

    # this dict contains existing files for each ID
    # useful to open external programs like tkmedit
    images_for_id = dict()

    for subject_id in id_list:
        path_list = { img: get_path_for_subject(in_dir, subject_id, name,
                                                vis_type, in_dir_type)
                        for img, name in required_files.items()
                    }
        invalid = [pfile for pfile in path_list.values() if
                   not pexists(pfile) or os.path.getsize(pfile) <= 0]
        if len(invalid) > 0:
            id_list_err.append(subject_id)
            invalid_list.extend(invalid)
        else:
            id_list_out.append(subject_id)
            images_for_id[subject_id] = path_list

    if len(id_list_err) > 0:
        warnings.warn('The following subjects do NOT have all the required files'
                      ' or some are empty - skipping them!')
        print('\n'.join(id_list_err))
        print('\n\nThe following files do not exist or empty:'
              ' \n {} \n\n'.format('\n'.join(invalid_list)))

    if len(id_list_out) < 1:
        raise ValueError('All the subject IDs do not have the required files'
                         ' - unable to proceed.')

    print('{} subjects are usable for review.'.format(len(id_list_out)))

    return np.array(id_list_out), images_for_id


def check_id_list_with_regex(id_list_in, in_dir, name_pattern):
    """Checks to ensure each subject listed has the required files and returns only those that can be processed."""

    if id_list_in is not None:
        if not pexists(id_list_in):
            raise IOError('Given ID list does not exist!')

        try:
            id_list = read_id_list(id_list_in)
        except:
            raise IOError('unable to read the ID list.')
    else:
        # get all IDs in the given folder
        id_list = [folder for folder in os.listdir(in_dir) if
                   os.path.isdir(pjoin(in_dir, folder))]

    id_list_out = list()
    id_list_err = list()
    invalid_list = list()

    # this dict contains existing files for each ID
    # useful to open external programs like tkmedit
    images_for_id = dict()

    for subject_id in id_list:
        results = expand_regex_paths(in_dir, subject_id, name_pattern)
        if len(results) < 1:
            print('No results for {} - skipping it.'.format(subject_id))
            continue
        for dp in results:
            if not pexists(dp) or os.path.getsize(dp) <= 0:
                id_list_err.append(subject_id)
                invalid_list.append(dp)
            else:
                new_id = splitext(basename(dp))[0]
                if subject_id not in new_id:
                    new_id = '{}_{}'.format(subject_id, new_id)
                id_list_out.append(new_id)
                images_for_id[new_id] = dp

    if len(id_list_err) > 0:
        warnings.warn('The following subjects do NOT have all the required files '
                      'or some are empty - skipping them!')
        print('\n'.join(id_list_err))
        print('\n\nThe following files do not exist or empty: \n {} \n\n'.format(
            '\n'.join(invalid_list)))

    if len(id_list_out) < 1:
        raise ValueError(
            'All the subject IDs do not have the required files - unable to proceed.')

    print('{} subjects/sessions/units are usable for review.'.format(len(id_list_out)))

    return np.array(id_list_out), images_for_id


def read_id_list(id_list_file):
    """Read all lines and strip them of newlines/spaces."""

    id_list = np.array([line.strip('\n ') for line in open(id_list_file)])

    return id_list


def write_id_list(id_list, file_path, delimiter='\n', file_mode='w'):
    """Writes a list of ids into a file path specified
        in the mode specified (w for overwrite and a for append),
        joined a given delimiter.

    """

    with open(realpath(file_path), file_mode) as fp:
        fp.write(delimiter.join(id_list))

    return


def expand_regex_paths(in_dir, subject_id, req_file):
    """Simple file finder with regex in name."""

    from glob import glob

    path_to_glob = pjoin(in_dir, subject_id, req_file)
    results = glob(path_to_glob, recursive=False)

    # file_path = results[0]
    # if len(results) > 1:
    #     print('Duplicate results found for {}\n'
    #           'Using {} \n'
    #           'igoring the follwing:\n'
    #           '{}'.format(subject_id, file_path, '\n'.join(results[1:])))

    return results


def get_path_for_subject(in_dir, subject_id, req_file, vis_type, in_dir_type=None):
    """Constructs the path for the image file based on chosen input and visualization type"""

    if vis_type is not None and (
        vis_type in freesurfer_vis_types or in_dir_type in ['freesurfer', ]):
        out_path = get_freesurfer_mri_path(in_dir, subject_id, req_file)
    else:
        out_path = realpath(pjoin(in_dir, subject_id, req_file))

    return out_path


def get_freesurfer_mri_path(in_dir, subject_id, req_file):

    return realpath(pjoin(in_dir, subject_id, 'mri', req_file))


def check_time(time_interval, var_name='time interval'):
    """"""

    time_interval = np.float(time_interval)
    if not np.isfinite(time_interval) or np.isclose(time_interval, 0.0):
        raise ValueError('Value of {} must be > 0 and be finite.'.format(var_name))

    return time_interval


def check_outlier_params(method, fraction, feat_types, disable_outlier_detection,
                         id_list, vis_type, type_of_features):
    """Validates parameters related to outlier detection"""

    if disable_outlier_detection:
        method = fraction = feat_types = None
        return method, fraction, feat_types, disable_outlier_detection

    method = method.lower()
    if method not in cfg.avail_outlier_detection_methods:
        raise NotImplementedError(
            'Chosen outlier detection method invalid or not implemented.'
            '\n\tChoose one of {}'.format(cfg.avail_outlier_detection_methods))

    if type_of_features not in cfg.avail_OLD_source_of_features:
        raise NotImplementedError(
            'Outlier detection based on current source of '
            'features ({}) is not implemented.\n Allowed: {} '
            ' '.format(type_of_features, cfg.avail_OLD_source_of_features))

    if type_of_features.lower() == 'freesurfer' and \
        vis_type not in cfg.freesurfer_vis_types:
        raise NotImplementedError(
            'Outlier detection based on current vis_type is not implemented.'
            '\nAllowed visualization types: {}'.format(cfg.freesurfer_vis_types))

    fraction = np.float64(fraction)
    # not clipping automatically to force the user to think about it.
    # fraction = min(max(1 / ns, fraction), 0.5)
    if id_list is not None:
        num_ids = len(id_list)
        if num_ids >= cfg.min_num_samples_needed:
            if fraction < 1 / num_ids:
                raise ValueError('Invalid fraction of outliers:'
                                 ' must be more than 1/n ({}) '
                                 ' (to enable detection of atleast 1 sample)'
                                 ''.format(1/num_ids))
        else:
            print('\nNumber of samples to review = {} < {}: '
                  ' insufficient for outlier detection, disabling it.\n'
                  ''.format(num_ids, cfg.min_num_samples_needed))
            disable_outlier_detection = True

    if fraction > 0.5:
        raise ValueError('Invalid fraction of outliers: can not be more than 50%')

    if not isinstance(feat_types, (list, tuple)):
        feat_types = [feat_types, ]

    feat_types = [feat.lower() for feat in feat_types]
    for feat in feat_types:
        if feat not in cfg.features_outlier_detection:
            raise NotImplementedError('{} features for outlier detection '
                                      'is not recognized / implemented'.format(feat))

    return method, fraction, feat_types, disable_outlier_detection


def check_labels(vis_type, label_set):
    """
    Validates the vis type and ensures appropriate labels are specified.
    Returns a set of unique labels for volumetric visualizations.

    """

    vis_type = vis_type.lower()
    if vis_type not in visualization_combination_choices:
        raise ValueError('Selected visualization type not recognized! '
                         'Choose one of:\n{}'.format(visualization_combination_choices))

    if label_set is not None:
        if vis_type not in cfg.label_types:
            raise ValueError(
                'Invalid selection of vis_type when labels are specifed. '
                'Choose one of {} for --vis_type'.format(cfg.label_types))

        label_set = np.unique(np.array(label_set).astype('int16'))
        if label_set.size < 1:
            raise ValueError('Atleast one unique label must be supplied!')

    elif vis_type in cfg.label_types:
        raise ValueError('When vis_type is one of {}, at least one label must be specified!'
                         ''.format(cfg.label_types))

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
        raise ValueError(
            'Atleast one valid view must be selected. Choose one or more of 0, 1, 2.')

    return out_views


def check_string_is_nonempty(string, string_type='string'):
    """Ensures input is a string of non-zero length"""

    if string is None or \
        (not isinstance(string, str)) or \
        len(string) < 1:
        raise ValueError('name of the {} must not be empty!'
                         ''.format(string_type))

    return string


def check_inputs_defacing(in_dir, defaced_name, mri_name, render_name, id_list_in):
    """Validates the integrity of the inputs"""

    in_dir = realpath(in_dir)
    if not pexists(in_dir):
        raise ValueError('user_dir does not exist : {}'.format(in_dir))

    defaced_name = check_string_is_nonempty(defaced_name, 'defaced MRI scan')
    mri_name = check_string_is_nonempty(mri_name, 'original MRI scan')
    render_name = check_string_is_nonempty(render_name, '3D rendered image')

    if id_list_in is not None:
        if not pexists(id_list_in):
            raise IOError('Given ID list does not exist @ \n'
                          ' {}'.format(id_list_in))

        try:
            id_list = read_id_list(id_list_in)
        except:
            raise IOError('unable to read the ID list @ {}'.format(id_list_in))
    else:
        # get all IDs in the given folder
        id_list = [folder for folder in os.listdir(in_dir) if
                   os.path.isdir(pjoin(in_dir, folder))]

    required_files = {'original': mri_name,
                      'defaced': defaced_name} # 'render': render_name

    id_list_out = list()
    id_list_err = list()
    invalid_list = list()

    # this dict contains existing files for each ID
    # useful to open external programs like tkmedit
    images_for_id = dict()

    for subject_id in id_list:
        path_list = { img_type: pjoin(in_dir, subject_id, name)
                        for img_type, name in required_files.items()
                    }

        # finding all rendered screenshots
        import fnmatch
        rendered_images = fnmatch.filter(os.listdir(pjoin(in_dir, subject_id)),
                                         '{}*'.format(render_name))
        rendered_images = [pjoin(in_dir, subject_id, img) for img in rendered_images]
        rendered_images = [ path for path in rendered_images
                            if pexists(path) and os.path.getsize(path) > 0]

        invalid = [pfile for pfile in path_list.values() if
                   not pexists(pfile) or os.path.getsize(pfile) <= 0]
        if len(invalid) > 0:
            id_list_err.append(subject_id)
            invalid_list.extend(invalid)
        else:
            id_list_out.append(subject_id)
            if len(rendered_images) < 1:
                raise ValueError(
                    'Atleast 1 non-empty rendered image is required!')
            path_list['render'] = rendered_images
            images_for_id[subject_id] = path_list

    if len(id_list_err) > 0:
        warnings.warn('The following subjects do NOT have all the required files, '
                       'or some are empty - skipping them!')
        print('\n'.join(id_list_err))
        print('\n\nThe following files do not exist or empty: \n {} \n\n'.format(
            '\n'.join(invalid_list)))

    if len(id_list_out) < 1:
        raise ValueError('All the subject IDs do not have the required files - '
                          'unable to proceed.')

    print('{} subjects are usable for review.'.format(len(id_list_out)))

    return in_dir, np.array(id_list_out), images_for_id, \
           defaced_name, mri_name, render_name


def compute_cell_extents_grid(bounding_rect=(0.03, 0.03, 0.97, 0.97),
                               num_rows=2, num_cols=6,
                               axis_pad=0.01):
    """
    Produces array of num_rows*num_cols elements each containing the rectangular extents of
    the corresponding cell the grid, whose position is within bounding_rect.
    """

    left, bottom, width, height = bounding_rect
    height_padding = axis_pad * (num_rows + 1)
    width_padding = axis_pad * (num_cols + 1)
    cell_height = float((height - height_padding) / num_rows)
    cell_width = float((width - width_padding) / num_cols)

    cell_height_padded = cell_height + axis_pad
    cell_width_padded = cell_width + axis_pad

    extents = list()
    for row in range(num_rows - 1, -1, -1):
        for col in range(num_cols):
            extents.append((left + col * cell_width_padded,
                            bottom + row * cell_height_padded,
                            cell_width, cell_height))

    return extents
