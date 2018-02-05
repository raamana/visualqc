# -*- coding: utf-8 -*-

"""Main module."""

import argparse
import os
import sys
import textwrap
import warnings
from os import makedirs
from os.path import join as pjoin, exists as pexists, realpath
from shutil import copyfile
import numpy as np
from visualqc.utils import read_image, void_subcortical_symmetrize_cortical, check_alpha_set, get_label_set, check_finite_int
from visualqc.viz import review_and_rate

# default values
default_out_dir_name = 'visualqc'
t1_mri_identifier = 'brainmask.mgz' # TODO make this an option to capture wider variety of use cases
fs_seg_identifier = 'aparc+aseg.mgz'
required_files = (t1_mri_identifier, fs_seg_identifier)
visualization_combination_choices = ('cortical_volumetric', 'labels',
                                     'cortical_surface',
                                     'cortical_composite', 'subcortical_volumetric')
default_label_set = None

default_alpha_set = (0.7, 0.7)

default_views = (0, 1, 2)
default_num_slices = 12
default_num_rows = 2

suffix_ratings_dir='ratings'
file_name_ratings = 'ratings.all.csv'
file_name_ratings_backup = 'backup_ratings.all.csv'

def run_workflow(vis_type, label_set, fs_dir, id_list, out_dir,
                 alpha_set=default_alpha_set,
                 views=default_views, num_slices=default_num_slices, num_rows=default_num_rows):
    """Generate the required visualizations for the specified subjects."""

    ratings, ratings_dir, incomplete_list, prev_done = get_ratings(out_dir, id_list)
    for subject_id in incomplete_list:
        print('Reviewing {}'.format(subject_id))
        t1_mri, overlay_seg, out_path = _prepare_images(fs_dir, subject_id, out_dir, vis_type, label_set)
        ratings[subject_id], quit_now = review_and_rate(t1_mri, overlay_seg, vis_type=vis_type,
                                                        views=views, num_rows=num_rows, num_slices=num_slices, output_path=out_path,
                                                        alpha_mri=alpha_set[0], alpha_seg=alpha_set[1],
                                                        annot='ID {}'.format(subject_id))
        # informing only when it was rated!
        if ratings[subject_id] is not None:
            print('id {} rating {}'.format(subject_id, ratings[subject_id]))
        else:
            ratings.pop(subject_id)

        if quit_now:
            print('user chosen to quit..')
            break

    print('Saving ratings .. \n')
    save_ratings(ratings, out_dir)

    return


def _prepare_images(fs_dir, subject_id, out_dir, vis_type, label_set):
    """Actual routine to generate the visualizations. """

    # we ensured these files exist and are not empty
    t1_mri_path = pjoin(fs_dir, subject_id, 'mri', t1_mri_identifier)
    fs_seg_path = pjoin(fs_dir, subject_id, 'mri', fs_seg_identifier)

    t1_mri = read_image(t1_mri_path, error_msg='T1 mri')
    fs_seg = read_image(fs_seg_path, error_msg='aparc+aseg segmentation')

    if t1_mri.shape != fs_seg.shape:
        raise ValueError('size mismatch! MRI: {} Seg: {}\n'
                         'Size must match in all dimensions.'.format(t1_mri.shape,fs_seg.shape))

    if label_set is not None:
        fs_seg = get_label_set(fs_seg, label_set)

    suffix = ''
    if vis_type in ('cortical_volumetric', ):
        out_seg = void_subcortical_symmetrize_cortical(fs_seg)
    elif vis_type in ('label_set', 'labels'):
        out_seg = fs_seg
        suffix = '_'.join([str(lbl) for lbl in list(label_set)])
    else:
        raise NotImplementedError('Other visualization combinations have not been implemented yet! Stay tuned.')

    out_path = pjoin(out_dir, 'visual_qc_{}_{}_{}'.format(vis_type, suffix, subject_id))

    return t1_mri, out_seg, out_path


def get_ratings(out_dir, id_list):
    """Creates a separate folder for ratings, backing up any previous sessions."""

    # making a copy
    incomplete_list = list(id_list)
    prev_done = [] # empty list

    ratings_dir = pjoin(out_dir, suffix_ratings_dir)
    if pexists(ratings_dir):
        prev_ratings = pjoin(ratings_dir,file_name_ratings)
        prev_ratings_backup = pjoin(ratings_dir, file_name_ratings_backup)
        if pexists(prev_ratings):
            ratings = load_ratings_csv(prev_ratings)
            copyfile(prev_ratings,prev_ratings_backup)
            # finding the remaining
            prev_done = set(ratings.keys())
            incomplete_list = list(set(id_list)-prev_done)
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

    ratings_file = pjoin(ratings_dir,file_name_ratings)
    prev_ratings_backup = pjoin(ratings_dir, file_name_ratings_backup)
    if pexists(ratings_file):
        copyfile(ratings_file,prev_ratings_backup)

    lines = '\n'.join(['{},{}'.format(sid, rating) for sid, rating  in ratings.items()])
    try:
        with open(ratings_file,'w') as cf:
            cf.write(lines)
    except:
        raise IOError('Error in saving ratings to file!!\nBackup might be helpful at:\n\t{}'.format(prev_ratings_backup))

    return


def get_parser():
    "Parser to specify arguments and their defaults."

    parser = argparse.ArgumentParser(prog="visualqc", formatter_class=argparse.RawTextHelpFormatter,
                                     description='visualqc: rate accuracy of anatomical segmentations and parcellations')

    help_text_fs_dir = textwrap.dedent("""
    Absolute path to ``SUBJECTS_DIR`` containing the finished runs of Freesurfer parcellation
    Each subject will be queried after its ID in the metadata file.

    E.g. ``--fs_dir /project/freesurfer_v5.3``
    \n""")

    help_text_id_list = textwrap.dedent("""
    Abs path to file containing list of subject IDs to be processed.
    If not provided, all the subjects with required files will be processed.

    E.g.
    .. parsed-literal::

        sub001
        sub002
        cn_003
        cn_004

    \n""")

    help_text_vis_type = textwrap.dedent("""
    Specifies the type of visualizations/overlay requested.
    Default: volumetric overlay of cortical segmentation on T1 mri.
    \n""")

    help_text_out_dir = textwrap.dedent("""
    Output folder to store the visualizations & ratings.
    Default: a new folder called ``visualqc`` will be created inside the ``fs_dir``
    \n""")

    help_text_alphas = textwrap.dedent("""
    Alpha values to control the transparency of MRI and aseg. 
    This must be a set of two values (between 0 and 1.0) separated by a space e.g. --alphas 0.7 0.5. 
    
    Default: 0.7 0.7
    Play with these values to find something that works for you and the dataset.
    \n""")

    help_text_label = textwrap.dedent("""
    Specifies the set of labels to include for overlay.
    
    Default: None (show all the labels in the selected segmentation)
    \n""")

    help_text_views = textwrap.dedent("""
    Specifies the set of views to display - could be just 1 view, or 2 or all 3.
    Example: --views 0 (typically sagittal) or --views 1 2 (axial and coronal)
    Default: 3 (show all the views in the selected segmentation)
    \n""")

    help_text_num_slices = textwrap.dedent("""
    Specifies the number of slices to display per each view. 
    This must be even to facilitate better division.
    Default: 12.
    \n""")

    help_text_num_rows = textwrap.dedent("""
    Specifies the number of rows to display per each axis. 
    Default: 2.
    \n""")

    parser.add_argument("-f", "--fs_dir", action="store", dest="fs_dir",
                        required=True, help=help_text_fs_dir)

    parser.add_argument("-i", "--id_list", action="store", dest="id_list",
                        default=None, required=False, help=help_text_id_list)

    parser.add_argument("-v", "--vis_type", action="store", dest="vis_type",
                        choices=visualization_combination_choices,
                        default='cortical_volumetric', required=False,
                        help=help_text_vis_type)

    parser.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        required=False, help=help_text_out_dir,
                        default=None)

    parser.add_argument("-a", "--alpha_set", action="store", dest="alpha_set",
                        metavar='alpha', nargs=2,
                        default=default_alpha_set,
                        required=False, help=help_text_alphas)

    parser.add_argument("-l", "--labels", action="store", dest="labels",
                        default=default_label_set, required=False, nargs='+',
                        help=help_text_label)

    parser.add_argument("-w", "--views", action="store", dest="views",
                        default=default_views, required=False, nargs='+',
                        help=help_text_views)

    parser.add_argument("-s", "--num_slices", action="store", dest="num_slices",
                        default=default_num_slices, required=False,
                        help=help_text_num_slices)

    parser.add_argument("-r", "--num_rows", action="store", dest="num_rows",
                        default=default_num_rows, required=False,
                        help=help_text_num_rows)

    return parser


def check_id_list(id_list_in, fs_dir):
    """Checks to ensure each subject listed has the required files and returns only those that can be processed."""

    if id_list_in is not None:
        if not pexists(id_list_in):
            raise IOError('Given ID list does not exist!')

        try:
            # read all lines and strip them of newlines/spaces
            id_list = [ line.strip('\n ') for line in open(id_list_in) ]
        except:
            raise IOError('unable to read the ID list.')
    else:
        # get all IDs in the given folder
        id_list = [ folder for folder in os.listdir(fs_dir) if os.path.isdir(pjoin(fs_dir,folder)) ]

    id_list_out = list()
    id_list_err = list()
    invalid_list = list()

    for subject_id in id_list:
        path_list = [realpath(pjoin(fs_dir, subject_id, 'mri', req_file)) for req_file in required_files]
        invalid = [ this_file for this_file in path_list if not pexists(this_file) or os.path.getsize(this_file)<=0 ]
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


def check_labels(vis_type, label_set):
    """Validates the selections."""

    vis_type = vis_type.lower()
    if vis_type not in visualization_combination_choices:
        raise ValueError('Selected visualization type not recognized! '
                         'Choose one of:\n{}'.format(visualization_combination_choices))

    if label_set is not None:
        if vis_type not in ['label_set', 'labels']:
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
        raise ValueError('Atleast one valid view must selected. Choose one or more of 0, 1, 2.')

    return out_views


def parse_args():
    """Parser/validator for the cmd line args."""

    parser = get_parser()

    if len(sys.argv) < 2:
        print('Too few arguments!')
        parser.print_help()
        parser.exit(1)

    # parsing
    try:
        user_args = parser.parse_args()
    except:
        parser.exit(1)

    fs_dir = user_args.fs_dir
    if not pexists(fs_dir):
        raise IOError('Freesurfer directory specified does not exist!')

    id_list = check_id_list(user_args.id_list, fs_dir)

    out_dir = user_args.out_dir
    if out_dir is None:
        out_dir = pjoin(fs_dir, default_out_dir_name)

    try:
        os.makedirs(out_dir, exist_ok=True)
    except:
        raise IOError('Unable to create the output directory as requested.')

    vis_type, label_set = check_labels(user_args.vis_type, user_args.labels)

    alpha_set = check_alpha_set(user_args.alpha_set)

    views = check_views(user_args.views)

    num_slices, num_rows = check_finite_int(user_args.num_slices, user_args.num_rows)

    return fs_dir, id_list, out_dir, vis_type, label_set, alpha_set, views, num_slices, num_rows


def cli_run():
    """Main entry point."""

    fs_dir, id_list, out_dir, vis_type, label_set, alpha_set, views, num_slices, num_rows = parse_args()

    if vis_type is not None:
        # matplotlib.interactive(True)
        run_workflow(vis_type=vis_type, label_set=label_set,
                     fs_dir=fs_dir, id_list=id_list,
                     out_dir=out_dir, alpha_set=alpha_set,
                     views=views, num_slices=num_slices, num_rows=num_rows)
        print('Results are available in:\n\t{}'.format(out_dir))
    else:
        raise ValueError('Invalid state for visualQC!\n\t Ensure proper combination of arguments is used.')

    return


if __name__ == '__main__':
    cli_run()
