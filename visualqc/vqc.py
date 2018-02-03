# -*- coding: utf-8 -*-

"""Main module."""

import os
import sys
import argparse
import textwrap
import nibabel as nib
import traceback
import warnings

import matplotlib
import matplotlib.pyplot as plt
from sys import version_info
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename

from visualqc.utils import read_image, void_subcortical_symmetrize_cortical, check_alpha_set
from visualqc.viz import review_and_rate

# default values
default_out_dir_name = 'visualqc'
t1_mri_identifier = 'orig.mgz' # TODO replace 'brainmask.mgz'
fs_seg_identifier = 'aparc+aseg.mgz'
required_files = (t1_mri_identifier, fs_seg_identifier)
visualization_combination_choices = ('cortical_volumetric', 'cortical_surface',
                                     'cortical_composite', 'subcortical_volumetric')

default_alpha_set = (0.7, 0.7)

def generate_visualizations(make_type, fs_dir, id_list, out_dir, alpha_set):
    """Generate the required visualizations for the specified subjects."""

    for subject_id in id_list:
        print('Processing {}'.format(subject_id))
        _generate_visualizations_per_subject(fs_dir, subject_id, out_dir, make_type, alpha_set)

    return


def _generate_visualizations_per_subject(fs_dir, subject_id, out_dir, make_type, alpha_set):
    """Actual routine to generate the visualizations. """

    # we ensured these files exist and are not empty
    t1_mri_path = pjoin(fs_dir, subject_id, 'mri', t1_mri_identifier)
    fs_seg_path = pjoin(fs_dir, subject_id, 'mri', fs_seg_identifier)

    t1_mri = read_image(t1_mri_path, error_msg='T1 mri')
    fs_seg = read_image(fs_seg_path, error_msg='aparc+aseg segmentation')

    if make_type in ('cortical_volumetric', ):
        ctx_aseg_symmetric = void_subcortical_symmetrize_cortical(fs_seg)
    else:
        raise NotImplementedError('Other visualization combinations have not been implemented yet! Stay tuned.')

    out_path = pjoin(out_dir, 'visual_qc_{}_{}'.format(make_type, subject_id))
    fig = review_and_rate(t1_mri, ctx_aseg_symmetric, output_path=out_path,
                          alpha_mri=alpha_set[0], alpha_seg=alpha_set[1])

    return fig, out_path


def rate_visualizations(rate_dir, id_list):
    """Rating routine."""


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

    help_text_make_type = textwrap.dedent("""
    Generates the visualizations.
    Default: volumetric overlay of cortical segmentation on T1 mri.
    \n""")

    help_text_rate_dir = textwrap.dedent("""
    Starts the review and rate workflow in the given directory.
    Requires that the visualizations be generated previously using --make.

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


    parser.add_argument("-f", "--fs_dir", action="store", dest="fs_dir",
                        required=True, help=help_text_fs_dir)

    parser.add_argument("-i", "--id_list", action="store", dest="id_list",
                        default=None, required=False, help=help_text_id_list)

    parser.add_argument("-m", "--make", action="store", dest="make_type",
                        choices=visualization_combination_choices,
                        default='cortical_volumetric', required=False,
                        help=help_text_make_type)

    parser.add_argument("-r", "--rate_dir", action="store", dest="rate_dir",
                        default=None, required=False,
                        help=help_text_rate_dir)

    parser.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        required=False, help=help_text_out_dir,
                        default=None)

    parser.add_argument("-a", "--alpha_set", action="store", dest="alpha_set",
                        metavar='alpha', nargs=2,
                        default=default_alpha_set,
                        required=False, help=help_text_alphas)

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

    return id_list_out


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

    if not pexists(out_dir):
        try:
            os.makedirs(out_dir)
        except:
            raise IOError('Unable to create the output directory as requested.')

    make_type = user_args.make_type.lower()
    rate_dir = user_args.rate_dir

    if make_type is not None and rate_dir is not None:
        raise ValueError('Only one of --make and --rate_dir can be specified at a time.\n Run --make first, and then --rate_dir afterwards.')

    if make_type is not None and rate_dir is None and not pexists(fs_dir):
        raise IOError('Given Freesurfer directory does not exist.')

    if make_type is None and rate_dir is not None and not pexists(rate_dir):
        raise IOError("""Given directory to review/rate does not exist! \nMake sure to generate visualizations with --make first.""")

    alpha_set = check_alpha_set(user_args.alpha_set)

    return fs_dir, id_list, out_dir, make_type, rate_dir, alpha_set


def cli_run():
    """Main entry point."""

    fs_dir, id_list, out_dir, make_type, rate_dir, alpha_set = parse_args()

    if make_type is not None and rate_dir is None:
        matplotlib.interactive(True)
        generate_visualizations(make_type=make_type, fs_dir=fs_dir, id_list=id_list,
                                out_dir=out_dir, alpha_set=alpha_set)
        print('Results are available in:\n\t{}'.format(out_dir))
    elif make_type is None and rate_dir is not None:
        rate_visualizations(rate_dir=rate_dir, id_list=id_list)
    else:
        raise ValueError('Invalid state for visualQC!\n\t Ensure proper combination of arguments is used.')

    return


if __name__ == '__main__':
    cli_run()
