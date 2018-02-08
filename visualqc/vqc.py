# -*- coding: utf-8 -*-

"""Main module."""

import argparse
import sys
import textwrap
from os.path import join as pjoin

from matplotlib.colors import is_color_like
from visualqc import config as cfg
from visualqc.config import default_out_dir_name, default_mri_name, default_seg_name, \
    visualization_combination_choices, default_label_set, default_alpha_set, default_views, default_num_slices, \
    default_num_rows, default_vis_type, default_freesurfer_dir, default_user_dir, \
    default_alpha_mri, default_alpha_seg
from visualqc.utils import read_image, void_subcortical_symmetrize_cortical, check_alpha_set, get_label_set, \
    check_finite_int, get_ratings, save_ratings, check_id_list, check_labels, check_views, check_input_dir, \
    check_out_dir, get_path_for_subject
from visualqc.viz import review_and_rate


def run_workflow(vis_type, label_set, fs_dir, id_list, out_dir,
                 mri_name=default_mri_name, seg_name=default_seg_name,
                 alpha_set=default_alpha_set, contour_color=cfg.default_contour_face_color,
                 views=default_views, num_slices=default_num_slices, num_rows=default_num_rows):
    """Generate the required visualizations for the specified subjects."""

    ratings, ratings_dir, incomplete_list, prev_done = get_ratings(out_dir, id_list)
    for subject_id in incomplete_list:
        print('Reviewing {}'.format(subject_id))
        t1_mri, overlay_seg, out_path = _prepare_images(fs_dir, subject_id, mri_name, seg_name,
                                                        out_dir, vis_type, label_set)
        ratings[subject_id], quit_now = review_and_rate(t1_mri, overlay_seg,
                                                        vis_type=vis_type, contour_color=contour_color,
                                                        out_dir=out_dir, fs_dir=fs_dir, subject_id=subject_id,
                                                        views=views, num_rows=num_rows, num_slices=num_slices,
                                                        output_path=out_path,
                                                        alpha_mri=alpha_set[0], alpha_seg=alpha_set[1],
                                                        annot='ID {}'.format(subject_id))
        # informing only when it was rated!
        if ratings[subject_id] is not None:
            print('id {} rating {}'.format(subject_id, ratings[subject_id]))
        else:
            ratings.pop(subject_id)

        if quit_now:
            print('\nUser chosen to quit..')
            break

    print('Saving ratings .. \n')
    save_ratings(ratings, out_dir)

    return


def _prepare_images(in_dir, subject_id, mri_name, seg_name, out_dir, vis_type, label_set):
    """Actual routine to generate the visualizations. """

    # we ensured these files exist and are not empty
    t1_mri_path = get_path_for_subject(in_dir, subject_id, mri_name, vis_type) # pjoin(in_dir, subject_id, 'mri', mri_name)
    fs_seg_path = get_path_for_subject(in_dir, subject_id, seg_name, vis_type)

    t1_mri = read_image(t1_mri_path, error_msg='T1 mri')
    fs_seg = read_image(fs_seg_path, error_msg='segmentation')

    if t1_mri.shape != fs_seg.shape:
        raise ValueError('size mismatch! MRI: {} Seg: {}\n'
                         'Size must match in all dimensions.'.format(t1_mri.shape, fs_seg.shape))

    if label_set is not None:
        fs_seg = get_label_set(fs_seg, label_set)

    suffix = ''
    if vis_type in ('cortical_volumetric','cortical_contour'):
        out_seg = void_subcortical_symmetrize_cortical(fs_seg)
        # generate pial surface

    elif vis_type in ('labels_volumetric', 'labels_contour'):
        out_seg = fs_seg
        if label_set is not None:
            suffix = '_'.join([str(lbl) for lbl in list(label_set)])
    else:
        raise NotImplementedError('Other visualization combinations have not been implemented yet! Stay tuned.')

    out_path = pjoin(out_dir, 'visual_qc_{}_{}_{}'.format(vis_type, suffix, subject_id))

    return t1_mri, out_seg, out_path


def get_parser():
    "Parser to specify arguments and their defaults."

    parser = argparse.ArgumentParser(prog="visualqc", formatter_class=argparse.RawTextHelpFormatter,
                                     description='visualqc: rate accuracy of anatomical segmentations and parcellations')

    help_text_fs_dir = textwrap.dedent("""
    Absolute path to ``SUBJECTS_DIR`` containing the finished runs of Freesurfer parcellation
    Each subject will be queried after its ID in the metadata file.

    E.g. ``--fs_dir /project/freesurfer_v5.3``
    \n""")

    help_text_user_dir = textwrap.dedent("""
    Absolute path to an input folder containing the MRI and segmentation images.
    Must specify 
     1) --vis_type to perform the right preprocessing and overlays.
     2) --mri_name and --seg_name (if they differ from defaults), as no specific input folder organization is expected - 
      unlike --fs_dir which looks for images in ``mri`` folder of the subject's Freesurfer folder)
    Each subject will be queried after its ID in the metadata file.

    E.g. ``--user_dir /project/images_to_QC``
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
    Default: {} (volumetric overlay of cortical segmentation on T1 mri).
    \n""".format(default_vis_type))

    help_text_out_dir = textwrap.dedent("""
    Output folder to store the visualizations & ratings.
    Default: a new folder called ``{}`` will be created inside the ``fs_dir``
    \n""".format(default_out_dir_name))

    help_text_alphas = textwrap.dedent("""
    Alpha values to control the transparency of MRI and aseg. 
    This must be a set of two values (between 0 and 1.0) separated by a space e.g. --alphas 0.7 0.5. 
    
    Default: {} {}.  Play with these values to find something that works for you and the dataset.
    \n""".format(default_alpha_mri, default_alpha_seg))

    help_text_label = textwrap.dedent("""
    Specifies the set of labels to include for overlay.
    
    Default: None (show all the labels in the selected segmentation)
    \n""")

    help_text_mri_name = textwrap.dedent("""
    Specifies the name of MRI image to serve as the reference slice.
    Typical options include orig.mgz, brainmask.mgz, T1.mgz etc.
    Make sure to choose the right vis_type.
    
    Default: {} (within the mri folder of Freesurfer format).
    \n""".format(default_mri_name))

    help_text_seg_name = textwrap.dedent("""
    Specifies the name of segmentation image (volumetric) to be overlaid on the MRI.
    Typical options include aparc+aseg.mgz, aseg.mgz, wmparc.mgz. 
    Make sure to choose the right vis_type. 

    Default: {} (within the mri folder of Freesurfer format).
    \n""".format(default_seg_name))

    help_text_views = textwrap.dedent("""
    Specifies the set of views to display - could be just 1 view, or 2 or all 3.
    Example: --views 0 (typically sagittal) or --views 1 2 (axial and coronal)
    Default: {} {} {} (show all the views in the selected segmentation)
    \n""".format(default_views[0], default_views[1], default_views[2]))

    help_text_num_slices = textwrap.dedent("""
    Specifies the number of slices to display per each view. 
    This must be even to facilitate better division.
    Default: {}.
    \n""".format(default_num_slices))

    help_text_num_rows = textwrap.dedent("""
    Specifies the number of rows to display per each axis. 
    Default: {}.
    \n""".format(default_num_rows))

    help_text_contour_color = textwrap.dedent("""
    Specifies the color to use for the contours overlaid on MRI (when vis_type requested prescribes contours). 
    Color can be specified in many ways as documented in https://matplotlib.org/users/colors.html
    Default: {}.
    \n""".format(cfg.default_contour_face_color))

    parser.add_argument("-f", "--fs_dir", action="store", dest="fs_dir",
                        default=default_freesurfer_dir,
                        required=False, help=help_text_fs_dir)

    parser.add_argument("-i", "--id_list", action="store", dest="id_list",
                        default=None, required=False, help=help_text_id_list)

    parser.add_argument("-v", "--vis_type", action="store", dest="vis_type",
                        choices=visualization_combination_choices,
                        default=default_vis_type, required=False,
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
                        help=help_text_label, metavar='label')

    parser.add_argument("-m", "--mri_name", action="store", dest="mri_name",
                        default=default_mri_name, required=False,
                        help=help_text_mri_name)

    parser.add_argument("-g", "--seg_name", action="store", dest="seg_name",
                        default=default_seg_name, required=False,
                        help=help_text_seg_name)

    parser.add_argument("-w", "--views", action="store", dest="views",
                        default=default_views, required=False, nargs='+',
                        help=help_text_views)

    parser.add_argument("-s", "--num_slices", action="store", dest="num_slices",
                        default=default_num_slices, required=False,
                        help=help_text_num_slices)

    parser.add_argument("-r", "--num_rows", action="store", dest="num_rows",
                        default=default_num_rows, required=False,
                        help=help_text_num_rows)

    parser.add_argument("-c", "--contour_color", action="store", dest="contour_color",
                        default=cfg.default_contour_face_color, required=False,
                        help=help_text_contour_color)

    parser.add_argument("-u", "--user_dir", action="store", dest="user_dir",
                        default=default_user_dir,
                        required=False, help=help_text_user_dir)

    return parser


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

    vis_type, label_set = check_labels(user_args.vis_type, user_args.labels)

    in_dir = check_input_dir(user_args.fs_dir, user_args.user_dir, vis_type)

    mri_name = user_args.mri_name
    seg_name = user_args.seg_name
    id_list = check_id_list(user_args.id_list, in_dir, vis_type, mri_name, seg_name)

    out_dir = check_out_dir(user_args.out_dir, in_dir)

    alpha_set = check_alpha_set(user_args.alpha_set)

    views = check_views(user_args.views)

    num_slices, num_rows = check_finite_int(user_args.num_slices, user_args.num_rows)

    contour_color = user_args.contour_color
    if not is_color_like(contour_color):
        raise ValueError('Specified color is not valid. Choose a valid spec from\n https://matplotlib.org/users/colors.html')

    return in_dir, id_list, out_dir, vis_type, label_set, \
           alpha_set, views, num_slices, num_rows, mri_name, seg_name, contour_color


def cli_run():
    """Main entry point."""

    fs_dir, id_list, out_dir, vis_type, label_set, alpha_set, \
    views, num_slices, num_rows, mri_name, seg_name, contour_color = parse_args()

    if vis_type is not None:
        # matplotlib.interactive(True)
        run_workflow(vis_type=vis_type, label_set=label_set,
                     fs_dir=fs_dir, id_list=id_list,
                     mri_name=mri_name, seg_name=seg_name, contour_color=contour_color,
                     out_dir=out_dir, alpha_set=alpha_set,
                     views=views, num_slices=num_slices, num_rows=num_rows)
        print('Results are available in:\n\t{}'.format(out_dir))
    else:
        raise ValueError('Invalid state for visualQC!\n\t Ensure proper combination of arguments is used.')

    return


if __name__ == '__main__':
    cli_run()
