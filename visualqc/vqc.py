# -*- coding: utf-8 -*-

"""Main module."""

import argparse
import sys
import textwrap
from os.path import join as pjoin
import warnings
from matplotlib.colors import is_color_like
from visualqc import config as cfg
from visualqc.config import default_out_dir_name, default_mri_name, default_seg_name, \
    visualization_combination_choices, default_label_set, default_alpha_set, default_views, default_num_slices, \
    default_num_rows, default_vis_type, default_freesurfer_dir, default_user_dir, \
    default_alpha_mri, default_alpha_seg
from visualqc.utils import read_image, void_subcortical_symmetrize_cortical, check_alpha_set, get_label_set, \
    check_finite_int, restore_previous_ratings, save_ratings_to_disk, check_id_list, check_labels, check_views, check_input_dir, \
    check_out_dir, get_path_for_subject, check_outlier_params
from visualqc.viz import review_and_rate, generate_required_visualizations
from visualqc.outliers import outlier_advisory


class QCWorkflow():
    """
    Class encapsulating the necessary parameters to run the workflow.
    """

    def __init__(self, in_dir, id_list, images_for_id, out_dir,
                 prepare_first, vis_type, source_of_features, label_set, alpha_set,
                 outlier_method, outlier_fraction, outlier_feat_types, disable_outlier_detection,
                 views, num_slices, num_rows,
                 mri_name, seg_name, contour_color,
                 rating_list=cfg.default_rating_list):
        """Constructor"""

        self.in_dir = in_dir
        self.id_list = id_list
        self.out_dir = out_dir

        self.vis_type = vis_type
        self.label_set = label_set
        self.source_of_features = source_of_features

        self.alpha_set = alpha_set
        self.alpha_mri = self.alpha_set[0]
        self.alpha_seg = self.alpha_set[1]

        self.views = views
        self.num_slices = num_slices
        self.num_rows = num_rows

        self.mri_name = mri_name
        self.seg_name = seg_name
        self.images_for_id = images_for_id
        self.contour_color = contour_color

        self.outlier_method = outlier_method
        self.outlier_fraction = outlier_fraction
        self.outlier_feat_types = outlier_feat_types
        self.disable_outlier_detection = disable_outlier_detection
        self.prepare_first = prepare_first

        self.rating_list = rating_list
        self._generate_vis_type_suffix()

    def _generate_vis_type_suffix(self):
        """Generates a distinct suffix for a given vis type and labels. """

        self.suffix = ''
        if self.vis_type in cfg.vis_types_with_multiple_ROIs:
            if self.label_set is not None:
                self.suffix = '_'.join([str(lbl) for lbl in list(self.label_set)])

    def save_cmd(self):
        """Saves the command issued by the user for debugging purposes"""

        cmd_file = pjoin(self.out_dir, 'cmd_issued.visualqc')
        with open(cmd_file, 'w') as cf:
            cf.write('{}\n'.format(' '.join(sys.argv)))

        return

    def save(self):
        """
        Saves the state of the QC workflow for restoring later on,
            as well as for future reference.

        """

        pass

    def reload(self):
        """Method to reload the saved state."""

        pass



def run_workflow(qcw):
    """Generate the required visualizations for the specified subjects."""

    if qcw.prepare_first:
        generate_required_visualizations(qcw)

    outliers_by_sample, outliers_by_feature = outlier_advisory(qcw)

    ratings, notes, incomplete_list = restore_previous_ratings(qcw)
    for subject_id in incomplete_list:
        flagged_as_outlier = subject_id in outliers_by_sample
        alerts_outlier = outliers_by_sample.get(subject_id, None) # None, if id not in dict
        outlier_alert_msg = '\n\tFlagged as a possible outlier by these measures:\n\t{}'.format(alerts_outlier) \
            if flagged_as_outlier else ' '
        print('\nReviewing {} {}'.format(subject_id, outlier_alert_msg))
        t1_mri, overlay_seg, out_path, skip_subject = _prepare_images(qcw, subject_id)

        if skip_subject:
            print('Skipping current subject ..')
            continue

        ratings[subject_id], notes[subject_id], quit_now = review_and_rate(qcw, t1_mri, overlay_seg,
                                                                           subject_id=subject_id,
                                                                           flagged_as_outlier=flagged_as_outlier,
                                                                           outlier_alerts=alerts_outlier,
                                                                           output_path=out_path,
                                                                           annot='ID {}'.format(subject_id))
        # informing only when it was rated!
        if ratings[subject_id] not in cfg.ratings_not_to_be_recorded:
            print('id {} rating {} notes {}'.format(subject_id, ratings[subject_id], notes[subject_id]))
        else:
            ratings.pop(subject_id)

        if quit_now:
            print('\nUser chosen to quit..')
            break

    print('Saving ratings .. \n')
    save_ratings_to_disk(ratings, notes, qcw)
    #TODO save QCW

    return


def _prepare_images(qcw, subject_id):
    """Actual routine to generate the visualizations. """

    # qcw.fs_dir, qcw.subject_id, qcw.mri_name, qcw.seg_name, qcw.out_dir, qcw.vis_type, qcw.label_set

    # we ensured these files exist and are not empty
    t1_mri_path = get_path_for_subject(qcw.in_dir, subject_id, qcw.mri_name, qcw.vis_type)
    fs_seg_path = get_path_for_subject(qcw.in_dir, subject_id, qcw.seg_name, qcw.vis_type)

    t1_mri = read_image(t1_mri_path, error_msg='T1 mri')
    fs_seg = read_image(fs_seg_path, error_msg='segmentation')

    if t1_mri.shape != fs_seg.shape:
        raise ValueError('size mismatch! MRI: {} Seg: {}\n'
                         'Size must match in all dimensions.'.format(t1_mri.shape, fs_seg.shape))

    skip_subject = False
    if qcw.label_set is not None:
        fs_seg, roi_set_empty = get_label_set(fs_seg, qcw.label_set)
        if roi_set_empty:
            skip_subject = True
            print('segmentation image for this subject does not contain requested label set!')

    if qcw.vis_type in ('cortical_volumetric', 'cortical_contour'):
        out_seg = void_subcortical_symmetrize_cortical(fs_seg)
    elif qcw.vis_type in ('labels_volumetric', 'labels_contour'):
        # TODO in addition to checking file exists, we need to requested labels exist, for label vis_type
        out_seg = fs_seg
    else:
        raise NotImplementedError('Other visualization combinations have not been implemented yet! Stay tuned.')

    out_path = pjoin(qcw.out_dir, 'visual_qc_{}_{}_{}'.format(qcw.vis_type, qcw.suffix, subject_id))

    return t1_mri, out_seg, out_path, skip_subject


def get_parser():
    """Parser to specify arguments and their defaults."""

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

    help_text_outlier_detection_method = textwrap.dedent("""
    Method used to detect the outliers.
    
    For more info, read http://scikit-learn.org/stable/modules/outlier_detection.html
    
    Default: {}.
    \n""".format(cfg.default_outlier_detection_method))

    help_text_outlier_fraction = textwrap.dedent("""
    Fraction of outliers expected in the given sample. Must be >= 1/n and <= (n-1)/n, 
    where n is the number of samples in the current sample.

    For more info, read http://scikit-learn.org/stable/modules/outlier_detection.html

    Default: {}.
    \n""".format(cfg.default_outlier_fraction))

    help_text_outlier_feat_types = textwrap.dedent("""
    Type of features to be employed in training the outlier detection method.  It could be one of  
    'cortical' (aparc.stats: mean thickness and other geometrical features from each cortical label), 
    'subcortical' (aseg.stats: volumes of several subcortical structures), 
    or 'both' (using both aseg and aparc stats).
    
    Default: {} {}.
    \n""".format(cfg.freesurfer_features_outlier_detection[0], cfg.freesurfer_features_outlier_detection[1]))

    help_text_disable_outlier_detection = textwrap.dedent("""
    This flag disables outlier detection and alerts altogether.
    \n""")

    help_text_prepare = textwrap.dedent("""
    This flag enables batch-generation of 3d surface visualizations, prior to starting any review and rating operations. 
    This makes the switch from one subject to the next, even more seamless (saving few seconds :) ).
    
    Default: False (required visualizations are generated only on demand, which can take 5-10 seconds for each subject).
    \n""")

    in_out = parser.add_argument_group('Input and output', ' ')
    in_out.add_argument("-f", "--fs_dir", action="store", dest="fs_dir",
                        default=default_freesurfer_dir,
                        required=False, help=help_text_fs_dir)

    in_out.add_argument("-i", "--id_list", action="store", dest="id_list",
                        default=None, required=False, help=help_text_id_list)

    in_out.add_argument("-u", "--user_dir", action="store", dest="user_dir",
                        default=default_user_dir,
                        required=False, help=help_text_user_dir)

    in_out.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        required=False, help=help_text_out_dir,
                        default=None)

    data_source = parser.add_argument_group('Sources of data', ' ')
    data_source.add_argument("-l", "--labels", action="store", dest="labels",
                             default=default_label_set, required=False, nargs='+',
                             help=help_text_label, metavar='label')

    data_source.add_argument("-m", "--mri_name", action="store", dest="mri_name",
                             default=default_mri_name, required=False,
                             help=help_text_mri_name)

    data_source.add_argument("-g", "--seg_name", action="store", dest="seg_name",
                             default=default_seg_name, required=False,
                             help=help_text_seg_name)

    vis_args = parser.add_argument_group('Overlay options', ' ')
    vis_args.add_argument("-v", "--vis_type", action="store", dest="vis_type",
                          choices=visualization_combination_choices,
                          default=default_vis_type, required=False,
                          help=help_text_vis_type)

    vis_args.add_argument("-c", "--contour_color", action="store", dest="contour_color",
                          default=cfg.default_contour_face_color, required=False,
                          help=help_text_contour_color)

    vis_args.add_argument("-a", "--alpha_set", action="store", dest="alpha_set",
                          metavar='alpha', nargs=2,
                          default=default_alpha_set,
                          required=False, help=help_text_alphas)

    outliers = parser.add_argument_group('Outlier detection',
                                         'options related to automatically detecting possible outliers')
    outliers.add_argument("-olm", "--outlier_method", action="store", dest="outlier_method",
                          default=cfg.default_outlier_detection_method, required=False,
                          help=help_text_outlier_detection_method)

    outliers.add_argument("-olf", "--outlier_fraction", action="store", dest="outlier_fraction",
                          default=cfg.default_outlier_fraction, required=False,
                          help=help_text_outlier_fraction)

    outliers.add_argument("-olt", "--outlier_feat_types", action="store", dest="outlier_feat_types",
                          default=cfg.freesurfer_features_outlier_detection, required=False,
                          help=help_text_outlier_feat_types)

    outliers.add_argument("-old", "--disable_outlier_detection", action="store_true", dest="disable_outlier_detection",
                          required=False, help=help_text_disable_outlier_detection)

    layout = parser.add_argument_group('Layout options', ' ')
    layout.add_argument("-w", "--views", action="store", dest="views",
                        default=default_views, required=False, nargs='+',
                        help=help_text_views)

    layout.add_argument("-s", "--num_slices", action="store", dest="num_slices",
                        default=default_num_slices, required=False,
                        help=help_text_num_slices)

    layout.add_argument("-r", "--num_rows", action="store", dest="num_rows",
                        default=default_num_rows, required=False,
                        help=help_text_num_rows)

    wf_args = parser.add_argument_group('Workflow', 'Options related to workflow e.g. to pre-generate all the visualizations required')
    wf_args.add_argument("-p", "--prepare_first", action="store_true", dest="prepare_first",
                          help=help_text_prepare)


    return parser


def parse_user_args():
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

    # TODO methods to restore from previous runs, without having re-enter all parameters

    vis_type, label_set = check_labels(user_args.vis_type, user_args.labels)

    in_dir, source_of_features = check_input_dir(user_args.fs_dir, user_args.user_dir, vis_type)

    mri_name = user_args.mri_name
    seg_name = user_args.seg_name
    id_list, images_for_id = check_id_list(user_args.id_list, in_dir, vis_type, mri_name, seg_name)

    out_dir = check_out_dir(user_args.out_dir, in_dir)

    alpha_set = check_alpha_set(user_args.alpha_set)

    views = check_views(user_args.views)

    num_slices, num_rows = check_finite_int(user_args.num_slices, user_args.num_rows)

    contour_color = user_args.contour_color
    if not is_color_like(contour_color):
        raise ValueError(
            'Specified color is not valid. Choose a valid spec from\n https://matplotlib.org/users/colors.html')

    outlier_method, outlier_fraction, outlier_feat_types, no_outlier_detection = check_outlier_params(user_args.outlier_method,
                                                                                user_args.outlier_fraction,
                                                                                user_args.outlier_feat_types,
                                                                                user_args.disable_outlier_detection,
                                                                                id_list, vis_type, source_of_features)

    qcw = QCWorkflow(in_dir, id_list, images_for_id, out_dir,
                     user_args.prepare_first,
                     vis_type, source_of_features, label_set, alpha_set,
                     outlier_method, outlier_fraction, outlier_feat_types, no_outlier_detection,
                     views, num_slices, num_rows,
                     mri_name, seg_name, contour_color)

    # if the workflow could be instantiated, it means things are in order!
    qcw.save_cmd()

    return qcw


def cli_run():
    """Main entry point."""

    qcw = parse_user_args()

    if qcw.vis_type is not None:
        # matplotlib.interactive(True)
        run_workflow(qcw)
        print('Results are available in:\n\t{}'.format(qcw.out_dir))
    else:
        raise ValueError('Invalid state for visualQC!\n\t Ensure proper combination of arguments is used.')

    return


if __name__ == '__main__':
    # disabling all not severe warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        cli_run()
