"""

Module to inspect the alignment and contrast of Quantitative MR Images (MTsat etc).

"""

import argparse
import sys
import textwrap
import warnings

import matplotlib

matplotlib.interactive(True)

from abc import ABC
from visualqc.interfaces import BaseReviewInterface
from visualqc.workflows import BaseWorkflowVisualQC
from visualqc import config as cfg
from visualqc.utils import (check_finite_int, check_id_list_quantitative,
                            check_input_dir_quantitative_MR,
                            check_out_dir, check_views, check_time)

# each rating is a set of labels, join them with a plus delimiter
_plus_join = lambda label_set: '+'.join(label_set)


class QuantitativeMrRatingWorkflow(BaseWorkflowVisualQC, ABC):
    """
    Rating workflow without any overlay.
    """


    def __init__(self,
                 id_list,
                 in_dir,
                 image_names,
                 out_dir,
                 in_dir_type='generic',
                 prepare_first=True,
                 vis_type=cfg.alignment_default_vis_type,
                 delay_in_animation=cfg.delay_in_animation,
                 views=cfg.default_views,
                 num_slices_per_view=cfg.default_num_slices,
                 num_rows_per_view=cfg.default_num_rows,
                 ):
        """Constructor"""

        super().__init__(id_list, in_dir, out_dir,
                         None, None, None, None) # might add outlier detection later

        self.vis_type = vis_type
        self.delay_in_animation = delay_in_animation
        self.continue_animation = True

        self.image_names = image_names
        self.in_dir_type = in_dir_type

        self.expt_id = 'QuantitativeMR_{}'.format('_'.join(self.image_names))
        self.suffix = self.expt_id
        self.current_alert_msg = None
        self.prepare_first = prepare_first

        self.init_layout(views, num_rows_per_view, num_slices_per_view)
        self.init_getters()


def get_parser():
    """Parser to specify arguments and their defaults."""

    parser = argparse.ArgumentParser(prog="visualqc_quantitative",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description='visualqc_quantitative: rate '
                                                 'quality '
                                                 'of alignment between two images.')

    help_text_in_dir = textwrap.dedent("""
    Absolute path to an input folder containing the images. 
    Each subject will be queried after its ID in the metadata file, 
    and is expected to have all the files (specified by ``--image_names``), 
    in its own folder under --user_dir.

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

    help_text_image_names = textwrap.dedent("""
    Specifies the names of 3d images to be QCed.
    \n""")

    help_text_out_dir = textwrap.dedent("""
    Output folder to store the visualizations & ratings.
    Default: a new folder called ``{}`` will be created inside the ``fs_dir``
    \n""".format(cfg.default_out_dir_name))

    help_text_delay_in_animation = textwrap.dedent("""
    Specifies the delay in animation of the display of two images (like in a GIF).

    Default: {} (units in seconds).
    \n""".format(cfg.delay_in_animation))

    help_text_views = textwrap.dedent("""
    Specifies the set of views to display - could be just 1 view, or 2 or all 3.
    Example: --views 0 (typically sagittal) or --views 1 2 (axial and coronal)
    Default: {} {} {} (show all the views in the selected segmentation)
    \n""".format(cfg.default_views[0], cfg.default_views[1], cfg.default_views[2]))

    help_text_num_slices = textwrap.dedent("""
    Specifies the number of slices to display per each view. 
    This must be even to facilitate better division.
    Default: {}.
    \n""".format(cfg.default_num_slices))

    help_text_num_rows = textwrap.dedent("""
    Specifies the number of rows to display per each axis. 
    Default: {}.
    \n""".format(cfg.default_num_rows))

    help_text_prepare = textwrap.dedent("""
    This flag does the heavy preprocessing first, prior to starting any review and 
    rating operations.
    Heavy processing can include computation of registration quality metrics and 
    outlier detection etc. 
    This makes the switch from one subject to the next, even more seamless (saving 
    few seconds :) ).

    Default: False.
    \n""")

    in_out = parser.add_argument_group('Input and output', ' ')

    in_out.add_argument("-i", "--in_dir", action="store", dest="in_dir",
                        default=cfg.default_user_dir,
                        required=True, help=help_text_in_dir)

    in_out.add_argument("-n", "--image_names", action="store",
                        dest="image_names",  nargs='+',
                        required=True, help=help_text_image_names)

    in_out.add_argument("-l", "--id_list", action="store", dest="id_list",
                        default=None, required=False, help=help_text_id_list)

    in_out.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        default=None, required=False, help=help_text_out_dir)

    vis = parser.add_argument_group('Visualization',
                                    'Customize behaviour of comparisons')

    vis.add_argument("-dl", "--delay_in_animation", action="store",
                     dest="delay_in_animation",
                     default=cfg.delay_in_animation, required=False,
                     help=help_text_delay_in_animation)

    layout = parser.add_argument_group('Layout options', ' ')
    layout.add_argument("-w", "--views", action="store", dest="views",
                        default=cfg.default_views, required=False, nargs='+',
                        help=help_text_views)

    layout.add_argument("-s", "--num_slices", action="store", dest="num_slices",
                        default=cfg.default_num_slices, required=False,
                        help=help_text_num_slices)

    layout.add_argument("-r", "--num_rows", action="store", dest="num_rows",
                        default=cfg.default_num_rows, required=False,
                        help=help_text_num_rows)

    wf_args = parser.add_argument_group('Workflow', 'Options related to workflow '
                                                    'e.g. to pre-compute '
                                                    'resource-intensive features, '
                                                    'and pre-generate all the '
                                                    'visualizations required')
    wf_args.add_argument("-p", "--prepare_first", action="store_true",
                         dest="prepare_first",
                         help=help_text_prepare)

    return parser


def make_workflow_from_user_options():
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

    vis_type = cfg.alignment_default_vis_type
    type_of_features = 'alignment'
    in_dir, in_dir_type = check_input_dir_quantitative_MR(user_args.in_dir)

    image_names = user_args.image_names
    id_list, images_for_id = check_id_list_quantitative(user_args.id_list, in_dir,
                                           image_names, in_dir_type=in_dir_type)

    delay_in_animation = check_time(user_args.delay_in_animation, var_name='Delay')

    out_dir = check_out_dir(user_args.out_dir, in_dir)
    views = check_views(user_args.views)
    num_slices_per_view, num_rows_per_view = check_finite_int(user_args.num_slices,
                                                              user_args.num_rows)

    wf = QuantitativeMrRatingWorkflow(id_list,
                                      in_dir,
                                      image_names,
                                      out_dir=out_dir,
                                      in_dir_type=in_dir_type,
                                      prepare_first=user_args.prepare_first,
                                      vis_type=vis_type,
                                      delay_in_animation=delay_in_animation,
                                      views=views,
                                      num_slices_per_view=num_slices_per_view,
                                      num_rows_per_view=num_rows_per_view)

    return wf


def cli_run():
    """Main entry point."""

    wf = make_workflow_from_user_options()

    if wf.vis_type is not None:
        # matplotlib.interactive(True)
        wf.run()
    else:
        raise ValueError('Invalid state for visualQC!\n'
                         '\t Ensure proper combination of arguments is used.')

    return


if __name__ == '__main__':
    # disabling all not severe warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        cli_run()
