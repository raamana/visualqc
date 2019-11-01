"""

Module to rate defaced MRI scans, optionally with their 3D renders

"""

import argparse
import sys
import textwrap
import warnings
from abc import ABC

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons
from mrivis.utils import crop_image
from mrivis.base import Collage
from os.path import join as pjoin, realpath

from visualqc import config as cfg
from visualqc.interfaces import BaseReviewInterface
from visualqc.utils import  check_inputs_defacing, check_out_dir, check_finite_int, \
    check_outlier_params, check_views, get_axis, pick_slices, \
    read_image, scale_0to1, saturate_brighter_intensities
from visualqc.workflows import BaseWorkflowVisualQC
from visualqc.image_utils import mask_image


class DefacingInterface(BaseReviewInterface):
    """Custom interface to rate the quality of defacing in an MRI scan"""



class RatingWorkflowDefacing(BaseWorkflowVisualQC, ABC):
    """Rating worklfow for defaced MRI scans"""

    def __init__(self,
                 id_list,
                 images_for_id,
                 in_dir,
                 out_dir,
                 defaced_name,
                 mri_name,
                 render_name,
                 issue_list=cfg.defacing_default_issue_list,
                 vis_type='defacing'):
        """Constructor"""

        super().__init__(id_list, in_dir, out_dir,
                         outlier_method=None, outlier_fraction=None,
                         outlier_feat_types=None, disable_outlier_detection=None)

        self.vis_type = vis_type
        self.issue_list = issue_list
        self.defaced_name = defaced_name
        self.mri_name = mri_name
        self.render_name = render_name
        self.images_for_id = images_for_id

        self.expt_id = 'rate_defaced_mri_{}'.format(self.defaced_name)
        self.suffix = self.expt_id
        self.current_alert_msg = None

        self.init_layout()


    def init_layout(self,
                    views=(0, 1, 2),
                    num_rows_per_view=1,
                    num_slices_per_view=5,
                    padding=cfg.default_padding):
        """initializes the layout"""

        plt.style.use('dark_background')

        # vmin/vmax are controlled, because we rescale all to [0, 1]
        self.display_params = dict(interpolation='none', aspect='equal',
                                   origin='lower', cmap='gray', vmin=0.0, vmax=1.0)
        self.figsize = cfg.default_review_figsize

        self.collage = Collage(view_set=views,
                               num_slices=num_slices_per_view, num_rows=num_rows_per_view,
                               display_params=self.display_params,
                               bounding_rect=cfg.bounding_box_review,
                               figsize=self.figsize)
        self.fig = self.collage.fig
        self.fig.canvas.set_window_title('VisualQC defacing : {} {} '
                                         ''.format(self.in_dir, self.defaced_name))

        self.padding = padding


    def prepare_UI(self):
        """Main method to run the entire workflow"""

        self.open_figure()
        self.add_UI()


    def open_figure(self):
        """Creates the master figure to show everything in."""

        plt.show(block=False)

    def add_UI(self):
        """Adds the review UI with defaults"""

        # two keys for same combinations exist to account for time delays in key presses
        map_key_to_callback = {'alt+s': self.show_saturated,
                               's+alt': self.show_saturated,
                               'alt+b': self.show_background_only,
                               'b+alt': self.show_background_only,
                               'alt+t': self.show_tails_trimmed,
                               't+alt': self.show_tails_trimmed,
                               'alt+o': self.show_original,
                               'o+alt': self.show_original}
        self.UI = DefacingInterface(self.collage.fig, self.collage.flat_grid, self.issue_list,
                                 next_button_callback=self.next,
                                 quit_button_callback=self.quit,
                                 processing_choice_callback=self.process_and_display,
                                 map_key_to_callback=map_key_to_callback)

        # connecting callbacks
        self.con_id_click = self.fig.canvas.mpl_connect('button_press_event',
                                                        self.UI.on_mouse)
        self.con_id_keybd = self.fig.canvas.mpl_connect('key_press_event',
                                                        self.UI.on_keyboard)
        # con_id_scroll = self.fig.canvas.mpl_connect('scroll_event', self.UI.on_scroll)

        self.fig.set_size_inches(self.figsize)



def get_parser():
    """Parser to specify arguments and their defaults."""

    parser = argparse.ArgumentParser(prog="visualqc_defacing",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description='visualqc_defacing: rate quality '
                                                 'of defaced MR scan.')

    help_text_user_dir = textwrap.dedent("""
    Absolute path to an input folder containing defaced MRI scans. 
    Each subject will be queried after its ID , 
    and is expected to have the MRI (specified ``--mri_name``), 
    in its own folder under --user_dir.

    E.g. ``--user_dir /project/images_to_QC``
    \n""")

    help_text_id_list = textwrap.dedent("""
    Absolute path to file containing list of subject IDs to be processed.
    If not provided, all the subjects with required files will be processed.

    E.g.

    .. parsed-literal::

        sub001
        sub002
        cn_003
        cn_004

    \n""")

    help_text_defaced_mri_name = textwrap.dedent("""
    Specifies the name of defaced MRI image to be rated.

    Default: {}
    \n""".format(cfg.default_defaced_mri_name))

    help_text_mri_name = textwrap.dedent("""
    Specifies the name of MRI image that is NOT defaced, to check the accuracy of 
    the defacing algorithm. 

    Default: {}
    \n""".format(cfg.default_mri_name))

    help_text_render_name = textwrap.dedent("""
    Specifies the name of 3D render of the MRI scan.

    Default: {}
    \n""".format(cfg.default_render_name))

    help_text_out_dir = textwrap.dedent("""
    Output folder to store the visualizations & ratings.
    Default: a new folder called ``{}`` will be created inside the ``fs_dir``
    \n""".format(cfg.default_out_dir_name))

    in_out = parser.add_argument_group('Input and output', ' ')

    in_out.add_argument("-u", "--user_dir", action="store", dest="user_dir",
                        default=cfg.default_user_dir,
                        required=False, help=help_text_user_dir)

    in_out.add_argument("-d", "--defaced_name", action="store", dest="defaced_name",
                        default=cfg.default_defaced_mri_name, required=False,
                        help=help_text_defaced_mri_name)

    in_out.add_argument("-m", "--mri_name", action="store", dest="render_name",
                        default=cfg.default_mri_name, required=False,
                        help=help_text_mri_name)

    in_out.add_argument("-r", "--render_name", action="store", dest="mri_name",
                        default=cfg.default_render_name, required=False,
                        help=help_text_render_name)

    in_out.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        required=False, help=help_text_out_dir,
                        default=None)

    in_out.add_argument("-i", "--id_list", action="store", dest="id_list",
                        default=None, required=False, help=help_text_id_list)

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

    vis_type = 'defacing'

    user_dir, id_list, images_for_id, defaced_name, mri_name, render_name \
        = check_inputs_defacing(user_args.user_dir, user_args.defaced_name,
                                user_args.mri_name, user_args.render_name,
                                user_args.id_list)

    out_dir = check_out_dir(user_args.out_dir, user_dir)

    wf = RatingWorkflowDefacing(id_list, images_for_id, user_dir, out_dir,
                                defaced_name, mri_name, render_name,
                                cfg.defacing_default_issue_list, vis_type)

    return wf


def cli_run():
    """Main entry point."""

    wf = make_workflow_from_user_options()

    if wf.vis_type is not None:
        # matplotlib.interactive(True)
        wf.run()
    else:
        raise ValueError('Invalid state for defacing visualQC!\n'
                         '\t Ensure proper combination of arguments is used.')

    return


if __name__ == '__main__':
    # disabling all not severe warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        cli_run()
