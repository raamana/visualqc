"""

Module to inspect the alignment and contrast of Quantitative MR Images (MTsat etc).

"""

import argparse
import asyncio
import sys
import textwrap
import time
import warnings

import matplotlib
import numpy as np

matplotlib.interactive(True)
from matplotlib.widgets import Button
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
from os.path import join as pjoin, realpath
from abc import ABC
from visualqc.interfaces import BaseReviewInterface
from visualqc.workflows import BaseWorkflowVisualQC
from visualqc import config as cfg
from visualqc.utils import (check_finite_int, check_id_list_quantitative,
                            check_input_dir_quantitative_MR,
                            check_out_dir, check_views, check_time)
from visualqc.utils import get_axis, pick_slices, read_image, scale_0to1
from mrivis.base import Collage

# each rating is a set of labels, join them with a plus delimiter
_plus_join = lambda label_set: '+'.join(label_set)


class QuantitativeMrInterface(BaseReviewInterface):
    """
    Custom interface for rating the quality of alignment between two 3d MR images
    """


    def __init__(self,
                 fig,
                 next_button_callback=None,
                 quit_button_callback=None,
                 change_vis_type_callback=None,
                 toggle_animation_callback=None,
                 alpha_seg=cfg.default_alpha_seg):
        """Constructor"""

        super().__init__(fig, None, next_button_callback, quit_button_callback)

        self.latest_alpha_seg = alpha_seg
        self.prev_axis = None
        self.prev_ax_pos = None
        self.zoomed_in = False

        self.next_button_callback = next_button_callback
        self.quit_button_callback = quit_button_callback
        self.user_contrast_callback = change_vis_type_callback
        self.toggle_animation_callback = toggle_animation_callback

        self.add_radio_buttons_alignment_rating()
        self.add_radio_buttons_contrast_rating()
        self.add_button_to_check_contrast()

        # this list of artists to be populated later
        # makes to handy to clean them all
        self.data_handles = list()

        self.unzoomable_axes = [self.radio_bt_rating.ax, self.radio_bt_vis_type.ax,
                                self.text_box.ax, self.bt_next.ax, self.bt_quit.ax, ]


    def add_radio_buttons_contrast_rating(self):

        ax_radio = plt.axes(cfg.position_alignment_radio_button,
                            facecolor=cfg.color_contrast_axis, aspect='equal')
        self.radio_bt_vis_type = RadioButtons(ax_radio,
                                              cfg.default_qmr_alignment_ratings,
                                              active=None, activecolor='orange')
        # no callback for now
        # self.radio_bt_vis_type.on_clicked(self.user_contrast_callback)

        for txt_lbl in self.radio_bt_vis_type.labels:
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for circ in self.radio_bt_vis_type.circles:
            circ.set(radius=0.06)


    def add_radio_buttons_alignment_rating(self):

        ax_radio = plt.axes(cfg.position_contrast_radio_button,
                            facecolor=cfg.color_alignment_axis, aspect='equal')
        self.radio_bt_rating = RadioButtons(ax_radio,
                                            cfg.default_qmr_contrast_ratings,
                                            active=None, activecolor='orange')
        self.radio_bt_rating.on_clicked(self.save_rating)
        for txt_lbl in self.radio_bt_rating.labels:
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for circ in self.radio_bt_rating.circles:
            circ.set(radius=0.06)


    def add_button_to_check_contrast(self):
        """Adds a button that invokes contrast checks"""

        ax_bt_contrast = self.fig.add_axes(cfg.position_contrast_button,
                              facecolor=cfg.color_contrast_button, aspect='equal')
        self.bt_contrast = Button(ax_bt_contrast, 'Check Contrast',
                                  hovercolor='blue')
        if self.user_contrast_callback is not None:
            self.bt_contrast.on_clicked(self.user_contrast_callback)
        else:
            raise RuntimeError('Callback for contrast check is not defined!')

    def save_rating(self, label):
        """Update the rating"""

        # print('  rating {}'.format(label))
        self.user_rating = label


    def get_ratings(self):
        """Returns the final set of checked ratings"""

        return self.user_rating


    def allowed_to_advance(self):
        """
        Method to ensure work is done for current iteration,
        before allowing the user to advance to next subject.

        Returns False if atleast one of the following conditions are not met:
            Atleast Checkbox is checked
        """

        return self.radio_bt_rating.value_selected is not None


    def reset_figure(self):
        "Resets the figure to prepare it for display of next subject."

        self.clear_data()
        self.clear_radio_buttons()
        self.clear_notes_annot()


    def clear_data(self):
        """clearing all data/image handles"""

        if self.data_handles:
            for artist in self.data_handles:
                artist.remove()
            # resetting it
            self.data_handles = list()


    def clear_radio_buttons(self):
        """Clears the radio button"""

        # enabling default rating encourages lazy advancing without review
        # self.radio_bt_rating.set_active(cfg.index_freesurfer_default_rating)
        for index, label in enumerate(self.radio_bt_rating.labels):
            if label.get_text() == self.radio_bt_rating.value_selected:
                self.radio_bt_rating.circles[index].set_facecolor(
                    cfg.color_rating_axis)
                break
        self.radio_bt_rating.value_selected = None


    def clear_notes_annot(self):
        """clearing notes and annotations"""

        self.text_box.set_val(cfg.textbox_initial_text)
        # text is matplotlib artist
        self.annot_text.remove()


    def on_mouse(self, event):
        """Callback for mouse events."""

        if self.prev_axis is not None:
            # include all the non-data axes here (so they wont be zoomed-in)
            if event.inaxes not in self.unzoomable_axes:
                self.prev_axis.set_position(self.prev_ax_pos)
                self.prev_axis.set_zorder(0)
                self.prev_axis.patch.set_alpha(0.5)
                self.zoomed_in = False

        # right click undefined
        if event.button in [3]:
            pass
        # double click to zoom in to any axis
        elif event.dblclick and event.inaxes is not None and \
            event.inaxes not in self.unzoomable_axes:
            # zoom axes full-screen
            self.prev_ax_pos = event.inaxes.get_position()
            event.inaxes.set_position(cfg.zoomed_position)
            event.inaxes.set_zorder(1)  # bring forth
            event.inaxes.set_facecolor('black')  # black
            event.inaxes.patch.set_alpha(1.0)  # opaque
            self.zoomed_in = True
            self.prev_axis = event.inaxes

        else:
            pass

        plt.draw()


    def on_keyboard(self, key_in):
        """Callback to handle keyboard shortcuts to rate and advance."""

        # ignore keyboard key_in when mouse within Notes textbox
        if key_in.inaxes == self.text_box.ax or key_in.key is None:
            return

        key_pressed = key_in.key.lower()
        # print(key_pressed)
        if key_pressed in ['right', ]:
            self.next_button_callback()
        elif key_pressed in ['ctrl+q', 'q+ctrl']:
            self.quit_button_callback()
        elif key_pressed in [' ', 'space']:
            self.toggle_animation_callback()
        # elif key_pressed in ['alt+1', '1+alt']:
        #     self.show_first_image_callback()
        # elif key_pressed in ['alt+2', '2+alt']:
        #     self.show_second_image_callback()
        else:
            if key_pressed in cfg.default_rating_list_shortform:
                self.user_rating = cfg.map_short_rating[key_pressed]
                index_to_set = cfg.default_rating_list.index(self.user_rating)
                self.radio_bt_rating.set_active(index_to_set)
            else:
                pass

        plt.draw()


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
                 delay_in_animation=cfg.delay_in_animation,
                 views=cfg.default_views,
                 num_slices_per_view=cfg.default_num_slices,
                 num_rows_per_view=cfg.default_num_rows,
                 ):
        """Constructor"""

        self.disable_outlier_detection = True
        super().__init__(id_list, in_dir, out_dir,
                         None, None, None,  # might add outlier detection later
                         disable_outlier_detection=self.disable_outlier_detection)

        self.current_cmap = 'gray'
        self.vis_type = 'Animate'
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


    def preprocess(self):
        """
        Preprocess the input data
            e.g. compute features, make complex visualizations etc.
            before starting the review process.
        """

        pass


    def prepare_UI(self):
        """Main method to run the entire workflow"""

        self.open_figure()
        self.add_UI()


    def init_layout(self, views, num_rows_per_view,
                    num_slices_per_view, padding=cfg.default_padding):

        self.views = views
        self.num_slices_per_view = num_slices_per_view
        self.num_rows_per_view = num_rows_per_view
        self.num_rows = len(self.views) * self.num_rows_per_view
        self.num_cols = int(
            (len(self.views) * self.num_slices_per_view) / self.num_rows)
        self.padding = padding


    def init_getters(self):
        """Initializes the getters methods for input paths and feature readers."""

        self.feature_extractor = None

        path_joiner = lambda sub_id, img_name: realpath(
            pjoin(self.in_dir, sub_id, img_name))
        self.path_getter_inputs = lambda sub_id: (path_joiner(sub_id, img) for img
                                                  in self.image_names)


    def open_figure(self):
        """Creates the master figure to show everything in."""

        self.figsize = cfg.default_review_figsize
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(self.num_rows, self.num_cols,
                                           figsize=self.figsize)
        self.axes = self.axes.flatten()

        self.fig.canvas.set_window_title('VisualQC Quantitative MR : {} '
                                         ''.format(self.in_dir))

        # vmin/vmax are controlled, because we rescale all to [0, 1]
        self.display_params = dict(interpolation='none', aspect='equal',
                                   origin='lower', cmap=self.current_cmap,
                                   vmin=0.0, vmax=1.0)

        # turning off axes, creating image objects
        self.h_images = [None] * len(self.axes)
        self.h_slice_numbers = [None] * len(self.axes)
        empty_image = np.full((100, 100, 3), 0.0)
        label_x, label_y = 5, 5  # in image space
        for ix, ax in enumerate(self.axes):
            ax.axis('off')
            self.h_images[ix] = ax.imshow(empty_image, **self.display_params)
            self.h_slice_numbers[ix] = ax.text(label_x, label_y, '',
                                               **cfg.slice_num_label_properties)

        self.fg_annot_h = self.fig.text(cfg.position_annotate_fg_quantMR[0],
                                        cfg.position_annotate_fg_quantMR[1],
                                        ' ', **cfg.annotate_fg_quantMR_properties)
        self.fg_annot_h.set_visible(False)

        # leaving some space on the right for review elements
        plt.subplots_adjust(**cfg.review_area)
        plt.show(block=False)

        # animation setup
        self.anim_loop = asyncio.get_event_loop()


    def add_UI(self):
        """Adds the review UI with defaults"""

        self.UI = QuantitativeMrInterface(self.fig,
                                          next_button_callback=self.next,
                                          quit_button_callback=self.quit,
                                          change_vis_type_callback=self.run_contrast_check,
                                          toggle_animation_callback=self.toggle_animation)

        # connecting callbacks
        self.con_id_click = self.fig.canvas.mpl_connect('button_press_event',
                                                        self.UI.on_mouse)
        self.con_id_keybd = self.fig.canvas.mpl_connect('key_press_event',
                                                        self.UI.on_keyboard)
        # con_id_scroll = self.fig.canvas.mpl_connect('scroll_event',
        # self.UI.on_scroll)

        self.fig.set_size_inches(self.figsize)


    def update_alerts(self):
        """Keeps a box, initially invisible."""

        if self.current_alert_msg is not None:
            h_alert_text = self.fig.text(cfg.position_outlier_alert[0],
                                         cfg.position_outlier_alert[1],
                                         self.current_alert_msg,
                                         **cfg.alert_text_props)
            # adding it to list of elements to cleared when advancing to next subject
            self.UI.data_handles.append(h_alert_text)


    def add_alerts(self):
        """Brings up an alert if subject id is detected to be an outlier."""

        pass


    def load_unit(self, unit_id):
        """Loads the image data for display."""

        skip_subject = False

        self.images = dict()
        for img_name in self.image_names:
            ipath = realpath(pjoin(self.in_dir, unit_id, img_name))
            img_data = read_image(ipath, error_msg=img_name)

            if np.count_nonzero(img_data) == 0:
                skip_subject = True
                print('image {} is empty!'.format(img_name, self.current_unit_id))

            self.images[img_name] = scale_0to1(img_data)

        if not skip_subject:
            # TODO implement crop to extents for more than 2 images
            # self.image_one, self.image_two = crop_to_seg_extents(self.image_one,
            #                                                      self.image_two,
            #                                                      self.padding)

            self.slices = pick_slices(self.images[self.image_names[0]], # first img
                                      self.views, self.num_slices_per_view)

        # # where to save the visualization to
        # out_vis_path = pjoin(self.out_dir, 'visual_qc_{}_{}'.format(
        # self.vis_type, unit_id))

        return skip_subject


    def display_unit(self):
        """Adds slice collage to the given axes"""

        if self.vis_type in ['GIF', 'Animate']:
            self.animate()
        else:
            raise RuntimeError('Invalid state for VisualQC - Quantitative MR!'
                               'Only Animation is implemented for alignment checks.')

        self.fg_annot_h.set_visible(False)


    def run_contrast_check(self, user_choice_vis_type):
        """Function to update vis type: alignment or contrast checks."""

        self.vis_type = user_choice_vis_type
        self.display_unit()


    def animate(self):
        """Displays the two images alternatively, until paused by external
        callbacks"""

        self.anim_loop.run_until_complete(self.alternate_images_with_delay_nTimes())


    @asyncio.coroutine
    def alternate_images_with_delay_nTimes(self):
        """Show image 1, wait, show image 2"""

        for _ in range(cfg.num_times_to_animate):
            for name, img in self.images.items():
                self.show_image(img, annot=name)
                plt.pause(0.05)
                time.sleep(self.delay_in_animation)


    def show_image(self, img, annot=None):
        """Display the requested slices of an image on the existing axes."""

        for ax_index, (dim_index, slice_index) in enumerate(self.slices):
            self.h_images[ax_index].set(data=get_axis(img, dim_index, slice_index),
                                        cmap=self.current_cmap)

        if annot is not None:
            self._identify_foreground(annot)
        else:
            self.fg_annot_h.set_visible(False)


    def show_specified_image(self, img_name):
        """Callback to show first image"""
        self.show_image(self.images[img_name], self.slices)
        self._identify_foreground(self.img_name)


    def _identify_foreground(self, text):
        """show the time point"""

        self.fg_annot_h.set_text(text)
        self.fg_annot_h.set_visible(True)


    def toggle_animation(self, input_event_to_ignore=None):
        """Callback to start or stop animation."""

        if self.anim_loop.is_running():
            self.anim_loop.stop()
        elif self.vis_type in ['GIF', 'Animate']:
            # run only when the vis_type selected in animatable.
            self.animate()


    def cleanup(self):
        """Preparating for exit."""

        # save ratings before exiting
        self.save_ratings()

        self.fig.canvas.mpl_disconnect(self.con_id_click)
        self.fig.canvas.mpl_disconnect(self.con_id_keybd)
        plt.close('all')

        self.anim_loop.run_until_complete(self.anim_loop.shutdown_asyncgens())
        self.anim_loop.close()


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
                        dest="image_names", nargs='+',
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
                                                        image_names,
                                                        in_dir_type=in_dir_type)

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
