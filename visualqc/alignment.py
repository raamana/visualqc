"""

Module to inspect the accuracy of spatial alignment (registration).

"""

import argparse
import asyncio
import sys
import textwrap
import time
import warnings
from abc import ABC
import numpy as np
from functools import partial
from scipy.ndimage import grey_erosion, sobel, \
    grey_opening, binary_opening
from scipy.ndimage.filters import median_filter, minimum_filter, maximum_filter
from scipy.signal import medfilt2d, medfilt

import matplotlib
matplotlib.interactive(True)

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.widgets import RadioButtons
from mrivis.utils import crop_to_seg_extents
from os.path import join as pjoin, realpath
from visualqc import config as cfg
from visualqc.interfaces import BaseReviewInterface
from visualqc.utils import check_finite_int, check_id_list, check_input_dir_alignment, \
    check_out_dir, check_outlier_params, check_views, get_axis, pick_slices, read_image, \
    scale_0to1, check_time
from visualqc.workflows import BaseWorkflowVisualQC
from visualqc.image_utils import overlay_edges, mix_color, diff_image, mix_slices_in_checkers

# each rating is a set of labels, join them with a plus delimiter
_plus_join = lambda label_set: '+'.join(label_set)

def mask_below_perc(img):
    """returns a mask of pixels below a percentile value"""
    # return img <= np.max(img)/10
    return img <= np.percentile(img, cfg.weak_edge_threshold)


class AlignmentInterface(BaseReviewInterface):
    """Custom interface for rating the quality of alignment between two 3d MR images"""


    def __init__(self,
                 fig,
                 rating_list=cfg.default_rating_list,
                 next_button_callback=None,
                 quit_button_callback=None,
                 change_vis_type_callback=None,
                 toggle_animation_callback=None,
                 show_first_image_callback=None,
                 show_second_image_callback=None,
                 alpha_seg=cfg.default_alpha_seg):
        """Constructor"""

        super().__init__(fig, None, next_button_callback, quit_button_callback)

        self.rating_list = rating_list

        self.latest_alpha_seg = alpha_seg
        self.prev_axis = None
        self.prev_ax_pos = None
        self.zoomed_in = False

        self.next_button_callback = next_button_callback
        self.quit_button_callback = quit_button_callback
        self.change_vis_type_callback = change_vis_type_callback
        self.toggle_animation_callback = toggle_animation_callback
        self.show_first_image_callback = show_first_image_callback
        self.show_second_image_callback = show_second_image_callback

        self.add_radio_buttons_rating()
        self.add_radio_buttons_comparison_method()

        # this list of artists to be populated later
        # makes to handy to clean them all
        self.data_handles = list()

        self.unzoomable_axes = [self.radio_bt_rating.ax, self.radio_bt_vis_type.ax,
                                self.text_box.ax, self.bt_next.ax, self.bt_quit.ax, ]


    def add_radio_buttons_comparison_method(self):

        ax_radio = plt.axes(cfg.position_alignment_radio_button_method,
                            facecolor=cfg.color_rating_axis, aspect='equal')
        self.radio_bt_vis_type = RadioButtons(ax_radio, cfg.choices_alignment_comparison,
                                              active=None, activecolor='orange')
        self.radio_bt_vis_type.on_clicked(self.change_vis_type_callback)
        for txt_lbl in self.radio_bt_vis_type.labels:
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for circ in self.radio_bt_vis_type.circles:
            circ.set(radius=0.06)


    def add_radio_buttons_rating(self):

        ax_radio = plt.axes(cfg.position_alignment_radio_button_rating,
                            facecolor=cfg.color_rating_axis, aspect='equal')
        self.radio_bt_rating = RadioButtons(ax_radio, self.rating_list,
                                            active=None, activecolor='orange')
        self.radio_bt_rating.on_clicked(self.save_rating)
        for txt_lbl in self.radio_bt_rating.labels:
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for circ in self.radio_bt_rating.circles:
            circ.set(radius=0.06)

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
                self.radio_bt_rating.circles[index].set_facecolor(cfg.color_rating_axis)
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
        elif key_pressed in ['alt+1', '1+alt']:
            self.show_first_image_callback()
        elif key_pressed in ['alt+2', '2+alt']:
            self.show_second_image_callback()
        else:
            if key_pressed in cfg.default_rating_list_shortform:
                self.user_rating = cfg.map_short_rating[key_pressed]
                index_to_set = cfg.default_rating_list.index(self.user_rating)
                self.radio_bt_rating.set_active(index_to_set)
            else:
                pass

        plt.draw()


class AlignmentRatingWorkflow(BaseWorkflowVisualQC, ABC):
    """
    Rating workflow without any overlay.
    """


    def __init__(self,
                 id_list,
                 in_dir,
                 image1_name,
                 image2_name,
                 out_dir,
                 issue_list=cfg.default_rating_list,
                 in_dir_type='generic',
                 prepare_first=True,
                 vis_type=cfg.alignment_default_vis_type,
                 delay_in_animation=cfg.delay_in_animation,
                 outlier_method=cfg.default_outlier_detection_method,
                 outlier_fraction=cfg.default_outlier_fraction,
                 outlier_feat_types=cfg.freesurfer_features_outlier_detection,
                 disable_outlier_detection=False,
                 views=cfg.default_views,
                 num_slices_per_view=cfg.default_num_slices,
                 num_rows_per_view=cfg.default_num_rows,
                 ):
        """Constructor"""

        super().__init__(id_list, in_dir, out_dir,
                         outlier_method, outlier_fraction,
                         outlier_feat_types, disable_outlier_detection)

        self.vis_type = vis_type
        self.current_cmap = cfg.alignment_cmap[self.vis_type]
        self.checker_size = cfg.default_checkerboard_size
        self.color_mix_alphas = cfg.default_color_mix_alphas
        self.delay_in_animation = delay_in_animation
        self.continue_animation = True
        self.set_mixer_method()

        self.issue_list = issue_list
        self.image1_name = image1_name
        self.image2_name = image2_name
        self.in_dir_type = in_dir_type

        self.expt_id = 'rate_alignment_{}_{}'.format(self.image1_name, self.image2_name)
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

        if not self.disable_outlier_detection:
            print('Preprocessing data - please wait .. '
                  '\n\t(or contemplate the vastness of universe! )')
            self.extract_features()
        self.detect_outliers()

        # no complex vis to generate - skipping


    def prepare_UI(self):
        """Main method to run the entire workflow"""

        self.open_figure()
        self.add_UI()
        self.add_histogram_panel()


    def init_layout(self, views, num_rows_per_view,
                    num_slices_per_view, padding=cfg.default_padding):

        self.views = views
        self.num_slices_per_view = num_slices_per_view
        self.num_rows_per_view = num_rows_per_view
        self.num_rows = len(self.views) * self.num_rows_per_view
        self.num_cols = int((len(self.views) * self.num_slices_per_view) / self.num_rows)
        self.padding = padding


    def init_getters(self):
        """Initializes the getters methods for input paths and feature readers."""

        self.feature_extractor = None

        path_joiner = lambda sub_id, img_name: realpath(
            pjoin(self.in_dir, sub_id, img_name))
        self.path_getter_inputs = lambda sub_id: (path_joiner(sub_id, self.image1_name),
                                                  path_joiner(sub_id, self.image2_name))


    def open_figure(self):
        """Creates the master figure to show everything in."""

        self.figsize = cfg.default_review_figsize
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(self.num_rows, self.num_cols,
                                           figsize=self.figsize)
        self.axes = self.axes.flatten()

        self.fig.canvas.set_window_title('VisualQC Alignment : {} {} {} '
                                         ''.format(self.in_dir, self.image1_name, self.image2_name))

        # vmin/vmax are controlled, because we rescale all to [0, 1]
        self.display_params = dict(interpolation='none', aspect='equal', origin='lower',
                                   cmap=self.current_cmap, vmin=0.0, vmax=1.0)

        # turning off axes, creating image objects
        self.h_images = [None] * len(self.axes)
        self.h_slice_numbers = [None] * len(self.axes)
        empty_image = np.full((100, 100, 3), 0.0)
        label_x, label_y = 5, 5 # in image space
        for ix, ax in enumerate(self.axes):
            ax.axis('off')
            self.h_images[ix] = ax.imshow(empty_image, **self.display_params)
            self.h_slice_numbers[ix]= ax.text(label_x, label_y, '',
                                              **cfg.slice_num_label_properties)

        self.fg_annot_h = self.fig.text(cfg.position_annotate_foreground[0],
                                        cfg.position_annotate_foreground[1],
                                        ' ', **cfg.annotate_foreground_properties)
        self.fg_annot_h.set_visible(False)

        # leaving some space on the right for review elements
        plt.subplots_adjust(**cfg.review_area)
        plt.show(block=False)

        # animation setup
        self.anim_loop = asyncio.get_event_loop()


    def add_UI(self):
        """Adds the review UI with defaults"""

        self.UI = AlignmentInterface(self.fig, self.issue_list,
                                     next_button_callback=self.next,
                                     quit_button_callback=self.quit,
                                     change_vis_type_callback=self.callback_display_update,
                                     toggle_animation_callback=self.toggle_animation,
                                     show_first_image_callback=self.show_first_image,
                                     show_second_image_callback=self.show_second_image)

        # connecting callbacks
        self.con_id_click = self.fig.canvas.mpl_connect('button_press_event',
                                                        self.UI.on_mouse)
        self.con_id_keybd = self.fig.canvas.mpl_connect('key_press_event',
                                                        self.UI.on_keyboard)
        # con_id_scroll = self.fig.canvas.mpl_connect('scroll_event', self.UI.on_scroll)

        self.fig.set_size_inches(self.figsize)


    def add_histogram_panel(self):
        """Extra axis for histogram"""

        self.ax_hist = plt.axes(cfg.position_histogram_alignment)
        self.ax_hist.set_xticks(cfg.xticks_histogram_alignment)
        self.ax_hist.set_yticks([])
        self.ax_hist.set_autoscaley_on(True)
        self.ax_hist.set_prop_cycle('color', cfg.color_histogram_alignment)
        self.ax_hist.set_title(cfg.title_histogram_alignment, fontsize='small')


    def update_histogram(self):
        """Updates histogram with current image data"""

        if not self._histogram_updated:
            diff_img = self.image_one - self.image_two
            nonzero_values = diff_img.ravel()[np.flatnonzero(diff_img)]
            _, _, patches_hist = self.ax_hist.hist(nonzero_values, density=True,
                                                   bins=cfg.num_bins_histogram_display)
            self.ax_hist.relim(visible_only=True)
            self.ax_hist.autoscale_view(scalex=False)  # xlim fixed to [0, 1]
            self.UI.data_handles.extend(patches_hist)
            self._histogram_updated = True


    def update_alerts(self):
        """Keeps a box, initially invisible."""

        if self.current_alert_msg is not None:
            h_alert_text = self.fig.text(cfg.position_outlier_alert[0],
                                         cfg.position_outlier_alert[1],
                                         self.current_alert_msg, **cfg.alert_text_props)
            # adding it to list of elements to cleared when advancing to next subject
            self.UI.data_handles.append(h_alert_text)


    def add_alerts(self):
        """Brings up an alert if subject id is detected to be an outlier."""

        flagged_as_outlier = self.current_unit_id in self.by_sample
        if flagged_as_outlier:
            alerts_list = self.by_sample.get(self.current_unit_id,
                                             None)  # None, if id not in dict
            print('\n\tFlagged as a possible outlier by these measures:\n\t\t{}'.format(
                '\t'.join(alerts_list)))

            strings_to_show = ['Flagged as an outlier:', ] + alerts_list
            self.current_alert_msg = '\n'.join(strings_to_show)
            self.update_alerts()
        else:
            self.current_alert_msg = None


    def load_unit(self, unit_id):
        """Loads the image data for display."""

        image1_path, image2_path = self.path_getter_inputs(unit_id)
        self.image_one = read_image(image1_path, error_msg='first image')
        self.image_two = read_image(image2_path, error_msg='second image')

        skip_subject = False
        if np.count_nonzero(self.image_one) == 0:
            skip_subject = True
            print(
                'image {} of {} is empty!'.format(self.image1_name, self.current_unit_id))

        if np.count_nonzero(self.image_two) == 0:
            skip_subject = True
            print(
                'image {} of {} is empty!'.format(self.image2_name, self.current_unit_id))

        if not skip_subject:
            # crop and rescale
            self.image_one, self.image_two = crop_to_seg_extents(self.image_one,
                                                                 self.image_two,
                                                                 self.padding)
            self.image_one = scale_0to1(self.image_one)
            self.image_two = scale_0to1(self.image_two)

            self.slices = pick_slices(self.image_one, self.views,
                                      self.num_slices_per_view)
            # flag to keep track of whether data has been changed.
            self._histogram_updated = False

        # # where to save the visualization to
        # out_vis_path = pjoin(self.out_dir, 'visual_qc_{}_{}'.format(self.vis_type, unit_id))

        return skip_subject


    def display_unit(self):
        """Adds slice collage to the given axes"""

        if self.vis_type in ['GIF', 'Animate']:
            self.animate()
        else:
            # the following isnt immediately effective
            self.anim_loop.stop()
            self.mix_and_display()

        self.fg_annot_h.set_visible(False)

        # updating histogram, if needed
        self.update_histogram()


    def callback_display_update(self, user_choice_vis_type):
        """Function to update vis type and display."""

        self.vis_type = user_choice_vis_type
        self.set_mixer_method()
        self.display_unit()

    def animate(self):
        """Displays the two images alternatively, until paused by external callbacks"""

        self.anim_loop.run_until_complete(self.alternate_images_with_delay_nTimes())

    @asyncio.coroutine
    def alternate_images_with_delay_nTimes(self):
        """Show image 1, wait, show image 2"""

        for _ in range(cfg.num_times_to_animate):
            for img in (self.image_one, self.image_two):
                self.show_image(img)
                plt.pause(0.05)
                time.sleep(self.delay_in_animation)

    def mix_and_display(self):
        """Static mix and display."""

        # TODO maintain a dict mixed[vis_type] to do computation only once
        for ax_index, (dim_index, slice_index) in enumerate(self.slices):
            slice_one = get_axis(self.image_one, dim_index, slice_index)
            slice_two = get_axis(self.image_two, dim_index, slice_index)
            mixed_slice = self.mixer(slice_one, slice_two)
            # mixed_slice is already in RGB mode m x p x 3, so
            #   prev. cmap (gray) has no effect on color_mixed data
            self.h_images[ax_index].set(data=mixed_slice, cmap=self.current_cmap)
            self.h_slice_numbers[ax_index].set_text(str(slice_index))


    def show_image(self, img, annot=None):
        """Display the requested slices of an image on the existing axes."""

        for ax_index, (dim_index, slice_index) in enumerate(self.slices):
            self.h_images[ax_index].set(data=get_axis(img, dim_index, slice_index),
                                        cmap=self.current_cmap)

        if annot is not None:
            self._identify_foreground(annot)
        else:
            self.fg_annot_h.set_visible(False)

    def show_first_image(self):
        """Callback to show first image"""
        self.show_image(self.image_one, self.slices)
        self._identify_foreground(self.image1_name)


    def show_second_image(self):
        """Callback to show second image"""
        self.show_image(self.image_two, self.slices)
        self._identify_foreground(self.image2_name)


    def _identify_foreground(self, text):
        """show the time point"""

        self.fg_annot_h.set_text(text)
        self.fg_annot_h.set_visible(True)


    def set_mixer_method(self):
        """Mixer to produce the image to be displayed."""

        if self.vis_type in ['Color_mix', 'color_mix', 'rgb']:
            self.mixer = partial(mix_color, alpha_channels=self.color_mix_alphas)
        elif self.vis_type in ['Checkerboard', 'checkerboard', 'checker', 'cb']:
            self.mixer = partial(mix_slices_in_checkers, checker_size=self.checker_size)
        elif self.vis_type in ['Voxelwise_diff', 'voxelwise_diff', 'vdiff']:
            self.mixer = diff_image
        elif self.vis_type in ['Edges_Sharp', 'Edges_Thinner']:
            self.mixer = partial(overlay_edges, sharper=True)
        elif self.vis_type in ['Edges_Diffused', ]:
            self.mixer = partial(overlay_edges, sharper=False)
        elif self.vis_type in ['GIF', 'Animate']:
            self.mixer = None # this is handled by self.display_unit()
        else:
            raise ValueError('Invalid mixer name chosen.')

        # update colormap
        self.current_cmap = cfg.alignment_cmap[self.vis_type]


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

    parser = argparse.ArgumentParser(prog="visualqc_alignment",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description='visualqc_alignment: rate quality of alignment between two images.')

    help_text_in_dir = textwrap.dedent("""
    Absolute path to an input folder containing the MRI scan. 
    Each subject will be queried after its ID in the metadata file, 
    and is expected to have the MRI (specified ``--mri_name``), 
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

    help_text_image1 = textwrap.dedent("""
    Specifies the name of the first 3d image to serve as the reference image.
    \n""")

    help_text_image2 = textwrap.dedent("""
    Specifies the name of second 3d image to serve as the comparison image
    Order of the two images does not typically matter.
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
    This flag does the heavy preprocessing first, prior to starting any review and rating operations.
    Heavy processing can include computation of registration quality metrics and outlier detection etc. 
    This makes the switch from one subject to the next, even more seamless (saving few seconds :) ).

    Default: False.
    \n""")

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
    Type of features to be employed in training the outlier detection method.  
    It could be one of .

    Default: {}.
    \n""".format(cfg.alignment_features_OLD))

    help_text_disable_outlier_detection = textwrap.dedent("""
    This flag disables outlier detection and alerts altogether.
    \n""")

    in_out = parser.add_argument_group('Input and output', ' ')

    in_out.add_argument("-d", "--in_dir", action="store", dest="in_dir",
                        default=cfg.default_user_dir,
                        required=True, help=help_text_in_dir)

    in_out.add_argument("-i1", "--image1", action="store", dest="image1",
                        required=True, help=help_text_image1)

    in_out.add_argument("-i2", "--image2", action="store", dest="image2",
                        required=True, help=help_text_image2)

    in_out.add_argument("-l", "--id_list", action="store", dest="id_list",
                        default=None, required=False, help=help_text_id_list)

    in_out.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        default=None, required=False, help=help_text_out_dir)

    vis = parser.add_argument_group('Visualization', 'Customize behaviour of comparisons')

    vis.add_argument("-dl", "--delay_in_animation", action="store",
                     dest="delay_in_animation",
                     default=cfg.delay_in_animation, required=False,
                     help=help_text_delay_in_animation)

    outliers = parser.add_argument_group('Outlier detection',
                                         'options related to automatically detecting possible outliers')
    outliers.add_argument("-olm", "--outlier_method", action="store",
                          dest="outlier_method",
                          default=cfg.default_outlier_detection_method, required=False,
                          help=help_text_outlier_detection_method)

    outliers.add_argument("-olf", "--outlier_fraction", action="store",
                          dest="outlier_fraction",
                          default=cfg.default_outlier_fraction, required=False,
                          help=help_text_outlier_fraction)

    outliers.add_argument("-olt", "--outlier_feat_types", action="store",
                          dest="outlier_feat_types",
                          default=cfg.t1_mri_features_OLD, required=False,
                          help=help_text_outlier_feat_types)

    outliers.add_argument("-old", "--disable_outlier_detection", action="store_true",
                          dest="disable_outlier_detection",
                          required=False, help=help_text_disable_outlier_detection)

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
                                                    'e.g. to pre-compute resource-intensive features, '
                                                    'and pre-generate all the visualizations required')
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
    in_dir, in_dir_type = check_input_dir_alignment(user_args.in_dir)

    image1 = user_args.image1
    image2 = user_args.image2
    id_list, images_for_id = check_id_list(user_args.id_list, in_dir, vis_type,
                                           image1, image2, in_dir_type=in_dir_type)

    delay_in_animation = check_time(user_args.delay_in_animation, var_name='Delay')

    out_dir = check_out_dir(user_args.out_dir, in_dir)
    views = check_views(user_args.views)
    num_slices_per_view, num_rows_per_view = check_finite_int(user_args.num_slices,
                                                              user_args.num_rows)

    outlier_method, outlier_fraction, \
    outlier_feat_types, disable_outlier_detection = check_outlier_params(
        user_args.outlier_method,
        user_args.outlier_fraction,
        user_args.outlier_feat_types,
        user_args.disable_outlier_detection,
        id_list, vis_type, type_of_features)

    wf = AlignmentRatingWorkflow(id_list,
                                 in_dir,
                                 image1,
                                 image2,
                                 out_dir=out_dir,
                                 in_dir_type=in_dir_type,
                                 prepare_first=user_args.prepare_first,
                                 vis_type=vis_type,
                                 delay_in_animation=delay_in_animation,
                                 outlier_method=outlier_method,
                                 outlier_fraction=outlier_fraction,
                                 outlier_feat_types=outlier_feat_types,
                                 disable_outlier_detection=disable_outlier_detection,
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
