"""

Module to present a base neuroimaging scan, currently T1 mri, without any overlay.

"""

import argparse
import sys
import textwrap
import warnings
from abc import ABC
from os.path import join as pjoin, realpath

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons
from mrivis.base import Collage
from mrivis.utils import crop_image

from visualqc import config as cfg
from visualqc.image_utils import mask_image
from visualqc.interfaces import BaseReviewInterface
from visualqc.utils import (check_finite_int, check_id_list, check_input_dir_T1,
                            check_out_dir, check_outlier_params, check_views,
                            read_image, saturate_brighter_intensities, scale_0to1,
                            check_bids_dir)
from visualqc.readers import find_anatomical_images_in_BIDS
from visualqc.workflows import BaseWorkflowVisualQC

# each rating is a set of labels, join them with a plus delimiter
_plus_join = lambda label_set: '+'.join(label_set)


class T1MriInterface(BaseReviewInterface):
    """Custom interface for rating the quality of T1 MRI scan."""


    def __init__(self,
                 fig,
                 axes,
                 issue_list=cfg.t1_mri_default_issue_list,
                 next_button_callback=None,
                 quit_button_callback=None,
                 processing_choice_callback=None,
                 map_key_to_callback=None):
        """Constructor"""

        super().__init__(fig, axes, next_button_callback, quit_button_callback)

        self.issue_list = issue_list

        self.prev_axis = None
        self.prev_ax_pos = None
        self.zoomed_in = False
        self.next_button_callback = next_button_callback
        self.quit_button_callback = quit_button_callback
        self.processing_choice_callback = processing_choice_callback
        if map_key_to_callback is None:
            self.map_key_to_callback = {}  # empty
        elif isinstance(map_key_to_callback, dict):
            self.map_key_to_callback = map_key_to_callback
        else:
            raise ValueError('map_key_to_callback must be a dict')

        self.add_checkboxes()
        self.add_process_options()
        # include all the non-data axes here (so they wont be zoomed-in)
        self.unzoomable_axes = [self.checkbox.ax, self.text_box.ax,
                                self.bt_next.ax, self.bt_quit.ax]
        # radio buttons may not exist in all interfaces
        if hasattr(self, 'radio_bt_vis_type'):
            self.unzoomable_axes.append(self.radio_bt_vis_type)

        # this list of artists to be populated later
        # makes to handy to clean them all
        self.data_handles = list()


    def add_checkboxes(self):
        """
        Checkboxes offer the ability to select multiple tags such as Motion,
        Ghosting, Aliasing etc, instead of one from a list of mutual exclusive
        rating options (such as Good, Bad, Error etc).
        """

        ax_checkbox = plt.axes(cfg.position_checkbox_t1_mri,
                               facecolor=cfg.color_rating_axis)
        # initially de-activating all
        actives = [False] * len(self.issue_list)
        self.checkbox = CheckButtons(ax_checkbox, labels=self.issue_list,
                                     actives=actives)
        self.checkbox.on_clicked(self.save_issues)
        for txt_lbl in self.checkbox.labels:
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for rect in self.checkbox.rectangles:
            rect.set_width(cfg.checkbox_rect_width)
            rect.set_height(cfg.checkbox_rect_height)

        # lines is a list of n crosses, each cross (x) defined by a tuple of lines
        for x_line1, x_line2 in self.checkbox.lines:
            x_line1.set_color(cfg.checkbox_cross_color)
            x_line2.set_color(cfg.checkbox_cross_color)

        self._index_pass = cfg.t1_mri_default_issue_list.index(
            cfg.t1_mri_pass_indicator)


    def add_process_options(self):
        """
        options to process the anatomical image in different ways to unearth issues!
        """

        ax_radio = plt.axes(cfg.position_radio_bt_t1_mri,
                            facecolor=cfg.color_rating_axis)
        self.radio_bt_vis_type = RadioButtons(ax_radio,
                                              cfg.processing_choices_t1_mri,
                                              active=None, activecolor='orange')
        self.radio_bt_vis_type.on_clicked(self.processing_choice_callback)
        for txt_lbl in self.radio_bt_vis_type.labels:
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for circ in self.radio_bt_vis_type.circles:
            circ.set(radius=0.06)


    def save_issues(self, label):
        """
        Update the rating

        This function is called whenever set_active() happens on any label,
        if checkbox.eventson is True.

        """

        if label == cfg.visual_qc_pass_indicator:
            self.clear_checkboxes(except_pass=True)
        else:
            self.clear_pass_only_if_on()

        self.fig.canvas.draw_idle()


    def clear_checkboxes(self, except_pass=False):
        """Clears all checkboxes.

        if except_pass=True,
            does not clear checkbox corresponding to cfg.t1_mri_pass_indicator
        """

        cbox_statuses = self.checkbox.get_status()
        for index, this_cbox_active in enumerate(cbox_statuses):
            if except_pass and index == self._index_pass:
                continue
            # if it was selected already, toggle it.
            if this_cbox_active:
                # not calling checkbox.set_active() as it calls the callback
                # self.save_issues() each time, if eventson is True
                self._toggle_visibility_checkbox(index)


    def clear_pass_only_if_on(self):
        """Clear pass checkbox only"""

        cbox_statuses = self.checkbox.get_status()
        if cbox_statuses[self._index_pass]:
            self._toggle_visibility_checkbox(self._index_pass)


    def _toggle_visibility_checkbox(self, index):
        """toggles the visibility of a given checkbox"""

        l1, l2 = self.checkbox.lines[index]
        l1.set_visible(not l1.get_visible())
        l2.set_visible(not l2.get_visible())


    def get_ratings(self):
        """Returns the final set of checked ratings"""

        cbox_statuses = self.checkbox.get_status()
        user_ratings = [self.checkbox.labels[idx].get_text()
                        for idx, this_cbox_active in
                        enumerate(cbox_statuses) if this_cbox_active]

        return user_ratings


    def allowed_to_advance(self):
        """
        Method to ensure work is done for current iteration,
        before allowing the user to advance to next subject.

        Returns False if atleast one of the following conditions are not met:
            Atleast Checkbox is checked
        """

        if any(self.checkbox.get_status()):
            allowed = True
        else:
            allowed = False

        return allowed


    def reset_figure(self):
        "Resets the figure to prepare it for display of next subject."

        self.clear_data()
        self.clear_checkboxes()
        self.clear_radio_buttons()
        self.clear_notes_annot()


    def clear_data(self):
        """clearing all data/image handles"""

        if self.data_handles:
            for artist in self.data_handles:
                artist.remove()
            # resetting it
            self.data_handles = list()


    def clear_notes_annot(self):
        """clearing notes and annotations"""

        self.text_box.set_val(cfg.textbox_initial_text)
        # text is matplotlib artist
        self.annot_text.remove()


    def clear_radio_buttons(self):
        """Clears the radio button"""

        # enabling default rating encourages lazy advancing without review
        # self.radio_bt_rating.set_active(cfg.index_freesurfer_default_rating)
        for index, label in enumerate(self.radio_bt_vis_type.labels):
            if label.get_text() == self.radio_bt_vis_type.value_selected:
                self.radio_bt_vis_type.circles[index].set_facecolor(
                    cfg.color_rating_axis)
                break
        self.radio_bt_vis_type.value_selected = None


    def on_mouse(self, event):
        """Callback for mouse events."""

        if self.prev_axis is not None:
            if event.inaxes not in self.unzoomable_axes:
                self.prev_axis.set_position(self.prev_ax_pos)
                self.prev_axis.set_zorder(0)
                self.prev_axis.patch.set_alpha(0.5)
                self.zoomed_in = False

        # right or double click to zoom in to any axis
        if (event.button in [3] or event.dblclick) and \
            (event.inaxes is not None) and \
            (event.inaxes not in self.unzoomable_axes):
            self.prev_ax_pos = event.inaxes.get_position()
            event.inaxes.set_position(cfg.zoomed_position)
            event.inaxes.set_zorder(1)  # bring forth
            event.inaxes.set_facecolor('black')  # black
            event.inaxes.patch.set_alpha(1.0)  # opaque
            self.zoomed_in = True
            self.prev_axis = event.inaxes
        else:
            pass

        self.fig.canvas.draw_idle()


    def on_keyboard(self, key_in):
        """Callback to handle keyboard shortcuts to rate and advance."""

        # ignore keyboard key_in when mouse within Notes textbox
        if key_in.inaxes == self.text_box.ax or key_in.key is None:
            return

        key_pressed = key_in.key.lower()
        # print(key_pressed)
        if key_pressed in ['right', ' ', 'space']:
            self.next_button_callback()
        elif key_pressed in ['ctrl+q', 'q+ctrl']:
            self.quit_button_callback()
        elif key_pressed in self.map_key_to_callback:
            # notice parentheses at the end
            self.map_key_to_callback[key_pressed]()
        else:
            if key_pressed in cfg.abbreviation_t1_mri_default_issue_list:
                checked_label = cfg.abbreviation_t1_mri_default_issue_list[
                    key_pressed]
                self.checkbox.set_active(
                    cfg.t1_mri_default_issue_list.index(checked_label))
            else:
                pass

        self.fig.canvas.draw_idle()


class RatingWorkflowT1(BaseWorkflowVisualQC, ABC):
    """
    Rating workflow without any overlay.
    """


    def __init__(self,
                 id_list,
                 in_dir,
                 out_dir,
                 issue_list,
                 mri_name,
                 in_dir_type,
                 images_for_id,
                 outlier_method, outlier_fraction,
                 outlier_feat_types, disable_outlier_detection,
                 prepare_first,
                 vis_type,
                 views, num_slices_per_view, num_rows_per_view):
        """Constructor"""

        super().__init__(id_list, in_dir, out_dir,
                         outlier_method, outlier_fraction,
                         outlier_feat_types, disable_outlier_detection)

        self.vis_type = vis_type
        self.issue_list = issue_list
        self.mri_name = mri_name
        self.in_dir_type = in_dir_type
        self.images_for_id = images_for_id
        self.expt_id = 'rate_mri_{}'.format(self.mri_name)
        self.suffix = self.expt_id
        self.current_alert_msg = None
        self.prepare_first = prepare_first

        self.init_layout(views, num_rows_per_view, num_slices_per_view)
        self.init_getters()

        self.__module_type__ = 't1_mri'


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

        plt.style.use('dark_background')

        # vmin/vmax are controlled, because we rescale all to [0, 1]
        self.display_params = dict(interpolation='none', aspect='equal',
                                   origin='lower', cmap='gray', vmin=0.0, vmax=1.0)
        self.figsize = cfg.default_review_figsize

        self.collage = Collage(view_set=views,
                               num_slices=num_slices_per_view,
                               num_rows=num_rows_per_view,
                               display_params=self.display_params,
                               bounding_rect=cfg.bounding_box_review,
                               figsize=self.figsize)
        self.fig = self.collage.fig
        self.fig.canvas.set_window_title('VisualQC T1 MRI : {} {} '
                                         ''.format(self.in_dir, self.mri_name))

        self.padding = padding


    def init_getters(self):
        """Initializes the getters methods for input paths and feature readers."""

        from visualqc.features import extract_T1_features
        self.feature_extractor = extract_T1_features

        if self.vis_type is not None and (
            self.vis_type in cfg.freesurfer_vis_types or self.in_dir_type in [
            'freesurfer', ]):
            self.path_getter_inputs = lambda sub_id: realpath(
                pjoin(self.in_dir, sub_id, 'mri', self.mri_name))
        else:
            if self.in_dir_type.upper() in ('BIDS',):
                self.path_getter_inputs = lambda sub_id: self.images_for_id[
                    sub_id]['image']
            else:
                self.path_getter_inputs = lambda sub_id: realpath(
                    pjoin(self.in_dir, sub_id, self.mri_name))


    def open_figure(self):
        """Creates the master figure to show everything in."""

        plt.show(block=False)


    def add_UI(self):
        """Adds the review UI with defaults"""

        # two keys for same combinations exist to account for time delays in key
        # presses
        map_key_to_callback = {'alt+s': self.show_saturated,
                               's+alt': self.show_saturated,
                               'alt+b': self.show_background_only,
                               'b+alt': self.show_background_only,
                               'alt+t': self.show_tails_trimmed,
                               't+alt': self.show_tails_trimmed,
                               'alt+o': self.show_original,
                               'o+alt': self.show_original}
        self.UI = T1MriInterface(self.collage.fig, self.collage.flat_grid,
                                 self.issue_list,
                                 next_button_callback=self.next,
                                 quit_button_callback=self.quit,
                                 processing_choice_callback=self.process_and_display,
                                 map_key_to_callback=map_key_to_callback)

        # connecting callbacks
        self.con_id_click = self.fig.canvas.mpl_connect('button_press_event',
                                                        self.UI.on_mouse)
        self.con_id_keybd = self.fig.canvas.mpl_connect('key_press_event',
                                                        self.UI.on_keyboard)
        # con_id_scroll = self.fig.canvas.mpl_connect('scroll_event',
        # self.UI.on_scroll)

        self.fig.set_size_inches(self.figsize)


    def add_histogram_panel(self):
        """Extra axis for histogram"""

        self.ax_hist = plt.axes(cfg.position_histogram_t1_mri)
        self.ax_hist.set_xticks(cfg.xticks_histogram_t1_mri)
        self.ax_hist.set_yticks([])
        self.ax_hist.set_autoscaley_on(True)
        self.ax_hist.set_prop_cycle('color', cfg.color_histogram_t1_mri)
        self.ax_hist.set_title(cfg.title_histogram_t1_mri, fontsize='small')


    def update_histogram(self, img):
        """Updates histogram with current image data"""

        nonzero_values = img.ravel()[np.flatnonzero(img)]
        _, _, patches_hist = self.ax_hist.hist(nonzero_values, density=True,
                                               bins=cfg.num_bins_histogram_display)
        self.ax_hist.relim(visible_only=True)
        self.ax_hist.autoscale_view(scalex=False)  # xlim fixed to [0, 1]
        self.UI.data_handles.extend(patches_hist)


    def update_alerts(self):
        """Keeps a box, initially invisible."""

        if self.current_alert_msg is not None:
            h_alert_text = self.fig.text(cfg.position_outlier_alert_t1_mri[0],
                                         cfg.position_outlier_alert_t1_mri[1],
                                         self.current_alert_msg,
                                         **cfg.alert_text_props)
            # adding it to list of elements to cleared when advancing to next subject
            self.UI.data_handles.append(h_alert_text)


    def add_alerts(self):
        """Brings up an alert if subject id is detected to be an outlier."""

        flagged_as_outlier = self.current_unit_id in self.by_sample
        if flagged_as_outlier:
            alerts_list = self.by_sample.get(self.current_unit_id,
                                             None)  # None, if id not in dict
            print('\n\tFlagged as a possible outlier by these measures:\n\t\t{}'
                  ''.format('\t'.join(alerts_list)))

            strings_to_show = ['Flagged as an outlier:', ] + alerts_list
            self.current_alert_msg = '\n'.join(strings_to_show)
            self.update_alerts()
        else:
            self.current_alert_msg = None


    def load_unit(self, unit_id):
        """Loads the image data for display."""

        # starting fresh
        for attr in ('current_img_raw', 'current_img',
                     'saturated_img', 'tails_trimmed_img', 'background_img'):
            if hasattr(self, attr):
                delattr(self, attr)

        t1_mri_path = self.path_getter_inputs(unit_id)
        self.current_img_raw = read_image(t1_mri_path, error_msg='T1 mri')
        # crop and rescale
        self.current_img = scale_0to1(crop_image(self.current_img_raw, self.padding))
        self.currently_showing = None

        skip_subject = False
        if np.count_nonzero(self.current_img) == 0:
            skip_subject = True
            print('MR image is empty!')

        # # where to save the visualization to
        # out_vis_path = pjoin(self.out_dir, 'visual_qc_{}_{}'.format(
        # self.vis_type, unit_id))

        return skip_subject


    def display_unit(self):
        """Adds slice collage to the given axes"""

        # showing the collage
        self.collage.attach(self.current_img)
        # updating histogram
        self.update_histogram(self.current_img)


    def process_and_display(self, user_choice):
        """Updates the display after applying the chosen method."""

        if user_choice in ('Saturate',):
            self.show_saturated(no_toggle=True)
        elif user_choice in ('Background only',):
            self.show_background_only(no_toggle=True)
        elif user_choice in ('Tails_trimmed', 'Tails trimmed'):
            self.show_tails_trimmed(no_toggle=True)
        elif user_choice in ('Original',):
            self.show_original()
        else:
            print('Chosen option seems to be not implemented!')


    def show_saturated(self, no_toggle=False):
        """Callback for ghosting specific review"""

        if not self.currently_showing in ['saturated', ] or no_toggle:
            if not hasattr(self, 'saturated_img'):
                self.saturated_img = saturate_brighter_intensities(
                    self.current_img, percentile=cfg.saturate_perc_t1)
            self.collage.attach(self.saturated_img)
            self.currently_showing = 'saturated'
        else:
            self.show_original()


    def show_background_only(self, no_toggle=False):
        """Callback for ghosting specific review"""

        if not self.currently_showing in ['Background only', ] or no_toggle:
            self._compute_background()
            self.collage.attach(self.background_img)
            self.currently_showing = 'Background only'
        else:
            self.show_original()


    def _compute_background(self):
        """Computes the background image for the current image."""

        if not hasattr(self, 'background_img'):
            # need to scale the mask, as Collage class does NOT automatically rescale
            self.foreground_mask = mask_image(self.current_img, out_dtype=bool)
            temp_background_img = np.copy(self.current_img)
            temp_background_img[self.foreground_mask] = 0.0
            self.background_img = scale_0to1(temp_background_img,
                                             exclude_outliers_below=1,
                                             exclude_outliers_above=1)


    def show_tails_trimmed(self, no_toggle=False):
        """Callback for ghosting specific review"""

        if not self.currently_showing in ['tails_trimmed', ] or no_toggle:
            if not hasattr(self, 'tails_trimmed_img'):
                self.tails_trimmed_img = scale_0to1(self.current_img,
                                                    exclude_outliers_below=1,
                                                    exclude_outliers_above=0.05)
            self.collage.attach(self.tails_trimmed_img)
            self.currently_showing = 'tails_trimmed'
        else:
            self.show_original()


    def show_original(self):
        """Show the original"""

        self.collage.attach(self.current_img)
        self.currently_showing = 'original'


    def cleanup(self):
        """Preparating for exit."""

        # save ratings before exiting
        self.save_ratings()

        self.fig.canvas.mpl_disconnect(self.con_id_click)
        self.fig.canvas.mpl_disconnect(self.con_id_keybd)
        plt.close('all')


def get_parser():
    """Parser to specify arguments and their defaults."""

    parser = argparse.ArgumentParser(prog="visualqc_t1_mri",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description='visualqc_t1_mri: rate quality of '
                                                 'anatomical MR scan.')

    help_text_bids_dir = textwrap.dedent("""
    Absolute path to the top-level BIDS folder containing the dataset.
    Each subject will be named after the longest/unique ID encoding info on
    sessions and anything else available in the filename in the deepest hierarchy etc

    E.g. ``--bids_dir /project/dataset_bids``
    \n""")

    help_text_fs_dir = textwrap.dedent("""
    Absolute path to ``SUBJECTS_DIR`` containing Freesurfer runs.
    Each subject will be queried after its ID in the metadata file.

    E.g. ``--fs_dir /project/freesurfer_v5.3``
    \n""")

    help_text_user_dir = textwrap.dedent("""
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

    help_text_mri_name = textwrap.dedent("""
    Specifies the name of MRI image to serve as the reference slice.
    Typical options include orig.mgz, brainmask.mgz, T1.mgz etc.
    Make sure to choose the right vis_type.

    Default: {} (within the mri folder of Freesurfer format).
    \n""".format(cfg.default_mri_name))

    help_text_out_dir = textwrap.dedent("""
    Output folder to store the visualizations & ratings.
    Default: a new folder called ``{}`` will be created inside the input folder
    \n""".format(cfg.default_out_dir_name))

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
    This flag enables batch-generation of 3d surface visualizations,
    prior to starting any review and rating operations. This makes the switch
    from one subject to the next, even more seamless (saving few seconds :) ).

    Default: False  (required visualizations are generated only on demand,
    which can take 5-10 seconds for each subject).
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
    It could be one of 'cortical' (aparc.stats: mean thickness and other
    geometrical features from each cortical label), 'subcortical' (aseg.stats:
    volumes of several subcortical structures), or 'both' (using both aseg and
    aparc stats).

    Default: {}.
    \n""".format(cfg.t1_mri_features_OLD))

    help_text_disable_outlier_detection = textwrap.dedent("""
    This flag disables outlier detection and alerts altogether.
    \n""")

    in_out = parser.add_argument_group('Input and output', ' ')

    in_out.add_argument("-b", "--bids_dir", action="store", dest="bids_dir",
                        default=cfg.default_bids_dir,
                        required=False, help=help_text_bids_dir)

    in_out.add_argument("-i", "--id_list", action="store", dest="id_list",
                        default=None, required=False, help=help_text_id_list)

    in_out.add_argument("-u", "--user_dir", action="store", dest="user_dir",
                        default=cfg.default_user_dir,
                        required=False, help=help_text_user_dir)

    in_out.add_argument("-m", "--mri_name", action="store", dest="mri_name",
                        default=cfg.default_mri_name, required=False,
                        help=help_text_mri_name)

    in_out.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        required=False, help=help_text_out_dir,
                        default=None)

    in_out.add_argument("-f", "--fs_dir", action="store", dest="fs_dir",
                        default=cfg.default_freesurfer_dir,
                        required=False, help=help_text_fs_dir)

    outliers = parser.add_argument_group('Outlier detection',
                                         'options related to automatically '
                                         'detecting '
                                         'possible outliers')
    outliers.add_argument("-olm", "--outlier_method", action="store",
                          dest="outlier_method",
                          default=cfg.default_outlier_detection_method,
                          required=False,
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

    wf_args = parser.add_argument_group('Workflow',
                                        'Options related to workflow e.g. to '
                                        'pre-compute resource-intensive features, '
                                        'and pre-generate all the visualizations '
                                        'required')
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

    vis_type = 'collage_t1_mri'
    type_of_features = 't1_mri'
    in_dir, in_dir_type = check_input_dir_T1(user_args.fs_dir, user_args.user_dir,
                                             user_args.bids_dir)

    if in_dir_type.upper() in ('BIDS',):
        mri_name = None
        in_dir, bids_dir_type = check_bids_dir(in_dir)
        id_list, images_for_id = find_anatomical_images_in_BIDS(in_dir)
    else:
        mri_name = user_args.mri_name
        id_list, images_for_id = check_id_list(user_args.id_list, in_dir, vis_type,
                                               mri_name, seg_name=None,
                                               in_dir_type=in_dir_type)

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

    wf = RatingWorkflowT1(id_list, in_dir, out_dir,
                          cfg.t1_mri_default_issue_list,
                          mri_name, in_dir_type, images_for_id,
                          outlier_method, outlier_fraction,
                          outlier_feat_types, disable_outlier_detection,
                          user_args.prepare_first,
                          vis_type,
                          views, num_slices_per_view, num_rows_per_view)

    return wf


def cli_run():
    """Main entry point."""

    print('\nAnatomical MRI module')
    from visualqc.utils import run_common_utils_before_starting
    run_common_utils_before_starting()

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
