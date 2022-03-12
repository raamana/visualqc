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
from matplotlib.image import imread
from matplotlib.widgets import CheckButtons, RadioButtons
from mrivis.base import Collage, SlicePicker
from visualqc import config as cfg
from visualqc.image_utils import rescale_without_outliers
from visualqc.interfaces import BaseReviewInterface
from visualqc.utils import (check_inputs_defacing, check_out_dir,
                            compute_cell_extents_grid, read_image,
                            pixdim_nifti_header, slice_aspect_ratio)
from visualqc.workflows import BaseWorkflowVisualQC


class DefacingInterface(BaseReviewInterface):
    """Custom interface to rate the quality of defacing in an MRI scan"""


    def __init__(self,
                 fig,
                 axes,
                 issue_list=cfg.defacing_default_issue_list,
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
                                self.bt_next.ax, self.bt_quit.ax,
                                self.radio_bt_vis_type]

        # this list of artists to be populated later
        # makes to handy to clean them all
        self.data_handles = list()


    def add_checkboxes(self):
        """
        Checkboxes offer the ability to select multiple tags such as Motion,
        Ghosting Aliasing etc, instead of one from a list of mutual exclusive
        rating options (such as Good, Bad, Error etc).

        """

        ax_checkbox = plt.axes(cfg.position_checkbox_t1_mri,
                               facecolor=cfg.color_rating_axis)
        # initially de-activating all
        check_box_status = [False] * len(self.issue_list)
        self.checkbox = CheckButtons(ax_checkbox, labels=self.issue_list,
                                     actives=check_box_status)
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

        self._index_pass = cfg.defacing_default_issue_list.index(
            cfg.defacing_pass_indicator)


    def add_process_options(self):

        ax_radio = plt.axes(cfg.position_radio_bt_t1_mri,
                            facecolor=cfg.color_rating_axis)
        self.radio_bt_vis_type = RadioButtons(ax_radio, cfg.vis_choices_defacing,
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
                #   self.save_issues() each time, if eventson is True
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
            event.inaxes not in self.unzoomable_axes:
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
                         show_unit_id=False,  # preventing bias/batch-effects
                         outlier_method=None, outlier_fraction=None,
                         outlier_feat_types=None,
                         disable_outlier_detection=None)

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

        self.__module_type__ = 'defacing'


    def preprocess(self):
        """Preprocessing if necessary."""

        pass


    def init_layout(self,
                    view_set=cfg.defacing_view_set,
                    num_rows_per_view=cfg.defacing_num_rows_per_view,
                    num_slices_per_view=cfg.defacing_num_slices_per_view,
                    padding=cfg.default_padding):
        """initializes the layout"""

        plt.style.use('dark_background')

        # vmin/vmax are controlled, because we rescale all to [0, 1]
        self.display_params = dict(interpolation='none',
                                   aspect='auto',
                                   origin='lower',
                                   cmap='gray',
                                   vmin=0.0, vmax=1.0)
        self.figsize = cfg.default_review_figsize

        self.collage = Collage(view_set=view_set,
                               num_slices=num_slices_per_view,
                               num_rows=num_rows_per_view,
                               display_params=self.display_params,
                               bounding_rect=cfg.bbox_defacing_MRI_review,
                               figsize=self.figsize)
        self.fig = self.collage.fig
        self.fig.canvas.set_window_title('VisualQC defacing : {} {} '
                                         ''.format(self.in_dir,
                                                   self.defaced_name))

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

        # 2 keys for same combination exist to account for time delays in key presses
        map_key_to_callback = {'alt+b': self.show_defaced,
                               'b+alt': self.show_defaced,
                               'alt+o': self.show_original,
                               'o+alt': self.show_original,
                               'alt+m': self.show_mixed,
                               'm+alt': self.show_mixed,
                               }
        self.UI = DefacingInterface(self.collage.fig,
                                    self.collage.flat_grid,
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


    def load_unit(self, unit_id):
        """Loads the image data for display."""

        # starting fresh
        for attr in ('defaced_img', 'orig_img', 'render_img'):
            if hasattr(self, attr):
                delattr(self, attr)

        self.defaced_img, self.defaced_hdr = read_image(
            self.images_for_id[unit_id]['defaced'], error_msg='defaced mri',
            return_header=True)
        self.orig_img, self.orig_hdr = read_image(
            self.images_for_id[unit_id]['original'], error_msg='T1 mri',
            return_header=True)

        self.current_pixdim = pixdim_nifti_header(self.orig_hdr)
        if not np.allclose(self.current_pixdim,
                           pixdim_nifti_header(self.defaced_hdr)):
            raise ValueError('pixel dimensions for the original and '
                             'defaced images do not match! They are: {}, {}'.format(
                self.current_pixdim, self.pixdim_nifti_header(defaced_hdr)))

        self.render_img_list = list()
        for rimg_path in self.images_for_id[unit_id]['render']:
            try:
                self.render_img_list.append(imread(rimg_path))
            except:
                raise IOError('Unable to read the 3D rendered image @\n {}'
                              ''.format(rimg_path))

        # crop, trim, and rescale
        from mrivis.utils import crop_to_extents
        self.defaced_img, self.orig_img = crop_to_extents(
            self.defaced_img, self.orig_img, padding=self.padding)
        self.defaced_img = rescale_without_outliers(
            self.defaced_img, trim_percentile=cfg.defacing_trim_percentile)
        self.orig_img = rescale_without_outliers(
            self.orig_img, trim_percentile=cfg.defacing_trim_percentile)
        self.currently_showing = None

        skip_subject = False
        if np.count_nonzero(self.defaced_img) == 0 or \
            np.count_nonzero(self.orig_img) == 0:
            skip_subject = True
            print('Defaced or original MR image is empty!')

        self.slice_picker = SlicePicker(self.orig_img,
                                        view_set=self.collage.view_set,
                                        num_slices=self.collage.num_slices,
                                        sampler=cfg.defacing_slice_locations)

        # # where to save the visualization to
        # out_vis_path = pjoin(self.out_dir,
        #   'visual_qc_{}_{}'.format(self.vis_type, unit_id))

        return skip_subject


    def process_and_display(self, user_choice):
        """Updates the display after applying the chosen method."""

        if user_choice in ('Defaced only',):
            self.show_defaced()
        elif user_choice in ('Original only',):
            self.show_original()
        elif user_choice in ('Mixed', 'Fused'):
            self.show_mixed()
        else:
            print('Chosen option seems to be not implemented!')


    def display_unit(self):
        """Adds slice collage to the given axes"""

        self.show_renders()
        self.show_mr_images()


    def show_renders(self):
        """Show all the rendered images"""

        num_cells = len(self.render_img_list)
        cell_extents = compute_cell_extents_grid(
            cfg.bbox_defacing_render_review,
            num_rows=cfg.defacing_num_rows_renders,
            num_cols=num_cells)

        self.ax_render = list()
        for img, ext in zip(self.render_img_list, cell_extents):
            ax = self.fig.add_axes(ext, frameon=False)
            ax.set_axis_off()
            ax.imshow(img)
            ax.set_visible(True)
            self.ax_render.append(ax)


    def show_defaced(self):
        """Show defaced only"""

        self.show_mr_images(vis_type='defaced')


    def show_original(self):
        """Show original only"""

        self.show_mr_images(vis_type='original')


    def show_mixed(self):
        """Show mixed"""

        self.show_mr_images(vis_type='mixed')


    def show_mr_images(self, vis_type='mixed'):
        """Generic router"""

        self.collage.clear()

        ax_counter = 0
        for dim, slice_num, (defaced, orig) in self.slice_picker.get_slices_multi(
            [self.defaced_img, self.orig_img], extended=True):

            ax = self.collage.flat_grid[ax_counter]
            if vis_type in ('mixed',):
                # TODO customizable colors: final_slice = mix_color(orig, df)
                red = 0.9 * orig
                grn = 1.0 * defaced
                blu = np.zeros_like(orig)
                ax.imshow(np.stack((red, grn, blu), axis=2),
                          **self.display_params)
            elif vis_type in ('defaced',):
                ax.imshow(defaced, **self.display_params)
            elif vis_type in ('original',):
                ax.imshow(orig, **self.display_params)
            else:
                raise ValueError('Invalid vis_type. Must be either mixed, '
                                 'defaced, or original')
            # TODO BUG individual slice-wise axes size is messed up for
            #   non-isotropic resolutions. Some subtle interaction of setting aspect
            #   ratio with axes scaling/extents, to be fixed/controlled
            ax.set_aspect(slice_aspect_ratio(self.current_pixdim, dim))
            ax_counter += 1

        self.collage.show()


    def mix_images(self, orig, defaced, color_orig, color_defaced):
        """Mixes the two images with different colors"""

        raise NotImplementedError()


    def add_alerts(self):
        pass


    def cleanup(self):
        """Cleanup before exit"""

        # save ratings
        self.save_ratings()

        self.fig.canvas.mpl_disconnect(self.con_id_click)
        self.fig.canvas.mpl_disconnect(self.con_id_keybd)
        plt.close('all')


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
    Default: a new folder called ``{}`` will be created inside the input folder
    \n""".format(cfg.default_out_dir_name))

    in_out = parser.add_argument_group('Input and output', ' ')

    in_out.add_argument("-u", "--user_dir", action="store", dest="user_dir",
                        default=cfg.default_user_dir,
                        required=False, help=help_text_user_dir)

    in_out.add_argument("-d", "--defaced_name", action="store", dest="defaced_name",
                        default=cfg.default_defaced_mri_name, required=False,
                        help=help_text_defaced_mri_name)

    in_out.add_argument("-m", "--mri_name", action="store", dest="mri_name",
                        default=cfg.default_mri_name, required=False,
                        help=help_text_mri_name)

    in_out.add_argument("-r", "--render_name", action="store", dest="render_name",
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

    print('\nDefacing module')
    from visualqc.utils import run_common_utils_before_starting
    run_common_utils_before_starting()

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
