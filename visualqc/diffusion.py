"""

Module to define interface, workflow and CLI for the review of diffusion MRI data.

"""
import argparse
import asyncio
import sys
import textwrap
import time
import warnings
from abc import ABC
from textwrap import wrap

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons
from mrivis.utils import crop_image
from os.path import basename, join as pjoin
from visualqc import config as cfg
from visualqc.image_utils import dwi_overlay_edges
from visualqc.readers import diffusion_traverse_bids
from visualqc.t1_mri import T1MriInterface
from visualqc.utils import check_bids_dir, check_finite_int, check_image_is_4d, \
    check_out_dir, check_outlier_params, check_time, check_views, get_axis, pick_slices, \
    scale_0to1
from visualqc.workflows import BaseWorkflowVisualQC

_z_score = lambda x: (x - np.mean(x)) / np.std(x)


def _prettify(filename, max_width=18):
    """Returns a easily displayable and readable multiline string"""

    parts = [s.replace('-', ' ') for s in filename.split('_')]
    fixed_width = list()
    for p in parts:
        if len(p) > max_width:
            # indenting by two spaace
            fixed_width.extend([' -' + s for s in wrap(p, max_width - 2)])
        else:
            fixed_width.append(p)

    return '\n'.join(fixed_width)


class DiffusionMRIInterface(T1MriInterface):
    """Interface for the review of fMRI images."""


    def __init__(self,
                 fig,
                 axes,
                 issue_list=cfg.diffusion_mri_default_issue_list,
                 next_button_callback=None,
                 quit_button_callback=None,
                 right_arrow_callback=None,
                 left_arrow_callback=None,
                 zoom_in_callback=None,
                 zoom_out_callback=None,
                 right_click_callback=None,
                 show_stdev_callback=None,
                 scroll_callback=None,
                 alignment_callback=None,
                 show_b0_vol_callback=None,
                 flip_first_last_callback=None,
                 stop_animation_callback=None,
                 axes_to_zoom=None,
                 total_num_layers=5):
        """Constructor"""

        super().__init__(fig, axes, issue_list,
                         next_button_callback,
                         quit_button_callback)
        self.issue_list = issue_list

        self.prev_axis = None
        self.prev_ax_pos = None
        self.prev_ax_zorder = None
        self.prev_visible = False
        self.zoomed_in = False
        self.nested_zoomed_in = False
        self.total_num_layers = total_num_layers
        self.axes_to_zoom = axes_to_zoom

        self.next_button_callback = next_button_callback
        self.quit_button_callback = quit_button_callback
        self.zoom_in_callback = zoom_in_callback
        self.zoom_out_callback = zoom_out_callback
        self.right_arrow_callback = right_arrow_callback
        self.left_arrow_callback = left_arrow_callback
        self.scroll_callback = scroll_callback,
        self.right_click_callback = right_click_callback
        self.show_stdev_callback = show_stdev_callback
        self.alignment_callback = alignment_callback
        self.flip_first_last_callback = flip_first_last_callback
        self.show_b0_vol_callback = show_b0_vol_callback
        self.stop_animation_callback = stop_animation_callback

        self.add_checkboxes()
        self.add_radio_buttons_comparison_method()

        # this list of artists to be populated later
        # makes to handy to clean them all
        self.data_handles = list()

        self.unzoomable_axes = [self.checkbox.ax, self.radio_bt_vis_type.ax,
                                self.text_box.ax, self.bt_next.ax, self.bt_quit.ax,
                                None]


    def add_checkboxes(self):
        """
        Checkboxes offer the ability to select multiple tags such as Motion,
        Ghosting, Aliasing etc, instead of one from a list of mutual exclusive
        rating options (such as Good, Bad, Error etc).

        """

        ax_checkbox = plt.axes(cfg.position_rating_checkbox_diffusion,
                               facecolor=cfg.color_rating_axis)
        # initially de-activating all
        actives = [False] * len(self.issue_list)
        self.checkbox = CheckButtons(ax_checkbox, labels=self.issue_list,
                                     actives=actives)
        self.checkbox.on_clicked(self.save_issues)
        for txt_lbl in self.checkbox.labels:
            txt_lbl.set(**cfg.checkbox_font_properties)

        for rect in self.checkbox.rectangles:
            rect.set_width(cfg.checkbox_rect_width_diffusion)
            rect.set_height(cfg.checkbox_rect_height_diffusion)

        # lines is a list of n crosses, each cross (x) defined by a tuple of lines
        for x_line1, x_line2 in self.checkbox.lines:
            x_line1.set_color(cfg.checkbox_cross_color)
            x_line2.set_color(cfg.checkbox_cross_color)

        self._index_pass = self.issue_list.index(cfg.diffusion_mri_pass_indicator)


    def add_radio_buttons_comparison_method(self):

        ax_radio = plt.axes(cfg.position_alignment_method_diffusion,
                            facecolor=cfg.color_rating_axis)
        self.radio_bt_vis_type = RadioButtons(ax_radio,
                                              cfg.choices_alignment_comparison_diffusion,
                                              active=None, activecolor='orange')
        for lbl in self.radio_bt_vis_type.labels:
            lbl.set_fontsize(cfg.fontsize_radio_button_align_method_diffusion)
        self.radio_bt_vis_type.on_clicked(self.alignment_callback)
        for txt_lbl in self.radio_bt_vis_type.labels:
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for circ in self.radio_bt_vis_type.circles:
            circ.set(radius=0.06)


    def add_process_options(self):
        """redefining it to void it's actions intended for T1w MRI interface"""
        pass

    def maximize_axis(self, ax):
        """zooms a given axes"""

        if not self.nested_zoomed_in:
            self.prev_ax_pos = ax.get_position()
            self.prev_ax_zorder = ax.get_zorder()
            self.prev_ax_alpha = ax.get_alpha()
            ax.set_position(cfg.zoomed_position_level2)
            ax.set_zorder(self.total_num_layers + 1)  # bring forth
            ax.patch.set_alpha(1.0)  # opaque
            self.nested_zoomed_in = True
            self.prev_axis = ax


    def restore_axis(self):

        if self.nested_zoomed_in:
            self.prev_axis.set(position=self.prev_ax_pos,
                               zorder=self.prev_ax_zorder,
                               alpha=self.prev_ax_alpha)
            self.nested_zoomed_in = False


    def on_mouse(self, event):
        """Callback for mouse events."""

        # if event occurs in non-data areas (or axis is None), do nothing
        if event.inaxes in self.unzoomable_axes:
            return

        # any mouse event in data-area stops the current animation
        self.stop_animation_callback()

        if self.zoomed_in:
            # include all the non-data axes here (so they wont be zoomed-in)
            if event.inaxes not in self.unzoomable_axes:
                if event.dblclick or event.button in [3]:
                    if event.inaxes in self.axes_to_zoom:
                        self.maximize_axis(event.inaxes)
                    else:
                        self.zoom_out_callback(event)
                else:
                    if self.nested_zoomed_in:
                        self.restore_axis()
                    else:
                        self.zoom_out_callback(event)

        elif event.button in [3]:
            self.right_click_callback(event)
        elif event.dblclick:
            self.zoom_in_callback(event)
        else:
            pass

        # redraw the figure - important
        self.fig.canvas.draw_idle()


    def on_keyboard(self, key_in):
        """Callback to handle keyboard shortcuts to rate and advance."""

        # ignore keyboard key_in when mouse within Notes textbox
        if key_in.inaxes == self.text_box.ax or key_in.key is None:
            return

        key_pressed = key_in.key.lower()
        # print(key_pressed)
        if key_pressed in ['right', 'up']:
            self.right_arrow_callback()
        elif key_pressed in ['left', 'down']:
            self.left_arrow_callback()
        elif key_pressed in [' ', 'space']:
            # space button stops the current animation
            self.stop_animation_callback()
        elif key_pressed in ['ctrl+q', 'q+ctrl']:
            self.quit_button_callback()
        elif key_pressed in ['alt+s', 's+alt']:
            self.show_stdev_callback()
        elif key_pressed in ['alt+0', '0+alt']:
            self.show_b0_vol_callback()
        elif key_pressed in ['alt+n', 'n+alt']:
            self.flip_first_last_callback()
        else:
            if key_pressed in cfg.abbreviation_diffusion_mri_default_issue_list:
                checked_label = cfg.abbreviation_diffusion_mri_default_issue_list[
                    key_pressed]
                # TODO if user chooses a different set of names, keyboard shortcuts might not work
                self.checkbox.set_active(self.issue_list.index(checked_label))
            else:
                pass

        self.fig.canvas.draw_idle()


    def on_scroll(self, scroll_event):
        """Implements the scroll callback"""

        self.scroll_callback(scroll_event)

    def reset_figure(self):
        "Resets the figure to prepare it for display of next subject."

        self.zoom_out_callback(None)
        self.restore_axis()
        self.clear_data()
        self.clear_checkboxes()
        self.clear_notes_annot()


class DiffusionRatingWorkflow(BaseWorkflowVisualQC, ABC):
    """
    Rating workflow for BOLD fMRI.
    """


    def __init__(self,
                 in_dir,
                 out_dir,
                 apply_preproc=False,
                 id_list=None,
                 name_pattern=None,
                 images_for_id=None,
                 delay_in_animation=cfg.delay_in_animation_diffusion_mri,
                 issue_list=cfg.diffusion_mri_default_issue_list,
                 in_dir_type='BIDS',
                 outlier_method=cfg.default_outlier_detection_method,
                 outlier_fraction=cfg.default_outlier_fraction,
                 outlier_feat_types=cfg.diffusion_mri_features_OLD,
                 disable_outlier_detection=True,
                 prepare_first=False,
                 vis_type=None,
                 views=cfg.default_views_diffusion,
                 num_slices_per_view=cfg.default_num_slices_diffusion,
                 num_rows_per_view=cfg.default_num_rows_diffusion):
        """
        Constructor.

        Parameters
        ----------
        in_dir : path
            must be a path to BIDS directory

        drop_start : int
            Number of frames to drop at the beginning of the time series.

        apply_preproc : bool
            Whether to apply basic preprocessing steps (detrend, slice timing correction etc)
                before building the carpet image.
                If the images are already preprocessed elsewhere, disable this with apply_preproc=True
            Default : False, to not apply any preprocessing before display for review.

        """

        if id_list is None and 'BIDS' in in_dir_type:
            id_list = pjoin(in_dir, 'participants.tsv')

        super().__init__(id_list, in_dir, out_dir,
                         outlier_method, outlier_fraction,
                         outlier_feat_types, disable_outlier_detection)

        # basic cleaning before display
        # whether to remove and detrend before making carpet plot
        self.apply_preproc = apply_preproc

        self.vis_type = vis_type
        self.issue_list = issue_list
        self.in_dir_type = in_dir_type
        self.name_pattern = name_pattern
        self.images_for_id = images_for_id
        self.expt_id = 'rate_diffusion'
        self.suffix = self.expt_id
        self.current_alert_msg = None
        self.prepare_first = prepare_first

        #
        self.current_grad_index = 0
        self.delay_in_animation = delay_in_animation

        self.init_layout(views, num_rows_per_view, num_slices_per_view)
        self.init_getters()

        self.__module_type__ = 'diffusion'


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

        from visualqc.features import diffusion_mri_features
        self.feature_extractor = diffusion_mri_features

        if 'BIDS' in self.in_dir_type.upper():
            from bids import BIDSLayout
            self.bids_layout = BIDSLayout(self.in_dir)
            self.units = diffusion_traverse_bids(self.bids_layout)

            if self.units is None or len(self.units) < 1:
                print('No valid subjects are found! Exiting.\n'
                      'Double check the format and integrity of the dataset '
                      'if this is unexpected.')
                import sys
                sys.exit(1)

            # file name of each scan is the unique identifier,
            #   as it essentially contains all the key info.
            self.unit_by_id = {basename(sub_data['image']): sub_data
                               for _, sub_data in self.units.items()}
            self.id_list = list(self.unit_by_id.keys())
        else:
            raise NotImplementedError('Only the BIDS format is supported for now!')


    def open_figure(self):
        """Creates the master figure to show everything in."""

        # number of stats to be overlaid on top of carpet plot
        self.num_stats = 3
        self.figsize = cfg.default_review_figsize

        # empty/dummy data for placeholding
        empty_image = np.full((200, 200), 0.0)
        label_x, label_y = (5, 5) # x, y in image data space
        empty_vec = np.full((200, 1), 0.0)
        gradients = list(range(200))

        # overlay order -- larger appears on top of smaller
        self.layer_order_carpet = 1
        self.layer_order_stats = 2
        self.layer_order_zoomedin = 3
        self.layer_order_to_hide = -1
        self.total_num_layers = 3

        plt.style.use('dark_background')

        # 1. main carpet, in the background
        self.fig, self.ax_carpet = plt.subplots(1, 1, figsize=self.figsize)
        self.fig.canvas.set_window_title('VisualQC Diffusion MRI :'
                                         ' {}'.format(self.in_dir))

        self.ax_carpet.set_zorder(self.layer_order_carpet)
        #   vmin/vmax are controlled, because we rescale all to [0, 1]
        self.imshow_params_carpet = dict(interpolation='none', aspect='auto',
                                         origin='lower', cmap='gray', vmin=0.0, vmax=1.0)

        self.ax_carpet.yaxis.set_visible(False)
        self.ax_carpet.set_xlabel('gradient')
        self.carpet_handle = self.ax_carpet.imshow(empty_image,
                                                   **self.imshow_params_carpet)
        self.ax_carpet.set_frame_on(False)
        self.ax_carpet.set_ylim(auto=True)

        # 2. temporal traces of image stats
        tmp_mat = self.fig.subplots(self.num_stats, 1, sharex=True)
        self.stats_axes = tmp_mat.flatten()
        self.stats_handles = [None] * len(self.stats_axes)

        stats = [(empty_vec, 'mean signal', 'cyan'),
                 (empty_vec, 'std. dev signal', 'xkcd:orange red'),
                 (empty_vec, 'DVARS', 'xkcd:mustard')]
        for ix, (ax, (stat, label, color)) in enumerate(zip(self.stats_axes, stats)):
            (vh,) = ax.plot(gradients, stat, color=color)
            self.stats_handles[ix] = vh
            vh.set_linewidth(cfg.linewidth_stats_diffusion)
            vh.set_linestyle(cfg.linestyle_stats_diffusion)
            ax.xaxis.set_visible(False)
            ax.set_frame_on(False)
            ax.set_ylim(auto=True)
            ax.set_ylabel(label, color=color)
            ax.set_zorder(self.layer_order_stats)
            ax.set_alpha(cfg.alpha_stats_overlay)
            ax.tick_params(color=color, labelcolor=color)
            ax.spines['left'].set_color(color)
            ax.spines['left'].set_position(('outward', 1))

        # sharing the time point axis
        self.stats_axes[0].get_shared_x_axes().join(self.ax_carpet.xaxis,
                                                    self.stats_axes[0].xaxis)
        # self.stats_axes[0].autoscale()

        # 3. axes to show slices in foreground when a time point is selected
        matrix_handles = self.fig.subplots(self.num_rows, self.num_cols,
                                           subplot_kw=dict(rasterized=True),
                                           gridspec_kw=dict(wspace=0.01, hspace=0.01))
        self.fg_axes = matrix_handles.flatten()

        # vmin/vmax are controlled, because we rescale all to [0, 1]
        self.imshow_params_zoomed = dict(interpolation='none', aspect='equal',
                                         rasterized=True, origin='lower', cmap='gray',
                                         vmin=0.0, vmax=1.0)

        # images to be shown in the forground
        self.images_fg = [None] * len(self.fg_axes)
        self.images_fg_label = [None] * len(self.fg_axes)
        for ix, ax in enumerate(self.fg_axes):
            ax.axis('off')
            self.images_fg[ix] = ax.imshow(empty_image, **self.imshow_params_zoomed)
            self.images_fg_label[ix] = ax.text(label_x, label_y, '',
                                               **cfg.slice_num_label_properties,
                                               zorder=self.layer_order_zoomedin+1)
            ax.set(visible=False, zorder=self.layer_order_zoomedin)

        self.foreground_h = self.fig.text(cfg.position_zoomed_gradient[0],
                                          cfg.position_zoomed_gradient[1],
                                          ' ', **cfg.annot_gradient)
        self.foreground_h.set_visible(False)

        # identifying axes that could be hidden to avoid confusion
        self.background_artists = list(self.stats_axes) + [self.ax_carpet, ]
        self.foreground_artists = list(self.fg_axes) + [self.foreground_h, ]

        # separating the list below to allow for differing x axes, while being background
        self.axes_common_xaxis = list(self.stats_axes) + [self.ax_carpet, ]

        # leaving some space on the right for review elements
        plt.subplots_adjust(**cfg.review_area)
        plt.show(block=False)

        self.anim_loop = asyncio.get_event_loop()


    def add_UI(self):
        """Adds the review UI with defaults"""

        self.UI = DiffusionMRIInterface(self.fig, self.ax_carpet, self.issue_list,
                                        next_button_callback=self.next,
                                        quit_button_callback=self.quit,
                                        right_click_callback=self.zoom_in_on_gradient,
                                        right_arrow_callback=self.show_next,
                                        left_arrow_callback=self.show_prev,
                                        scroll_callback=self.change_gradient_by_step,
                                        zoom_in_callback=self.zoom_in_on_gradient,
                                        zoom_out_callback=self.zoom_out_callback,
                                        show_stdev_callback=self.show_stdev,
                                        show_b0_vol_callback=self.show_b0_gradient,
                                        flip_first_last_callback=self.flip_first_last,
                                        alignment_callback=self.alignment_check,
                                        stop_animation_callback=self.stop_animation,
                                        axes_to_zoom=self.fg_axes,
                                        total_num_layers=self.total_num_layers)

        # connecting callbacks
        self.con_id_click = self.fig.canvas.mpl_connect('button_press_event',
                                                        self.UI.on_mouse)
        self.con_id_keybd = self.fig.canvas.mpl_connect('key_press_event',
                                                        self.UI.on_keyboard)
        self.con_id_scroll = self.fig.canvas.mpl_connect('scroll_event',
                                                         self.UI.on_scroll)

        self.fig.set_size_inches(self.figsize)


    def add_histogram_panel(self):
        """Extra axis for histogram"""
        pass


    def update_histogram(self, img):
        """
        Updates histogram with current image data.

        Mimic behaviour in T1 mri workflow if helpful!
        """
        pass


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

        img_path = self.unit_by_id[unit_id]['image']
        bval_path = self.unit_by_id[unit_id]['bval']
        try:
            hdr = nib.load(img_path)
            self.hdr_this_unit = nib.as_closest_canonical(hdr)
            self.img_this_unit_raw = self.hdr_this_unit.get_data()
            self.b_values_this_unit = np.loadtxt(bval_path).flatten()
        except Exception as exc:
            print(exc)
            print('Unable to read image at \n\t{}'.format(img_path))
            skip_subject = True
        else:
            check_image_is_4d(self.img_this_unit_raw)

            self.b0_indices = np.flatnonzero(self.b_values_this_unit == 0)
            if len(self.b0_indices) < 1:
                skip_subject = True
                print('There are no b=0 volumes for {}! Skipping it..'.format(unit_id))
                return skip_subject

            if len(self.b0_indices) == 1:
                self.b0_volume = self.img_this_unit_raw[..., self.b0_indices].squeeze()
            else:
                # TODO which is the correct b=0 volumes are available
                # TODO is there a way to reduce multiple into one
                print('Multiple b=0 volumes found for {} '
                      '- choosing the first!'.format(unit_id))
                self.b0_volume = self.img_this_unit_raw[..., self.b0_indices[0]].squeeze()
            # need more thorough checks on whether image loaded is indeed DWI

            self.dw_indices = np.flatnonzero(self.b_values_this_unit != 0)
            self.dw_volumes = self.img_this_unit_raw[:, :, :, self.dw_indices]
            self.num_gradients = self.dw_volumes.shape[3]
            # to check alignment
            self.current_grad_index = 0

            skip_subject = False
            if np.count_nonzero(self.img_this_unit_raw) == 0:
                skip_subject = True
                print('Diffusion image is empty!')

        return skip_subject


    def display_unit(self):
        """Adds multi-layered composite."""

        # TODO show median signal instead of mean - or option for both?
        self.stdev_this_unit, self.mean_this_unit = self.stats_over_gradients()

        # TODO what about slice timing correction?

        num_voxels = np.prod(self.dw_volumes.shape[0:3])
        # TODO better way to label each gradient would be with unit vector/direction
        gradients = list(range(self.num_gradients))

        # 1. compute necessary stats/composites
        carpet, mean_signal_spatial, stdev_signal_spatial, dvars = self.compute_stats()

        # 2. display/update the data
        self.carpet_handle.set_data(carpet)
        self.stats_handles[0].set_data(gradients, mean_signal_spatial)
        self.stats_handles[1].set_data(gradients, stdev_signal_spatial)
        # not displaying DVARS for t=0, as its always 0
        self.stats_handles[2].set_data(gradients[1:], dvars[1:])

        # 3. updating axes limits and views
        self.update_axes_limits(self.num_gradients, carpet.shape[0])
        self.refresh_layer_order()

        # clean up
        del carpet, mean_signal_spatial, stdev_signal_spatial, dvars


    def zoom_in_on_gradient(self, event):
        """Brings up selected time point"""

        if event.x is None:
            return

        self.checking_alignment = False  # to distinguish between no or alignment overlay

        # computing x in axes data coordinates myself, to avoid overlaps with other axes
        # retrieving the latest transform after to ensure its accurate at click time
        x_in_carpet, _y = self._event_location_in_axis(event, self.ax_carpet)
        # clipping it to [0, T]
        self.current_grad_index = max(0, min(self.dw_volumes.shape[3],
                                             int(round(x_in_carpet))))
        self.show_gradient()


    def change_gradient_by_step(self, step):
        """Changes the index of the gradient being shown.

        Step could be negative to move in opposite direction.
        """

        # skipping unnecessary computation
        new_index = self.current_grad_index + step
        if (new_index > self.num_gradients - 1) or (new_index < 0):
            return

        # clipping from 0 to num_gradients
        self.current_grad_index = max(0, min(self.num_gradients, new_index))
        self.show_gradient()


    def show_b0_gradient(self):
        """Shows the b=0 volume"""
        # TODO what if more than one b=0 volumees are available
        if self.current_grad_index == self.b0_indices[0] and \
            self.UI.zoomed_in:
            return  # do nothing

        self.show_3dimage(self.b0_volume.squeeze(), 'b=0 volume')


    def animate_through_gradients(self):
        """Loops through all the gradients, in mulit-slice view, to help spot artefacts"""

        self.anim_loop.run_until_complete(self._animate_through_gradients())


    @asyncio.coroutine
    def _animate_through_gradients(self):
        """Show image 1, wait, show image 2"""

        # fixing the same slices for all gradients
        slices = pick_slices(self.b0_volume, self.views, self.num_slices_per_view)

        for grad_idx in range(self.num_gradients):
            self.show_3dimage(self.dw_volumes[:, :, :, grad_idx].squeeze(),
                              slices=slices, annot='gradient {}'.format(grad_idx))
            plt.pause(0.05)
            time.sleep(self.delay_in_animation)


    def flip_first_last(self):
        """Flips between first and last volume to identify any pulsation artefacts"""

        # 0 and -1 are indexing into self.dw_volumes, not b0_volumes
        self.flip_between_two(0, -1)


    def flip_between_two(self, index_one, index_two, first_index_in_b0=False):
        """Flips between first and last volume to identify any pulsation artefacts"""

        self.anim_loop.run_until_complete(self._flip_between_two_nTimes(
            index_one, index_two, first_index_in_b0=first_index_in_b0))


    @asyncio.coroutine
    def _flip_between_two_nTimes(self, index_one, index_two, first_index_in_b0=False):
        """Show first, wait, show last, repeat"""

        if first_index_in_b0:
            _first_vol = self.b0_volume  # [:, :, :, index_one].squeeze()
            _id_first = 'b=0'  # index {}'.format(index_one)
        else:
            _first_vol = self.dw_volumes[:, :, :, index_one].squeeze()
            _id_first = 'DW gradient {}'.format(index_one)

        _second_vol = self.dw_volumes[:, :, :, index_two].squeeze()
        if index_two < 0:
            # -1 would be confusing to the user
            index_two = self.num_gradients+index_two
        _id_second = 'DW gradient {}'.format(index_two)

        for _ in range(cfg.num_times_to_animate_diffusion_mri):
            for img, txt in ((_first_vol, _id_first),
                             (_second_vol, _id_second)):
                self.show_3dimage(img, txt)
                plt.pause(0.05)
                time.sleep(self.delay_in_animation)


    def alignment_check(self, label=None):
        """Chooses between the type of alignment check to show"""

        if label is not None:
            self.current_alignment_check = label

        if label in ['Align to b=0 (animate)', 'Alignment to b=0', 'Align to b=0',
                     'Align to b=0 (edges)']:
            self.alignment_to_b0()
        elif label in ['Animate all', 'Flip through all']:
            self.animate_through_gradients()
        elif label in ['Flip first & last', ]:
            self.flip_first_last()
        else:
            raise NotImplementedError('alignment check:{} not implemented.')


    def alignment_to_b0(self):
        """Overlays a given gradient on b0 volume to check for alignment isses"""

        self.checking_alignment = True

        if self.current_alignment_check in ['Align to b=0 (animate)', 'Align to b=0']:
            self.flip_between_two(self.b0_indices, self.current_grad_index,
                                  first_index_in_b0=True)
        elif self.current_alignment_check in ['Align to b=0 (edges)', ]:
            self.overlay_dwi_edges()


    def overlay_dwi_edges(self):

        # not cropping to help checking align in full FOV
        overlaid = scale_0to1(self.b0_volume)
        base_img = scale_0to1(self.dw_volumes[..., self.current_grad_index].squeeze())
        slices = pick_slices(base_img, self.views, self.num_slices_per_view)
        for ax_index, (dim_index, slice_index) in enumerate(slices):
            mixed = dwi_overlay_edges(get_axis(base_img, dim_index, slice_index),
                                      get_axis(overlaid, dim_index, slice_index))
            self.images_fg[ax_index].set(data=mixed)
            self.images_fg_label[ax_index].set_text(str(slice_index))

        # the following needs to be done outside show_image3d, as we need custom mixing
        self._set_backgrounds_visibility(False)
        self._set_foregrounds_visibility(True)
        self._identify_foreground('Alignment check to b=0, '
                                  'grad index {}'.format(self.current_grad_index))

    def stop_animation(self):
        # TODO this not working - run_until_complete() is likely the reason
        # call_soon does not start animation right away or at all
        if self.anim_loop.is_running():
            self.anim_loop.stop()


    def show_next(self):

        if self.current_grad_index == self.dw_volumes.shape[3] - 1:
            return  # do nothing

        self.current_grad_index = min(self.dw_volumes.shape[3] - 1,
                                      self.current_grad_index + 1)
        if self.checking_alignment:
            self.alignment_to_b0()
        else:
            self.show_gradient()


    def show_prev(self):

        if self.current_grad_index == 0:
            return  # do nothing

        self.current_grad_index = max(self.current_grad_index - 1, 0)
        if self.checking_alignment:
            self.alignment_to_b0()
        else:
            self.show_gradient()


    def zoom_out_callback(self, event):
        """Hides the zoomed-in axes (showing frame)."""

        self._set_foregrounds_visibility(False)
        self._set_backgrounds_visibility(True)


    @staticmethod
    def _event_location_in_axis(event, axis):
        """returns (x_in_axis, y_in_axis)"""

        # display pixels to axis coords
        return axis.transData.inverted().transform_point((event.x, event.y))


    def show_gradient(self):
        """Exhibits a selected timepoint on top of stats/carpet"""

        if self.current_grad_index < 0 or self.current_grad_index >= self.num_gradients:
            print('Requested time point outside '
                  'range [0, {}]'.format(self.num_gradients))
            return

        self.show_3dimage(self.dw_volumes[:, :, :, self.current_grad_index].squeeze(),
                          'zoomed-in gradient {}'.format(self.current_grad_index))


    def show_3dimage(self, image, annot, slices=None):
        """generic display method."""

        self.attach_image_to_foreground_axes(image, slices=slices)
        self._identify_foreground(annot)
        self._set_backgrounds_visibility(False)
        self._set_foregrounds_visibility(True)


    def _identify_foreground(self, text):
        """show the time point"""

        self.foreground_h.set_text(text)
        self.foreground_h.set_visible(True)


    def _set_backgrounds_visibility(self, visibility=True):

        for ax in self.background_artists:
            ax.set(visible=visibility)


    def _set_foregrounds_visibility(self, visibility=False):

        if visibility:
            zorder = self.layer_order_zoomedin
        else:
            zorder = self.layer_order_to_hide

        for ax in self.foreground_artists:
            ax.set(visible=visibility, zorder=zorder)
        # this state flag in important
        self.UI.zoomed_in = visibility


    def show_stdev(self):
        """Shows the image of temporal std. dev"""

        if self.stdev_this_unit is not None:
            self.attach_image_to_foreground_axes(self.stdev_this_unit,
                                                 cmap=cfg.colormap_stdev_diffusion)
            self._identify_foreground('Std. dev over gradients')
            self.UI.zoomed_in = True
        else:
            # if the number of b0 volumes are not sufficient
            print('SD for this unit is not available')


    def attach_image_to_foreground_axes(self, image3d, slices=None, cmap='gray'):
        """Attaches a given image to the foreground axes and bring it forth"""

        image3d = crop_image(image3d, self.padding)
        # TODO is it always acceptable to rescale diffusion data?
        image3d = scale_0to1(image3d)
        if slices is None:
            slices = pick_slices(image3d, self.views, self.num_slices_per_view)
        for ax_index, (dim_index, slice_index) in enumerate(slices):
            slice_data = get_axis(image3d, dim_index, slice_index)
            self.images_fg[ax_index].set(data=slice_data, cmap=cmap)
            self.images_fg_label[ax_index].set_text(str(slice_index))


    def compute_stats(self):
        """Computes the necessary stats to be displayed."""

        mean_signal_spatial, stdev_signal_spatial = spatial_stats(self.dw_volumes)
        dvars = self.compute_DVARS()

        for stat, sname in zip((mean_signal_spatial, stdev_signal_spatial, dvars),
                               ('mean_signal_spatial', 'stdev_signal_spatial', 'dvars')):
            if len(stat) != self.dw_volumes.shape[3]:
                raise ValueError('ERROR: lengths of different stats do not match!')
            if any(np.isnan(stat)):
                raise ValueError('ERROR: invalid values in stat : {}'.format(sname))

        carpet = self.make_carpet()

        return carpet, mean_signal_spatial, stdev_signal_spatial, dvars


    def make_carpet(self, row_order=None):
        """Makes the carpet image
        """

        carpet = self.dw_volumes.reshape(-1, self.dw_volumes.shape[3])
        if self.apply_preproc:
            # no cleaning implemented so far
            raise NotImplementedError

        # TODO is rescaled over gradients allowed?
        carpet = _rescale_over_gradients(carpet)

        # TODO reorder the carper in interesting groups of rows?

        return carpet


    def stats_over_b0(self, indices_b0):
        """Computes voxel-wise stats over B=0 volumes (no diffusion) data
            --> single volume over space.
        """

        # TODO connect this
        b0_subset = self.dw_volumes[:, :, :, self.b0_indices]
        mean_img = np.mean(b0_subset, axis=3)
        sd_img = np.std(b0_subset, axis=3)

        return mean_img, sd_img


    def stats_over_gradients(self):
        """Computes voxel-wise stats over gradients of diffusion data
            --> single volume over space.
        """

        mean_img = np.mean(self.dw_volumes, axis=3)
        sd_img = np.std(self.dw_volumes, axis=3)

        return mean_img, sd_img


    def compute_DVARS(self):
        """
        Computes the DVARS for a given diffusion image.
        Statistic adapted from the fMRI world.
        """
        # TODO need to use a common function across usecases

        RMS_diff = lambda img2, img1: np.sqrt(np.mean(np.square(img2 - img1)))
        DVARS_1_to_N = [RMS_diff(self.dw_volumes[:, :, :, grad_idx],
                                 self.dw_volumes[:, :, :, grad_idx - 1])
                        for grad_idx in range(1, self.dw_volumes.shape[-1])]

        DVARS = np.full(self.dw_volumes.shape[-1], np.nan)
        # dvars value at time point 0 is set to 0
        DVARS[0] = 0.0
        DVARS[1:] = DVARS_1_to_N

        return DVARS


    def update_axes_limits(self, num_gradients, num_voxels_shown):
        """Synchronizes the x-axis limits and updates the carpet image extents"""

        for a in self.axes_common_xaxis:
            a.set_xlim(-0.5, num_gradients - 0.5)
            a.set_ylim(auto=True)
            a.relim()
            a.autoscale_view()
        self.carpet_handle.set_extent(
            (-0.5, num_gradients - 0.5, -0.5, num_voxels_shown - 0.5))
        self.ax_carpet.set_xticks(np.linspace(0, num_gradients - 1, num=20, dtype='int'))


    def refresh_layer_order(self):
        """Ensures the expected order for layers"""

        for a in self.stats_axes:
            a.set_zorder(self.layer_order_stats)
        self.ax_carpet.set_zorder(self.layer_order_carpet)
        if not self.UI.zoomed_in:
            for a in self.fg_axes:
                a.set_zorder(self.layer_order_to_hide)


    def identify_unit(self, unit_id, counter):
        """
        Method to inform the user which unit (subject or scan) they are reviewing.
        """

        str_list = _prettify(unit_id)
        id_with_counter = '{}\n({}/{})'.format(str_list, counter + 1,
                                               self.num_units_to_review)
        if len(id_with_counter) < 1:
            return
        self.UI.annot_text = self.fig.text(cfg.position_annot_text[0],
                                           cfg.position_annot_text[1],
                                           id_with_counter, **cfg.annot_text_props)


    def cleanup(self):
        """Preparing for exit."""

        # save ratings before exiting
        self.save_ratings()
        for cid in (self.con_id_click, self.con_id_keybd, self.con_id_scroll):
            self.fig.canvas.mpl_disconnect(cid)
        plt.close('all')

        self.anim_loop.run_until_complete(self.anim_loop.shutdown_asyncgens())
        self.anim_loop.close()


def pis_map(diffn_img, index_low_b_val, index_high_b_val):
    """
    Produces the physically implausible signal (PIS) map [1].

    Parameters
    ----------
    diffn_img

    index_low_b_val : int
        index into the DWI identifying the image with lower b-value.
        Usually 0 referring to the b=0 (non-DW) image

    index_high_b_val : int
        index into the DWI identifying the image with higher b-value

    Returns
    -------
    pis_map : ndarray
        Binary 3d image identifying voxels to be PIS.

    References
    -----------
    1. D. Perrone et al. / NeuroImage 120 (2015) 441–455

    """

    pis = diffn_img[:, :, :, index_low_b_val] < diffn_img[:, :, :, index_high_b_val]

    return pis


def spatial_stats(diffn_img):
    """Computes volume-wise stats over space of diffusion data
        --> single vector over time.
    """

    num_gradients = diffn_img.shape[3]
    mean_signal = np.array(
        [np.nanmean(diffn_img[:, :, :, t]) for t in range(num_gradients)])
    stdev_signal = np.array(
        [np.nanstd(diffn_img[:, :, :, t]) for t in range(num_gradients)])

    return mean_signal, stdev_signal


def _rescale_over_gradients(matrix):
    """
    Voxel-wise normalization over gradients.

    Input: num_voxels x num_gradients
    """

    if matrix.shape[0] <= matrix.shape[1]:
        raise ValueError('Number of voxels is less than the number of gradients!! '
                         'Are you sure data is reshaped correctly?')

    min_ = matrix.min(axis=1)
    range_ = matrix.ptp(axis=1)  # ptp : peak to peak, max-min
    min_tile = np.tile(min_, (matrix.shape[1], 1)).T
    range_tile = np.tile(range_, (matrix.shape[1], 1)).T
    # avoiding any numerical difficulties
    range_tile[range_tile < np.finfo(np.float).eps] = 1.0

    normed = (matrix - min_tile) / range_tile

    del min_, range_, min_tile, range_tile

    return normed


def _within_frame_rescale(matrix):
    """Rescaling within a given grame"""

    min_colwise = matrix.min(axis=0)
    range_colwise = matrix.ptp(axis=0)  # ptp : peak to peak, max-min
    normed_matrix = (matrix - min_colwise) / range_colwise

    return normed_matrix


def get_parser():
    """Parser to specify arguments and their defaults."""

    parser = argparse.ArgumentParser(prog="visualqc_diffusion",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description='visualqc_diffusion_mri: rate quality of diffusion MR scan.')

    help_text_bids_dir = textwrap.dedent("""
    Absolute path to the root folder of the dataset formatted with the BIDS spec.
    See bids.neuroimaging.io for more info.

    E.g. ``--bids_dir /project/new_big_idea/ds042``
    \n""")

    help_text_user_dir = textwrap.dedent("""
    Absolute path to an input folder containing the MRI scan.
    Each subject will be queried after its ID in the metadata file,
    and is expected to have a file, uniquely specified ``--name_pattern``),
    in its own folder under this path ``--user_dir``.

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

    help_text_name_pattern = textwrap.dedent("""
    Specifies the regex to be used to search for the image to be reviewed.
    Typical options include:

        - ``'dwi.nii'``, when name is common across subjects
        - ``'*_preproc_*.nii'``, when filenames have additional info encoded (such as redundant subject ID as in BIDS format)
         - ``'func/sub*_dwi_*space-MNI152*_preproc.nii.gz'`` when you need to additional levels deeper (with a / in regex)
            or control different versions (atlas space) of the same type of file.

    Ensure the regex is *tight* enough to result in only one file for each ID in the id_list. You can do this by giving it a try in the shell and counting the number of results against the number of IDs in id_list. If you have more results than the IDs, then there are duplicates. You can use https://regex101.com to construct your pattern to tightly match your requirements. If multiple matches are found, the first one will be used.

    Make sure to use single quotes to avoid the shell globbing before visualqc receives it.

    Default: '{}'
    \n""".format(cfg.default_name_pattern))

    help_text_out_dir = textwrap.dedent("""
    Output folder to store the visualizations & ratings.
    Default: a new folder called ``{}`` will be created inside the input dir.
    \n""".format(cfg.default_out_dir_name))

    help_text_apply_preproc = textwrap.dedent("""
    Whether to apply basic preprocessing steps (detrend, slice timing correction etc), before building the carpet image.

    If the images are already preprocessed elsewhere, use this flag ``--apply_preproc``

    Default is NOT to apply any preprocessing (detrending etc) before showing images for review.
    \n""")

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
    This flag enables batch-generation of 3d surface visualizations, prior to starting any review and rating operations.

    This makes the switch from one subject to the next, even more seamless (saving few seconds :) ).

    Default: False (required visualizations are generated only on demand, which can take 5-10 seconds for each subject).
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
    Type of features to be employed in training the outlier detection method.  It could be one of
    'cortical' (aparc.stats: mean thickness and other geometrical features from each cortical label),
    'subcortical' (aseg.stats: volumes of several subcortical structures),
    or 'both' (using both aseg and aparc stats).

    Default: {}.
    \n""".format(cfg.diffusion_mri_features_OLD))

    help_text_disable_outlier_detection = textwrap.dedent("""
    This flag disables outlier detection and alerts altogether.
    \n""")

    help_text_delay_in_animation = textwrap.dedent("""
    Specifies the delay in animation of the display of two images (like in a GIF).

    Default: {} (units in seconds).
    \n""".format(cfg.delay_in_animation_diffusion_mri))

    in_out = parser.add_argument_group('Input and output', ' ')

    in_out.add_argument("-b", "--bids_dir", action="store", dest="bids_dir",
                        default=cfg.default_user_dir,
                        required=False, help=help_text_bids_dir)

    # in_out.add_argument("-u", "--user_dir", action="store", dest="user_dir",
    #                     default=cfg.default_user_dir,
    #                     required=False, help=help_text_user_dir)

    in_out.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        required=False, help=help_text_out_dir,
                        default=None)

    in_out.add_argument("-i", "--id_list", action="store", dest="id_list",
                        default=None, required=False, help=help_text_id_list)

    in_out.add_argument("-n", "--name_pattern", action="store", dest="name_pattern",
                        default=cfg.default_name_pattern, required=False,
                        help=help_text_name_pattern)

    preproc = parser.add_argument_group('Preprocessing',
                                        'options related to preprocessing before review')

    preproc.add_argument("-ap", "--apply_preproc", action="store_true",
                         dest="apply_preproc",
                         required=False, help=help_text_apply_preproc)

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
                          default=cfg.diffusion_mri_features_OLD, required=False,
                          help=help_text_outlier_feat_types)

    # outliers.add_argument("-old", "--disable_outlier_detection", action="store_true",
    #                       dest="disable_outlier_detection",
    #                       required=False, help=help_text_disable_outlier_detection)

    # TODO re-enable it when OLD is ready for DWI
    outliers.add_argument("-old", "--disable_outlier_detection", action="store_false",
                          dest="disable_outlier_detection",
                          required=False, help=help_text_disable_outlier_detection)

    layout = parser.add_argument_group('Layout options',
                                       'Slice layout arragement when zooming in on a time point,\n'
                                       ' or show to the std. dev plot.')
    layout.add_argument("-w", "--views", action="store", dest="views",
                        default=cfg.default_views, required=False, nargs='+',
                        help=help_text_views)

    layout.add_argument("-s", "--num_slices", action="store", dest="num_slices",
                        default=cfg.default_num_slices, required=False,
                        help=help_text_num_slices)

    layout.add_argument("-r", "--num_rows", action="store", dest="num_rows",
                        default=cfg.default_num_rows, required=False,
                        help=help_text_num_rows)

    _wf_descr = 'Options related to workflow e.g. to pre-compute resource-intensive features,  ' \
                'and pre-generate all the visualizations required before initiating the review.'
    wf_args = parser.add_argument_group('Workflow', _wf_descr)
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

    vis_type = 'diffusion_mri'
    type_of_features = 'diffusion_mri'

    if user_args.bids_dir is None:
        raise ValueError('Invalid args: Only BIDS format input is supported at present.')

    in_dir, in_dir_type = check_bids_dir(user_args.bids_dir)
    id_list = None
    name_pattern = None
    images_for_id = None

    # elif user_args.bids_dir is None and user_args.user_dir is not None:
    #     name_pattern = user_args.name_pattern
    #     in_dir = realpath(user_args.user_dir)
    #     in_dir_type = 'generic'
    #     id_list, images_for_id = check_id_list_with_regex(user_args.id_list, in_dir, name_pattern)

    out_dir = check_out_dir(user_args.out_dir, in_dir)
    apply_preproc = user_args.apply_preproc
    delay_in_animation = check_time(user_args.delay_in_animation, var_name='Delay')

    views = check_views(user_args.views)
    num_slices_per_view, num_rows_per_view = check_finite_int(user_args.num_slices,
                                                              user_args.num_rows)

    outlier_method, outlier_fraction, \
    outlier_feat_types, disable_outlier_detection = check_outlier_params(
        user_args.outlier_method, user_args.outlier_fraction,
        user_args.outlier_feat_types, user_args.disable_outlier_detection,
        id_list, vis_type, type_of_features)

    wf = DiffusionRatingWorkflow(in_dir, out_dir,
                                 id_list=id_list,
                                 images_for_id=images_for_id,
                                 issue_list=cfg.diffusion_mri_default_issue_list,
                                 name_pattern=name_pattern, in_dir_type=in_dir_type,
                                 apply_preproc=apply_preproc,
                                 delay_in_animation=delay_in_animation,
                                 outlier_method=outlier_method,
                                 outlier_fraction=outlier_fraction,
                                 outlier_feat_types=outlier_feat_types,
                                 disable_outlier_detection=disable_outlier_detection,
                                 prepare_first=user_args.prepare_first, vis_type=vis_type,
                                 views=views, num_slices_per_view=num_slices_per_view,
                                 num_rows_per_view=num_rows_per_view)

    return wf


def cli_run():
    """Main entry point."""

    print('\nDiffusion MRI module')
    from visualqc.utils import run_common_utils_before_starting
    run_common_utils_before_starting()

    wf = make_workflow_from_user_options()
    wf.run()

    return


if __name__ == '__main__':
    # disabling all not severe warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        cli_run()
