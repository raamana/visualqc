"""

Module to define interface, workflow and CLI for the review of functional MRI data.

"""
import argparse
import sys
import textwrap
import warnings
from abc import ABC
from textwrap import wrap

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons
from mrivis.utils import crop_image
from os.path import basename, join as pjoin, realpath, splitext

from visualqc import config as cfg
from visualqc.image_utils import mask_image
from visualqc.readers import func_mri_traverse_bids
from visualqc.t1_mri import T1MriInterface
from visualqc.utils import check_bids_dir, check_finite_int, check_id_list_with_regex, \
    check_image_is_4d, check_out_dir, check_outlier_params, check_views, get_axis, \
    pick_slices, scale_0to1
from visualqc.workflows import BaseWorkflowVisualQC


def _unbidsify(filename, max_width = 18):
    """Returns a easily displayable and readable multiline string"""

    parts = [s.replace('-', ' ') for s in filename.split('_')]
    fixed_width = list()
    for p in parts:
        if len(p) > max_width:
            # indenting by two spaace
            fixed_width.extend([' -'+s for s in wrap(p,max_width-2)])
        else:
            fixed_width.append(p)

    return  '\n'.join(fixed_width)

_z_score = lambda x: (x - np.mean(x)) / np.std(x)


class FunctionalMRIInterface(T1MriInterface):
    """Interface for the review of fMRI images."""


    def __init__(self,
                 fig,
                 axes,
                 issue_list=cfg.func_mri_default_issue_list,
                 next_button_callback=None,
                 quit_button_callback=None,
                 right_arrow_callback=None,
                 left_arrow_callback=None,
                 zoom_in_callback=None,
                 zoom_out_callback=None,
                 right_click_callback=None,
                 show_stdev_callback=None,
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
        self.right_click_callback = right_click_callback
        self.show_stdev_callback = show_stdev_callback

        self.add_checkboxes()

        # this list of artists to be populated later
        # makes to handy to clean them all
        self.data_handles = list()


    def add_checkboxes(self):
        """
        Checkboxes offer the ability to select multiple tags such as Motion, Ghosting Aliasing etc,
            instead of one from a list of mutual exclusive rating options (such as Good, Bad, Error etc).

        """

        ax_checkbox = plt.axes(cfg.position_checkbox, facecolor=cfg.color_rating_axis)
        # initially de-activating all
        actives = [False] * len(self.issue_list)
        self.checkbox = CheckButtons(ax_checkbox, labels=self.issue_list, actives=actives)
        self.checkbox.on_clicked(self.save_issues)
        for txt_lbl in self.checkbox.labels:
            txt_lbl.set(**cfg.checkbox_font_properties)

        for rect in self.checkbox.rectangles:
            rect.set_width(cfg.checkbox_rect_width)
            rect.set_height(cfg.checkbox_rect_height)

        # lines is a list of n crosses, each cross (x) defined by a tuple of lines
        for x_line1, x_line2 in self.checkbox.lines:
            x_line1.set_color(cfg.checkbox_cross_color)
            x_line2.set_color(cfg.checkbox_cross_color)

        self._index_pass = self.issue_list.index(cfg.func_mri_pass_indicator)


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

        # if event occurs in non-data areas, do nothing
        if event.inaxes in [self.checkbox.ax, self.text_box.ax,
                            self.bt_next.ax, self.bt_quit.ax]:
            return

        if self.zoomed_in:
            # include all the non-data axes here (so they wont be zoomed-in)
            if event.inaxes not in [self.checkbox.ax, self.text_box.ax,
                                    self.bt_next.ax, self.bt_quit.ax]:
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
        elif event.dblclick and event.inaxes is not None:
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
        elif key_pressed in ['left', 'down' ]:
            self.left_arrow_callback()
        elif key_pressed in [' ', 'space']:
            self.next_button_callback()
        elif key_pressed in ['ctrl+q', 'q+ctrl']:
            self.quit_button_callback()
        elif key_pressed in ['alt+s', 's+alt']:
            self.show_stdev_callback()
        else:
            if key_pressed in cfg.abbreviation_func_mri_default_issue_list:
                checked_label = cfg.abbreviation_func_mri_default_issue_list[key_pressed]
                # TODO if user chooses a different set of names, keyboard shortcuts might not work
                self.checkbox.set_active(self.issue_list.index(checked_label))
            else:
                pass

        self.fig.canvas.draw_idle()

    def reset_figure(self):
        """Resets the figure to prepare it for display of next subject."""

        self.zoom_out_callback(None)
        self.restore_axis()
        self.clear_data()
        self.clear_checkboxes()
        self.clear_notes_annot()


class FmriRatingWorkflow(BaseWorkflowVisualQC, ABC):
    """
    Rating workflow for BOLD fMRI.
    """


    def __init__(self,
                 in_dir,
                 out_dir,
                 drop_start=1,
                 drop_end=None,
                 no_preproc=False,
                 id_list=None,
                 name_pattern=None,
                 images_for_id=None,
                 issue_list=cfg.func_mri_default_issue_list,
                 in_dir_type='BIDS',
                 outlier_method=cfg.default_outlier_detection_method,
                 outlier_fraction=cfg.default_outlier_fraction,
                 outlier_feat_types=cfg.func_mri_features_OLD,
                 disable_outlier_detection=True,
                 prepare_first=False,
                 vis_type=None,
                 views=cfg.default_views_fmri,
                 num_slices_per_view=cfg.default_num_slices_fmri,
                 num_rows_per_view=cfg.default_num_rows_fmri):
        """
        Constructor.

        Parameters
        ----------
        in_dir : path
            must be a path to BIDS directory

        drop_start : int
            Number of frames to drop at the beginning of the time series.

        no_preproc : bool
            Whether to apply basic preprocessing steps (detrend, slice timing correction etc)
                before building the carpet image.
                If the images are already preprocessed elsewhere, disable this with no_preproc=True
            Default : True , apply to basic preprocessing before display for review.


        """

        if id_list is None and 'BIDS' in in_dir_type:
            id_list = pjoin(in_dir, 'participants.tsv')

        super().__init__(id_list, in_dir, out_dir,
                         outlier_method, outlier_fraction,
                         outlier_feat_types, disable_outlier_detection)

        # proper checks
        self.drop_start = drop_start
        self.drop_end = drop_end
        if self.drop_start is None:
            self.drop_start = 0
        if self.drop_end is None:
            self.drop_end = 0

        # basic cleaning before display
        # whether to remove and detrend before making carpet plot
        self.no_preproc = no_preproc

        self.vis_type = vis_type
        self.issue_list = issue_list
        self.in_dir_type = in_dir_type
        self.name_pattern = name_pattern
        self.images_for_id = images_for_id
        self.expt_id = 'rate_fmri'
        self.suffix = self.expt_id
        self.current_alert_msg = None
        self.prepare_first = prepare_first

        #
        self.current_time_point = 0

        self.init_layout(views, num_rows_per_view, num_slices_per_view)
        self.init_getters()

        self.__module_type__ = 'functional_mri'


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

        from visualqc.features import functional_mri_features
        self.feature_extractor = functional_mri_features

        if 'BIDS' in self.in_dir_type.upper():
            from bids import BIDSLayout
            self.bids_layout = BIDSLayout(self.in_dir)
            self.units = func_mri_traverse_bids(self.bids_layout,
                                                **cfg.func_mri_BIDS_filters)

            if self.units is None or len(self.units) < 1:
                print('No valid subjects are found! Exiting.\n'
                      'Double check the format and integrity of the dataset '
                      'if this is unexpected.')
                import sys
                sys.exit(1)

            # file name of each BOLD scan is the unique identifier,
            #   as it essentially contains all the key info.
            self.unit_by_id = {basename(sub_data['image']): sub_data
                               for _, sub_data in self.units.items()}
            self.id_list = list(self.unit_by_id.keys())

        elif 'GENERIC' in self.in_dir_type.upper():
            if self.id_list is None or self.images_for_id is None:
                raise ValueError('id_list or images_for_id can not be None '
                                 'for generic in_dir')
            self.unit_by_id = self.images_for_id.copy()
        else:
            raise NotImplementedError('Only two formats are supported: BIDS and ' \
                                      'GENERIC with regex spec for filenames')


    def open_figure(self):
        """Creates the master figure to show everything in."""

        # number of stats to be overlaid on top of carpet plot
        self.num_stats = 3
        self.figsize = cfg.default_review_figsize

        # empty/dummy data for placeholding
        empty_image = np.full((200, 200), 0.0)
        empty_vec = np.full((200, 1), 0.0)
        time_points = list(range(200))

        # overlay order -- larger appears on top of smaller
        self.layer_order_carpet = 1
        self.layer_order_stats = 2
        self.layer_order_zoomedin = 3
        self.layer_order_to_hide = -1
        self.total_num_layers = 3

        plt.style.use('dark_background')

        # 1. main carpet, in the background
        self.fig, self.ax_carpet = plt.subplots(1, 1, figsize=self.figsize)
        self.fig.canvas.set_window_title('VisualQC Functional MRI :'
                                         ' {}'.format(self.in_dir))

        self.ax_carpet.set_zorder(self.layer_order_carpet)
        #   vmin/vmax are controlled, because we rescale all to [0, 1]
        self.imshow_params_carpet = dict(interpolation='none', aspect='auto',
                                         origin='lower', cmap='gray',
                                         vmin=0.0, vmax=1.0)

        self.ax_carpet.yaxis.set_visible(False)
        self.ax_carpet.set_xlabel('time point')
        self.carpet_handle = self.ax_carpet.imshow(empty_image,
                                                   **self.imshow_params_carpet)
        self.ax_carpet.set_frame_on(False)
        self.ax_carpet.set_ylim(auto=True)

        # 2. temporal traces of image stats
        tmp_mat = self.fig.subplots(self.num_stats, 1, sharex=True)
        self.stats_axes = tmp_mat.flatten()
        self.stats_handles = [None] * len(self.stats_axes)

        stats = [(empty_vec, 'mean BOLD', 'cyan'),
                 (empty_vec, 'SD BOLD', 'xkcd:orange red'),
                 (empty_vec, 'DVARS', 'xkcd:mustard')]
        for ix, (ax, (stat, label, color)) in enumerate(zip(self.stats_axes, stats)):
            (vh,) = ax.plot(time_points, stat, color=color)
            self.stats_handles[ix] = vh
            vh.set_linewidth(cfg.linewidth_stats_fmri)
            vh.set_linestyle(cfg.linestyle_stats_fmri)
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
                                           gridspec_kw=dict(wspace=0.0, hspace=0.0))
        self.fg_axes = matrix_handles.flatten()

        # vmin/vmax are controlled, because we rescale all to [0, 1]
        # TODO aspect auto here covers carpet so the user can focus on the frame,
        #   not accurately representing geometry underneath
        self.imshow_params_zoomed = dict(interpolation='none', aspect='auto',
                                         rasterized=True, origin='lower',
                                         cmap='gray', vmin=0.0, vmax=1.0)

        # images to be shown in the forground
        self.images_fg = [None] * len(self.fg_axes)
        for ix, ax in enumerate(self.fg_axes):
            ax.axis('off')
            self.images_fg[ix] = ax.imshow(empty_image, **self.imshow_params_zoomed)
            ax.set_visible(False)
            ax.set_zorder(self.layer_order_zoomedin)

        self.foreground_h = self.fig.text(cfg.position_zoomed_time_point[0],
                                          cfg.position_zoomed_time_point[1],
                                          ' ', **cfg.annot_time_point)
        self.foreground_h.set_visible(False)

        # leaving some space on the right for review elements
        plt.subplots_adjust(**cfg.review_area)
        plt.show(block=False)


    def add_UI(self):
        """Adds the review UI with defaults"""

        self.UI = FunctionalMRIInterface(self.fig, self.ax_carpet, self.issue_list,
                                         next_button_callback=self.next,
                                         quit_button_callback=self.quit,
                                         right_click_callback=self.zoom_in_on_time_point,
                                         right_arrow_callback=self.show_next_time_point,
                                         left_arrow_callback=self.show_prev_time_point,
                                         zoom_in_callback=self.zoom_in_on_time_point,
                                         zoom_out_callback=self.zoom_out_callback,
                                         show_stdev_callback=self.show_stdev,
                                         axes_to_zoom=self.fg_axes,
                                         total_num_layers=self.total_num_layers)

        # connecting callbacks
        self.con_id_click = self.fig.canvas.mpl_connect('button_press_event',
                                                        self.UI.on_mouse)
        self.con_id_keybd = self.fig.canvas.mpl_connect('key_press_event',
                                                        self.UI.on_keyboard)

        # TODO implement the scrolling movement to scroll in time
        # con_id_scroll = self.fig.canvas.mpl_connect('scroll_event', self.UI.on_scroll)

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
        params_path = self.unit_by_id[unit_id]['params']
        try:
            hdr = nib.load(img_path)
            self.hdr_this_unit = nib.as_closest_canonical(hdr)
            self.img_this_unit_raw = self.hdr_this_unit.get_data()
        except Exception as exc:
            print(exc)
            print('Unable to read image at \n\t{}'.format(img_path))
            skip_subject = True
        else:
            check_image_is_4d(self.img_this_unit_raw)
            self.TR_this_unit = self.hdr_this_unit.header.get_zooms()[-1]

            skip_subject = False
            if np.count_nonzero(self.img_this_unit_raw) == 0:
                skip_subject = True
                print('Functional image is empty!')

        return skip_subject


    def display_unit(self):
        """Adds multi-layered composite."""

        # if frames are to be dropped
        end_frame = self.img_this_unit_raw.shape[3] - self.drop_end
        self.img_this_unit = self.img_this_unit_raw[:, :, :, self.drop_start:end_frame]
        # TODO show median signal instead of mean - or option for both?
        self.stdev_this_unit, self.mean_this_unit = temporal_stats(self.img_this_unit)

        # TODO should we perform head motion correction before any display at all?
        # TODO what about slice timing correction?

        num_voxels = np.prod(self.img_this_unit.shape[0:3])
        num_time_points = self.img_this_unit.shape[3]
        time_points = list(range(num_time_points))

        # 1. compute necessary stats/composites
        carpet, mean_signal_spatial, stdev_signal_spatial, dvars = self.compute_stats()

        # 2. display/update the data
        self.carpet_handle.set_data(carpet)
        self.stats_handles[0].set_data(time_points, mean_signal_spatial)
        self.stats_handles[1].set_data(time_points, stdev_signal_spatial)
        # not displaying DVARS for t=0, as its always 0
        self.stats_handles[2].set_data(time_points[1:], dvars[1:])

        # 3. updating axes limits and views
        self.update_axes_limits(num_time_points, carpet.shape[0])
        self.refresh_layer_order()

        # clean up
        del carpet, mean_signal_spatial, stdev_signal_spatial, dvars

        print()


    def make_carpet(self, mask, row_order=None):
        """
        Makes the carpet image


        Parameters
        ----------
        func_img

        Returns
        -------

        """

        carpet = self.img_this_unit.reshape(-1, self.img_this_unit.shape[3])
        if not self.no_preproc:
            from nilearn.signal import clean
            # notice the transpose before clean and after
            carpet = clean(carpet.T, t_r=self.TR_this_unit,
                           detrend=True, standardize=False).T

        # Removes voxels with low variance
        cropped_carpet = np.delete(carpet, np.where(mask.flatten() == 0), axis=0)
        normed_carpet = _rescale_over_time(cropped_carpet)

        del carpet, cropped_carpet

        # TODO blurring within tissue segmentations and other deeper subcortical areas
        # TODO reorder rows either using anatomical seg, or using clustering

        # dropping alternating voxels if it gets too big
        # to save on memory and avoid losing signal
        if normed_carpet.shape[1] > 600:
            print('Too many frames (n={}) to display: dropping alternating frames'.format(normed_carpet.shape[1]))
            normed_carpet = normed_carpet[:, ::2]

        return normed_carpet


    def zoom_in_on_time_point(self, event):
        """Brings up selected time point"""

        if event.x is None:
            return
        # computing x in axes data coordinates myself, to avoid overlaps with other axes
        # retrieving the latest transform after to ensure its accurate at click time
        x_in_carpet, _y = self._event_location_in_axis(event, self.ax_carpet)
        # clipping it to [0, T]
        self.current_time_point = max(0,
                                      min(self.img_this_unit.shape[3],
                                          int(round(x_in_carpet))))
        self.show_timepoint(self.current_time_point)


    def show_next_time_point(self):

        if self.current_time_point == self.img_this_unit.shape[3] - 1:
            return  # do nothing

        self.current_time_point = min(self.img_this_unit.shape[3] - 1,
                                      self.current_time_point + 1)
        self.show_timepoint(self.current_time_point)


    def show_prev_time_point(self):

        if self.current_time_point == 0:
            return  # do nothing

        self.current_time_point = max(self.current_time_point - 1, 0)
        self.show_timepoint(self.current_time_point)


    def zoom_out_callback(self, event):
        """Hides the zoomed-in axes (showing frame)."""

        for ax in self.fg_axes:
            ax.set(visible=False, zorder=self.layer_order_to_hide)
        self.foreground_h.set_visible(False)
        self.UI.zoomed_in = False

    @staticmethod
    def _event_location_in_axis(event, axis):
        """returns (x_in_axis, y_in_axis)"""

        # display pixels to axis coords
        return axis.transData.inverted().transform_point((event.x, event.y))


    def show_timepoint(self, time_pt):
        """Exhibits a selected timepoint on top of stats/carpet"""

        if time_pt < 0 or time_pt >= self.img_this_unit.shape[3]:
            print('Requested time point outside '
                  'range [0, {}]'.format(self.img_this_unit.shape[3]))
            return

        # print('Time point zoomed-in {}'.format(time_pt))
        image3d = np.squeeze(self.img_this_unit[:, :, :, time_pt])
        self.attach_image_to_foreground_axes(image3d)
        self._identify_foreground('zoomed-in time point {}'.format(time_pt))
        # this state flag in important
        self.UI.zoomed_in = True


    def _identify_foreground(self, text):
        """show the time point"""

        self.foreground_h.set_text(text)
        self.foreground_h.set_visible(True)


    def show_stdev(self):
        """Shows the image of temporal std. dev"""

        self.attach_image_to_foreground_axes(self.stdev_this_unit, cfg.colormap_stdev_fmri)
        self._identify_foreground('Std. dev over time')
        self.UI.zoomed_in = True


    def attach_image_to_foreground_axes(self, image3d, cmap='gray'):
        """Attaches a given image to the foreground axes and bring it forth"""

        image3d = crop_image(image3d, self.padding)
        image3d = scale_0to1(image3d)
        slices = pick_slices(image3d, self.views, self.num_slices_per_view)
        for ax_index, (dim_index, slice_index) in enumerate(slices):
            slice_data = get_axis(image3d, dim_index, slice_index)
            self.images_fg[ax_index].set(data=slice_data, cmap=cmap)
        for ax in self.fg_axes:
            ax.set(visible=True, zorder=self.layer_order_zoomedin)


    def compute_stats(self):
        """Computes the necessary stats to be displayed."""

        mean_img_temporal, stdev_img_temporal = temporal_stats(self.img_this_unit)
        mean_signal_spatial, stdev_signal_spatial = spatial_stats(self.img_this_unit)
        dvars = compute_DVARS(self.img_this_unit)

        for stat, sname in zip((mean_signal_spatial, stdev_signal_spatial, dvars),
                               ('mean_signal_spatial', 'stdev_signal_spatial', 'dvars')):
            if len(stat) != self.img_this_unit.shape[3]:
                raise ValueError('ERROR: lengths of different stats do not match!')
            if any(np.isnan(stat)):
                raise ValueError('ERROR: invalid values in stat : {}'.format(sname))

        mask = mask_image(mean_img_temporal, update_factor=0.9, init_percentile=5)
        carpet = self.make_carpet(mask)

        return carpet, mean_signal_spatial, stdev_signal_spatial, dvars


    def update_axes_limits(self, num_time_points, num_voxels_shown):
        """Synchronizes the x-axis limits and updates the carpet image extents"""

        for a in list(self.stats_axes) + [self.ax_carpet, ]:
            a.set_xlim(-0.5, num_time_points - 0.5)
            a.set_ylim(auto=True)
            a.relim()
            a.autoscale_view()
        self.carpet_handle.set_extent(
            (-0.5, num_time_points - 0.5, -0.5, num_voxels_shown - 0.5))
        self.ax_carpet.set_xticks(np.linspace(0, num_time_points-1,
                                              num=20, dtype='int'))


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

        str_list = _unbidsify(unit_id)
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

        self.fig.canvas.mpl_disconnect(self.con_id_click)
        self.fig.canvas.mpl_disconnect(self.con_id_keybd)
        plt.close('all')


def compute_DVARS(func_img, mean_img=None, mask=None, apply_mask=False):
    """Computes the DVARS for a given fMRI image."""

    if mean_img is None:
        mean_img = np.mean(func_img, axis=3)

    if apply_mask:
        if mask is None:
            mask = mask_image(mean_img)
        mean_img[np.logical_not(mask)] = 0.0

    num_time_points = func_img.shape[3]

    RMS_diff = lambda img2, img1: np.sqrt(np.mean(np.square(img2 - img1)))
    DVARS_1_to_N = [RMS_diff(func_img[:, :, :, t], func_img[:, :, :, t - 1]) for t in
                    range(1, num_time_points)]

    DVARS = np.full(num_time_points, np.nan)
    # dvars value at time point 0 is set to 0
    DVARS[0] = 0.0
    DVARS[1:] = DVARS_1_to_N

    return DVARS


def temporal_stats(func_img):
    """Computes voxel-wise temporal average of functional data --> single volume over space."""

    mean_img = np.mean(func_img, axis=3)
    sd_img = np.std(func_img, axis=3)

    return mean_img, sd_img


def spatial_stats(func_img):
    """Computes volume-wise spatial average of functional data --> single vector over time."""

    num_time_points = func_img.shape[3]
    mean_signal = np.array(
        [np.nanmean(func_img[:, :, :, t]) for t in range(num_time_points)])
    stdev_signal = np.array(
        [np.nanstd(func_img[:, :, :, t]) for t in range(num_time_points)])

    return mean_signal, stdev_signal


def _rescale_over_time(matrix):
    """
    Voxel-wise normalization over time.

    Input: num_voxels x num_time_points
    """

    if matrix.shape[0] <= matrix.shape[1]:
        raise ValueError('Number of voxels is less than the number of time points!! '
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

    parser = argparse.ArgumentParser(prog="visualqc_func_mri",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description='visualqc_func_mri: rate quality of functional MR scan.')

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

        - ``'bold.nii'``, when name is common across subjects
        - ``'*_preproc_*.nii'``, when filenames have additional info encoded (such as redundant subject ID as in BIDS format)
         - ``'func/sub*_bold_*space-MNI152*_preproc.nii.gz'`` when you need to additional levels deeper (with a / in regex)
            or control different versions (atlas space) of the same type of file.

    Ensure the regex is *tight* enough to result in only one file for each ID in the id_list. You can do this by giving it a try in the shell and counting the number of results against the number of IDs in id_list. If you have more results than the IDs, then there are duplicates. You can use https://regex101.com to construct your pattern to tightly match your requirements. If multiple matches are found, the first one will be used.

    Make sure to use single quotes to avoid the shell globbing before visualqc receives it.

    Default: '{}'
    \n""".format(cfg.default_name_pattern))

    help_text_out_dir = textwrap.dedent("""
    Output folder to store the visualizations & ratings.
    Default: a new folder called ``{}`` will be created inside the input folder
    \n""".format(cfg.default_out_dir_name))

    help_text_no_preproc = textwrap.dedent("""
    Whether to apply basic preprocessing steps (detrend, slice timing correction etc), before building the carpet image.

    If the images are already preprocessed elsewhere, use this flag ``--no_preproc``

    Default is to apply minimal preprocessing (detrending etc) before showing images for review.
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
    \n""".format(cfg.func_mri_features_OLD))

    help_text_disable_outlier_detection = textwrap.dedent("""
    This flag disables outlier detection and alerts altogether.
    \n""")

    in_out = parser.add_argument_group('Input and output', ' ')

    in_out.add_argument("-b", "--bids_dir", action="store", dest="bids_dir",
                        default=cfg.default_user_dir,
                        required=False, help=help_text_bids_dir)

    in_out.add_argument("-u", "--user_dir", action="store", dest="user_dir",
                        default=cfg.default_user_dir,
                        required=False, help=help_text_user_dir)

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

    preproc.add_argument("-np", "--no_preproc", action="store_true", dest="no_preproc",
                          required=False, help=help_text_no_preproc)

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
                          default=cfg.func_mri_features_OLD, required=False,
                          help=help_text_outlier_feat_types)

    outliers.add_argument("-old", "--disable_outlier_detection", action="store_true",
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

    wf_args = parser.add_argument_group('Workflow', 'Options related to workflow '
                                                    'e.g. to pre-compute resource-intensive features, '
                                                    'and pre-generate all the visualizations required '
                                                    'before bringin up the review interface.')
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

    vis_type = 'func_mri'
    type_of_features = 'func_mri'

    if user_args.bids_dir is not None and user_args.user_dir is None:
        in_dir, in_dir_type = check_bids_dir(user_args.bids_dir)
        id_list = None
        name_pattern = None
        images_for_id = None
    elif user_args.bids_dir is None and user_args.user_dir is not None:
        name_pattern = user_args.name_pattern
        in_dir = realpath(user_args.user_dir)
        in_dir_type = 'generic'
        id_list, images_for_id = check_id_list_with_regex(user_args.id_list, in_dir, name_pattern)
    else:
        raise ValueError('Invalid args: specify only one of bids_dir or user_dir, not both.')

    out_dir = check_out_dir(user_args.out_dir, in_dir)
    no_preproc = user_args.no_preproc

    views = check_views(user_args.views)
    num_slices_per_view, num_rows_per_view = check_finite_int(user_args.num_slices,
                                                              user_args.num_rows)

    outlier_method, outlier_fraction, \
    outlier_feat_types, disable_outlier_detection = check_outlier_params(
        user_args.outlier_method, user_args.outlier_fraction,
        user_args.outlier_feat_types, user_args.disable_outlier_detection,
        id_list, vis_type, type_of_features)

    wf = FmriRatingWorkflow(in_dir, out_dir,
                            id_list=id_list,
                            images_for_id=images_for_id,
                            issue_list=cfg.func_mri_default_issue_list,
                            name_pattern=name_pattern, in_dir_type=in_dir_type,
                            no_preproc=no_preproc,
                            outlier_method=outlier_method, outlier_fraction=outlier_fraction,
                            outlier_feat_types=outlier_feat_types,
                            disable_outlier_detection=disable_outlier_detection,
                            prepare_first=user_args.prepare_first, vis_type=vis_type,
                            views=views, num_slices_per_view=num_slices_per_view,
                            num_rows_per_view=num_rows_per_view)

    return wf


def cli_run():
    """Main entry point."""

    print('\nFunctional MRI module')
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
