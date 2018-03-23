"""

Module to define interface, workflow and CLI for the review of functional MRI data.

"""


from visualqc.t1_mri import T1MriInterface
import argparse
import sys
import textwrap
import warnings
from os import makedirs
from os.path import join as pjoin, exists as pexists, splitext, realpath, basename
from shutil import copyfile
from abc import ABC
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons
from mrivis.utils import crop_image
import nibabel as nib
from visualqc import config as cfg
from visualqc.interfaces import BaseReviewInterface
from visualqc.utils import check_image_is_4d,  scale_0to1, pick_slices, get_axis
from visualqc.workflows import BaseWorkflowVisualQC
from visualqc.readers import traverse_bids
from visualqc.image_utils import mask_image

_unbidsify = lambda string : '\n'.join([ s.replace('-',' ') for s in string.split('_') ])
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
                 total_num_layers=2):
        """Constructor"""

        super().__init__(fig, axes, issue_list, next_button_callback, quit_button_callback)
        self.issue_list = issue_list

        self.prev_axis = None
        self.prev_ax_pos = None
        self.prev_ax_zorder = None
        self.prev_visible = False
        self.zoomed_in = False
        self.total_num_layers = total_num_layers

        self.next_button_callback = next_button_callback
        self.quit_button_callback = quit_button_callback
        self.zoom_in_callback = zoom_in_callback
        self.zoom_out_callback = zoom_out_callback
        self.right_arrow_callback = right_arrow_callback
        self.left_arrow_callback = left_arrow_callback
        self.right_click_callback = right_click_callback

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
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for rect in self.checkbox.rectangles:
            rect.set_width(cfg.checkbox_rect_width)
            rect.set_height(cfg.checkbox_rect_height)

        # lines is a list of n crosses, each cross (x) defined by a tuple of lines
        for x_line1, x_line2 in self.checkbox.lines:
            x_line1.set_color(cfg.checkbox_cross_color)
            x_line2.set_color(cfg.checkbox_cross_color)

        self._index_pass = cfg.func_mri_default_issue_list.index(cfg.func_mri_pass_indicator)

    def on_mouse(self, event):
        """Callback for mouse events."""

        if self.prev_axis is not None:
            # include all the non-data axes here (so they wont be zoomed-in)
            if event.inaxes not in [self.checkbox.ax, self.text_box.ax,
                                    self.bt_next.ax, self.bt_quit.ax]:
                self.zoom_out_callback(event)

        # right click ignored
        if event.button in [3]:
            self.right_click_callback(event)
        # double click to zoom in to any axis
        elif event.dblclick and event.inaxes is not None and \
            event.inaxes not in [self.checkbox.ax, self.text_box.ax,
                                 self.bt_next.ax, self.bt_quit.ax]:
            self.zoom_in_callback(event)
        else:
            pass

    def on_keyboard(self, key_in):
        """Callback to handle keyboard shortcuts to rate and advance."""

        # ignore keyboard key_in when mouse within Notes textbox
        if key_in.inaxes == self.text_box.ax or key_in.key is None:
            return

        key_pressed = key_in.key.lower()
        # print(key_pressed)
        if key_pressed in ['right', ]:
            self.right_arrow_callback()
        elif key_pressed in ['left', ]:
            self.left_arrow_callback()
        elif key_pressed in [' ', 'space']:
            self.next_button_callback()
        elif key_pressed in ['ctrl+q', 'q+ctrl']:
            self.quit_button_callback()
        else:
            if key_pressed in cfg.abbreviation_func_mri_default_issue_list:
                checked_label = cfg.abbreviation_func_mri_default_issue_list[key_pressed]
                self.checkbox.set_active(cfg.func_mri_default_issue_list.index(checked_label))
            else:
                pass

class FmriRatingWorkflow(BaseWorkflowVisualQC, ABC):
    """
    Rating workflow for BOLD fMRI.
    """


    def __init__(self,
                 in_dir,
                 out_dir,
                 drop_start=1,
                 drop_end=None,
                 id_list=None,
                 issue_list=cfg.func_mri_default_issue_list,
                 in_dir_type='BIDS',
                 outlier_method=cfg.default_outlier_detection_method,
                 outlier_fraction=cfg.default_outlier_fraction,
                 outlier_feat_types=cfg.func_outlier_features,
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


        """

        if id_list is None and in_dir_type=='BIDS':
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

        self.vis_type = vis_type
        self.issue_list = issue_list
        self.in_dir_type = in_dir_type
        self.expt_id = 'rate_fmri'
        self.suffix = self.expt_id
        self.current_alert_msg = None
        self.prepare_first = prepare_first

        #
        self.current_time_point = 0

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
        self.num_rows = len(self.views)*self.num_rows_per_view
        self.num_cols = int((len(self.views) * self.num_slices_per_view) / self.num_rows)
        self.padding = padding

    def init_getters(self):
        """Initializes the getters methods for input paths and feature readers."""

        from visualqc.features import functional_mri_features
        self.feature_extractor = functional_mri_features

        from bids.grabbids import BIDSLayout
        if self.in_dir_type.upper() == 'BIDS':
            self.bids_layout = BIDSLayout(self.in_dir)
            self.field_names, self.units = traverse_bids(self.bids_layout, **cfg.func_mri_BIDS_filters)

            # file name of each BOLD scan is the unique identifier, as it essentially contains all the key info.
            self.unit_by_id = { splitext(basename(fpath))[0] : realpath(fpath) for _, fpath in self.units}
            self.id_list = list(self.unit_by_id.keys())
        else:
            raise NotImplementedError('Only BIDS format is supported for input directory at the moment!')

    def open_figure(self):
        """Creates the master figure to show everything in."""

        # number of stats to be overlaid on top of carpet plot
        self.num_stats = 3
        self.figsize = cfg.default_review_figsize

        # empty/dummy data for placeholding
        empty_image = np.full((200,200), 0.0)
        empty_vec = np.full((200, 1), 0.0)
        time_points = list(range(200))

        # overlay order -- larger appears on top of smaller
        self.layer_order_carpet   = 1
        self.layer_order_stats    = 2
        self.layer_order_zoomedin = 3

        plt.style.use('dark_background')

        # 1. main carpet, in the background
        self.fig, self.ax_carpet = plt.subplots(1, 1, figsize=self.figsize)
        self.ax_carpet.set_zorder(self.layer_order_carpet)
        #   vmin/vmax are controlled, because we rescale all to [0, 1]
        self.imshow_params_carpet = dict(interpolation='none', aspect='auto',
                              origin='lower', cmap='gray', vmin=0.0, vmax=1.0)

        self.ax_carpet.yaxis.set_visible(False)
        self.ax_carpet.set_xlabel('time point')
        self.carpet_handle = self.ax_carpet.imshow(empty_image, **self.imshow_params_carpet)
        self.ax_carpet.set_frame_on(False)
        self.ax_carpet.set_ylim(auto=True)

        # 2. temporal traces of image stats
        tmp_mat = self.fig.subplots(self.num_stats, 1, sharex=True)
        self.stats_axes = tmp_mat.flatten()
        self.stats_handles = [None] * len(self.stats_axes)

        stats = [(empty_vec, 'mean BOLD', 'cyan'),
                 (empty_vec, 'SD BOLD'  , 'xkcd:orange red'),
                 (empty_vec, 'DVARS'    , 'xkcd:mustard')]
        for ix, (ax, (stat, label, color)) in enumerate(zip(self.stats_axes, stats)):
            (vh, ) = ax.plot(time_points, stat, color=color)
            self.stats_handles[ix] = vh
            vh.set_linewidth(cfg.linewidth_stats_fmri)
            vh.set_linestyle(cfg.linestyle_stats_fmri)
            ax.xaxis.set_visible(False)
            ax.set_frame_on(False)
            ax.spines['left'].set_color(color)
            ax.set_ylim(auto=True)
            ax.set_ylabel(label, color=color)
            ax.set_zorder(self.layer_order_stats)
            ax.set_alpha(cfg.alpha_stats_overlay)
            ax.tick_params(color=color, labelcolor=color)
            ax.spines['left'].set_position(('outward', 1))

        # sharing the time point axis
        self.stats_axes[0].get_shared_x_axes().join(self.ax_carpet.xaxis, self.stats_axes[0].xaxis)
        self.stats_axes[0].autoscale()

        # 3. axes to show slices in foreground when a time point is selected
        matrix_handles = self.fig.subplots(self.num_rows, self.num_cols, gridspec_kw=dict(wspace=0.0, hspace=0.0))
        self.fg_axes = matrix_handles.flatten()

        # vmin/vmax are controlled, because we rescale all to [0, 1]
        self.imshow_params_zoomed = dict(interpolation='none', aspect='equal',
                              origin='lower', cmap='gray', vmin=0.0, vmax=1.0)

        # images to be shown in the forground
        self.images_fg = [None] * len(self.fg_axes)
        for ix, ax in enumerate(self.fg_axes):
            ax.axis('off')
            self.images_fg[ix] = ax.imshow(empty_image, **self.imshow_params_zoomed)
            ax.set_visible(False)
            ax.set_zorder(self.layer_order_zoomedin)

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
                                         zoom_in_callback=self.zoom_in_on_time_point)

        # connecting callbacks
        self.con_id_click = self.fig.canvas.mpl_connect('button_press_event', self.UI.on_mouse)
        self.con_id_keybd = self.fig.canvas.mpl_connect('key_press_event', self.UI.on_keyboard)
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
            h_alert_text= self.fig.text(cfg.position_outlier_alert[0], cfg.position_outlier_alert[1],
                                         self.current_alert_msg, **cfg.alert_text_props)
            # adding it to list of elements to cleared when advancing to next subject
            self.UI.data_handles.append(h_alert_text)

    def add_alerts(self):
        """Brings up an alert if subject id is detected to be an outlier."""

        flagged_as_outlier = self.current_unit_id in self.by_sample
        if flagged_as_outlier:
            alerts_list = self.by_sample.get(self.current_unit_id, None)  # None, if id not in dict
            print('\n\tFlagged as a possible outlier by these measures:\n\t\t{}'.format('\t'.join(alerts_list)))

            strings_to_show = ['Flagged as an outlier:', ] + alerts_list
            self.current_alert_msg = '\n'.join(strings_to_show)
            self.update_alerts()
        else:
            self.current_alert_msg = None

    def load_unit(self, unit_id):
        """Loads the image data for display."""

        img_path = self.unit_by_id[unit_id]
        try:
            hdr = nib.load(img_path)
            hdr = nib.as_closest_canonical(hdr)
            func_img_raw = hdr.get_data()
        except:
            raise IOError('Unable to read image at \n\t{}'.format(img_path))

        check_image_is_4d(func_img_raw)
        self.TR_current_scan = hdr.header.get_zooms()[-1]

        # if frames are to be dropped
        func_img = func_img_raw[:, :, :, self.drop_start:func_img_raw.shape[3] - self.drop_end]

        skip_subject = False
        if np.count_nonzero(func_img)==0:
            skip_subject = True
            print('Functional image is empty!')

        # # where to save the visualization to
        # out_vis_path = pjoin(self.out_dir, 'visual_qc_{}_{}'.format(self.vis_type, unit_id))

        return func_img, skip_subject

    def display_unit(self, func_img):
        """Adds slice collage to the given axes"""

        # to help with any callbacks who do not get image data as an argument.
        self.current_func_img = func_img

        # TODO should we perform head motion correction before any display at all?

        num_voxels = np.prod(func_img.shape[0:3])
        num_time_points = func_img.shape[3]
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
        del func_img, carpet, mean_signal_spatial, stdev_signal_spatial, dvars

        print()


    def zoom_in_on_time_point(self, event):
        """Brings up selected time point"""

        click_location = int(event.ydata)  # imshow
        self.current_time_point = max(0, min(self.current_func_img.shape[3], click_location))

        if self.current_time_point < 0 or self.current_time_point > self.current_func_img.shape[3]-1:
            return # do nothing

        self.show_timepoint(self.current_time_point)

    def show_next_time_point(self):

        if self.current_time_point == self.current_func_img.shape[3]-1:
            return # do nothing

        self.current_time_point = min(self.current_func_img.shape[3]-1, self.current_time_point+1)
        self.show_timepoint(self.current_time_point)

    def show_prev_time_point(self):

        if self.current_time_point == 0:
            return # do nothing

        self.current_time_point = max(self.current_time_point-1, 0)
        self.show_timepoint(self.current_time_point)

    def zoom_out_callback(self, event):
        """Hides the zoomed-in axes (showing frame)."""

        for ax in self.fg_axes:
            ax.set_visible(False)
        self.UI.zoomed_in = False

    def show_timepoint(self, time_pt):
        """Exhibits a selected timepoint on top of stats/carpet"""

        print('Time point zoomed-in {}'.format(time_pt))
        image3d = np.squeeze(self.current_func_img[:,:,:,time_pt])
        image3d = crop_image(image3d, self.padding)
        image3d = scale_0to1(image3d)
        slices = pick_slices(image3d, self.views, self.num_slices_per_view)
        for ax_index, (dim_index, slice_index) in enumerate(slices):
            slice_data = get_axis(image3d, dim_index, slice_index)
            self.images_fg[ax_index].set_data(slice_data)
            self.images_fg[ax_index].set_zorder(self.layer_order_zoomedin)

        for ax in self.fg_axes:
            ax.set_visible(True)

        self.UI.zoomed_in = True

    def compute_stats(self):
        """Computes the necessary stats to be displayed."""

        mean_img_temporal, stdev_img_temporal = temporal_stats(self.current_func_img)
        mean_signal_spatial, stdev_signal_spatial = spatial_stats(self.current_func_img)
        dvars = compute_DVARS(self.current_func_img)
        for stat, sname in zip((mean_signal_spatial, stdev_signal_spatial, dvars),
                               ('mean_signal_spatial', 'stdev_signal_spatial', 'dvars')):
            if len(stat) != self.current_func_img.shape[3]:
                raise ValueError('ERROR: lengths of different stats do not match!')
            if any(np.isnan(stat)):
                raise ValueError('ERROR: invalid values in stat : {}'.format(sname))
        mask = mask_image(mean_img_temporal, update_factor=0.9, init_percentile=5)
        carpet = make_carpet(self.current_func_img, mask)

        return carpet, mean_signal_spatial, stdev_signal_spatial, dvars

    def update_axes_limits(self, num_time_points, num_voxels_shown):
        """Synchronizes the x-axis limits and updates the carpet image extents"""

        for a in list(self.stats_axes)+[self.ax_carpet, ]:
            a.set_xlim(-0.5, num_time_points-0.5)
            a.set_ylim(auto=True)
            a.relim()
            a.autoscale_view()
        self.carpet_handle.set_extent((-0.5, num_time_points-0.5, -0.5, num_voxels_shown-0.5))

    def refresh_layer_order(self):
        """Ensures the expected order for layers"""

        for a in self.stats_axes:
            a.set_zorder(self.layer_order_stats)
        self.ax_carpet.set_zorder(self.layer_order_carpet)


    def identify_unit(self, unit_id, counter):
        """
        Method to inform the user which unit (subject or scan) they are reviewing.
        """

        str_list = _unbidsify(unit_id)
        id_with_counter = '{}\n({}/{})'.format(str_list, counter + 1, self.num_units_to_review)
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

    RMS_diff = lambda img2, img1 : np.sqrt(np.mean(np.square(img2-img1)))
    DVARS_1_to_N = [ RMS_diff(func_img[:,:,:,t], func_img[:,:,:,t-1]) for t in range(1, num_time_points)]

    DVARS = np.full(num_time_points, np.nan)
    # dvars value at time point 0 is set to 0
    DVARS[0] = 0.0
    DVARS[1:] = DVARS_1_to_N

    return DVARS


def make_carpet(func_img, mask, row_order=None):
    """
    Makes the carpet image


    Parameters
    ----------
    func_img

    Returns
    -------

    """

    num_voxels = np.prod(func_img.shape[0:3])
    num_time_points = func_img.shape[3]

    carpet = np.reshape(func_img, (num_voxels, num_time_points))
    # Removes voxels with low variance
    cropped_carpet = np.delete(carpet, np.where(mask.flatten() == 0), axis=0)

    min_colwise = cropped_carpet.min(axis=0)
    range_colwise = cropped_carpet.ptp(axis=0) # ptp : peak to peak, max-min
    normed_carpet = (cropped_carpet - min_colwise)/range_colwise

    del carpet, cropped_carpet

    # TODO blurring within tissue segmentations and other deeper subcortical areas

    return normed_carpet


def temporal_stats(func_img):
    """Computes voxel-wise temporal average of functional data --> single volume over space."""

    mean_img = np.mean(func_img, axis=3)
    sd_img = np.std(func_img, axis=3)

    return mean_img, sd_img


def spatial_stats(func_img):
    """Computes volume-wise spatial average of functional data --> single vector over time."""

    num_time_points = func_img.shape[3]
    mean_signal  = np.array([ np.nanmean(func_img[:,:,:,t]) for t in range(num_time_points) ])
    stdev_signal = np.array([ np.nanstd( func_img[:,:,:,t]) for t in range(num_time_points) ])

    return mean_signal, stdev_signal
