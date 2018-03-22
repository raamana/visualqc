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

from visualqc import config as cfg
from visualqc.interfaces import BaseReviewInterface
from visualqc.utils import check_id_list, check_input_dir_T1, check_views, \
    check_finite_int, check_out_dir, check_outlier_params, get_path_for_subject, \
    read_image, scale_0to1, pick_slices, get_axis, get_ratings_path_info, load_ratings_csv
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
                 zoom_in_callback=None,
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
        self.right_click_callback = right_click_callback

        super().add_checkboxes()

        # this list of artists to be populated later
        # makes to handy to clean them all
        self.data_handles = list()

    def on_mouse(self, event):
        """Callback for mouse events."""

        if self.prev_axis is not None:
            # include all the non-data axes here (so they wont be zoomed-in)
            if event.inaxes not in [self.checkbox.ax, self.text_box.ax,
                                    self.bt_next.ax, self.bt_quit.ax]:
                self.prev_axis.set_position(self.prev_ax_pos)
                self.prev_axis.set_zorder(self.prev_ax_zorder)
                self.prev_axis.set_visible(self.prev_visible)
                self.prev_axis.patch.set_alpha(0.5)
                self.zoomed_in = False

        # right click ignored
        if event.button in [3]:
            self.right_click_callback()
        # double click to zoom in to any axis
        elif event.dblclick and event.inaxes is not None and \
            event.inaxes not in [self.checkbox.ax, self.text_box.ax,
                                 self.bt_next.ax, self.bt_quit.ax]:
            # zoom axes full-screen
            self.prev_ax_pos = event.inaxes.get_position()
            self.prev_ax_zorder = event.inaxes.get_zorder()
            self.prev_visible = event.inaxes.get_visible()
            event.inaxes.set_position(cfg.zoomed_position)
            event.inaxes.set_zorder(self.total_num_layers) # bring forth
            # event.inaxes.set_facecolor('black') # black
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
        if key_pressed in ['right', ' ', 'space']:
            self.next_button_callback()
        if key_pressed in ['ctrl+q', 'q+ctrl']:
            self.quit_button_callback()
        else:
            if key_pressed in cfg.func_mri_default_rating_list_shortform:
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
                 id_list=None,
                 issue_list=cfg.func_mri_default_issue_list,
                 in_dir_type='BIDS',
                 outlier_method=cfg.default_outlier_detection_method,
                 outlier_fraction=cfg.default_outlier_fraction,
                 outlier_feat_types=cfg.func_outlier_features,
                 disable_outlier_detection=True,
                 prepare_first=False,
                 vis_type=None,
                 views=cfg.default_views,
                 num_slices_per_view=cfg.default_num_slices,
                 num_rows_per_view=cfg.default_num_rows):
        """
        Constructor.

        Parameters
        ----------
        in_dir : path
            must be a path to BIDS directory


        """

        if id_list is None and in_dir_type=='BIDS':
            id_list = pjoin(in_dir, 'participants.tsv')

        super().__init__(id_list, in_dir, out_dir,
                         outlier_method, outlier_fraction,
                         outlier_feat_types, disable_outlier_detection)

        self.vis_type = vis_type
        self.issue_list = issue_list
        self.in_dir_type = in_dir_type
        self.expt_id = 'rate_fmri'
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
        self.layer_order_carpet   = 0
        self.layer_order_stats    = 1
        self.layer_order_zoomedin = 2

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

        # 2. temporal traces of image stats
        tmp_mat = self.fig.subplots(self.num_stats, 1, sharex=True)
        self.stats_axes = tmp_mat.flatten()
        self.stats_handles = [None] * len(self.stats_axes)

        stats = [(empty_vec, 'mean BOLD', 'cyan'),
                 (empty_vec, 'SD BOLD'  , 'xkcd:orange red'),
                 (empty_vec, 'DVARS'    , 'xkcd:pine green')]
        for ix, (ax, (stat, label, color)) in enumerate(zip(self.stats_axes, stats)):
            (vh, ) = ax.plot(time_points, stat, color=color)
            self.stats_handles[ix] = vh
            vh.set_linewidth(cfg.linewidth_stats_fmri)
            vh.set_linestyle(cfg.linestyle_stats_fmri)
            ax.xaxis.set_visible(False)
            ax.set_frame_on(False)
            ax.spines['left'].set_color(color)
            ax.set_ylabel(label, color=color)
            ax.set_zorder(self.layer_order_stats)
            ax.set_alpha(cfg.alpha_stats_overlay)
            ax.tick_params(color=color, labelcolor=color)
            ax.spines['left'].set_position(('outward', 1))

        # sharing the time point axis
        self.stats_axes[0].get_shared_x_axes().join(self.ax_carpet, self.stats_axes[0])
        self.stats_axes[0].autoscale()

        # 3. axes to show slices in foreground when a time point is selected
        matrix_handles = self.fig.subplots(self.num_rows, self.num_cols)
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

        self.UI = FunctionalMRIInterface(self.fig, self.ax_carpet,
                                         self.issue_list, self.next, self.quit)

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
        func_img = read_image(img_path, num_dims=4, error_msg='functional mri')

        skip_subject = False
        if np.count_nonzero(func_img)==0:
            skip_subject = True
            print('Functional image is empty!')

        # # where to save the visualization to
        # out_vis_path = pjoin(self.out_dir, 'visual_qc_{}_{}'.format(self.vis_type, unit_id))

        return func_img, skip_subject

    def display_unit(self, func_img):
        """Adds slice collage to the given axes"""

        # TODO should we perform head motion correction before any display at all?

        mean_img_temporal, stdev_img_temporal = temporal_stats(func_img)
        mean_signal_spatial, stdev_signal_spatial = spatial_stats(func_img)
        dvars = compute_DVARS(func_img)

        mask = mask_image(mean_img_temporal, update_factor=0.9, init_percentile=5)
        masked_func_img = np.full_like(func_img, 0.0)
        masked_func_img[mask] = func_img[mask]

        # crop and rescale
        stdev_img_temporal = crop_image(stdev_img_temporal, self.padding)
        stdev_img_temporal = scale_0to1(stdev_img_temporal)*cfg.scale_factor_BOLD

        num_voxels = np.prod(func_img.shape[0:3])
        num_time_points = func_img.shape[3]
        time_points = list(range(num_time_points))

        #
        carpet = make_carpet(func_img, mask)
        self.carpet_handle.set_data(carpet)

        # overlay stats on top
        self.stats_handles[0].set_data(time_points, mean_signal_spatial)
        self.stats_handles[1].set_data(time_points, stdev_signal_spatial)
        # not displaying DVARS for t=0, as its always 0
        self.stats_handles[2].set_data(time_points[1:], dvars[1:])
        # updating axes limits
        [(a.relim(), a.autoscale_view()) for a in list(self.stats_axes)+[self.ax_carpet, ]]

        # adding slices
        slices = pick_slices(sd_img, self.views, self.num_slices_per_view)
        for ax_index, (dim_index, slice_index) in enumerate(slices):
            slice_data = get_axis(sd_img, dim_index, slice_index)
            self.images[ax_index].set_data(slice_data)


    def identify_unit(self, unit_id):
        """
        Method to inform the user which unit (subject or scan) they are reviewing.
        """

        str_list = _unbidsify(unit_id)
        if len(str_list) < 1:
            return
        self.UI.annot_text = self.fig.text(cfg.position_annot_text[0],
                                        cfg.position_annot_text[1],
                                        str_list, **cfg.annot_text_props)

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

    DVARS = np.nan(num_time_points)
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
    mean_signal  = [ np.nanmean(func_img[:,:,:,t]) for t in range(num_time_points) ]
    stdev_signal = [ np.nanstd( func_img[:,:,:,t]) for t in range(num_time_points) ]

    return mean_signal, stdev_signal
