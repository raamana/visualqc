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

_unbidsify = lambda string : '\n'.join([ s.replace('-',' ') for s in string.split('_') ])

class FunctionalMRIInterface(T1MriInterface):
    """Interface for the review of fMRI images."""

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

        self.figsize = cfg.default_review_figsize
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(self.num_rows, self.num_cols, figsize=self.figsize)
        self.axes = self.axes.flatten()

        # vmin/vmax are controlled, because we rescale all to [0, 1]
        self.display_params = dict(interpolation='none', aspect='equal',
                              origin='lower', cmap='gray', vmin=0.0, vmax=1.0)

        # turning off axes, creating image objects
        self.images = [None] * len(self.axes)
        empty_image = np.full((10,10), 0.0)
        for ix, ax in enumerate(self.axes):
            ax.axis('off')
            self.images[ix] = ax.imshow(empty_image, **self.display_params)

        # leaving some space on the right for review elements
        plt.subplots_adjust(**cfg.review_area)
        plt.show(block=False)

    def add_UI(self):
        """Adds the review UI with defaults"""

        self.UI = FunctionalMRIInterface(self.fig, self.axes, self.issue_list, self.next, self.quit)

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

        sd_img = stdev_bold(func_img)

        # crop and rescale
        sd_img = crop_image(sd_img, self.padding)
        sd_img = scale_0to1(sd_img)

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
        """Preparating for exit."""

        # save ratings before exiting
        self.save_ratings()

        self.fig.canvas.mpl_disconnect(self.con_id_click)
        self.fig.canvas.mpl_disconnect(self.con_id_keybd)
        plt.close('all')


def stdev_bold(func_img):
    """Computes the SD over time."""

    # TODO checks for sufficient number of time points before computing SD

    return np.std(func_img, axis=3)
