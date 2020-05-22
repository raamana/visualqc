"""

Module to present a base neuroimaging scan, currently T1 mri, without any overlay.

"""

import argparse
import subprocess
import sys
import textwrap
import traceback
import warnings
from abc import ABC
from os import makedirs
from subprocess import check_output

import matplotlib.image as mpimg
import numpy as np
from matplotlib import cm, colors, pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.widgets import RadioButtons, Slider
from mrivis.color_maps import get_freesurfer_cmap
from mrivis.utils import crop_to_seg_extents
from os.path import exists as pexists, join as pjoin

from visualqc import config as cfg
from visualqc.interfaces import BaseReviewInterface
from visualqc.readers import read_aparc_stats_wholebrain
from visualqc.utils import check_alpha_set, check_finite_int, check_id_list, \
    check_input_dir, check_labels, check_out_dir, check_outlier_params, check_views, \
    freesurfer_installed, get_axis, get_freesurfer_mri_path, get_label_set, pick_slices, \
    read_image, scale_0to1, void_subcortical_symmetrize_cortical
from visualqc.workflows import BaseWorkflowVisualQC

# each rating is a set of labels, join them with a plus delimiter
_plus_join = lambda label_set: '+'.join(label_set)


class FreesurferReviewInterface(BaseReviewInterface):
    """Custom interface for rating the quality of Freesurfer parcellation."""


    def __init__(self,
                 fig,
                 axes_seg,
                 rating_list=cfg.default_rating_list,
                 next_button_callback=None,
                 quit_button_callback=None,
                 alpha_seg=cfg.default_alpha_seg):
        """Constructor"""

        super().__init__(fig, axes_seg, next_button_callback, quit_button_callback)

        self.rating_list = rating_list

        self.overlaid_artists = axes_seg
        self.latest_alpha_seg = alpha_seg
        self.prev_axis = None
        self.prev_ax_pos = None
        self.zoomed_in = False
        self.add_radio_buttons()
        self.add_alpha_slider()

        self.next_button_callback = next_button_callback
        self.quit_button_callback = quit_button_callback

        # this list of artists to be populated later
        # makes to handy to clean them all
        self.data_handles = list()

        self.unzoomable_axes = [self.radio_bt_rating.ax, self.text_box.ax,
                                self.bt_next.ax, self.bt_quit.ax]


    def add_radio_buttons(self):

        ax_radio = plt.axes(cfg.position_radio_buttons,
                            facecolor=cfg.color_rating_axis, aspect='equal')
        self.radio_bt_rating = RadioButtons(ax_radio, self.rating_list,
                                            active=None, activecolor='orange')
        self.radio_bt_rating.on_clicked(self.save_rating)
        for txt_lbl in self.radio_bt_rating.labels:
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for circ in self.radio_bt_rating.circles:
            circ.set(radius=0.06)


    def add_alpha_slider(self):
        """Controls the transparency level of overlay"""

        # alpha slider
        ax_slider = plt.axes(cfg.position_slider_seg_alpha,
                             facecolor=cfg.color_slider_axis)
        self.slider = Slider(ax_slider, label='transparency',
                             valmin=0.0, valmax=1.0, valinit=0.7, valfmt='%1.2f')
        self.slider.label.set_position((0.99, 1.5))
        self.slider.on_changed(self.set_alpha_value)


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

        # this is populated for each unit during display
        self.overlaid_artists.clear()


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

        # right click toggles overlay
        if event.button in [3]:
            self.toggle_overlay()
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
        if key_pressed in ['right', ' ', 'space']:
            self.next_button_callback()
        if key_pressed in ['ctrl+q', 'q+ctrl']:
            self.quit_button_callback()
        else:
            if key_pressed in cfg.default_rating_list_shortform:
                self.user_rating = cfg.map_short_rating[key_pressed]
                index_to_set = cfg.default_rating_list.index(self.user_rating)
                self.radio_bt_rating.set_active(index_to_set)
            elif key_pressed in ['t']:
                self.toggle_overlay()
            else:
                pass


    def toggle_overlay(self):
        """Toggles the overlay by setting alpha to 0 and back."""

        if self.latest_alpha_seg != 0.0:
            self.prev_alpha_seg = self.latest_alpha_seg
            self.latest_alpha_seg = 0.0
        else:
            self.latest_alpha_seg = self.prev_alpha_seg
        self.update()


    def set_alpha_value(self, latest_value):
        """" Use the slider to set alpha."""

        self.latest_alpha_seg = latest_value
        self.update()


    def update(self):
        """updating seg alpha for all axes"""

        for art in self.overlaid_artists:
            art.set_alpha(self.latest_alpha_seg)

        # update figure
        plt.draw()


class FreesurferRatingWorkflow(BaseWorkflowVisualQC, ABC):
    """
    Rating workflow without any overlay.
    """


    def __init__(self,
                 id_list,
                 images_for_id,
                 in_dir,
                 out_dir,
                 vis_type=cfg.default_vis_type,
                 label_set=cfg.default_label_set,
                 issue_list=cfg.default_rating_list,
                 mri_name=cfg.default_mri_name,
                 seg_name=cfg.default_seg_name,
                 alpha_set=cfg.default_alpha_set,
                 outlier_method=cfg.default_outlier_detection_method,
                 outlier_fraction=cfg.default_outlier_fraction,
                 outlier_feat_types=cfg.freesurfer_features_outlier_detection,
                 source_of_features=cfg.default_source_of_features_freesurfer,
                 disable_outlier_detection=False,
                 no_surface_vis=False,
                 views=cfg.default_views,
                 num_slices_per_view=cfg.default_num_slices,
                 num_rows_per_view=cfg.default_num_rows):
        """Constructor"""

        super().__init__(id_list, in_dir, out_dir,
                         outlier_method, outlier_fraction,
                         outlier_feat_types, disable_outlier_detection)

        self.issue_list = issue_list
        # in_dir_type must be freesurfer; vis_type must be freesurfer

        self.mri_name = mri_name
        self.seg_name = seg_name
        self.label_set = label_set
        self.vis_type = vis_type
        self.images_for_id = images_for_id

        self.expt_id = 'rate_freesurfer_{}'.format(self.mri_name)
        self.suffix = self.expt_id
        self.current_alert_msg = None
        self.no_surface_vis = no_surface_vis

        self.alpha_mri = alpha_set[0]
        self.alpha_seg = alpha_set[1]
        self.contour_color = 'yellow'

        self.init_layout(views, num_rows_per_view, num_slices_per_view)

        self.source_of_features = source_of_features
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

        if not self.no_surface_vis:
            self.generate_surface_vis()


    def generate_surface_vis(self):
        """Generates surface visualizations."""

        print('Attempting to generate the surface visualizations of parcellation ...')
        if not freesurfer_installed():  # needs tksurfer
            print('Freesurfer does not seem to be installed '
                  '- skipping surface visualizations.')
            self._freesurfer_installed = False
        else:
            self._freesurfer_installed = True

        self.surface_vis_paths = dict()
        for sid in self.id_list:
            self.surface_vis_paths[sid] = \
                make_vis_pial_surface(self.in_dir, sid, self.out_dir,
                                      self._freesurfer_installed)


    def prepare_UI(self):
        """Main method to run the entire workflow"""

        self.open_figure()
        self.add_UI()
        self.add_histogram_panel()


    def init_layout(self, views, num_rows_per_view,
                    num_slices_per_view, padding=cfg.default_padding):

        self.padding = padding
        self.views = views
        self.num_slices_per_view = num_slices_per_view
        self.num_rows_per_view = num_rows_per_view
        num_cols_volumetric = num_slices_per_view / num_rows_per_view
        num_rows_volumetric = len(self.views) * self.num_rows_per_view
        total_num_panels_vol = int(num_cols_volumetric * num_rows_volumetric)

        # surf vis generation happens at the beginning - no option to defer for user
        if not self.no_surface_vis and 'cortical' in self.vis_type:
            extra_panels_surface_vis = cfg.num_cortical_surface_vis
            extra_rows_surface_vis = 1
            self.volumetric_start_index = extra_panels_surface_vis
        else:
            extra_panels_surface_vis = 0
            extra_rows_surface_vis = 0
            self.volumetric_start_index = 0

        total_num_panels = total_num_panels_vol + extra_panels_surface_vis
        self.num_rows_total = num_rows_volumetric + extra_rows_surface_vis
        self.num_cols_final = int(np.ceil(total_num_panels / self.num_rows_total))


    def init_getters(self):
        """Initializes the getters methods for input paths and feature readers."""

        from visualqc.readers import gather_freesurfer_data
        self.feature_extractor = gather_freesurfer_data
        self.path_getter_inputs = get_freesurfer_mri_path


    def open_figure(self):
        """Creates the master figure to show everything in."""

        self.figsize = cfg.default_review_figsize
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(self.num_rows_total, self.num_cols_final,
                                           figsize=self.figsize)
        self.axes = self.axes.flatten()
        self.fig.canvas.set_window_title('VisualQC {} {} : {}'
                                         ' '.format(self.vis_type, self.seg_name, self.in_dir))

        self.display_params_mri = dict(interpolation='none', aspect='equal',
                                       origin='lower',
                                       alpha=self.alpha_mri)
        self.display_params_seg = dict(interpolation='none', aspect='equal',
                                       origin='lower',
                                       alpha=self.alpha_seg)

        normalize_mri = colors.Normalize(vmin=cfg.min_cmap_range_t1_mri,
                                         vmax=cfg.max_cmap_range_t1_mri, clip=True)
        self.mri_mapper = cm.ScalarMappable(norm=normalize_mri, cmap='gray')

        fs_cmap = get_freesurfer_cmap(self.vis_type)
        # deciding colors for the whole image
        if self.label_set is not None and self.vis_type in cfg.label_types:

            # re-numbering as remapping happens for each vol. seg via utils.get_label_set()
            unique_labels = np.arange(self.label_set.size)+1

            # NOTE vmin and vmax can't be the same as Normalize would simply return 0,
            #   even for nonzero values
            # NOTE PREV BUG although norm method here is set up to rescale values from
            #   (np.min(unique_labels), np.max(unique_labels)) to (0, 1)
            #   it was being applied to values outside this range (typically in [0, 5] range)
            #   as unique_labels tend be > 10.. so setting it up correctly now
            #   from 0 to L, L being number of unique labels
            #   this interacts with utils.get_label_set(), so they need to be handled together
            normalize_labels = colors.Normalize(vmin=0,
                                                vmax=unique_labels.size,
                                                clip=True)

        elif self.vis_type in cfg.cortical_types:
            # TODO this might become a bug, if num of cortical labels become more than 34
            num_cortical_labels = len(fs_cmap.colors)
            if num_cortical_labels < 1:
                raise ValueError('number of cortical labels seem to be zero:\n'
                                 ' invalid colormap/labels set for Freesurfer!\n'
                                 'Must be typically 34 or higher - '
                                 'report this bug to the developers. Thanks.')
            unique_labels = np.arange(num_cortical_labels, dtype='int8')
            # as the cortical labels
            normalize_labels = colors.Normalize(vmin=0,
                                                vmax=num_cortical_labels,
                                                clip=True)

        self.seg_mapper = cm.ScalarMappable(norm=normalize_labels, cmap=fs_cmap)

        # removing background - 0 stays 0
        self.unique_labels_display = np.setdiff1d(unique_labels, 0)
        if len(self.unique_labels_display) == 1:
            self.color_for_label = [self.contour_color]
        else:
            self.color_for_label = self.seg_mapper.to_rgba(self.unique_labels_display)

        # doing all the one-time operations, to improve speed later on
        # specifying 3rd dim for empty_image to avoid any color mapping
        empty_image = np.full((10, 10, 3), 0.0)

        self.h_surfaces = [None]*cfg.num_cortical_surface_vis

        num_volumetric_panels = len(self.axes) - cfg.num_cortical_surface_vis
        for ix, ax in enumerate(self.axes[:cfg.num_cortical_surface_vis]):
            ax.axis('off')
            self.h_surfaces[ix] = ax.imshow(empty_image)

        self.h_images_mri = [None] * num_volumetric_panels
        if 'volumetric' in self.vis_type:
            self.h_images_seg = [None] * num_volumetric_panels

        for ix, ax in enumerate(self.axes[cfg.num_cortical_surface_vis:]):
            ax.axis('off')
            self.h_images_mri[ix] = ax.imshow(empty_image, **self.display_params_mri)
            if 'volumetric' in self.vis_type:
                self.h_images_seg[ix] = ax.imshow(empty_image, **self.display_params_seg)

        self.togglable_handles = list()

        # leaving some space on the right for review elements
        plt.subplots_adjust(**cfg.review_area)
        plt.show(block=False)


    def add_UI(self):
        """Adds the review UI with defaults"""

        self.UI = FreesurferReviewInterface(self.fig, self.togglable_handles,
                                            self.issue_list, self.next, self.quit)

        # connecting callbacks
        self.con_id_click = self.fig.canvas.mpl_connect('button_press_event',
                                                        self.UI.on_mouse)
        self.con_id_keybd = self.fig.canvas.mpl_connect('key_press_event',
                                                        self.UI.on_keyboard)
        # con_id_scroll = self.fig.canvas.mpl_connect('scroll_event', self.UI.on_scroll)

        self.fig.set_size_inches(self.figsize)


    def add_histogram_panel(self):
        """Extra axis for histogram of cortical thickness!"""

        if not self.vis_type in cfg.cortical_types:
            return

        self.ax_hist = plt.axes(cfg.position_histogram_freesurfer)
        self.ax_hist.set(xticks=cfg.xticks_histogram_freesurfer,
                         xticklabels=cfg.xticks_histogram_freesurfer,
                         yticks=[], autoscaley_on=True)
        self.ax_hist.set_prop_cycle('color', cfg.color_histogram_freesurfer)
        self.ax_hist.set_title(cfg.title_histogram_freesurfer, fontsize='small')


    def update_histogram(self):
        """Updates histogram with current image data"""

        # to update thickness histogram, you need access to full FS output or aparc.stats
        try:
            distribution_to_show = read_aparc_stats_wholebrain(self.in_dir, self.current_unit_id,
                                                   subset=(cfg.statistic_in_histogram_freesurfer,))
        except:
            # do nothing
            return

        # number of vertices is too high - so presenting mean ROI thickness is smarter!
        _, _, patches_hist = self.ax_hist.hist(distribution_to_show, density=True,
                                               bins=cfg.num_bins_histogram_display)
        self.ax_hist.set_xlim(cfg.xlim_histogram_freesurfer)
        self.ax_hist.relim(visible_only=True)
        self.ax_hist.autoscale_view(scalex=False)
        self.UI.data_handles.extend(patches_hist)


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

        t1_mri_path = get_freesurfer_mri_path(self.in_dir, unit_id, self.mri_name)
        fs_seg_path = get_freesurfer_mri_path(self.in_dir, unit_id, self.seg_name)

        temp_t1_mri = read_image(t1_mri_path, error_msg='T1 mri')
        temp_fs_seg = read_image(fs_seg_path, error_msg='segmentation')

        if temp_t1_mri.shape != temp_fs_seg.shape:
            raise ValueError('size mismatch! MRI: {} Seg: {}\n'
                             'Size must match in all dimensions.'.format(
                self.current_t1_mri.shape,
                temp_fs_seg.shape))

        skip_subject = False
        if self.vis_type in ('cortical_volumetric', 'cortical_contour'):
            temp_seg_uncropped, roi_set_is_empty = void_subcortical_symmetrize_cortical(temp_fs_seg)
        elif self.vis_type in ('labels_volumetric', 'labels_contour'):
            if self.label_set is not None:
                # TODO same colors for same labels is not guaranteed
                #   if one subject fewer labels than others
                #   due to remapping of labels for each subject
                temp_seg_uncropped, roi_set_is_empty = get_label_set(temp_fs_seg,
                                                                     self.label_set)
            else:
                raise ValueError('--label_set must be specified for visualization types: '
                                 ' labels_volumetric and labels_contour')
        else:
            raise NotImplementedError('Invalid visualization type - '
                                      'choose from: {}'.format(
                cfg.visualization_combination_choices))

        if roi_set_is_empty:
            skip_subject = True
            print('segmentation image for {} '
                  'does not contain requested label set!'.format(unit_id))
            return skip_subject

        # T1 mri must be rescaled - to avoid strange distributions skewing plots
        rescaled_t1_mri = scale_0to1(temp_t1_mri, cfg.max_cmap_range_t1_mri)
        self.current_t1_mri, self.current_seg = crop_to_seg_extents(rescaled_t1_mri,
                                                                    temp_seg_uncropped,
                                                                    self.padding)

        out_vis_path = pjoin(self.out_dir,
                             'visual_qc_{}_{}_{}'.format(self.vis_type, self.suffix,
                                                         unit_id))

        return skip_subject


    def display_unit(self):
        """Adds slice collage, with seg overlays on MRI in each panel."""

        if 'cortical' in self.vis_type:
            if not self.no_surface_vis and self.current_unit_id in self.surface_vis_paths:
                surf_paths = self.surface_vis_paths[self.current_unit_id] # is a dict of paths
                for sf_ax_index, ((hemi, view), spath) in enumerate(surf_paths.items()):
                    plt.sca(self.axes[sf_ax_index])
                    img = mpimg.imread(spath)
                    # img = crop_image(img)
                    h_surf = plt.imshow(img)
                    self.axes[sf_ax_index].text(0, 0, '{} {}'.format(hemi, view))
                    self.UI.data_handles.append(h_surf)
            else:
                msg = 'no surface visualizations\navailable or disabled'
                print('{} for {}'.format(msg, self.current_unit_id))
                self.axes[1].text(0.5, 0.5, msg)

        slices = pick_slices(self.current_seg, self.views, self.num_slices_per_view)
        for vol_ax_index, (dim_index, slice_index) in enumerate(slices):
            panel_index = self.volumetric_start_index + vol_ax_index
            plt.sca(self.axes[panel_index])
            slice_mri = get_axis(self.current_t1_mri, dim_index, slice_index)
            slice_seg = get_axis(self.current_seg, dim_index, slice_index)

            mri_rgba = self.mri_mapper.to_rgba(slice_mri, alpha=self.alpha_mri)
            # self.h_images_mri[ax_index].set_data(mri_rgb)
            h_m = plt.imshow(mri_rgba, interpolation='none',
                             aspect='equal', origin='lower')
            self.UI.data_handles.append(h_m)

            if 'volumetric' in self.vis_type:
                seg_rgba = self.seg_mapper.to_rgba(slice_seg, alpha=self.alpha_seg)
                # self.h_images_seg[ax_index].set_data(seg_rgb)
                h_seg = plt.imshow(seg_rgba, interpolation='none',
                                   aspect='equal', origin='lower')
                self.togglable_handles.append(h_seg)
                # self.UI.data_handles.append(h_seg)
                del seg_rgba
            elif 'contour' in self.vis_type:
                h_seg = self.plot_contours_in_slice(slice_seg, self.axes[panel_index])
                for contours in h_seg:
                    self.togglable_handles.extend(contours.collections)
                    # for clearing upon review
                    self.UI.data_handles.extend(contours.collections)

            del slice_seg, slice_mri, mri_rgba

        # histogram shown only for cortical parcellation QC
        if self.vis_type in cfg.cortical_types:
            self.update_histogram()


    def plot_contours_in_slice(self, slice_seg, target_axis):
        """Plots contour around the data in slice (after binarization)"""

        plt.sca(target_axis)
        contour_handles = list()
        for index, label in enumerate(self.unique_labels_display):
            binary_slice_seg = slice_seg == index
            if not binary_slice_seg.any():
                continue
            ctr_h = plt.contour(binary_slice_seg,
                                levels=[cfg.contour_level, ],
                                colors=(self.color_for_label[index],),
                                linewidths=cfg.contour_line_width,
                                alpha=self.alpha_seg,
                                zorder=cfg.seg_zorder_freesurfer)
            contour_handles.append(ctr_h)

        return contour_handles


    def cleanup(self):
        """Preparating for exit."""

        # save ratings before exiting
        self.save_ratings()

        self.fig.canvas.mpl_disconnect(self.con_id_click)
        self.fig.canvas.mpl_disconnect(self.con_id_keybd)
        plt.close('all')


def make_vis_pial_surface(in_dir, subject_id, out_dir,
                          FREESURFER_INSTALLED,
                          annot_file='aparc.annot'):
    """Generate screenshot for the pial surface in different views"""

    out_vis_dir = pjoin(out_dir, cfg.annot_vis_dir_name)
    makedirs(out_vis_dir, exist_ok=True)
    hemis = ('lh', 'rh')
    hemis_long = ('left', 'right')
    vis_list = dict()

    print('Processing {}'.format(subject_id))
    for hemi, hemi_l in zip(hemis, hemis_long):
        vis_list[hemi_l] = dict()
        script_file, vis_files = make_tcl_script_vis_annot(subject_id, hemi_l, out_vis_dir, annot_file)
        try:
            # run the script only if all the visualizations were not generated before
            all_vis_exist = all([pexists(vis_path) for vis_path in vis_files.values()])
            if not all_vis_exist and FREESURFER_INSTALLED:
                _, _ = run_tksurfer_script(in_dir, subject_id, hemi, script_file)

            vis_list[hemi_l].update(vis_files)
        except:
            traceback.print_exc()
            print('unable to generate tksurfer visualizations for {} hemi - skipping'.format(hemi))

    # flattening it for easier use later on
    out_vis_list = dict()
    pref_order = [ ('right', 'lateral'), ('left', 'lateral'),
                   ('right', 'medial'), ('left', 'medial'),
                   ('right', 'transverse'), ('left', 'transverse')]
    for hemi_l, view in pref_order:
        if pexists(vis_list[hemi_l][view]):
            out_vis_list[(hemi_l, view)] = vis_list[hemi_l][view]

    return out_vis_list


def make_tcl_script_vis_annot(subject_id, hemi, out_vis_dir, annot_file='aparc.annot'):
    """Generates a tksurfer script to make visualizations"""

    script_file = pjoin(out_vis_dir, 'vis_annot_{}.tcl'.format(hemi))
    vis = dict()
    for view in cfg.surface_view_angles:
        vis[view] = pjoin(out_vis_dir, '{}_{}_{}.tif'.format(subject_id, hemi, view))

    img_format = 'tiff'  # rgb does not work

    cmds = list()
    # cmds.append("resize_window 1000")
    cmds.append("labl_import_annotation {}".format(annot_file))
    cmds.append("scale_brain 1.37")
    cmds.append("redraw")
    cmds.append("save_{} {}".format(img_format, vis['lateral']))
    cmds.append("rotate_brain_y 180.0")
    cmds.append("redraw")
    cmds.append("save_{} {}".format(img_format, vis['medial']))
    cmds.append("rotate_brain_z -90.0")
    cmds.append("rotate_brain_y 135.0")
    cmds.append("redraw")
    cmds.append("save_{} {}".format(img_format, vis['transverse']))
    cmds.append("exit 0")

    try:
        with open(script_file, 'w') as sf:
            sf.write('\n'.join(cmds))
    except:
        raise IOError('Unable to write the script file to\n {}'.format(script_file))

    return script_file, vis


def run_tksurfer_script(in_dir, subject_id, hemi, script_file):
    """Runs a given TCL script to generate visualizations"""

    try:
        cmd_args = ['tksurfer', '-sdir', in_dir, subject_id, hemi, 'pial', '-tcl', script_file]
        txt_out = check_output(cmd_args, shell=False, stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as tksurfer_exc:
        print('Error running tksurfer to generate 3d surface visualizations - skipping them.')
        exit_code = tksurfer_exc.returncode
        txt_out = tksurfer_exc.output
    else:
        exit_code = 0

    return txt_out, exit_code


def get_parser():
    """Parser to specify arguments and their defaults."""

    parser = argparse.ArgumentParser(prog="visualqc_freesurfer",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description='visualqc_freesurfer: rate quality '
                                                 'of Freesurfer reconstruction.')

    help_text_fs_dir = textwrap.dedent("""
    Absolute path to ``SUBJECTS_DIR`` containing the finished runs of Freesurfer parcellation
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

    help_text_seg_name = textwrap.dedent("""
    Specifies the name of segmentation image (volumetric) to be overlaid on the MRI.
    Typical options include aparc+aseg.mgz, aseg.mgz, wmparc.mgz. 
    Make sure to choose the right vis_type. 

    Default: {} (within the mri folder of Freesurfer format).
    \n""".format(cfg.default_seg_name))

    help_text_out_dir = textwrap.dedent("""
    Output folder to store the visualizations & ratings.
    Default: a new folder called ``{}`` will be created inside the ``fs_dir``
    \n""".format(cfg.default_out_dir_name))

    help_text_vis_type = textwrap.dedent("""
    Specifies the type of visualizations/overlay requested.
    Default: {} (volumetric overlay of cortical segmentation on T1 mri).
    \n""".format(cfg.default_vis_type))

    help_text_label = textwrap.dedent("""
    Specifies the set of labels to include for overlay.

    Atleast one label must be specified when vis_type is labels_volumetric or labels_contour

    Default: None (show nothing)
    \n""")

    help_text_contour_color = textwrap.dedent("""
    Specifies the color to use for the contours overlaid on MRI (when vis_type requested prescribes contours). 
    Color can be specified in many ways as documented in https://matplotlib.org/users/colors.html
    Default: {}.
    \n""".format(cfg.default_contour_face_color))

    help_text_alphas = textwrap.dedent("""
    Alpha values to control the transparency of MRI and aseg. 
    This must be a set of two values (between 0 and 1.0) separated by a space e.g. --alphas 0.7 0.5. 

    Default: {} {}.  Play with these values to find something that works for you and the dataset.
    \n""".format(cfg.default_alpha_mri, cfg.default_alpha_seg))

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

    help_text_no_surface_vis = textwrap.dedent("""
    This flag disables batch-generation of 3d surface visualizations, which are shown along with cross-sectional overlays. This is not recommended, but could be used in situations where you do not have Freesurfer installed or want to focus solely on cross-sectional views.

    Default: False (required visualizations are generated at the beginning, which can take 5-10 seconds for each subject).
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
    \n""".format(cfg.t1_mri_features_OLD))

    help_text_disable_outlier_detection = textwrap.dedent("""
    This flag disables outlier detection and alerts altogether.
    \n""")

    in_out = parser.add_argument_group('Input and output', ' ')

    in_out.add_argument("-i", "--id_list", action="store", dest="id_list",
                        default=None, required=False, help=help_text_id_list)

    in_out.add_argument("-f", "--fs_dir", action="store", dest="fs_dir",
                        default=cfg.default_freesurfer_dir,
                        required=False, help=help_text_fs_dir)

    in_out.add_argument("-o", "--out_dir", action="store", dest="out_dir",
                        required=False, help=help_text_out_dir,
                        default=None)

    in_out.add_argument("-m", "--mri_name", action="store", dest="mri_name",
                        default=cfg.default_mri_name, required=False,
                        help=help_text_mri_name)

    in_out.add_argument("-g", "--seg_name", action="store", dest="seg_name",
                        default=cfg.default_seg_name, required=False,
                        help=help_text_seg_name)

    in_out.add_argument("-l", "--labels", action="store", dest="label_set",
                        default=cfg.default_label_set,
                        nargs='+', metavar='label',
                        required=False, help=help_text_label)

    vis_args = parser.add_argument_group('Overlay options', ' ')
    vis_args.add_argument("-v", "--vis_type", action="store", dest="vis_type",
                          choices=cfg.visualization_combination_choices,
                          default=cfg.default_vis_type, required=False,
                          help=help_text_vis_type)

    vis_args.add_argument("-c", "--contour_color", action="store", dest="contour_color",
                          default=cfg.default_contour_face_color, required=False,
                          help=help_text_contour_color)

    vis_args.add_argument("-a", "--alpha_set", action="store", dest="alpha_set",
                          metavar='alpha', nargs=2,
                          default=cfg.default_alpha_set,
                          required=False, help=help_text_alphas)

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
                          default=cfg.freesurfer_features_outlier_detection,
                          required=False, help=help_text_outlier_feat_types)

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

    wf_args.add_argument("-ns", "--no_surface_vis", action="store_true",
                         dest="no_surface_vis", help=help_text_no_surface_vis)

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

    vis_type, label_set = check_labels(user_args.vis_type, user_args.label_set)
    in_dir, source_of_features = check_input_dir(user_args.fs_dir, None, vis_type,
                                                 freesurfer_install_required=False)

    mri_name = user_args.mri_name
    seg_name = user_args.seg_name
    id_list, images_for_id = check_id_list(user_args.id_list, in_dir, vis_type, mri_name,
                                           seg_name)

    out_dir = check_out_dir(user_args.out_dir, in_dir)

    alpha_set = check_alpha_set(user_args.alpha_set)
    views = check_views(user_args.views)
    num_slices, num_rows = check_finite_int(user_args.num_slices, user_args.num_rows)

    contour_color = user_args.contour_color
    if not is_color_like(contour_color):
        raise ValueError('Specified color is not valid. Choose a valid spec from\n'
                         ' https://matplotlib.org/users/colors.html')

    outlier_method, outlier_fraction, outlier_feat_types, disable_outlier_detection = \
        check_outlier_params(user_args.outlier_method, user_args.outlier_fraction,
                             user_args.outlier_feat_types,
                             user_args.disable_outlier_detection,
                             id_list, vis_type, source_of_features)

    wf = FreesurferRatingWorkflow(id_list,
                                  images_for_id,
                                  in_dir,
                                  out_dir,
                                  vis_type=vis_type,
                                  label_set=label_set,
                                  issue_list=cfg.default_rating_list,
                                  mri_name=mri_name,
                                  seg_name=seg_name,
                                  alpha_set=alpha_set,
                                  outlier_method=outlier_method,
                                  outlier_fraction=outlier_fraction,
                                  outlier_feat_types=outlier_feat_types,
                                  disable_outlier_detection=disable_outlier_detection,
                                  source_of_features=source_of_features,
                                  no_surface_vis=user_args.no_surface_vis,
                                  views=views,
                                  num_slices_per_view=num_slices,
                                  num_rows_per_view=num_rows)

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
