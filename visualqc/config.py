"""
Central place to store the config info

"""
from collections import OrderedDict

import numpy as np

# default values
default_out_dir_name = 'visualqc'
annot_vis_dir_name = 'annot_visualizations'
default_mri_name = 'orig.mgz'  # brainmask would not help check expansion of surfaces into skull
default_seg_name = 'aparc+aseg.mgz'
required_files = (default_mri_name, default_seg_name)
default_source_of_features_freesurfer = 'whole_brain'

num_cortical_surface_vis = 6
position_histogram_freesurfer = [0.905, 0.7, 0.09, 0.1]
statistic_in_histogram_freesurfer = 'ThickAvg'
title_histogram_freesurfer = 'mean thickness (label-wise)'
num_bins_histogram_display = 30
xlim_histogram_freesurfer_all = { 'ThickAvg' : [1.0, 6.0], }
xlim_histogram_freesurfer = xlim_histogram_freesurfer_all[statistic_in_histogram_freesurfer]
xticks_histogram_freesurfer = np.arange(1.5, 6.01, 1.0)
color_histogram_freesurfer = ('#c9ae74')  # sandstone

freesurfer_features_outlier_detection = ('cortical', 'subcortical')
outlier_list_prefix = 'possible_outliers'
alert_background_color = 'xkcd:coral'
alert_colors_outlier = dict(cortical='xkcd:hot pink',
                            subcortical='xkcd:periwinkle')

position_annot_text = (0.990, 0.98)
annot_text_props = dict(ha='right', va='top', multialignment='left',
                        wrap=True, fontsize='large', color='#c65102')

alert_text_props = dict(horizontalalignment='center', fontsize='medium',
                        color='white', backgroundcolor=alert_background_color)

slice_num_label_properties = dict(color='xkcd:beige', ha='left', fontsize='small')

default_outlier_detection_method = 'isolation_forest'
default_outlier_fraction = 0.2
avail_outlier_detection_methods = ('isolation_forest',)
# OLD -> OutLier Detection
avail_OLD_source_of_features = ('freesurfer', 't1_mri', 'func_mri', 'diffusion_mri')

default_freesurfer_dir = None
cortical_types = ('cortical_volumetric', 'cortical_contour')
label_types = ('labels_volumetric', 'labels_contour')
freesurfer_vis_types = cortical_types + label_types
visualization_combination_choices = cortical_types + label_types
default_vis_type = 'cortical_contour'

# these vis types would need to be identified by more than one label
vis_types_with_multiple_ROIs = ('labels_volumetric', 'labels_contour')

surface_view_angles = ['lateral', 'medial', 'transverse']

freesurfer_vis_cmd = 'tksurfer'

default_label_set = None

default_bids_dir = None
default_user_dir = None

default_alpha_mri = 1.0
default_alpha_seg = 0.7
default_alpha_set = (default_alpha_mri, default_alpha_seg)

min_cmap_range_t1_mri = 0
max_cmap_range_t1_mri = 1

mri_zorder_freesurfer = 0
seg_zorder_freesurfer = 1

background_value = 0 # for segmentations or MRI

default_views = (0, 1, 2)
default_num_slices = 12
default_num_rows = 2
default_padding = 5  # pixels/voxels

default_review_figsize = [15, 11]

default_navigation_options = ("Next", "Quit")
# shortcuts L, F, S have actions on matplotlib interface, so choosing other words
freesurfer_default_rating = 'rEVIEW LATER'
map_short_rating = OrderedDict(g='gOOD',
                               n='MInOR error',
                               m='mAJOR error',
                               a='FaIL',
                               d="I'M TIREd",
                               r=freesurfer_default_rating)
default_rating_list = tuple(map_short_rating.values())
index_freesurfer_default_rating = default_rating_list.index(freesurfer_default_rating)
default_rating_list_shortform = map_short_rating.keys()
ratings_not_to_be_recorded = [None,
                              '',
                              'something else',
                              "i'm tired",
                              'review later']

# for serialization
delimiter = ','
# when ratings or notes contain the above delimiter, it will be replaced by this
delimiter_replacement = ';'
# when ratings are multiple (in some use cases), how to concat them into a single string without a delimiter
rating_joiner = '+'

textbox_title = ''
textbox_initial_text = 'Notes: '  # Text(text='Your Notes:', )

color_rating_axis = 'xkcd:slate'
color_textbox_input = '#009b8c'
color_quit_axis = '#009b8c'
color_slider_axis = '#fa8072'
text_box_color = 'xkcd:grey'
text_box_text_color = 'black'
text_option_color = 'white'
color_navig_text = 'black'

position_outlier_alert = (0.950, 0.92)
position_outlier_alert_box = [0.902, 0.87, 0.097, 0.07]
position_rating_axis = [0.905, 0.65, 0.09, 0.2]
position_radio_buttons = [0.905, 0.45, 0.09, 0.23]
position_checkbox = [0.905, 0.42, 0.09, 0.25]
position_slider_seg_alpha = [0.905, 0.4, 0.07, 0.03]
position_text_input = [0.900, 0.18, 0.095, 0.10]
position_next_button = [0.905, 0.11, 0.07, 0.04]
position_quit_button = [0.905, 0.05, 0.07, 0.04]
position_navig_options = [0.905, 0.21, 0.07, 0.12]

position_zoomed_time_point = [0.7, 0.02]
position_zoomed_gradient = [0.7, 0.02]
annot_time_point = dict(fontsize='medium', color='xkcd:pale orange')
annot_gradient = dict(fontsize='medium', color='xkcd:pale orange')

review_area = dict(left=0.06, right=0.88,
                   bottom=0.06, top=0.98,
                   wspace=0.0, hspace=0.0)
bounding_box_review = (review_area['left'], review_area['bottom'],
                     review_area['right']-review_area['left'],
                     review_area['top']-review_area['bottom'])
no_blank_area = dict(left=0.01, right=0.99,
                     bottom=0.01, top=0.99,
                     wspace=0.05, hspace=0.02)

suffix_ratings_dir = 'ratings'
file_name_ratings = 'ratings.all.csv'
prefix_backup = 'backup'

# visualization layout
zoomed_position = [0.15, 0.15, 0.7, 0.7]
zoomed_position_level2 = [0.22, 0.22, 0.50, 0.50]
default_contour_face_color = 'yellow'  # '#cccc00' # 'yellow'
contour_line_width = 1
binary_pixel_value = 1
contour_level = 0.5
line_break = [np.NaN, np.NaN]

## ----------------------------------------------------------------------------
#       T1 mri specific
## ----------------------------------------------------------------------------
visual_qc_pass_indicator = 'Pass'
t1_mri_pass_indicator = visual_qc_pass_indicator
abbreviation_t1_mri_default_issue_list = {'p': t1_mri_pass_indicator,
                                          'm': 'Motion',
                                          'r': 'Ringing',
                                          'g': 'Ghosting',
                                          'c': 'Contrast',
                                          'y': 'bLurrY',
                                          'b': 'Bright',
                                          'd': 'Dark',
                                          'o': 'Orient/FOV',
                                          'w': 'Weird',
                                          's': 'Something else',
                                          't': "i'm Tired",
                                          'l': 'reView Later'}
t1_mri_default_issue_list = list(abbreviation_t1_mri_default_issue_list.values())
t1_mri_default_rating_list_shortform = abbreviation_t1_mri_default_issue_list.keys()

bounding_box_full_t1_mri = (0.01, 0.01, 0.88, 0.98)
bounding_box_main_t1_mri = (0.01, 0.12, 0.88, 0.87)
bounding_box_suppl_t1_mri = (0.01, 0.01, 0.88, 0.1)

position_checkbox_t1_mri = [0.905, 0.31, 0.09, 0.25]
position_radio_bt_t1_mri = [0.905, 0.57, 0.09, 0.10]
position_histogram_t1_mri = [0.905, 0.70, 0.09, 0.1]
position_outlier_alert_t1_mri = (0.950, 0.85)

processing_choices_t1_mri = ('Saturate',
                             'Background only',
                             'Tails_trimmed',
                             'Original')
saturate_perc_t1 = 33 # supra-threshold values are saturated.
num_bins_histogram_intensity_distribution = 100
num_bins_histogram_contrast_enhancement = 256

# outlier detection (OLD)
min_num_samples_needed = 10
t1_mri_features_OLD = ('histogram_whole_scan',)
checkbox_rect_width = 0.05
checkbox_rect_height = 0.05
checkbox_cross_color = 'xkcd:goldenrod'
checkbox_font_properties = dict(color=text_option_color,
                                fontweight='normal')  # , fontname='Arial Narrow')

title_histogram_t1_mri = 'nonzero intensities'
xticks_histogram_t1_mri = np.arange(0.1, 1.01, 0.2)
color_histogram_t1_mri = ('#c9ae74')  # sandstone

## ----------------------------------------------------------------------------
#           Functional mri specific
## ----------------------------------------------------------------------------

func_mri_pass_indicator = visual_qc_pass_indicator
# TODO Tired and Review Later must also be handled separately??
abbreviation_func_mri_default_issue_list = OrderedDict(p=func_mri_pass_indicator,
                                                       m='Motion',
                                                       r='Ringing',
                                                       s='Spikes',
                                                       g='Ghosting',
                                                       o='Orient/FOV',
                                                       w='Weird',
                                                       e='othEr',
                                                       t="i'm Tired",
                                                       v='reView later')
func_mri_default_issue_list = list(abbreviation_func_mri_default_issue_list.values())
func_mri_default_rating_list_shortform = abbreviation_func_mri_default_issue_list.keys()

func_outlier_features = None

func_mri_BIDS_filters = dict(modalities='func', types='bold')
# usually done in analyses to try keep the numbers in numerical calculations away from small values
# not important here, just for display, doing it anyways.
scale_factor_BOLD = 1000

alpha_stats_overlay = 0.5
linewidth_stats_fmri = 2
linestyle_stats_fmri = '-'

default_views_fmri = (2,)
default_num_slices_fmri = 30
default_num_rows_fmri = 5

default_name_pattern = '*.nii'

func_mri_features_OLD = ('dvars',)
colormap_stdev_fmri = 'seismic'

## ----------------------------------------------------------------------------
#           Diffusion mri specific
## ----------------------------------------------------------------------------

checkbox_rect_width_diffusion = 0.06
checkbox_rect_height_diffusion = 0.051

diffusion_mri_pass_indicator = visual_qc_pass_indicator
# f, s, l are matplotlib builtin shortcuts
abbreviation_diffusion_mri_default_issue_list = OrderedDict(p=diffusion_mri_pass_indicator,
                                                            m='Motion',
                                                            d='Dropout',
                                                            v='Vibration',
                                                            r='Ringing',
                                                            s='Spikes',
                                                            f='driFt (scanner)',
                                                            l='puLsation',
                                                            g='Geometric',
                                                            h='gHosting',
                                                            o='Orient/FOV',
                                                            i='Implausible',
                                                            e='othEr',
                                                            t="i'm Tired",
                                                            w='revieW later')
diffusion_mri_default_issue_list = list(abbreviation_diffusion_mri_default_issue_list.values())
diffusion_mri_default_rating_list_shortform = abbreviation_diffusion_mri_default_issue_list.keys()

diffusion_outlier_features = None

diffusion_mri_BIDS_filters = dict(modalities='dwi', types='dwi')
# usually done in analyses to try keep the numbers in numerical calculations away from small values
# not important here, just for display, doing it anyways.
scale_factor_diffusion = 1000

delay_in_animation_diffusion_mri = 0.3
num_times_to_animate_diffusion_mri = 4

alpha_stats_overlay_diffusion = 0.5
linewidth_stats_diffusion = 2
linestyle_stats_diffusion = '-'

default_views_diffusion = (0, 1, 2,)
default_num_slices_diffusion = 30
default_num_rows_diffusion = 5

default_name_pattern_diffusion = '*.nii'

diffusion_mri_features_OLD = ('dvars',)
colormap_stdev_diffusion = 'seismic'

choices_alignment_comparison_diffusion = ('Animate all',
                                          'Flip first & last',
                                          'Align to b=0 (animate)',
                                          'Align to b=0 (edges)')

position_rating_checkbox_diffusion  = [0.899, 0.30, 0.095, 0.35]
position_alignment_method_diffusion = [0.899, 0.66, 0.095, 0.10]
## ----------------------------------------------------------------------------


## ----------------------------------------------------------------------------
#           Registration and alignment specific
## ----------------------------------------------------------------------------

alignment_features_OLD = ('MSE', )
alignment_cmap = OrderedDict(Animate='gray',
                             Checkerboard='gray',
                             Voxelwise_diff='seismic',
                             Edges_Thinner=None,
                             Edges_Diffused=None,
                             Color_mix=None)
choices_alignment_comparison = alignment_cmap.keys()
alignment_default_vis_type = 'Edges_Thinner' # 'Checkerboard' # 'Animate'

default_checkerboard_size = None # 25
edge_threshold_alignment = 0.4
default_color_mix_alphas = (1, 1)

position_alignment_radio_button_method = [0.895, 0.45, 0.1, 0.19]
position_alignment_radio_button_rating = [0.895, 0.25, 0.1, 0.25]
position_text_input_alignment  = [0.900, 0.20, 0.09, 0.1]
position_next_button_alignment = [0.905, 0.10, 0.07, 0.03]
position_quit_button_alignment = [0.905, 0.03, 0.07, 0.03]
position_toggle_animation = [0.925, 0.63, 0.07, 0.05]

position_annotate_foreground = [0.7, 0.02]
annotate_foreground_properties = dict(fontsize='medium', color='xkcd:pale orange')

position_histogram_alignment = [0.905, 0.7, 0.09, 0.1]
title_histogram_alignment  = 'voxel-wise diff'
num_bins_histogram_alignment = 20
xticks_histogram_alignment = np.arange(0.1, 1.01, 0.2)
color_histogram_alignment  = ('#c9ae74')  # sandstone

delay_in_animation = 0.5
num_times_to_animate = 5

# edge detection and filtering
weak_edge_threshold = 60 # percentile: must be integer >1 and < 100
num_iter_weak_edge_removal = 10
median_filter_size = 3

alpha_background_slice_alignment = 1.0
alpha_edge_overlay_alignment = 1.0

## ----------------------------------------------------------------------------
#          Groupwise comparison of registration/alignment
## ----------------------------------------------------------------------------


alignment_features_groupwise = ('MSE', )
alignment_cmap_groupwise = 'seismic'
choices_alignment_comparison_groupwise = ('Std. dev map',
                                          'Animate through',
                                          'Animate with ref',
                                          'show outliers')
alignment_groupwise_default_vis_type = 'Std. dev map' # 'Checkerboard' # 'Animate'

## ----------------------------------------------------------------------------

outlier_feature_folder_name = 'features_outlier_detection'
features_outlier_detection = freesurfer_features_outlier_detection + t1_mri_features_OLD + func_mri_features_OLD


## ----------------------------------------------------------------------------
#          defacing MRI scan quality
## ----------------------------------------------------------------------------

default_defaced_mri_name = 'defaced.nii'
default_render_name = 'render.png'

defacing_pass_indicator = visual_qc_pass_indicator
abbreviation_defacing_default_issue_list = {'p': defacing_pass_indicator,
                                          'f': 'Fail',
                                          'o': 'Overstripped',
                                          'u': 'Understripped',
                                          'e': 'something Else',
                                          't': "i'm Tired",
                                          'l': 'reView Later'}
defacing_default_issue_list = list(abbreviation_defacing_default_issue_list.values())
defacing_default_rating_list_shortform = abbreviation_defacing_default_issue_list.keys()

bbox_defacing_MRI_review = (0.02, 0.02, 0.88, 0.7)
bbox_defacing_render_review = (0.02, 0.72, 0.88, 0.3)

defacing_num_rows_renders = 1

defacing_slice_locations = (20, 30, 35, 40, 50, 60, 65, 70, 80)
defacing_num_slices_per_view = len(defacing_slice_locations)
defacing_num_rows_per_view = 1
defacing_view_set = (0, 1, 2)

vis_choices_defacing = ('Defaced only',
                        'Original only',
                        'Mixed')

defacing_trim_percentile = 1
