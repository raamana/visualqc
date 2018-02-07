"""
Central place to store the config info

"""

# default values
default_out_dir_name = 'visualqc'
annot_vis_dir_name = 'annot_visualizations'
default_mri_name = 'brainmask.mgz'
default_seg_name = 'aparc+aseg.mgz'
required_files = (default_mri_name, default_seg_name)

default_freesurfer_dir = None
freesurfer_vis_types = ('cortical_volumetric', 'cortical_contour',
                        'labels_volumetric', 'labels_contour')
visualization_combination_choices = ('cortical_volumetric', 'cortical_contour',
                                     'labels_volumetric', 'labels_contour')
default_vis_type = 'cortical_contour'

freesurfer_vis_cmd = 'tksurfer'

default_label_set = None

default_user_dir = None

default_alpha_mri = 1.0
default_alpha_seg = 0.7
default_alpha_set = (default_alpha_mri, default_alpha_seg)

default_views = (0, 1, 2)
default_num_slices = 12
default_num_rows = 2
default_padding = 5 # pixels/voxels

default_rating_list = ('Good', 'Suspect', 'Bad', 'Failed', 'Later')
default_navigation_options = ("Next", "Quit")

position_rating_axis = [0.905, 0.8, 0.085, 0.18]
position_navig_options = [0.905, 0.59, 0.065, 0.1]
position_slider_seg_alpha =  [0.905, 0.73, 0.07, 0.02]
annot_position = (0.95, 0.02)

suffix_ratings_dir = 'ratings'
file_name_ratings = 'ratings.all.csv'
file_name_ratings_backup = 'backup_ratings.all.csv'

# visualization layout
zoomed_position = [0.15, 0.15, 0.7, 0.7]
contour_face_color = '#cccc00' # 'yellow'
contour_line_width = 1
binary_pixel_value = 1
contour_level = 0.5
