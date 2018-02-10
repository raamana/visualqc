"""
Central place to store the config info

"""
import numpy as np

# default values
default_out_dir_name = 'visualqc'
annot_vis_dir_name = 'annot_visualizations'
default_mri_name = 'brainmask.mgz'
default_seg_name = 'aparc+aseg.mgz'
required_files = (default_mri_name, default_seg_name)

default_freesurfer_dir = None
cortical_types = ('cortical_volumetric', 'cortical_contour')
label_types = ('labels_volumetric', 'labels_contour')
freesurfer_vis_types = cortical_types + label_types
visualization_combination_choices = cortical_types + label_types
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
default_padding = 5  # pixels/voxels

default_navigation_options = ("Next", "Quit")
default_rating_list = ('Good', 'Suspect', 'Bad', 'Failed', 'Later')
map_short_rating = dict(g='Good', s='Suspect', b='Bad', f='Failed', l='Later')
default_rating_list_shortform = map_short_rating.keys()

textbox_title = ''
textbox_initial_text = 'Your Notes:' #Text(text='Your Notes:', )

color_rating_axis = '#009b8c'
color_textbox_input = '#009b8c'
color_quit_axis = '#0084b4'
color_slider_axis = '#fa8072'
text_box_color = 'xkcd:grey'
text_box_text_color = 'black'
text_option_color = 'white'

position_rating_axis = [0.905, 0.78, 0.085, 0.2]
position_slider_seg_alpha = [0.905, 0.74, 0.07, 0.02]
position_text_input = [0.9, 0.4, 0.095, 0.3]
position_navig_options = [0.905, 0.27, 0.07, 0.12]
annot_position = (0.95, 0.03)

review_area = dict(left=0.01, right=0.9,
                   bottom=0.01, top=0.99,
                   wspace=0.05, hspace=0.02)
no_blank_area = dict(left=0.01, right=0.99,
                     bottom=0.01, top=0.99,
                     wspace=0.05, hspace=0.02)

suffix_ratings_dir = 'ratings'
file_name_ratings = 'ratings.all.csv'
file_name_ratings_backup = 'backup_ratings.all.csv'

# visualization layout
zoomed_position = [0.15, 0.15, 0.7, 0.7]
default_contour_face_color = 'yellow'  # '#cccc00' # 'yellow'
contour_line_width = 1
binary_pixel_value = 1
contour_level = 0.5
line_break = [np.NaN, np.NaN]
