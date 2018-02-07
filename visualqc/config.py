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

default_alpha_mri = 0.8
default_alpha_seg = 0.7
default_alpha_set = (0.7, 0.7)

default_views = (0, 1, 2)
default_num_slices = 12
default_num_rows = 2
default_padding = 5 # pixels/voxels

suffix_ratings_dir = 'ratings'
file_name_ratings = 'ratings.all.csv'
file_name_ratings_backup = 'backup_ratings.all.csv'

# visualization layout
zoomed_position = [0.15, 0.15, 0.7, 0.7]
contour_face_color = '#cccc00' # 'yellow'
contour_line_width = 1
binary_pixel_value = 1
contour_level = 0.5
