
from os import mkdir, makedirs
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename
from visualqc.vqc import generate_visualizations

aseg_mri_dir = '/home/praamana/visualqc/example_datasets/id_001'
aseg_path = pjoin(aseg_mri_dir, 'aparc+aseg.mgz')
mri_path = pjoin(aseg_mri_dir, 'orig.mgz')
out_aseg_mri_path = '/home/praamana/aseg_overlay_on_mri_test'

make_type = 'cortical_volumetric'
fs_dir = '/home/praamana/visualqc/example_datasets'
id_list = ('id_001', )
out_dir = pjoin(fs_dir,'qc_vis')
makedirs(out_dir)

generate_visualizations(make_type=make_type, fs_dir=fs_dir, id_list=id_list, out_dir=out_dir)
