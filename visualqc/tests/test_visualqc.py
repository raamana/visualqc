
from os import mkdir, makedirs
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename, dirname
from visualqc.vqc import generate_visualizations

test_dir = dirname(realpath(__file__))
base_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))

aseg_mri_dir = pjoin(base_dir,'id_001')
aseg_path = pjoin(aseg_mri_dir, 'aparc+aseg.mgz')
mri_path = pjoin(aseg_mri_dir, 'orig.mgz')
out_aseg_mri_path = pjoin(base_dir,'aseg_overlay_on_mri_test')

make_type = 'cortical_volumetric'
fs_dir = base_dir
id_list = ('id_001', )
out_dir = pjoin(fs_dir,'qc_vis')
makedirs(out_dir, exist_ok=True)

generate_visualizations(make_type=make_type, fs_dir=fs_dir, id_list=id_list, out_dir=out_dir)
