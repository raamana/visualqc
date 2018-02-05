
from os import mkdir, makedirs
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename, dirname
from visualqc.vqc import run_workflow

test_dir = dirname(realpath(__file__))
base_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))

aseg_mri_dir = pjoin(base_dir,'id_001')
aseg_path = pjoin(aseg_mri_dir, 'aparc+aseg.mgz')
mri_path = pjoin(aseg_mri_dir, 'orig.mgz')
out_aseg_mri_path = pjoin(base_dir,'aseg_overlay_on_mri_test')

vis_type = 'cortical_volumetric'
fs_dir = base_dir
id_list = ('id_001', )
out_dir = pjoin(fs_dir,'qc_vis')
makedirs(out_dir, exist_ok=True)

def test_gen():
    """super basic run."""

    run_workflow(vis_type=vis_type, label_set=None,
                 fs_dir=fs_dir, id_list=id_list, out_dir=out_dir,
                 alpha_set=(0.8, 0.7))

def test_gen_label_focus():
    """super basic run."""

    run_workflow(vis_type='labels', fs_dir=fs_dir, id_list=id_list, out_dir=out_dir,
                 label_set=[17,])


def test_gen_single_view():
    """super basic run."""

    run_workflow(vis_type=vis_type, label_set=None,
                 fs_dir=fs_dir, id_list=id_list, out_dir=out_dir,
                 views=[0, ], num_slices=24, num_rows=4,
                 alpha_set=(0.8, 0.7))


# test_gen()
# test_gen_label_focus()
test_gen_single_view()
