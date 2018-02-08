
from os import mkdir, makedirs
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename, dirname
from visualqc.vqc import run_workflow, cli_run
import shlex
import sys

test_dir = dirname(realpath(__file__))
base_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))

aseg_mri_dir = pjoin(base_dir,'id_001')
aseg_path = pjoin(aseg_mri_dir, 'aparc+aseg.mgz')
mri_path = pjoin(aseg_mri_dir, 'orig.mgz')
out_aseg_mri_path = pjoin(base_dir,'aseg_overlay_on_mri_test')

vis_type = 'cortical_contour'
fs_dir = '/data1/strother_lab/praamana/ABIDE/processed/freesurfer_v5.1'
id_list = '/data1/strother_lab/praamana/ABIDE/processed/target_lists/list.visualqc_n10.csv'
out_dir = pjoin(fs_dir,'qc_vis')
makedirs(out_dir, exist_ok=True)


def test_gen():
    "function to hit the CLI lines."

    sys.argv = shlex.split('visualqc -f {} -i {} -o {} -v {}'.format(fs_dir, id_list, out_dir, vis_type))
    cli_run()


def test_gen_label_focus():
    """super basic run."""

    sys.argv = shlex.split('visualqc -f {} -i {} -o {} -v {} -l 17'.format(fs_dir, id_list, out_dir, 'labels_contour'))
    cli_run()


def test_gen_single_view():
    """super basic run."""

    sys.argv = shlex.split('visualqc -f {} -i {} -o {} -v {} '
                           '-w 0 1 -s 16 -r 6 -a 1.0 0.7'.format(fs_dir, id_list, out_dir, 'cortical_contour'))
    cli_run()



# test_gen()
test_gen_label_focus()
# test_gen_single_view()
