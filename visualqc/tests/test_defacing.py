import shlex
import sys
from os import makedirs
from os.path import dirname, join as pjoin, realpath

from visualqc.defacing import cli_run

test_dir = dirname(realpath(__file__))
base_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))

id_list = pjoin(base_dir, 'id_list_defacing')
name_defaced = 'defaced_noniso2.nii.gz'  # 'defaced.nii.gz'
name_mri = 'orig_noniso2.nii.gz'  # 'orig.nii.gz'
name_rendered = 'rendered.png'

slice_locations = ' -s 40 45 47 50 55 57 60 '

out_dir = pjoin(base_dir, 'out_defacing')
makedirs(out_dir, exist_ok=True)

sys.argv = shlex.split('visualqc_defacing -u {} -i {} -o {} '
                       '-d {} -m {} -r {} '
                       ' {} '  # slice locations
                       ''.format(base_dir, id_list, out_dir,
                                 name_defaced, name_mri, name_rendered,
                                 slice_locations))
cli_run()
