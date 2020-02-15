
import shlex
import sys
from os import makedirs
from os.path import dirname, join as pjoin, realpath

from visualqc.freesurfer import cli_run

test_dir = dirname(realpath(__file__))

fs_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))
id_list = pjoin(fs_dir, 'id_list')

# fs_dir = '/data1/strother_lab/praamana/ABIDE/processed/freesurfer_v5.1' # base_dir
# id_list = '/data1/strother_lab/praamana/ABIDE/processed/target_lists/list.visualqc_n10.csv'

out_dir = pjoin(fs_dir,'vqc_test')
vis_type =  'cortical_contour' # 'cortical_volumetric' #

makedirs(out_dir, exist_ok=True)

sys.argv = shlex.split('visualqc_freesurfer -f {} -i {} -o {} -v {} '
                       ''.format(fs_dir, id_list, out_dir, vis_type))
cli_run()
