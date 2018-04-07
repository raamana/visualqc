
from os import makedirs
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename, dirname
from visualqc.freesurfer import cli_run
import shlex
import sys
import numpy as np
from pytest import raises, warns

test_dir = dirname(realpath(__file__))
base_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))
id_list = pjoin(base_dir, 'id_list')

# base_dir = '/data1/strother_lab/praamana/ABIDE/processed/freesurfer_v5.1' # base_dir
# id_list = '/data1/strother_lab/praamana/ABIDE/processed/target_lists/list.visualqc_n10.csv'

out_dir = pjoin(base_dir,'vqc_test')
vis_type =  'cortical_contour' # 'cortical_volumetric' #

makedirs(out_dir, exist_ok=True)

sys.argv = shlex.split('visualqc_freesurfer -f {} -i {} --seg_name aparc+aseg.mgz '
                       '-o {} -v {} -old '
                       ''.format(base_dir, id_list, out_dir, vis_type))
cli_run()
