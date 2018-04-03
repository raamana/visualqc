from os import makedirs
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename, dirname
from visualqc.t1_mri import cli_run
import shlex
import sys
import numpy as np
from pytest import raises, warns

test_dir = dirname(realpath(__file__))
base_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))

id_list = pjoin(base_dir, 'id_list')
fs_dir = base_dir

out_dir = pjoin(fs_dir,'new_classes_test_t1_mri')
makedirs(out_dir, exist_ok=True)

olf = 0.5

sys.argv = shlex.split('visualqc -f {} -i {} -o {} -old -s 5 -r 1'.format(fs_dir, id_list, out_dir))
cli_run()
