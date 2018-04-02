
from os import makedirs
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename, dirname
from visualqc.t1_mri import cli_run
import shlex
import sys
import numpy as np
from visualqc.functional_mri import FmriRatingWorkflow

test_dir = dirname(realpath(__file__))
# base_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))

base_dir = '/home/praamana/BIDS'
bids_dir = pjoin(base_dir, 'ds114_R2')

out_dir = pjoin(base_dir,'vqc_func_ds114')
makedirs(out_dir, exist_ok=True)

wf = FmriRatingWorkflow(in_dir=bids_dir, out_dir=out_dir)
wf.run()
