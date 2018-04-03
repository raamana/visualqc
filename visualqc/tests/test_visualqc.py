
from os import makedirs
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename, dirname
from visualqc.vqc import cli_run
import shlex
import sys
import numpy as np
from pytest import raises, warns

test_dir = dirname(realpath(__file__))
base_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))

id_list = pjoin(base_dir, 'id_list')
fs_dir = base_dir

out_dir = pjoin(fs_dir,'vqc_test')
makedirs(out_dir, exist_ok=True)

# id_list = '/data1/strother_lab/praamana/ABIDE/processed/target_lists/list.visualqc.outlier_detect'
# fs_dir = '/data1/strother_lab/praamana/ABIDE/processed/freesurfer_v5.1'
# out_dir = '/data1/strother_lab/praamana/ABIDE/processed/freesurfer_v5.1/qc_vis'
# sys.argv = shlex.split('visualqc -f {} -i {} -o {} --prepare'.format(fs_dir, id_list, out_dir))
# cli_run()

print('Due to nature of the application and design requiring UI and graphics for the app to run, '
      ' core functionality testing can not be auto tested in a typical CI framework!! '
      ' Contributors are greatly welcome.')

# sys.exit(0)

# def test_gen():
#     "function to hit the CLI lines."
#
#     sys.argv = shlex.split('visualqc -f {} -i {} -o {}'.format(fs_dir, id_list, out_dir))
#     cli_run()
#
#
# def test_gen_label_focus():
#     """super basic run."""
#
#     sys.argv = shlex.split('visualqc -f {} -i {} -o {} -v {} -l 17'.format(fs_dir, id_list, out_dir, 'labels_contour'))
#     cli_run()
#
#
# def test_gen_single_view():
#     """super basic run."""
#
#     sys.argv = shlex.split('visualqc -f {} -i {} -o {} -v {} '
#                            '-w 0 1 -s 16 -r 6 -a 1.0 0.7'.format(fs_dir, id_list, out_dir, 'cortical_contour'))
#     cli_run()


def test_invalid_usage_vis_type():
    ""

    with raises((NotImplementedError, AttributeError, SystemExit)):
        sys.argv = shlex.split('visualqc -f {} -i {} -o {} --vis_type random_name_sngkkjfdk'.format(fs_dir, id_list, out_dir))
        cli_run()


    for color in ['invalid_color_skjfsw034', 'redBlah', 'bluuuu', 'xkcd:random']:
        with raises((ValueError, NotImplementedError, AttributeError, SystemExit)):
            sys.argv = shlex.split('visualqc -f {} -i {} -o {} --contour_color {color}'.format(fs_dir, id_list, out_dir, color=color))
            cli_run()

    for al in [np.nan, np.Inf, -1, +2, 'lol']:
        with raises((ValueError, NotImplementedError, AttributeError, SystemExit)):
            sys.argv = shlex.split('visualqc -f {} -i {} -o {} --alpha_set {al} {al}'.format(fs_dir, id_list, out_dir, al=al))
            cli_run()


def test_invalid_usage_old():
    "Ensures invalid usages are always caught in outlier detection (old)."

    # OLD is not implemented for generic usage
    with raises(NotImplementedError):
        sys.argv = shlex.split('visualqc -u {} -i {} -o {} --vis_type labels_contour'.format(fs_dir, id_list, out_dir))
        cli_run()

    with raises(NotImplementedError):
        sys.argv = shlex.split('visualqc -f {} -i {} -o {} --outlier_method random_name_sngkkjfdk'.format(fs_dir, id_list, out_dir))
        cli_run()

    with raises(NotImplementedError):
        sys.argv = shlex.split('visualqc -f {} -i {} -o {} --outlier_feat_types random_name_sngkkjfdk'.format(fs_dir, id_list, out_dir))
        cli_run()

    for invalid_val in [-0.1, +1.1, 1.0, 0.0, np.nan, np.Inf]:
        with raises(ValueError):
            sys.argv = shlex.split('visualqc -f {} -i {} -o {} '
                                   '--outlier_fraction {val}'.format(fs_dir, id_list, out_dir, val=invalid_val))
            cli_run()

# test_gen()
# test_gen_label_focus()
# test_gen_single_view()

# test_invalid_usage_old()
