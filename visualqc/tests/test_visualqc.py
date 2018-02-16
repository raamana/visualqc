
from os import makedirs
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename, dirname
from visualqc.vqc import cli_run
import shlex
import sys

test_dir = dirname(realpath(__file__))
base_dir = realpath(pjoin(test_dir, '..', '..', 'example_datasets'))

id_list = pjoin(base_dir, 'id_list')
fs_dir = base_dir

out_dir = pjoin(fs_dir,'vqc_test')
makedirs(out_dir, exist_ok=True)


print('Due to nature of the application and design requiring UI and graphics for the app to run, core functionality testing can not be auto tested in a typical CI framework!!')

# sys.exit(0)
#
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



# test_gen()
# test_gen_label_focus()
# test_gen_single_view()
