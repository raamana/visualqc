"""

Data reader module.

"""
import numpy as np
from os.path import join as pjoin, exists as pexists, realpath

def read_aseg_stats(fs_dir, id):
    """
    Returns the volumes of both the subcortical and whole brain segmentations, found in Freesurfer output: subid/stats/aseg.stats

    Parameters
    ----------
    fs_dir : str
        Abs path to Freesurfer's SUBJECTS_DIR

    id : str
        String identifying a given subject

    """

    seg_stats_file = realpath(pjoin(fs_dir, id, 'stats', 'aseg.stats'))
    if not pexists(seg_stats_file):
        raise IOError('given path does not exist : {}'.format(seg_stats_file))

    stats = np.loadtxt(seg_stats_file, dtype="i1,i1,i4,f4,S50,f4,f4,f4,f4,f4")
    # returning volumes only:
    subcortical_data = np.array([seg[3] for seg in stats])

    wb_regex_pattern = r'# Measure ([\w/+_\- ]+), ([\w/+_\- ]+), ([\w/+_\- ]+), ([\d\.]+), ([\w/+_\-^]+)'
    datatypes = np.dtype('U100,U100,U100,f8,U10')
    stats = np.fromregex(seg_stats_file, wb_regex_pattern, dtype=datatypes)
    wb_data = np.array([seg[3] for seg in stats])

    out_data = np.hstack((subcortical_data.flatten(), wb_data.flatten()))

    return out_data


def read_aparc_stats_wholebrain(fs_dir, id):
    """Convenient routine to obtain the whole brain cortical ROI stats."""

    aparc_stats = list()
    for hm in ('lh', 'rh'):
        stats_path = pjoin(fs_dir, id, 'stats', '{}.aparc.stats'.format(hm))
        hm_data = read_aparc_stats(stats_path)
        aparc_stats.append(hm_data)

    return np.hstack(aparc_stats)


def read_aparc_stats(stats_file):
    """Read statistics on cortical features (such as thickness, curvature etc) produced by Freesurfer.

    file_path would contain whether it is from the right or left hemisphere.

    """

    stats_file = realpath(stats_file)
    if not pexists(stats_file):
        raise IOError('given path does not exist : {}'.format(stats_file))

    # ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
    aparc_roi_dtype = [('StructName', 'S50'), ('NumVert', '<i4'), ('SurfArea', '<i4'), ('GrayVol', '<i4'),
                       ('ThickAvg', '<f4'), ('ThickStd', '<f4'), ('MeanCurv', '<f4'), ('GausCurv', '<f4'),
                       ('FoldInd', '<f4'), ('CurvInd', '<f4')]
    roi_stats = np.genfromtxt(stats_file, dtype=aparc_roi_dtype, filling_values=np.NaN)
    subset = ['SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']
    roi_stats_values = np.full((len(roi_stats), len(subset)), np.NaN)
    for idx, stat in enumerate(roi_stats):
        roi_stats_values[idx, :] = [stat[feat] for feat in subset]

    # whole cortex
    # Measure Cortex, NumVert, Number of Vertices, 120233, unitless
    # Measure Cortex, WhiteSurfArea, White Surface Total Area, 85633.5, mm^2
    # Measure Cortex, MeanThickness, Mean Thickness, 2.59632, mm
    wb_regex_pattern = r'# Measure Cortex, ([\w/+_\- ]+), ([\w/+_\- ]+), ([\d\.]+), ([\w/+_\-^]+)'
    wb_aparc_dtype = np.dtype('U100,U100,f8,U10')
    # wb_aparc_dtype = [('f0', '<U100'), ('f1', '<U100'), ('f2', '<f8'), ('f3', '<U10')]
    wb_stats = np.fromregex(stats_file, wb_regex_pattern, dtype=wb_aparc_dtype)

    # concatenating while surf total area and global mean thickness
    stats = np.hstack((roi_stats_values.flatten(), (wb_stats[1][2], wb_stats[2][2])))

    return stats

