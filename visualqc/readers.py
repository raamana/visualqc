"""

Data reader module.

"""
import numpy as np
from os.path import join as pjoin, exists as pexists, realpath
from visualqc.utils import read_id_list

def read_aseg_stats(fs_dir, id, include_global_areas=False):
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
    out_data = subcortical_data.flatten()

    if include_global_areas:
        wb_data = read_volumes_global_areas(seg_stats_file)
        out_data = np.hstack((out_data, wb_data))

    return out_data


def read_volumes_global_areas(seg_stats_file):
    """Returns the volumes of big global areas such as the ICV, Left/Right hemisphere cortical gray/white matter volume, Subcortical gray matter volume and Supratentorial volume etc.


    Order of the return values is as it appears in the original aseg.stats file (not as mentioned above).
    """

    # Snippet from the relevant part of the aseg.stats
    # Measure lhCortex, lhCortexVol, Left hemisphere cortical gray matter volume, 234615.987869, mm^3
    # Measure rhCortex, rhCortexVol, Right hemisphere cortical gray matter volume, 260948.684264, mm^3
    # Measure Cortex, CortexVol, Total cortical gray matter volume, 495564.672133, mm^3
    # Measure lhCorticalWhiteMatter, lhCorticalWhiteMatterVol, Left hemisphere cortical white matter volume, 222201.531250, mm^3
    # Measure rhCorticalWhiteMatter, rhCorticalWhiteMatterVol, Right hemisphere cortical white matter volume, 232088.671875, mm^3
    # Measure CorticalWhiteMatter, CorticalWhiteMatterVol, Total cortical white matter volume, 454290.203125, mm^3
    # Measure SubCortGray, SubCortGrayVol, Subcortical gray matter volume, 188561.000000, mm^3
    # Measure TotalGray, TotalGrayVol, Total gray matter volume, 684125.672133, mm^3
    # Measure SupraTentorial, SupraTentorialVol, Supratentorial volume, 1046623.140109, mm^3
    # Measure IntraCranialVol, ICV, Intracranial Volume, 1137205.249190, mm^3

    wb_regex_pattern = r'# Measure ([\w/+_\- ]+), ([\w/+_\- ]+), ([\w/+_\- ]+), ([\d\.]+), ([\w/+_\-^]+)'
    datatypes = np.dtype('U100,U100,U100,f8,U10')
    stats = np.fromregex(seg_stats_file, wb_regex_pattern, dtype=datatypes)
    wb_data = np.array([seg[3] for seg in stats])

    return wb_data.flatten()


def read_aparc_stats_wholebrain(fs_dir, id):
    """Convenient routine to obtain the whole brain cortical ROI stats."""

    aparc_stats = list()
    for hm in ('lh', 'rh'):
        stats_path = pjoin(fs_dir, id, 'stats', '{}.aparc.stats'.format(hm))
        hm_data = read_aparc_stats_in_hemi(stats_path)
        aparc_stats.append(hm_data)

    return np.hstack(aparc_stats)


def read_aparc_stats_in_hemi(stats_file, include_whole_brain_stats=False):
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

    stats = roi_stats_values.flatten()
    if include_whole_brain_stats:
        wb_stats = read_global_mean_surf_area_thickness(stats_file)
        stats = np.hstack((stats, wb_stats))

    return stats


def read_global_mean_surf_area_thickness(stats_file):
    """Returns total surface area of the white surface, and global mean cortical thickness"""

    # Snippet from the relevant part of aparc.stats
    # Measure Cortex, NumVert, Number of Vertices, 120233, unitless
    # Measure Cortex, WhiteSurfArea, White Surface Total Area, 85633.5, mm^2
    # Measure Cortex, MeanThickness, Mean Thickness, 2.59632, mm
    wb_regex_pattern = r'# Measure Cortex, ([\w/+_\- ]+), ([\w/+_\- ]+), ([\d\.]+), ([\w/+_\-^]+)'
    wb_aparc_dtype = np.dtype('U100,U100,f8,U10')
    # wb_aparc_dtype = [('f0', '<U100'), ('f1', '<U100'), ('f2', '<f8'), ('f3', '<U10')]
    wb_stats = np.fromregex(stats_file, wb_regex_pattern, dtype=wb_aparc_dtype)

    # concatenating while surf total area and global mean thickness
    stats = [wb_stats[1][2], wb_stats[2][2]]

    return stats


def gather_freesurfer_data(fs_dir,
                           id_list,
                           feature_type='whole_brain'):
    """
    Reads all the relevant features to perform outlier detection on.

    feature_type could be cortical, subcortical, or whole_brain.

    """

    feature_type = feature_type.lower()
    if feature_type in ['cortical', ]:
        features = np.vstack([read_aparc_stats_wholebrain(fs_dir, id) for id in id_list])
    elif feature_type in ['subcortical', ]:
        features = np.vstack([read_aseg_stats(fs_dir, id) for id in id_list])
    elif feature_type in ['whole_brain', 'wholebrain']:
        cortical = np.vstack([read_aparc_stats_wholebrain(fs_dir, id) for id in id_list])
        sub_ctx = np.vstack([read_aseg_stats(fs_dir, id) for id in id_list])
        features = np.hstack((cortical, sub_ctx))
    else:
        raise ValueError('Invalid type of features requested.')

    return features
