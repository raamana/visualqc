"""

Data reader module.

"""
import numpy as np
from os.path import exists as pexists, join as pjoin, realpath, splitext, basename
from itertools import product
from collections import Sequence
from visualqc import config as cfg


def read_aseg_stats(fs_dir, subject_id, include_global_areas=False):
    """
    Returns the volumes of both the subcortical and whole brain segmentations, found in Freesurfer output: subid/stats/aseg.stats

    Parameters
    ----------
    fs_dir : str
        Abs path to Freesurfer's SUBJECTS_DIR

    subject_id : str
        String identifying a given subject

    """

    seg_stats_file = realpath(pjoin(fs_dir, subject_id, 'stats', 'aseg.stats'))
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


def read_aparc_stats_wholebrain(fs_dir, subject_id, subset=None):
    """Convenient routine to obtain the whole brain cortical ROI stats."""

    aparc_stats = list()
    for hm in ('lh', 'rh'):
        stats_path = pjoin(fs_dir, subject_id, 'stats', '{}.aparc.stats'.format(hm))
        hm_data = read_aparc_stats_in_hemi(stats_path, subset)
        aparc_stats.append(hm_data)

    return np.hstack(aparc_stats)


def read_aparc_stats_in_hemi(stats_file,
                             subset=None,
                             include_whole_brain_stats=False):
    """Read statistics on cortical features (such as thickness, curvature etc) produced by Freesurfer.

    file_path would contain whether it is from the right or left hemisphere.

    """

    stats_file = realpath(stats_file)
    if not pexists(stats_file):
        raise IOError('given path does not exist : {}'.format(stats_file))

    # ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
    aparc_roi_dtype = [('StructName', 'S50'),
                       ('NumVert', '<i4'),
                       ('SurfArea', '<i4'),
                       ('GrayVol', '<i4'),
                       ('ThickAvg', '<f4'),
                       ('ThickStd', '<f4'),
                       ('MeanCurv', '<f4'),
                       ('GausCurv', '<f4'),
                       ('FoldInd', '<f4'),
                       ('CurvInd', '<f4')]

    subset_all = ['SurfArea', 'GrayVol',
                  'ThickAvg', 'ThickStd',
                  'MeanCurv', 'GausCurv',
                  'FoldInd', 'CurvInd']
    if subset is None or not isinstance(subset, Sequence):
        subset_return = subset_all
    else:
        subset_return = [st for st in subset if st in subset_all]
        if len(subset_return) < 1:
            raise ValueError('Atleast 1 valid stat must be chosen! '
                             'From: \n{}'.format(subset_all))

    roi_stats = np.genfromtxt(stats_file, dtype=aparc_roi_dtype, filling_values=np.NaN)
    roi_stats_values = np.full((len(roi_stats), len(subset_return)), np.NaN)
    for idx, stat in enumerate(roi_stats):
        roi_stats_values[idx, :] = [stat[feat] for feat in subset_return]

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


def gather_freesurfer_data(qcw,
                           feature_type='whole_brain'):
    """
    Reads all the relevant features to perform outlier detection on.

    feature_type could be cortical, subcortical, or whole_brain.

    """

    if qcw.source_of_features not in cfg.avail_OLD_source_of_features:
        raise NotImplementedError('Reader for the given source of features ({}) '
                                  'is currently not implemented.'.format(
            qcw.source_of_features))

    feature_type = feature_type.lower()
    if feature_type in ['cortical', ]:
        features = np.vstack(
            [read_aparc_stats_wholebrain(qcw.in_dir, id) for id in qcw.id_list])
    elif feature_type in ['subcortical', ]:
        features = np.vstack([read_aseg_stats(qcw.fs_dir, id) for id in qcw.id_list])
    elif feature_type in ['whole_brain', 'wholebrain']:
        cortical = np.vstack(
            [read_aparc_stats_wholebrain(qcw.in_dir, id) for id in qcw.id_list])
        sub_ctx = np.vstack([read_aseg_stats(qcw.fs_dir, id) for id in qcw.id_list])
        features = np.hstack((cortical, sub_ctx))
    else:
        raise ValueError('Invalid type of features requested.')

    return features


def gather_data(path_list, id_list):
    """
    Takes in a list of CSVs, and return a table of features.

    id_list is to ensure the row order in the matrix.

    """

    features = np.vstack([np.genfromtxt(path_list[sid]) for sid in id_list])

    return features


def anatomical_traverse_bids(bids_layout,
                            modalities='anat',
                            subjects=None,
                            sessions=None,
                            extensions=('nii', 'nii.gz', 'json'),
                            param_files_required=False,
                            **kwargs):
    """
    Builds a convenient dictionary of usable anatomical subjects/sessions.
    """

    meta_types = {'datatype'  : modalities,
                  'extensions': extensions,
                  'subjects'  : subjects,
                  'sessions'  : sessions}

    meta_types.update(kwargs)
    non_empty_types = {type_: values for type_, values in meta_types.items() if values}

    __FIELDS_TO_IGNORE__ = ('filename', 'modality', 'type')
    __TYPES__ = ['subjects', 'sessions',]

    results = bids_layout.get(**non_empty_types)
    if len(results) < 1:
        print('No results found!')
        return None, None

    all_subjects = bids_layout.get_subjects()
    all_sessions = bids_layout.get_sessions()
    if len(all_sessions) > 1:
        sessions_exist = True
        combinations = product(all_subjects, all_sessions)
    else:
        sessions_exist = False
        combinations = all_subjects


    reqd_exts_params = ('.json', )
    named_exts_params = ('params', )
    reqd_exts_images = ('.nii', '.gz')
    named_exts_images = ('image', 'image')

    files_by_id = dict()
    for sub in combinations:
        if sessions_exist:
            # sub is a tuple of subject,session
            results = bids_layout.get(subject=sub[0], session=sub[1],
                                      datatype='anat')
            final_sub_id = '_'.join(sub)
        else:
            results = bids_layout.get(subject=sub,  datatype='anat')
            final_sub_id = sub

        temp = {splitext(file.filename)[-1] : realpath(file.path)
                for file in results}

        param_files_exist = all([file_ext in temp for file_ext in reqd_exts_params])
        image_files_exist = any([file_ext in temp for file_ext in reqd_exts_images])
        if param_files_required and (not param_files_exist):
            print('parameter files are required, but do not exist for {}'
                  ' - skipping it.'.format(sub))
            continue

        if not image_files_exist:
            print('Image file is required, but does not exist for {}'
                  ' - skipping it.'.format(sub))
            continue

        files_by_id[final_sub_id] = dict()
        # only when all the files required exist, do we include it for review
        # adding parameter files, only if they exist
        if param_files_exist:
            files_by_id[final_sub_id] = { new_ext : temp[old_ext]
                                 for old_ext, new_ext in zip(reqd_exts_params,
                                                             named_exts_params)}
        else:
            files_by_id[final_sub_id]['params'] = 'None'

        # adding the image file
        files_by_id[final_sub_id]['image'] = temp['.nii'] \
            if 'nii' in temp else temp['.gz']

    return files_by_id


def find_anatomical_images_in_BIDS(bids_dir):
    """Traverses the BIDS structure to find all the relevant anatomical images."""

    from bids import BIDSLayout
    bids_layout = BIDSLayout(bids_dir)
    images = anatomical_traverse_bids(bids_layout)
    # file name of each scan is the unique identifier,
    #   as it essentially contains all the key info.
    images_by_id = {basename(sub_data['image']): sub_data
                       for _, sub_data in images.items()}
    id_list = np.array(list(images_by_id.keys()))

    return id_list, images_by_id


def diffusion_traverse_bids(bids_layout,
                            modalities='dwi',
                            subjects=None,
                            sessions=None,
                            extensions=('nii', 'nii.gz',
                                        'bval', 'bvec', 'json'),
                            param_files_required=False,
                            **kwargs):
    """
    Builds a convenient dictionary of usable DWI subjects/sessions.

    """

    meta_types = {'datatype'  : modalities,
                  'extensions': extensions,
                  'subjects'  : subjects,
                  'sessions'  : sessions}

    meta_types.update(kwargs)
    non_empty_types = {type_: values for type_, values in meta_types.items() if values}

    __FIELDS_TO_IGNORE__ = ('filename', 'modality', 'type')
    __TYPES__ = ['subjects', 'sessions',]

    results = bids_layout.get(**non_empty_types)
    if len(results) < 1:
        print('No results found!')
        return None, None

    all_subjects = bids_layout.get_subjects()
    all_sessions = bids_layout.get_sessions()
    if len(all_sessions) > 1:
        sessions_exist = True
        combinations = product(all_subjects, all_sessions)
    else:
        sessions_exist = False
        combinations = all_subjects


    reqd_exts_params = ('.bval', '.bvec', '.json')
    named_exts_params = ('bval', 'bvec', 'params')
    reqd_exts_images = ('.nii', '.gz')
    named_exts_images = ('image', 'image')

    files_by_id = dict()
    for sub in combinations:
        if sessions_exist:
            # sub is a tuple of subject,session
            results = bids_layout.get(subject=sub[0], session=sub[1], datatype='dwi')
            final_sub_id = '_'.join(sub)
        else:
            results = bids_layout.get(subject=sub,  datatype='dwi')
            final_sub_id = sub

        temp = {splitext(file.filename)[-1] : realpath(file.path)
                for file in results}

        param_files_exist = all([file_ext in temp for file_ext in reqd_exts_params])
        image_files_exist = any([file_ext in temp for file_ext in reqd_exts_images])
        if param_files_required and not param_files_exist:
            print('b-value/b-vec are required, but do not exist for {}'
                  ' - skipping it.'.format(sub))
            continue

        if not image_files_exist:
            print('Image file is required, but does not exist for {}'
                  ' - skipping it.'.format(sub))
            continue

        files_by_id[final_sub_id] = dict()
        # only when all the files required exist, do we include it for review
        # adding parameter files, only if they exist
        if param_files_exist:
            files_by_id[final_sub_id] = { new_ext : temp[old_ext]
                                 for old_ext, new_ext in zip(reqd_exts_params,
                                                             named_exts_params)}
        else:
            # assuming the first volume is b=0
            files_by_id[final_sub_id]['bval'] = 'assume_first'
            # indicating the absence with None
            files_by_id[final_sub_id]['bvec'] = None

        # adding the image file
        files_by_id[final_sub_id]['image'] = temp['.nii'] if 'nii' in temp else temp['.gz']

    return files_by_id


def traverse_bids(bids_layout, modalities='func', types='bold',
                  subjects=None, sessions=None, runs=None,
                  tasks=None, events=None, extensions=('nii', 'nii.gz'),
                  **kwargs):
    """
    Dataset traverser.

    Args:
        subjects: list of subjects
        sessions: list of sessions
        runs: list of runs
        ...
        kwargs : values for the particular type chosen.

    Returns:
        tuple of existing combinations
            first item: list of type names identifying the file
            second item: path to the file identified by the above types.

    """

    meta_types = {'modality'  : modalities,
                  'type'      : types,
                  'extensions': extensions,
                  'subjects'  : subjects,
                  'sessions'  : sessions,
                  'runs'      : runs,
                  'tasks'     : tasks,
                  'events'    : events}
    meta_types.update(kwargs)
    non_empty_types = {type_: values for type_, values in meta_types.items() if values}

    __FIELDS_TO_IGNORE__ = ('filename', 'modality', 'type')
    __TYPES__ = ['subjects', 'sessions', 'tasks', 'runs', 'events']

    results = bids_layout.get(**non_empty_types)
    if len(results) < 1:
        print('No results found!')
        return None, None

    common_field_set = _unique_in_order(results[0]._fields)
    if len(results) > 1:
        for res in results[1:]:
            _field_set = _unique_in_order(res._fields)
            common_field_set = [ff for ff in common_field_set if ff in _field_set]

    final_fields = [unit for unit in common_field_set if unit not in __FIELDS_TO_IGNORE__]
    # TODO final_fields can still have duplicates like: ( 'acquisition', 'acq'); handle it.

    if len(final_fields) < 1:
        return None, None

    # print('Dataset will be traversed for different values of:\n {}'.format(final_fields))
    unit_paths = [[[file.__getattribute__(unit) for unit in final_fields], file.filename]
                  for file in results]

    return final_fields, unit_paths


def _unique_in_order(seq):
    """
    Utility to preserver order while making a set of unique elements.

    Copied from Markus Jarderot's answer at
     https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order

    Args:
        seq : sequence
    Returns:
        unique_list : list
            List of unique elements in their original order

    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
