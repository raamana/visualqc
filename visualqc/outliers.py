"""

Outlier detection module.

"""
__all__ = ['outlier_advisory']

from genericpath import exists as pexists
from os import makedirs

import numpy as np
from os.path import join as pjoin
from scipy import stats
from sklearn.ensemble import IsolationForest

import visualqc.config as cfg
from visualqc.readers import gather_freesurfer_data


def outlier_advisory(qcw):
    """
    Performs outlier detection based on chosen types of data and technique.

    Returns
    -------
    outliers_by_sample : dict
        Keyed in by sample id, each element is a list of features that identified
        a given ID as a possible outlier.

    outliers_by_feature : dict
        Keyed in by feature, each element is a list of IDs that feature identified
        as possible outliers.

    """

    if not pexists(qcw.out_dir):
        makedirs(qcw.out_dir)

    outliers_by_feature = dict()
    outliers_by_sample = dict()

    if qcw.disable_outlier_detection:
        print('outlier detection: disabled, as requested.')
        return outliers_by_sample, outliers_by_feature

    for feature_type in qcw.outlier_feat_types:
        print('\nRunning outlier detection based on {} measures:'.format(feature_type))
        features = gather_freesurfer_data(qcw, feature_type)
        out_file = pjoin(qcw.out_dir, '{}_{}_{}.txt'.format(cfg.outlier_list_prefix,
                                                            qcw.outlier_method,
                                                            feature_type))
        outliers_by_feature[feature_type] = detect_outliers(
            features, qcw.id_list, method=qcw.outlier_method,
            out_file=out_file, fraction_of_outliers=qcw.outlier_fraction)

    # re-organizing the identified outliers by sample
    for sid in qcw.id_list:
        # each id contains a list of all feature types that flagged it as an outlier
        outliers_by_sample[sid] = [feat for feat in qcw.outlier_feat_types if
                                   sid in outliers_by_feature[feat]]

    # dropping the IDs that were not flagged by any feature
    # so a imple ID in dict would reveal whether it was ever suspected as an outlier
    outliers_by_sample = {id: flag_list
                          for id, flag_list in outliers_by_sample.items()
                          if flag_list}

    return outliers_by_sample, outliers_by_feature


def detect_outliers(features,
                    id_list,
                    method='isolation_forest',
                    fraction_of_outliers=.3,
                    out_file=None):
    """Performs outlier detection based on chosen feature type and OD technique."""

    method = method.lower()
    if method == 'isolation_forest':
        outlying_ids = run_isolation_forest(features, id_list,
                                            fraction_of_outliers=fraction_of_outliers)
    else:
        raise NotImplementedError(
            'Chosen detection method {} not implemented or invalid.'.format(method))

    # printing out info on detected outliers
    if len(outlying_ids) > 0:
        print('\nPossible outliers ({} / {}):'.format(len(outlying_ids),len(id_list)))
        print('\n'.join(outlying_ids))
    else:
        print('\nNo outliers were detected!\n\n')

    # writing out to a file, if requested
    if out_file is not None:
        np.savetxt(out_file, outlying_ids, fmt='%s', delimiter='\n')

    return outlying_ids


def run_isolation_forest(features, id_list, fraction_of_outliers=.3):
    """Performs anomaly detection based on Isolation Forest."""

    rng = np.random.RandomState(1984)

    num_samples = features.shape[0]
    iso_f = IsolationForest(max_samples=num_samples,
                            contamination=fraction_of_outliers,
                            random_state=rng)
    iso_f.fit(features)
    pred_scores = iso_f.decision_function(features)

    threshold = stats.scoreatpercentile(pred_scores, 100 * fraction_of_outliers)
    outlying_ids = id_list[pred_scores < threshold]

    return outlying_ids
