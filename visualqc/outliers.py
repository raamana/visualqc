"""

Outlier detection module.

"""

import numpy as np
from scipy import stats
from visualqc.readers import read_aparc_stats_wholebrain, read_aseg_stats
from sklearn.ensemble import IsolationForest


def gather_freesurfer_data(fs_dir, id_list, feature_type='whole_brain'):
    """Reads all the relevant features to perform outlier detection on.

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


def detect_outliers(fs_dir, id_list_file,
                    feature_type='whole_brain',
                    fraction_of_outliers=.3):
    """Performs outlier detection based on chosen types of features."""

    outlying_ids = list()
    rng = np.random.RandomState(1984)

    id_list = [line.strip('\n ') for line in open(id_list_file)]
    num_samples = len(id_list)

    num_inliers  = int((1. - fraction_of_outliers) * num_samples)
    num_outliers = int(fraction_of_outliers * num_samples)
    given_labels = np.ones(num_samples, dtype=int)
    given_labels[-num_outliers:] = -1

    features = gather_freesurfer_data(fs_dir, id_list, feature_type)

    clf = IsolationForest(max_samples=num_samples,
                          contamination=fraction_of_outliers,
                          random_state=rng)
    clf.fit(features)
    pred_scores = clf.decision_function(features)
    pred_labels = clf.predict(features)

    threshold = stats.scoreatpercentile(pred_scores,
                                        100 * fraction_of_outliers)
    n_errors = (pred_labels != given_labels).sum()

    return outlying_ids
