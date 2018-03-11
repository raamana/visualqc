"""

Module to define base classes.

"""

import sys
from abc import ABC, abstractmethod
from os.path import join as pjoin
from visualqc import config as cfg



class BaseWorkflow(ABC):
    """
    Class defining the base workflow for visualqc.
    """

    def __init__(self,
                 id_list,
                 in_dir,
                 out_dir,
                 outlier_method, outlier_fraction,
                 outlier_feat_types, disable_outlier_detection):
        """Constructor"""

        # super().__init__()

        self.id_list = id_list
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.outlier_method = outlier_method
        self.outlier_fraction = outlier_fraction
        self.outlier_feat_types = outlier_feat_types
        self.disable_outlier_detection = disable_outlier_detection

        self.feature_extractor = None

    def save_cmd(self):
        """Saves the command issued by the user for debugging purposes"""

        cmd_file = pjoin(self.out_dir,
                         'cmd_issued.visualqc.{}'.format(self.__name__))
        with open(cmd_file, 'w') as cf:
            cf.write('{}\n'.format(' '.join(sys.argv)))

        return

    def save(self):
        """
        Saves the state of the QC workflow for restoring later on,
            as well as for future reference.

        """

        pass

    def reload(self):
        """Method to reload the saved state."""

        pass

    def extract_features(self):
        """
        Feature extraction method (as part of pre-processing),
        producing the input to outlier detection module.

        Could be redefined by child class to be empty if no need (like Freesurfer).

        """

        self.feature_paths = dict()
        for feat_type in self.outlier_feat_types:
            try:
                print('Extracting feature type: {}'.format(feat_type))
                self.feature_paths[feat_type] = self.feature_extractor(self, feat_type)
            except:
                print('Unable to extract {} features - skipping them.'.format(feat_type))

    def detect_outliers(self):
        """Runs outlier detection and reports the ids flagged as outliers."""

        # outliers categorized
        self.by_feature = dict()
        self.by_sample = dict()

        if self.disable_outlier_detection:
            print('outlier detection: disabled, as requested.')
            return

        if len(self.feature_paths)<1:
            print('Features required for outlier detection are not available - skipping it.')
            return

        from visualqc.outliers import detect_outliers
        from visualqc.readers import gather_data
        for feature_type in self.outlier_feat_types:
            features = gather_data(self.feature_paths[feature_type])
            if features.shape[0] > self.outlier_fraction*len(self.id_list):
                print('\nRunning outlier detection based on {} measures:'.format(feature_type))
                out_file = pjoin(self.out_dir, '{}_{}_{}.txt'.format(cfg.outlier_list_prefix,
                                                                     self.outlier_method, feature_type))
                self.by_feature[feature_type] = detect_outliers(features, self.id_list,
                                                           method=self.outlier_method,
                                                           out_file=out_file,
                                                           fraction_of_outliers=self.outlier_fraction)
            else:
                print('Insufficient number of samples (with features: {}) \n'
                      ' \t to run outlier detection - skipping it.'.format(feature_type))

        # re-organizing the identified outliers by sample
        for sid in self.id_list:
            # each id contains a list of all feature types that flagged it as an outlier
            self.by_sample[sid] = [feat for feat in self.outlier_feat_types if sid in self.by_feature[feat]]

        # dropping the IDs that were not flagged by any feature
        # so a imple ID in dict would reveal whether it was ever suspected as an outlier
        self.by_sample = {id: flag_list for id, flag_list in self.by_sample.items() if flag_list}

        return


