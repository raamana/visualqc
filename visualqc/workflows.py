"""

Module to define base classes.

"""

import sys
import traceback
from abc import ABC, abstractmethod
from shutil import copyfile

from os.path import exists as pexists, join as pjoin
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
from visualqc import config as cfg
from visualqc.utils import get_ratings_path_info, load_ratings_csv, summarize_ratings


class DummyCallable(object):
    """Class to define placeholder callable. """


    def __init__(self, *args, **kwargs):
        pass


    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            'This callable must be overridden before being used!')


class BaseWorkflowVisualQC(ABC):
    """
    Class defining the base workflow for visualqc.
    """


    def __init__(self,
                 id_list,
                 in_dir,
                 out_dir,
                 outlier_method,
                 outlier_fraction,
                 outlier_feat_types,
                 disable_outlier_detection,
                 show_unit_id=True):
        """Constructor"""

        # super().__init__()

        self.id_list = id_list
        self.in_dir = in_dir
        self.out_dir = out_dir
        print(
            'Input folder: {}\nOutput folder: {}'.format(self.in_dir, self.out_dir))

        self.ratings = dict()
        self.notes = dict()
        self.timer = dict()

        self.outlier_method = outlier_method
        self.outlier_fraction = outlier_fraction
        self.outlier_feat_types = outlier_feat_types
        self.disable_outlier_detection = disable_outlier_detection

        # option to hide the ID, which may contain meta data such as site/time
        # hiding ID reduces bias or batch effects
        self.show_unit_id = show_unit_id

        # following properties must be instantiated
        self.feature_extractor = DummyCallable()
        self.fig = None
        self.UI = None

        self.quit_now = False


    def run(self):
        """Entry point after init."""

        self.preprocess()
        self.restore_ratings()
        self.prepare_UI()
        self.loop_through_units()
        self.cleanup()

        print('\nAll Done - results are available in:\n\t{}'.format(self.out_dir))


    @abstractmethod
    def preprocess(self):
        """
        Method to get all required preprocessing done,
         to get ready to start the review interface.

         """


    @abstractmethod
    def prepare_UI(self):
        """
        Method to prepare UI and add all the elements required for review.

        This is where you
        - open a figure with the required layout,
        - must save the figure handle to self.fig
        - add :class:BaseReviewInterface and save handle to self.UI
        - add additional ones on top the base review interface.


        """


    def restore_ratings(self):
        """Method to restore ratings from previous sessions, if any."""

        print('Restoring ratings from previous session(s), if they exist ..')

        # making a copy
        self.incomplete_list = list(self.id_list)
        prev_done = []  # empty list

        ratings_file, backup_name_ratings = get_ratings_path_info(self)

        if pexists(ratings_file):
            self.ratings, self.notes = load_ratings_csv(ratings_file)
            # finding the remaining
            prev_done = set(self.ratings.keys())
            self.incomplete_list = list(set(self.id_list) - prev_done)
        else:
            self.ratings = dict()
            self.notes = dict()

        if len(prev_done) > 0:
            print('\nRatings for {} sessions were restored'.format(len(prev_done)))

        if len(self.incomplete_list) < 1:
            print('No subjects to review/rate - exiting.')
            sys.exit(0)
        else:
            self.num_units_to_review = len(self.incomplete_list)
            print('To be reviewed : {}\n'.format(self.num_units_to_review))


    def save_ratings(self):
        """Saves ratings to disk """

        print('Saving ratings .. \n')
        ratings_file, prev_ratings_backup = get_ratings_path_info(self)

        if pexists(ratings_file):
            copyfile(ratings_file, prev_ratings_backup)

        # add column names: subject_id,issue1:issue2:issue3,...,notes etc
        # TODO add path(s) to data (images etc) that produced the review
        lines = '\n'.join(['{},{},{}'.format(sid, self._join_ratings(rating_set),
                                             self.notes[sid])
                           for sid, rating_set in self.ratings.items()])
        try:
            with open(ratings_file, 'w') as cf:
                cf.write(lines)
        except:
            raise IOError(
                'Error in saving ratings to file!!\n'
                'Backup might be helpful at:\n\t{}'.format(prev_ratings_backup))

        # summarize ratings to stdout and id lists
        summarize_ratings(ratings_file)
        self.save_time_spent()


    @staticmethod
    def _join_ratings(str_list):

        if isinstance(str_list, (list, tuple)):
            return cfg.rating_joiner.join(str_list)
        else:
            return str_list

    def save_time_spent(self):
        """Saves time spent on each unit"""

        ratings_dir = Path(self.out_dir).resolve() / cfg.suffix_ratings_dir
        if not ratings_dir.exists():
            makedirs(ratings_dir)

        timer_file = ratings_dir / '{}_{}_{}'.format(
            self.vis_type, self.suffix, cfg.file_name_timer)

        lines = '\n'.join(['{},{}'.format(sid, elapsed_time)
                           for sid, elapsed_time in self.timer.items()])

        # saving to disk
        try:
            with open(timer_file, 'w') as tf:
                tf.write(lines)
        except:
            print('Unable to save timer info to disk -- printing them to log:')
            print(lines)
            raise IOError('Error in saving timer info to file!')

        # printing summary
        times = np.array(list(self.timer.values()))
        if len(times) < 10:
            print('\n\ntimes spent per subject in seconds:\n{}'.format(lines))

        print('\nMedian time per subject : {} seconds'.format(np.median(times)))
        print('\t5th and 95th percentile of distribution of times spent '
              ': {} seconds'.format(np.nanpercentile(times, [5, 95])))


    def loop_through_units(self):
        """Core loop traversing through the units (subject/session/run) """

        for counter, unit_id in enumerate(self.incomplete_list):

            print('\nReviewing {}'.format(unit_id))
            self.current_unit_id = unit_id
            self.identify_unit(unit_id, counter)
            self.add_alerts()

            skip_subject = self.load_unit(unit_id)

            if skip_subject:
                print('Skipping current subject ..')
                continue

            self.display_unit()

            timer_start = timer()

            # this is where all the reviewing/rating/notes happen
            self.show_fig_and_wait()

            # capturing time elapsed by ID, in seconds
            self.timer[unit_id] = timedelta(seconds=timer() - timer_start).seconds

            # TODO save each rating to disk to avoid loss of work due to crach etc
            self.print_rating(unit_id)

            if self.quit_now:
                print('\nUser chosen to quit..')
                break


    def identify_unit(self, unit_id, counter):
        """
        Method to inform the user which unit (subject or scan) they are reviewing.

        Deafult location is to the top right.

        This can be overridden by the child class for fancier presentation.

        """

        if self.show_unit_id:
            annot_text = '{}\n({}/{})'.format(unit_id, counter + 1,
                                              self.num_units_to_review)
        else:
            annot_text = '{}/{}'.format(counter + 1, self.num_units_to_review)

        self.UI.add_annot(annot_text)


    def show_fig_and_wait(self):
        """Show figure and let interaction happen"""

        # window management
        self.fig.canvas.manager.show()
        self.fig.canvas.draw_idle()
        # starting a 'blocking' loop to let the user interact
        self.fig.canvas.start_event_loop(timeout=-1)


    @abstractmethod
    def load_unit(self, unit_id):
        """Method to load necessary data for a given subject.

        Parameters
        ----------
        unit_id : str
            Identifier to locate the data for the given unit in self.in_dir.
            Unit could be a subject, session or run depending on the task.

        Returns
        -------
        skip_subject : bool
            Flag to indicate whether to skip the display and review of subject e.g.
            when necessary data was not available or usable.
            When returning True, must print a message informing the user why.

        """


    @abstractmethod
    def display_unit(self):
        """Display routine."""


    @abstractmethod
    def add_alerts(self):
        """
        Method to appropriately alert the reviewer
            e.g. when subject was flagged as an outlier
        """


    def quit(self, input_event_to_ignore=None):
        "terminator"

        if self.UI.allowed_to_advance():
            self.prepare_to_advance()
            self.quit_now = True
        else:
            print('You have not rated the current subject! '
                  'Please rate it before you can advance '
                  'to next subject, or to quit..')


    def next(self, input_event_to_ignore=None):
        "advancer"

        if self.UI.allowed_to_advance():
            self.prepare_to_advance()
            self.quit_now = False
        else:
            print('You have not rated the current subject! '
                  'Please rate it before you can advance '
                  'to next subject, or to quit..')


    def prepare_to_advance(self):
        """Work needed before moving to next subject"""

        self.capture_user_input()
        self.UI.reset_figure()
        # stopping the blocking event loop
        self.fig.canvas.stop_event_loop()


    def capture_user_input(self):
        """Updates all user input to class"""

        self.ratings[self.current_unit_id] = self.UI.get_ratings()
        self.notes[self.current_unit_id] = self.UI.user_notes


    def print_rating(self, subject_id):
        """Method to print the rating recorded for the current subject."""

        # checking if "i'm tired" or 'review later' appear in ratings
        do_not_save = any([ rt.lower() in cfg.ratings_not_to_be_recorded
              for rt in self.ratings[subject_id]])

        # not saving ratings meant not to be saved!
        if do_not_save:
            self.ratings.pop(subject_id)
        else:
            print('    id: {}\n'
                  'rating: {}\n'
                  ' notes: {}'.format(subject_id, self.ratings[subject_id],
                                      self.notes[subject_id]))


    @abstractmethod
    def cleanup(self):
        """
        Clean up routine to
        1) save ratings,
        2) close all callbacks and figures etc.
        """


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
                traceback.print_exc()
                print('Unable to extract {} features! skipping..'.format(feat_type))


    def detect_outliers(self):
        """Runs outlier detection and reports the ids flagged as outliers."""

        # outliers categorized
        self.by_feature = dict()
        self.by_sample = dict()

        if self.disable_outlier_detection:
            print('outlier detection: disabled, as requested.')
            return

        if len(self.feature_paths) < 1:
            print('Features required for outlier detection are not available -'
                  ' skipping it.')
            return

        try:
            from visualqc.outliers import detect_outliers
            from visualqc.readers import gather_data
            for feature_type in self.outlier_feat_types:

                if len(self.feature_paths[feature_type]) < 1:
                    print('{} features for outlier detection are not available ...'
                          ' '.format(feature_type))
                    continue

                try:
                    if self.__module_type__.lower() == 'freesurfer':
                        # they're already assembled into an array, ordered by id_list
                        features = self.feature_paths[feature_type]
                    elif self.__module_type__.lower() == 't1_mri':
                        # features will be read from filepaths by id
                        features = gather_data(self.feature_paths[feature_type],
                                               self.id_list)
                    else:
                        raise ValueError('outlier detection not implemented for'
                                         ' {} module'.format(self.__module_type__))
                except:
                    raise IOError('Unable to read/assemble features for outlier '
                                  'detection. Skipping them!')

                if features.shape[0] > self.outlier_fraction * len(self.id_list):
                    print('\nRunning outlier detection based on {} measures:'
                          ''.format(feature_type))
                    out_file = pjoin(self.out_dir, '{}_{}_{}.txt'.format(
                        cfg.outlier_list_prefix, self.outlier_method, feature_type))
                    self.by_feature[feature_type] = \
                        detect_outliers(features, self.id_list,
                                        method=self.outlier_method,
                                        out_file=out_file,
                                        fraction_of_outliers=self.outlier_fraction)
                else:
                    print('Insufficient number of samples (with features: {}) \n'
                          ' \t to run outlier detection - skipping it.'
                          ''.format(feature_type))

            # re-organizing the identified outliers by sample
            for sid in self.id_list:
                # each id --> list of all feature types that flagged it as an outlier
                self.by_sample[sid] = [feat for feat in self.outlier_feat_types
                                       if sid in self.by_feature[feat]]

            # dropping the IDs that were not flagged by any feature
            # so a simple ID in dict implies --> it was ever suspected as an outlier
            self.by_sample = {id_: flag_list
                              for id_, flag_list in self.by_sample.items()
                              if flag_list}
        except:
            self.disable_outlier_detection = False
            self.by_feature = dict()
            self.by_sample = dict()
            print('Assitance with outlier detection did not succeed.\n'
                  'Proceeding by disabling it. Stack trace below:\n')
            import traceback
            traceback.print_exc()

        return
