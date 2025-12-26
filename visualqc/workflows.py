"""

Module to define base classes.

"""

import sys
import traceback
import logging
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from os import makedirs
from os.path import exists as pexists, join as pjoin
from pathlib import Path
from timeit import default_timer as timer

import numpy as np

from visualqc import config as cfg
from visualqc.utils import get_ratings_path_info, load_ratings_csv


class DummyCallable(object):
    """Class to define placeholder callable. """


    def __init__(self, *args, **kwargs):
        pass


    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            'This callable must be overridden before being used!')


def reconstruct_command_line(user_args, parser, program_name: str) -> str:
    """
    Reconstruct command line from parsed arguments for reproducibility.
    
    This creates a clean, reproducible command line from the parsed arguments,
    which is more useful than the raw sys.argv (which may have typos, etc.).
    
    Parameters
    ----------
    user_args : argparse.Namespace
        Parsed command line arguments
    parser : argparse.ArgumentParser
        Argument parser instance
    program_name : str
        Program name (e.g., 'visualqc_alignment')
        
    Returns
    -------
    str
        Reconstructed command line
    """
    parts = [program_name]
    
    for action in parser._actions:
        if action.dest == 'help' or action.dest is None:
            continue
        
        # Skip if this is a positional argument (we'll handle those separately)
        if not action.option_strings:
            continue
        
        value = getattr(user_args, action.dest, None)
        
        # Skip if value is None, False (for flags), or matches default
        if value is None:
            continue
        
        # Handle boolean flags
        if isinstance(value, bool):
            if value and not action.default:
                # Only add if it's True and default is False
                parts.append(action.option_strings[0])
            continue
        
        # Handle lists/tuples
        if isinstance(value, (list, tuple)):
            if value:
                parts.append(action.option_strings[0])
                parts.extend(str(v) for v in value)
            continue
        
        # Skip if value matches default
        if hasattr(action, 'default') and value == action.default:
            continue
        
        # Add the option and value
        parts.append(action.option_strings[0])
        if isinstance(value, Path):
            parts.append(str(value))
        else:
            parts.append(str(value))
    
    # Add positional arguments at the end
    for action in parser._actions:
        if action.dest == 'help' or action.dest is None:
            continue
        if not action.option_strings:  # Positional argument
            value = getattr(user_args, action.dest, None)
            if value is not None:
                if isinstance(value, (list, tuple)):
                    parts.extend(str(v) for v in value)
                else:
                    parts.append(str(value))
    
    return ' '.join(parts)


def setup_ratings_logger(out_dir: Path):
    """
    Set up structured logging for ratings.
    
    Parameters
    ----------
    out_dir : Path
        Output directory for log files
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    log_dir = Path(out_dir) / cfg.suffix_ratings_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"ratings_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create logger
    logger = logging.getLogger('visualqc.ratings')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler with structured format
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    logger.addHandler(file_handler)
    
    # Console handler (only INFO and above to avoid clutter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings/errors on console
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s: %(message)s')
    )
    logger.addHandler(console_handler)
    
    return logger


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
                 show_unit_id=True,
                 screenshot_only=False):
        """Constructor"""

        # super().__init__()

        self.id_list = id_list
        self.in_dir = Path(in_dir).resolve()
        self.out_dir = Path(out_dir).resolve()
        print(f'Input folder: {self.in_dir}\nOutput folder: {self.out_dir}')

        self.ratings = dict()
        self.notes = dict()
        self.timer = dict()

        self.outlier_method = outlier_method
        self.outlier_fraction = outlier_fraction
        self.outlier_feat_types = outlier_feat_types
        self.disable_outlier_detection = disable_outlier_detection

        self.screenshot_only = screenshot_only
        if self.screenshot_only:
            # to enable batch generation without any windows
            from visualqc.utils import set_noninteractive_backend
            set_noninteractive_backend()

            self.screenshot_dir = self.out_dir / cfg.screenshot_out_dir_name
            if not self.screenshot_dir.exists():
                self.screenshot_dir.mkdir(exist_ok=True)

        # option to hide the ID, which may contain meta data such as site/time
        # hiding ID reduces bias or batch effects
        self.show_unit_id = show_unit_id

        # following properties must be instantiated
        self.feature_extractor = DummyCallable()
        self.fig = None
        self.UI = None

        self.quit_now = False
        
        # Capture command line for metadata (will be updated after parsing/validation)
        self.command_line = ' '.join(sys.argv)
        
        # Initialize ratings database for immediate checkpointing
        # Only initialize if not in screenshot-only mode
        self.ratings_db = None
        self.session_id = None
        if not self.screenshot_only:
            from visualqc.ratings_db import RatingsDatabase
            ratings_dir = Path(self.out_dir) / cfg.suffix_ratings_dir
            # Generate database filename based on vis_type and suffix
            # These will be set by child classes, so we'll initialize later
            # For now, just set up the logger
            self.logger = setup_ratings_logger(self.out_dir)
        else:
            self.logger = None


    def capture_command_metadata(self, command_line: str):
        """
        Capture command line for session tracking.
        
        This should be called after workflow instantiation with the validated
        command line (after parsing and validation).
        
        Parameters
        ----------
        command_line : str
            Full command line after parsing/validation
        """
        self.command_line = command_line
    
    def capture_command_metadata_from_args(self, user_args, parser):
        """
        Reconstruct and capture command line from parsed arguments.
        
        Automatically extracts program name from parser.prog.
        
        Parameters
        ----------
        user_args : argparse.Namespace
            Parsed command line arguments
        parser : argparse.ArgumentParser
            Argument parser instance (must have prog set)
        """
        program_name = parser.prog if hasattr(parser, 'prog') and parser.prog else sys.argv[0]
        command_line = reconstruct_command_line(user_args, parser, program_name)
        self.capture_command_metadata(command_line)
    
    def _init_ratings_database(self):
        """Initialize the ratings database after vis_type and suffix are set."""
        if self.screenshot_only or self.ratings_db is not None:
            return
        
        from visualqc.ratings_db import RatingsDatabase
        ratings_dir = Path(self.out_dir) / cfg.suffix_ratings_dir
        # Generate database filename based on vis_type and suffix
        db_filename = f"{self.vis_type}_{self.suffix}_ratings.db"
        db_path = ratings_dir / db_filename
        self.ratings_db = RatingsDatabase(db_path)
        
        # Generate session ID for tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save session metadata (command line and versions)
        if self.command_line:
            self.ratings_db.save_session_metadata(
                session_id=self.session_id,
                command_line=self.command_line
            )
        
        if self.logger:
            self.logger.info(f"Initialized ratings database: {db_path}")
            self.logger.info(f"Session ID: {self.session_id}")

    def run(self):
        """Entry point after init."""

        self.preprocess()
        # Initialize database after vis_type and suffix are set by child class
        self._init_ratings_database()
        self.restore_ratings()
        self.prepare_UI()
        self.loop_through_units()
        self.cleanup()

        print('\nAll Done - results are available in:\n\t{}'.format(self.out_dir))


    def cleanup(self):
        """Cleanup before exit"""

        if not self.screenshot_only:
            self.save_ratings()

        self.close_UI()


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


    @abstractmethod
    def close_UI(self):
        """Method to close all figures and UI elements."""


    def restore_ratings(self):
        """Method to restore ratings from previous sessions, if any."""

        # making a copy
        self.incomplete_list = list(self.id_list)

        if self.screenshot_only:
            # processing every available subject
            return

        print('\nRestoring ratings from previous session(s), if they exist ..')
        prev_done = set()  # Track restored ratings

        # Load from SQLite database (primary source)
        if self.ratings_db is not None:
            try:
                db_ratings = self.ratings_db.get_all_ratings()
                if db_ratings:
                    self.ratings = {k: v[0] for k, v in db_ratings.items()}
                    self.notes = {k: v[1] for k, v in db_ratings.items()}
                    prev_done = set(self.ratings.keys())
                    self.incomplete_list = list(set(self.id_list) - prev_done)
                    print(f'\n✓ Restored {len(prev_done)} ratings from database')
                    
                    # Auto-regenerate JSON if stale
                    if self.ratings_db.is_json_stale():
                        print('Regenerating JSON export (was stale)...')
                        self.ratings_db.export_json_if_needed(force=True)
                    
                    if self.logger:
                        self.logger.info(f"Restored {len(prev_done)} ratings from database")
                else:
                    # Database exists but empty - check for legacy CSV to migrate
                    self._migrate_csv_to_db()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load from database: {e}")
                # Try to migrate CSV if database fails
                self._migrate_csv_to_db()
        else:
            # Database not initialized - shouldn't happen in normal operation
            self.ratings = dict()
            self.notes = dict()
    
    def _migrate_csv_to_db(self):
        """
        One-time migration: Load from CSV if it exists, then migrate to SQLite.
        This is only called once when database is empty or doesn't exist.
        """
        ratings_file, _ = get_ratings_path_info(self)
        prev_done = set()  # Initialize to avoid NameError
        
        if pexists(ratings_file):
            print('Migrating legacy CSV ratings to database...')
            try:
                csv_ratings, csv_notes = load_ratings_csv(ratings_file)
                if csv_ratings and self.ratings_db is not None:
                    # Migrate CSV data to database
                    migrated_count = 0
                    for unit_id, rating in csv_ratings.items():
                        notes = csv_notes.get(unit_id, '')
                        success = self.ratings_db.save_rating(
                            unit_id=unit_id,
                            rating=rating,
                            notes=notes,
                            session_id='migrated_from_csv',
                            module_type=getattr(self, '__module_type__', None),
                            vis_type=self.vis_type
                        )
                        if success:
                            migrated_count += 1
                    
                    if migrated_count > 0:
                        print(f'✓ Migrated {migrated_count} ratings from CSV to database')
                        # Reload from database
                        db_ratings = self.ratings_db.get_all_ratings()
                        self.ratings = {k: v[0] for k, v in db_ratings.items()}
                        self.notes = {k: v[1] for k, v in db_ratings.items()}
                        prev_done = set(self.ratings.keys())
                        self.incomplete_list = list(set(self.id_list) - prev_done)
                        
                        if self.logger:
                            self.logger.info(f"Migrated {migrated_count} ratings from CSV to database")
                    else:
                        # Fallback: use CSV data in memory
                        self.ratings = csv_ratings
                        self.notes = csv_notes
                        prev_done = set(self.ratings.keys())
                        self.incomplete_list = list(set(self.id_list) - prev_done)
                else:
                    # No CSV or no database - use CSV in memory if available
                    self.ratings = csv_ratings if csv_ratings else dict()
                    self.notes = csv_notes if csv_notes else dict()
                    prev_done = set(self.ratings.keys())
                    self.incomplete_list = list(set(self.id_list) - prev_done)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to migrate CSV: {e}")
                self.ratings = dict()
                self.notes = dict()
        else:
            # No CSV, no database - start fresh
            self.ratings = dict()
            self.notes = dict()

        if len(prev_done) > 0:
            print('\nRatings for {} subjects were restored'.format(len(prev_done)))

        if len(self.incomplete_list) < 1:
            print('No subjects to review/rate - exiting.')
            sys.exit(0)
        else:
            self.num_units_to_review = len(self.incomplete_list)
            print('To be reviewed : {}\n'.format(self.num_units_to_review))


    def save_ratings(self, export_json: bool = True):
        """
        Export ratings to JSON (SQLite is source of truth).
        
        Note: Ratings are already checkpointed immediately when captured.
        This method exports them to JSON format for human readability.
        
        Parameters
        ----------
        export_json : bool
            Whether to export to JSON (default: True)
        """

        print('\nSaving ratings .. \n')
        
        if self.ratings_db is not None:
            # Export to JSON (human-readable with full metadata)
            if export_json:
                json_path = self.ratings_db.get_json_path()
                success = self.ratings_db.export_to_json(json_path, include_metadata=True)
                if success:
                    print(f'✓ Exported ratings to JSON: {json_path}')
                else:
                    print('✗ Warning: Failed to export to JSON')
            
            # Summarize ratings directly from SQLite database
            from visualqc.utils import summarize_ratings
            summarize_ratings(self.ratings_db)
        else:
            # Fallback: database not initialized (shouldn't happen in normal operation)
            if self.logger:
                self.logger.warning("No database available for export")
            print('✗ Warning: No ratings database available')
        
        self.save_time_spent()
        
        if self.logger:
            self.logger.info("Ratings export completed")


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
            makedirs(ratings_dir, exist_ok=True)

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

        if self.screenshot_only:
            self.UI.remove_UI()

        self.num_units_to_review = len(self.incomplete_list)
        for counter, unit_id in enumerate(self.incomplete_list):

            self.current_unit_id = unit_id
            self.identify_unit(unit_id, counter)
            self.add_alerts()

            skip_subject = self.load_unit(unit_id)

            if skip_subject:
                print('Skipping current subject ..')
                continue

            self.display_unit()

            # checking if batch generation of screenshots is requested
            if not self.screenshot_only:

                print('\nReviewing {}'.format(unit_id))
                timer_start = timer()

                # this is where all the reviewing/rating/notes happen
                self.show_fig_and_wait()

                # capturing time elapsed by ID, in seconds
                self.timer[unit_id] = timedelta(seconds=timer() - timer_start).seconds

                # Rating is already checkpointed in capture_user_input()
                # No need to save here - it's already on disk!
                self.print_rating(unit_id)

                if self.quit_now:
                    print('\nUser chosen to quit..')
                    break
            else:
                self.export_screenshot()
                # annot text is unit specific
                self.UI.annot_text.remove()


    def identify_unit(self, unit_id, counter):
        """
        Method to inform the user which unit (subject or scan) they are reviewing.

        Deafult location is to the top right.

        This can be overridden by the child class for fancier presentation.

        """

        if self.show_unit_id:
            annot_text = f'{unit_id}\n({counter + 1}/{self.num_units_to_review})'
        else:
            annot_text = f'{counter + 1}/{self.num_units_to_review}'

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


    def export_screenshot(self):
        """Exports the screenshot of current visualization to disk"""

        if self.vis_type is None or len(self.vis_type) < 1:
            vis_type_suffix = ''
        else:
            vis_type_suffix = self.vis_type

        print("exporting screenshot for {}".format(self.current_unit_id))
        ss_out_file = self.screenshot_dir / "{}_{}_{}.{}".format(
            self.current_unit_id, vis_type_suffix,
            cfg.screenshot_suffix, cfg.screenshot_format_ext)
        self.fig.savefig(ss_out_file, bbox_inches='tight', dpi=cfg.dpi_export_fig)


    @abstractmethod
    def add_alerts(self):
        """
        Method to appropriately alert the reviewer
            e.g. when subject was flagged as an outlier
        """


    def quit(self, input_event_to_ignore=None):
        """terminator"""

        if self.UI.allowed_to_advance():
            self.prepare_to_advance()
            self.quit_now = True
        else:
            print('You have not rated the current subject! '
                  'Please rate it before you can advance '
                  'to next subject, or to quit..')


    def next(self, input_event_to_ignore=None):
        """advancer"""

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
        """Updates all user input to class and checkpoints immediately."""

        self.ratings[self.current_unit_id] = self.UI.get_ratings()
        self.notes[self.current_unit_id] = self.UI.user_notes
        
        # IMMEDIATE CHECKPOINT - write to disk right away
        rating_value = self.ratings[self.current_unit_id]
        
        # Skip checkpointing if rating is None or empty
        if rating_value is None:
            return
        
        # Check if this is a "do not save" rating
        do_not_save = False
        if isinstance(rating_value, (list, tuple)):
            do_not_save = any([
                str(rt).lower() in cfg.ratings_not_to_be_recorded
                for rt in rating_value
            ])
        elif isinstance(rating_value, str):
            # Handle string ratings that might contain multiple values
            rating_parts = rating_value.split(cfg.rating_joiner)
            do_not_save = any([
                rt.lower() in cfg.ratings_not_to_be_recorded
                for rt in rating_parts
            ])
        
        if not do_not_save and self.ratings_db is not None:
            # Convert rating to string format
            rating_str = self._join_ratings(rating_value)
            notes_str = self.notes.get(self.current_unit_id, '')
            
            # Write immediately to database
            success = self.ratings_db.save_rating(
                unit_id=self.current_unit_id,
                rating=rating_str,
                notes=notes_str,
                session_id=self.session_id,
                module_type=getattr(self, '__module_type__', None),
                vis_type=self.vis_type
            )
            
            # Only log/print failures (quiet on success)
            if not success:
                if self.logger:
                    self.logger.error(f"Failed to checkpoint rating for {self.current_unit_id}: {rating_str}")
                print(f'✗ Warning: Failed to checkpoint rating for {self.current_unit_id}')
        elif do_not_save:
            # Remove from in-memory dict if it's a "do not save" rating
            if self.current_unit_id in self.ratings:
                self.ratings.pop(self.current_unit_id)
            if self.current_unit_id in self.notes:
                self.notes.pop(self.current_unit_id)


    def print_rating(self, subject_id):
        """Method to print the rating recorded for the current subject."""

        if subject_id in self.ratings and (self.ratings[subject_id] is not None):
            # Note: "do not save" ratings are already handled in capture_user_input()
            print('    id: {}\n'
                  'rating: {}\n'
                  ' notes: {}'.format(subject_id, self.ratings[subject_id],
                                      self.notes[subject_id]))
        else:
            print(f'rating for {subject_id} has not been recorded!')


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
            print('Assistance with outlier detection did not succeed.\n'
                  'Proceeding by disabling it. Stack trace below:\n')
            import traceback
            traceback.print_exc()

        return
