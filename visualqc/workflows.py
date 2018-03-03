"""

Module to define base classes.

"""

import sys
from abc import ABC
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
                 rating_list=cfg.default_rating_list):
        """Constructor"""

        # super().__init__()

        self.id_list = id_list
        self.in_dir = in_dir
        self.out_dir = out_dir

        self.rating_list = rating_list

    def save_cmd(self):
        """Saves the command issued by the user for debugging purposes"""

        cmd_file = pjoin(self.out_dir, 'cmd_issued.visualqc.{}'.format(self.__name__))
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


