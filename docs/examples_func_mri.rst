Example usage - Functional MRI
------------------------------

The support for the visualQC of fMRI scans is offered for datasets stored in the BIDS format. Usage is as simple as passing on the path to the root folder of the BIDS dataset via the ``--bids_dir`` or ``-b`` flag:

.. code-block:: bash

    visualqc_func_mri --bids_dir /new_great_project/ds042

which searches that dataset to identify various subjects and sessions that have the necessary imaging data ready for inspection. VisualQC shows one session at a time for a given subject.

The identifying details for that session are shown in the top right corner as usual. The interface allows you to review the data in great detail, either as a carpet plot entirely (all time points/frames at once), or one time-point/frame at a time (by right clicking on a specific time point). You can further zoom into any slice within that time point shown on the screen -- please refer to the :doc:`interface` page to learn some shortcuts and tricks to familiarize yourself with the interface.


Please read the documentation in :doc:`cli_func_mri` on the remaining options/flags for this module as they are more or less self-explanatory.

Notable flags are

 - ``--no_preproc`` to avoid applying any preprocessing before showing the data
 - ``--name_pattern`` or ``-n`` to specify a regex pattern to limit which sessions/subjects to include for your review.

