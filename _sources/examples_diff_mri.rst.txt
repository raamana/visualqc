Example usage - Diffusion MRI
------------------------------

The support for the visualQC of diffusion-weighted MRI scans is offered for datasets stored in the BIDS format. Usage is as simple as passing on the path to the root folder of the BIDS dataset via the ``--bids_dir`` or ``-b`` flag:

.. code-block:: bash

    visualqc_diffusion --bids_dir /new_great_project/ds042

which searches that dataset to identify and presents various subjects that have the necessary imaging data ready for inspection.

The identifying details for that subject are shown in the top right corner as usual. The interface allows you to review the data in great detail, either as a carpet plot entirely (all gradients at once), or one gradient at a time (by right clicking on a specific time point). You can further zoom into any slice within that gradient shown on the screen -- please refer to the :doc:`interface` page to learn some shortcuts and tricks to familiarize yourself with the interface.


Please read the documentation in :doc:`cli_diff_mri` on the remaining options/flags for this module as they are more or less self-explanatory.

Notable flags are

 - ``--no_preproc`` to avoid applying any preprocessing before showing the data
 - ``--name_pattern`` or ``-n`` to specify a regex pattern to limit which sessions/subjects to include for your review.

Note the outlier detection part is not yet fully tested/validated for this module.

As with other modules of ``VisualQC``, you can choose which views, how many slices and rows to display using the Layout command line arguments i.e. ``--views``, ``--num_slices`` and ``--num_rows``.