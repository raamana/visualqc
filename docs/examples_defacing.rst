Example usage - Defacing
----------------------------

To run ``vqcdeface`` on a dataset, make sure that each subject has their own folder, containing 1) the original pre-defaced image, 2) the defaced image, and 3) png snapshot(s) of the 3D rendered defaced image. At a minimum, one snapshot of the front of the defaced scan is required, but it is recommended to have a couple of different angles, to capture a more complete image of the scan. All snapshots should have the same prefix to view them at the same time.


Rendered images of the MRI volume can easily be generated using the ``generate_rendered_images_3Dvolume_deface.m`` in the `VisualQC scripts folder <https://github.com/raamana/visualqc/tree/master/docs>`_. This script requires two Matlab toolboxes including `imResizeN <https://www.mathworks.com/matlabcentral/fileexchange/64516-imresizen-resize-an-n-dimensional-array>`_ and `Viewer3D <https://www.mathworks.com/matlabcentral/fileexchange/21993-viewer3d?s_tid=srchtitle>`_. If you have a better solution/script, ideally in Python, let me know.

.. note::

    make sure the prefix is different from all of the mri files in the same folder, e.g. defaced.png and defaced.nii.gz in the same folder for one subject will cause an error, but Render_deface.png and defaced.nii.gz will be fine.


A rough example of usage can be:

.. code-block:: bash

    visualqc_defacing --user_dir /project/protect_privacy --defaced_name defaced.nii --mri_name orig.mgz  --render_name rendered.png --id_list subject_ids.txt

which searches the specified directory ``/project/protect_privacy`` for all subjects (using one ID at a time read from ``subject_ids.txt``), that has the required files with names as specified above i.e. ``orig.mgz``, ``defaced.nii`` and ``rendered.png``. You can then review these visualizations for one subject at a time, rate the accuracy of defacing for this subject, and move on to the next subject.

The default visualization would be composite overlay of 1) what's been removed in red along with 2) what's been retained in green to easily emphasize if the defacing algorithm over- or under-stripped. Use the radio button to switch between different types of visualizations to get a better sense of the brain before and after defacing, to be confident in your assessment. You can use the checkboxes and the Notes section to record your ratings and other comments.

Note the outlier detection part is not yet fully tested/validated for this module.

As with other modules of ``VisualQC``, you can choose which views, how many slices and rows to display using the Layout command line arguments i.e. ``--views``, ``--num_slices`` and ``--num_rows``.