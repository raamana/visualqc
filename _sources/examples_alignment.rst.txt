Example usage - Alignment
----------------------------

``vqcalignment`` compares the alignment of one image to another. The images are typically from the same source e.g. subject at different times or stages, but they don't have to be. To run it on a dataset, it should have all the pairs of images to be reviewed in the same folder, identified by ``--user_dir``, wherein each subject/unit is identified by one line in the ``--id_list``. Make sure that each subject has their own folder, containing the two images you would like to check alignment against each other. In the command line, they are identified by the ``-i1`` or ``--image1`` for the first and by ``-i2`` or ``--image2`` for the second.


A rough example of usage can be:

.. code-block:: bash

    visualqc_alignment --user_dir /project/registration  -i1 orig.nii -i2 aligned.nii --id_list subject_ids.txt

which searches the specified directory ``/project/registration`` for all subjects (using one ID at a time read from ``subject_ids.txt``), that has the required files with names as specified by i1 and i2 above i.e. ``orig.nii``, ``aligned.nii``. You can then review the composite visualizations for one subject at a time, rate the accuracy of registration between the two images for this subject, and move on to the next subject.

The default visualization would be composite overlay of 1) edges computed from image 2 onto image 1. Use the radio button to switch between different types of visualizations to get a better sense of the registration accuracy, and make sure zoom on to different slices with a double click to closely inspect *busy* neighbourhoods. You can use the checkboxes and the Notes section to record your ratings and other comments.

Note the outlier detection part is not yet fully tested/validated for this module.

As with other modules of ``VisualQC``, you can choose which views, how many slices and rows to display using the Layout command line arguments i.e. ``--views``, ``--num_slices`` and ``--num_rows``.