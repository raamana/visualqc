Examples
--------

A rough example of usage can be:

.. code-block:: bash

    visualqc --fs_dir /project/freesurfer

which searches the specified directory for all subjects with successfully completed runs of Freesurfer and presents with cortical parcellation in all the three views, along with surface rendering (annotated with cortical ROIs). You can then review these visualizations for one subject at a time, rate their accuracy and move on to the next subject.

Note:
 - right click toggles the overlay
 - slider changes the transparency of the overlay in all slices
 - double click in a given slice zooms it full-screen

You can choose to work on pre-selected list of subjects, as you choose:

.. code-block:: bash

    visualqc --fs_dir /project/freesurfer --id_list list_complete.txt

Note the default visualization type is ``cortical_contour`` and you can change it to QC a particular ROI e.g. left hippocampus (label 17 in `Freesurfer color lookup table <https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT>`_:

.. code-block:: bash

    visualqc -f /project/freesurfer -i list_hippo.txt --vis_type labels_contour --labels 17

You can add as many ROIs as you wish

.. code-block:: bash

    visualqc -f /project/freesurfer -i list_hippo.txt --vis_type labels_contour --labels 10 11 12 13

However, be sure your layout is providing you with enough detail and convenience to get a quick glance of whole structure(s). If you don't need all the 3 views, or you'd like to focus on one particular view that is ideal for a given ROI, you can specify that with --views. It accepts a list of dimensions (max 3), from 0 to 2, inclusive.

.. code-block:: bash

    visualqc -f /project/freesurfer -i list_hippo.txt --vis_type labels_contour -l 10 11 12 13 --views 0


If you would like many more slices or rows (or fewer), you can control that with --num_rows and --num_slices. These parameters are applied separately for each view (not the entire layout). So if you choose -r 3 and -s 10 for -w 0 2, then `visualqc` displays 10 slices in 3 rows for each of dimensions 0 and 2 (20 panels in total).

.. code-block:: bash

    visualqc -f /project/freesurfer -i list_hippo.txt --vis_type labels_contour -l 10 11 12 13 --views 0 -w 0 2 -r 3 -s 10









