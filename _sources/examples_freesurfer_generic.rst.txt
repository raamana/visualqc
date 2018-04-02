Example usage - Segmentation
-----------------------------

In some use cases, you may want to overlay images from arbitrary locations without any pre-defined structure or hierarchy like Freesurfer. This is possible in ``visualqc_freesurfer`` by specifying:

 1. path to parent folder of the images using ``--user_dir`` or ``-u`` option, which contains a separate folder for each subject ID
 2. name of the anatomical MRI with ``--mri_name`` (or ``-m``) and
 3. name of the segmentation with ``--seg_name`` (or ``-g``) that is to be overlaid on the MRI.


If you would like to review all the subjects (each with their own folder) in ``/project/MR_segmentation``, whose segmentation(s) are stored in ``roi_set.nii`` whose T1/anatomical MRI is stored in ``mri.nii``. The folder hierarchy (within ``/project/MR_segmentation``) might look like this:

.. code-block:: bash

    .
    |-- atlas1
    |   |-- mri.nii
    |   `-- roi_set.nii
    |-- atlas2
    |   |-- mri.nii
    |   `-- roi_set.nii
    |-- sub_01
    |   |-- mri.nii
    |   `-- roi_set.nii
    `-- sub_04
        |-- mri.nii
        `-- roi_set.nii


In that case, you would issue the following command:

.. code-block:: bash

    visualqc_freesurfer --in_dir /project/MR_segmentation --mri_name mri.nii --seg_name roi_set.nii


This will process the four subjects (atlas1, atlas2, sub_01, sub_04) sequentially, and creates an output directory called ``visualqc`` in the input directory specified ``/project/MR_segmentation``, to store the visualizations generated, along with the ratings and notes provided by the user. You can also change the output directory with the ``-o`` option. You can also limit the review to a subject of IDs, by using a predefined list by a specifying an id list with ``--id_list`` or ``-i`` option, containing one ID per line. An example (focusing only on the 2 atlases) could like:

.. code-block:: bash

    atlas1
    atlas2

