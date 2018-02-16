Examples (Generic)
----------------------

In some use cases, you may want to overlay images from arbitrary locations without any pre-defined structure or heirarchy like Freesurfer. This is possible in ``visualqc`` by specifying:

 1. path to parent folder of the images using ``--user_dir`` or ``-u`` option, which contains a seprate folder for each subject ID
 2. name of the anatomical MRI with ``--mri_name`` (or ``-m``) and
 3. name of the segmentation with ``--seg_name`` (or ``-g``) that is to be overlaid on the MRI.


If you would like to review all the subjects (each with their own folder) in ``/project/MR_segmentation``, who segmentation(s) are stored in ``roi_set.nii`` whose T1/anatomical MRI is stored in ``mri.nii``, you would issue the following command:

.. code-block:: bash

    visualqc --in_dir /project/MR_segmentation --mri_name mri.nii --seg_name roi_set.nii


This creates an output directory called ``visualqc`` in the input directory specified ``/project/MR_segmentation``, to store the visualizations generated, along with the ratings and notes provided by the user. You can also change the output directory with the ``-o`` option.
