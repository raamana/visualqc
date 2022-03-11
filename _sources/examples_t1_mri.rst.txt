Example usage - T1w MRI
----------------------------


``vqcanat`` (short for ``visualqc_anatomical``) presents a mosaic or collage of various slices from the 3D volume in different view on the same page for quick inspection and zooming of different slices. The user can then try identify any aretefacts and rate their presence or severity using the checkboxes as well as free form notes.

To run it on a dataset, you need to choose one of the input format available such as ``--user_dir``, ``--fs_dir`` or ``--bids_dir``. When ``--user_dir`` is chosen, each subject/unit is identified by one line in the ``--id_list``, which is expected to be in their own folder, containing an image with the name specified by ``--mri_name``. The ``--fs_dir``, referring to the Freesurfer's ``$SUBJECTS_DIR``, and ``--bids_dir`` referring the root path of BIDS dataset, have their own formats that VisualQC can traverse. In all the formats, image to be displayed will be identified by the name specified by ``--mri_name``.

you would like to check alignment against each other. In the command line, they are identified by the ``-i1`` or ``--image1`` for the first and by ``-i2`` or ``--image2`` for the second.


A rough example of usage can be:

.. code-block:: bash

    vqcanat --user_dir /project/morphological  --mri_name T1w.nii --id_list subject_ids.txt

which searches the specified directory ``/project/registration`` for all subjects (using one ID at a time read from ``subject_ids.txt``), that has the required image (in this case T1w.nii). You can then review the composite visualization for one subject at a time, rate the quality of image and/or identify any artefacts, and move on to the next subject.

It is recommended to specify the subject IDs you would like to review to achieve better grouping (by demographics such healthy vs. disease, men vs. women etc), which leads to much better results from the outlier detection module. However, you could choose to omit the ``id_list`` altogether, whence VisualQC would search for subjects with all the required files, and then present them one by one. So the simpler possibilities include

.. code-block:: bash

    vqcanat --fs_dir /project/freesurfer_v6
    vqcanat --bids_dir /project/raw_data_bids


As with other modules of ``VisualQC``, you can choose which views, how many slices and rows to display using the Layout command line arguments i.e. ``--views``, ``--num_slices`` and ``--num_rows``.
