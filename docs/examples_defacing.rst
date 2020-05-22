Example usage - Defacing
----------------------------

To run ``vqcdeface`` on a dataset, make sure that each subject has their own folder, containing the original pre-defaced image, the defaced image, and png snapshot(s) of the 3D rendered defaced image. At a minimum, one snapshot of the front of the defaced scan is required, but it is recommended to have a couple of different angles, to capture a more complete image of the scan. All snapshots should have the same prefix to view them at the same time. Note: make sure the prefix is different from all of the mri files in the same folder, e.g. defaced.png and defaced.nii.gz will cause an error but Render_deface.png is fine

A rough example of usage can be:

.. code-block:: bash

    visualqc_defacing --user_dir /project/defaced --defaced_name defaced.nii --mri_name orig.mgz  --render_name rendered.png --id_list subject_ids.txt

which searches the specified directory for all subjects (using IDs in ``subject_ids.txt``), that has the required files with names as specified above. You can then review these visualizations for one subject at a time, rate their accuracy and move on to the next subject. Use the radio button
