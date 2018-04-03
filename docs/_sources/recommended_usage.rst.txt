Recommended Usage
------------------

General suggestions:

 - For the best results from outlier detection process, it is recommended to divide list of IDs into known groups (healthy, disease1, disease2, young, old etc) based on non-imaging parameters (such as clinical diagnosis, age etc), to perform the QC process independently on each group.
 - Be generous in the number of slices you use to review in each view (even if they appear small in the collage), as you have the ability to zoom-in anywhere you please for detailed inspection.
 - Routinely toggle overlays to ensure composite overlays are not affecting your perception of GM/WB boundaries in scans with unusual intensity distributions (low or high contrast, dark or too bright etc).

For Freesurfer outputs:

 - Inspect the quality of raw T1 MRI scans first, using visualqc, for presence of any artefacts, such as motion, ringing, ghosting, and anything else.
 - Install and `run Freesufer <https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferWiki>`_, on ALL subjects in your dataset.
 - Follow the `troubleshooting guide <https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/TroubleshootingData>`_ by the Freesurfer team, that includes atleast the following checks. These `slides <http://surfer.nmr.mgh.harvard.edu/pub/docs/freesurfer.failure_modes.ppt>`_ are a fantastic start to get an idea of what to focus on.

   - Review the accuracy of white and pial surfaces (this is the default), and identify subjects for further inspection (errors in the preceding steps of the pipeline)
   - Review the segmentation of white matter is accurate (overlay wm.mgz on T1.mgz) for each subject, and identify those to be rerun or to be corrected for minor errors.
   - Review the accuracy of skull-stripping for each subject, and identify the subjects that need to be rerun with special flags (for major errors), or corrected manually (for minor errors).
