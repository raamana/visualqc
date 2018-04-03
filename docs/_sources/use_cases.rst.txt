Use-cases supported
===============================

VisualQC supports the following use cases:

 * Functional MRI scans (focused visual review, with rich and custom-built visualizations)
 * Freesurfer cortical parcellations (accuracy of pial/white surfaces on T1w mri)
 * Structural T1w MRI scans (artefact rating)
 * Volumetric segmenatation accuracy (against T1w MRI)
 * Registration quality (spatial alignment) within a single modality (multimodal support coming)


More detailed specifics for each modality are noted below:

Structural MRI usecases
----------------------------

 - Checking the accuracy of white and pial surfaces (from Freesurfer and other algorithms)
 - Evaluate the accuracy of skull-stripping (e.g. from Freesurfer or BET or 3dSkullStrip or SPM)
 - Assess the accuracy of tissue segmentation (gray matter, white matter or CSF masks)
 - Inspect the quality of raw T1 MRI scan (for motion, ringing, ghosting, or other artefacts)
 - Comparison of the registration quality or accuracy of the spatial alignment.

Functional MRI usecases
----------------------------

 - Visual review of scan quality, identification of artefactual frames (motion, spikes, etc)
 - Quality control of the impact of different pre-processing steps

Registration/Alignment usecases
--------------------------------

 - Within-modality assessment of the accuracy of the spatial alignment
   - e.g. T1w to T1w, EPI to EPI etc.
 - Cross-modal comparison coming soon.


**Other modalities to be supported soon.**
