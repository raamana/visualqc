Use-cases supported
===============================

VisualQC supports the following use cases:

 * Functional MRI scans (focused visual review, with rich and custom-built visualizations)
 * Freesurfer cortical parcellations (accuracy of pial/white surfaces on T1w mri)
 * Structural T1w MRI scans (artefact rating)
 * Volumetric segmentation accuracy (against T1w MRI)
 * Registration quality (spatial alignment) within a single modality (multimodal support coming)
 * Defacing accuracy


More detailed specifics for each modality are noted below:

Structural MRI use-cases
----------------------------

 - Checking the accuracy of white and pial surfaces (from Freesurfer and other algorithms)
 - Evaluate the accuracy of defacing or skull-stripping
 - Assess the accuracy of voxel-wise ROI or tissue segmentation (subcortical structures, gray matter, white matter or CSF masks)
 - Inspect the quality of raw T1 MRI scan (for motion, ringing, ghosting, or other artefacts)
 - Comparison of the registration quality or accuracy of the spatial alignment.

Functional MRI use-cases
----------------------------

 - Visual review of scan quality, identification of artefactual frames (motion, spikes, etc)
 - Quality control of the impact of different pre-processing steps

Diffusion MRI use-cases
----------------------------

 - Visual review of scan quality, identification of artefactual gradients (motion, spikes, etc)
 - Quality control of the impact of different pre-processing steps

Registration/Alignment use-cases
--------------------------------

 - Within-modality assessment of the accuracy of the spatial alignment
   - e.g. T1w to T1w, EPI to EPI etc.
 - Cross-modal comparison coming soon.


**Other modalities to be supported soon.**
