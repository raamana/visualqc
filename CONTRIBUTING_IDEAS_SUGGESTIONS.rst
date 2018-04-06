.. highlight:: shell

================================================
Ideas and suggestions for contribution
================================================

You can contribute in different ways to different modalities, as noted below.

Alignment / Registration
-------------------------

 - Improved support for cross-modal comparison (PET on T1w, or fMRI on T1w etc)
    - edge detection algorithm performance is variable in different modalities due to variations in the properties of noise and contrast levels - hence tweaking it to different modalities e.g. PET , fMRI etc would be great.
 - Additional blending methods
 - Implementation of intensity renormalization methods
 - in the zoomed mode, ability to navigate through different slices with an arrow keys or scrollbar.
 - *TO BE UPDATED*


Image Quality Metrics (IQM)
----------------------------
 - implementation of relevant IQMs in pure python - some are described well here: `QAP <http://preprocessed-connectomes-project.org/quality-assessment-protocol/#taxonomy-of-qa-measures>`_.


Tissue segmentation algorithms
------------------------------

Computation of IQMs, esp. more advanced ones, require tissue segmentation in the anatomical T1 mri space. While this can easily be via Freesurfer, FSL or AFNI, the design inclination of ``VisualQC`` is:

 - to reduce the number of external dependencies (esp. big installations that are hard to acquire and maintain for a novice user without any system admin experience or help)
 - to being able to *compute only what is needed* and **only when required**

So implementation of tissue segmentation algorithms natively in Python would greatly help with a smooth user experience over time for a novice user e.g. reducing the installation difficulties, delivering new features and bug fixes and reducing the amount of computation.

 - implementation of *reasonably fast* and *sufficiently accurate* tissue segmentation in pure Python (integrating existing python packages that are well-tested is fine also). This is helpful to QC both fMRI as well T1w MRI scans.

 - *TO BE UPDATED*

fMRI Preprocessing
-------------------

 - Implementation of algorithms for minimal preprocessing of fMRI scans such as head-motion correction, slice-timing correction etc. Some work has been done already at `pyaffineprep <https://github.com/dohmatob/pyaffineprep>`_ wherein we could contribute to making it more stable and testing it thoroughly.
 - Reorder the rows in the carpet plots in interesting groups:
  - such as within each tissue class (from above)
  - using arbitray parcellations, that are either data-driven (e.g. clustering voxels by time-series) or motivated by another user-chosen criteria
 - Incorporating task design to highlight event borders, or to check for task-coupled artefacts (such as motion)
 - Ability to mark individual frames (as "corrupted", or for "further review") for subsequent processing (such as censoring) and analyses.
  - This can be achieved with the Notes functionality e.g. including appropriate text ``bad_frames:{3,17}`` that can later be parsed programmatically.
  - However, this can be made much easier with clever interface and programming e.g. `Ctrl+click` on a particular frame can mark it is one idea.
 - *TO BE UPDATED*


**TO BE UPDATED**
