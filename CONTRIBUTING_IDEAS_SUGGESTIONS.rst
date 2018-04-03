.. highlight:: shell

================================================
Ideas and suggestions for contribution
================================================

You can contribute in different ways to different modalities, as noted below.

Alignment / Registration
-------------------------

 - Support for cross-modal comparison (e.g. PET on T1w MRI)
 - Additional blending methods
 - Implementation of intensity renormalization methods
 - in the zoomed mode, ability to navigate through different slices with an arrow keys or scrollbar.
 - *TO BE UPDATED*

Image Quality Metrics (IQM)
----------------------------

Computation of IQMs, esp. more advanced ones, require tissue segmentation in the anatomical T1 mri space. While this can easily be via Freesurfer, FSL or AFNI, the design inclination of ``VisualQC`` is:

 - to reduce the number of external dependencies (esp. big installations that are hard to acquire and maintain for a novice user without any system admin experience or help)
 - to being able to *compute only what is needed* and **only when required**

So implementation of tissue segmentation algorithms natively in Python would greatly help with a smooth user experience over time for a novice user e.g. reducing the installation difficulties, delivering new features and bug fixes and reducing the amount of computation.

 - implementation of fast tissue segmentation in pure Python (integrating existing python packages that are well-tested is fine also). This is helpful to QC both fMRI as well T1w MRI scans.
 - implementation of relevant IQMs in pure python - some are described well here: `QAP <http://preprocessed-connectomes-project.org/quality-assessment-protocol/#taxonomy-of-qa-measures>`_.
 - *TO BE UPDATED*

fMRI Preprocessing
-------------------

 - Implementation of algorithms for minimal preprocessing of fMRI scans such as head-motion correction, slice-timing correction etc. Some work has been done already at `pyaffineprep <https://github.com/dohmatob/pyaffineprep>`_ wherein we could contribute to making it more stable and testing it thoroughly.
 - *TO BE UPDATED*


**TO BE UPDATED**
