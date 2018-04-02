Data formats and requirements
===============================

``visualqc`` relies on `nibabel <http://nipy.org/nibabel/>`_ to read the input image data, and supports all the formats that nibabel can read.

**The only requirement being the two images to be overlaid must be of the same shape, in dimensions and size.**

And, for a given subject ID, these two images must be in the same folder (although this might be relaxed in the future with a more generic input mechanism).

Following imaging formats are strongly encouraged:

 - Nifti
 - MGH/Freesurfer

while the following formats are supported, as they can be ready via ``nibabel``, but they are not routinely tested:

 - MINC (1/2)
 - gifti
 - Analyze
 - DICOM
 - PAR/REC
 - ECAT
 - TrackVis
 - Streamlines
