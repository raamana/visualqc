Gallery - Diffusion MRI artefact detection and rating
--------------------------------------------------------------

**This app/interface is in continuous development - your feedback to improve will be greatly appreciated:** `feedback welcome. <https://github.com/raamana/visualqc/issues/new>`_

The carpet plot [1]_ can easily reveal abnormal gradients to the human eye, which is what VisualQC starts with for diffusion MRI volumes - see below. VisualQC also overlay few useful statistics for each gradient to further help you review that particular gradient for any issues. We currently show 3 stats (mean, SD, and DVARS, but plan to few more such as framewise displacement (FD) and other motion parameters. Your suggestions and contributions are very `welcome <https://github.com/raamana/visualqc/issues/new>`_.

.. image:: vis/diffusion/dwi_vis_vqc_default1.png


Another example from another can be seen here:

.. image:: vis/diffusion/dwi_vis_vqc_default1.png


However, you may wish to dig into these "interesting" gradients further to get a full view, so you can see what is going on in that gradient. You can do that simply with a right click on any vertical line and VisualQC presents a full view of that gradient in different views (each with multiple slices):

.. image:: vis/diffusion/dwi_vis_specific_gradient_slice_1.png

Given the focus of VisualQC and purpose of quality control, we need to be able to examine every detail before ruling out or rating any artefacts, so you can right click again on any slice to zoom it further to reveal all the voxels:

.. image:: vis/diffusion/dwi_vis_specific_gradient_zoomed_in.png


You can already guess you can zoom in any of these slices too! :).


You can easily get back to the home screen with single clicks. Once you are done inspecting the run, you can rate the scan on various items, or approve it with `Pass`. FYI: You can not advance to next scan without rating the current scan. Then, click ``Next`` button to retrieve the next subject. And then, you can repeat the aforementioned process to thoroughly QC this run, without worrying about opening and closing and resizing multiple viewers and spreadsheets:



.. [1]  Power, J. D. (2017). A simple but useful way to assess fMRI scan qualities. NeuroImage, 154, 150–158. http://doi.org/10.1016/j.neuroimage.2016.08.009
