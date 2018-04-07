Gallery - Freesurfer Parcellation
----------------------------------

Some examples of how the interface looks are shown below. The first screenshot showcases the use case wherein we can review the accuracy of Freesufer's cortical parcellation against the original MRI:

.. image:: vis/fs_contour/fs_cortical_contour_with_hist.png

You could zoom in on any panel you wish:

.. image:: vis/fs_contour/fs_cortical_contour_with_hist_zoomed.png

You could decrease color transparency level to see GM/WM boundaries better:

.. image:: vis/fs_contour/fs_contour_high_transparency.png

Or increase it to suit your preference:

.. image:: vis/fs_contour/fs_contour_low_transparency.png

Or turn it off completely in case really low contrast:

.. image:: vis/fs_contour/fs_contour_overlay_disabled.png

Further, we can choose show a single label (subcortical segmentation, tissue class or cortical ROI) - shown here are hippocampus and amygdala:

.. image:: vis/fs_contour/visual_qc_labels_contour_53_Pitt_0050039_v012_ns18_9x6.png

We can also add nearby amygdala:

.. image:: vis/fs_contour/visual_qc_labels_contour_53_54_Pitt_0050036_v02_ns21_6x7.png

And you can add as many ROIs as you like:

.. image:: vis/fs_contour/visual_qc_labels_contour_10_11_12_13_NYU_0051036.png

ROIs could be from anywhere in the MRI (including big cortical labels too!). For example, let's look at Insula (label 1035 in Freesurfer ColorLUT) in the left hemi-sphere :

.. image:: vis/fs_contour/visual_qc_labels_contour_1035_Pitt_0050032_v02_ns21_6x7.png

And, how about middle temporal?

.. image:: vis/fs_contour/visual_qc_labels_contour_2015_Pitt_0050035_v02_ns21_6x7.png

Let's just focus on axial view to get more detail:

.. image:: vis/fs_contour/visual_qc_labels_contour_2015_Pitt_0050039_v2_ns27_3x9.png


Fore more visualizations e.g. those with filled labels instead of contours, refer to :doc:`gallery_freesurfer_filled`.
