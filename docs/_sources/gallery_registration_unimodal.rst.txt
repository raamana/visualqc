Gallery - Registration : comparison of spatial alignment
--------------------------------------------------------------------

The alignment module of VisualQC provides many options to visualize the accuracy of registration. For example, using a checkerboard pattern:

.. image:: vis/alignment/alignment_mismatched_checkers.png

Once displayed, you can zoom in on any slice:

.. image:: vis/alignment/alignment_mismatched_checkers_zoomed.png

You can change the type of blending on the fly with a single click:

.. image:: vis/alignment/alignment_mismatched_colormix.png

A simple voxel-wise  difference map can reveal mismatch in an intuitive way:

.. image:: vis/alignment/alignment_mismatched_diff.png

Some people find the edge overlay especially useful to highlight any gross mismatch in the affine parameters:

.. image:: vis/alignment/alignment_mismatched_edge_overlay.png

.. image:: vis/alignment/alignment_mismatched_edge_overlay_zoomed.png

In the four screenshots below, you can see more examples of the interface when the images being compared are highly matched - in this particular instance, I am overlaying the smoothed version on the original:

.. image:: vis/alignment/alignment_similar_checkers.png

.. image:: vis/alignment/alignment_similar_checkers_zoomed.png

.. image:: vis/alignment/alignment_similar_diff.png

.. image:: vis/alignment/alignment_similar_edge_overlay.png
