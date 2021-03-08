Gallery - Defacing algorithm accuracy
--------------------------------------------------------------------

VisualQC's Defacing module can help quickly evaluate the accuracy of defacing algorithms by presenting a comprehensive picture of the result of "defacing", and whether it led to data loss, or it failed to remove potentially reidentifiable info (such as facial features). Various examples below illustrate the defacing QC interface along with different types of failures by the defacing algorithm. Please note 1) the overlay in red on top of green MRI helps to highlight what has been removed/stripped to achieve defacing, and 2) that we may have blurred and/or greyed out some areas to avoid revealing any info remotely helping with patient reidentification.

An example result that correctly defaced a T1w MR image without over- or under-stripping:

.. image:: vis/defacing/defacing_illustration_1.png

Below we observe overstripping of the brain the frontal areas, which can be noticed both in the 3D render as well as in the composite overlay:

.. image:: vis/defacing/defacing_illustration_overstrip_frontal.png

The example below shows the defacing algorithm *understripping* the brain wherein eyes and few other facial features haven't been removed when they should have been. We greyed out some areas to protect the patient's privacy as much as possible.

.. image:: vis/defacing/defacing_illustration_cutbehindface.png


This example shows the algorithm going completely bonkers:

.. image:: vis/defacing/defacing_illustration_maskmisalign.png
