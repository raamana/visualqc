.. highlight:: shell

============
Installation
============


Stable release
--------------

To install `visualqc`, run this command in your terminal:

.. code-block:: console

    $ pip install -U visualqc

This is the preferred method to install ``visualqc``, as it will always install the most recent stable release.


.. note::

 As VisualQC is GUI-based interface for rating, it needs access to GUI backend (tkinter etc). So, if you are trying to run it on a remote server over an SSH connection, make sure to start the SSH connection with the `-X` or `-Y` flags. As it's a complex operation, it might run into few issues, so check the errors in the terminal, search the internet for solutions and contact your server admin first to try resolve any environmental and configurational issues. If you still run into issues, please `open an issue <https://github.com/raamana/visualqc/issues/new/>`_ at the `visualqc` repo.

 It must be noted I/O speed would be lower over SSH relative to your local computation. In addition, there might be other difficulties to maintain a stable SSH connection for hours, it is a very risky choice to use VisualQC over SSH. You could lose hours and days of hardwork for multiple reasons nothing to do with you, such as wifi dropping out, laptop dying etc. Hence, *for small datasets (n<200)*, I discourage the usage of VisualQC over SSH. Please download the required data to your laptop/desktop, and run VisualQC locally.

 For larger datasets, it may not be possible to download the entire dataset locally, for lack of sufficient storage and/or skills to manage such a large transfer. Then you can attempt to get VisualQC's pre-processing done over SSH (generating the surface visualizations, feature extraction for outlier detection etc) on the remote server itself. This would produce various subfolders in the chosen output folder,such as `annot_visualizations` for the Freesufer module and some others for other modules. Then you can download this output folder locally along with any required input files (e.g. `mri/{orig,aparc+aseg}.mgz` for the Freesurfer module) for each subject, and you would be ready to run it locally. Check the end of :doc:`examples_freesurfer` and relevant documentation for other modules for more instructions and scripts to limit the amount of data you need to download, to save time and speed.

 If you are still confused, please `open an issue <https://github.com/raamana/visualqc/issues/new/>`_ at the `visualqc` repo.


If you don't have `Python`_ or `pip`_ installed, follow the following guides:

.. _pip: https://pip.pypa.io
.. _Python: _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Requirements
-------------

 - Python version 3.6 or higher is required. I strongly recommend upgrading to python 3 if possible. I recommend `conda to manage different versions of python in a separate virtual environment <https://conda.io/docs/user-guide/tasks/manage-python.html>`_.

 - The following python packages are required, which will be automatically installed when you issue the above command `pip install -U visualqc`:

    - nibabel
    - matplotlib>=2.1.1
    - mrivis
    - scipy
    - numpy
    - scikit-learn
    - nilearn


From sources
------------

The sources for visualqc can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/raamana/visualqc

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/raamana/visualqc/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/raamana/visualqc
.. _tarball: https://github.com/raamana/visualqc/tarball/master
