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

If you don't have `Python`_ or `pip`_ installed, follow the following guides:

.. _pip: https://pip.pypa.io
.. _Python: _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


Requirements
-------------

 - Python version 3 or higher is required. I strongly recommend upgrading to python 3 if possible. If not, I recommend `conda to manage different versions of python <https://conda.io/docs/user-guide/tasks/manage-python.html>`_.

 - The following python packages are required, which will be automatically installed when you issue the above command:

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
