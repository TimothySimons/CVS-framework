.. _install:

============
Installation
============

Currently, cogvis is not part of the Anaconda distribution or PyPI. You can 
build from source or install the package via wheel, which is available on our
`Github page`_. 

To build from source, find the ``setup.py`` project file and execute::

    python -m pip install --upgrade build
    python -m build

To install via wheel::

    pip install cogvis-0.0.1-py3-none-any.whl

.. _Github page: https://github.com/TimothySimons/CVS_framework


Documentation
-------------

To build documentation, navigate to the docs folder and execute::

    make html
