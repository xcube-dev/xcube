.. xcube-documentation documentation master file, created by
   sphinx-quickstart on Tue May  7 13:21:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the xcube documentation!
************************************
Xcube can be used for DataCube creation.

Installation
=============
1. Clone the GitHub repository with

``$ git clone https://github.com/dcs4cop/xcube.git``

2. Change into the directory of xcube and do the following steps:

.. codeblock::

   $ cd xcube
   $ conda env create
   $ activate xcube
   $ python setup.py develop

3. To update the project:

.. codeblock::

   $ activate xcube
   $ git pull --force
   $ python setup.py develop


Tools
*****


.. toctree::
   :maxdepth: 2
   :caption: Contents:

xcube gen
==========
.. automodule:: xcube.api.gen.gen
    :members:

xcube serve
============
.. automodule:: xcube.webapi.app
    :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
