.. _`xcube viewer demo`: https://xcube-viewer.s3.eu-central-1.amazonaws.com/index.html
.. _`xcube-viewer`: https://github.com/dcs4cop/xcube-viewer
.. _`README`: https://github.com/dcs4cop/xcube-viewer/blob/master/README.md

.. _`Earth System Data Lab`: https://www.earthsystemdatalab.net/
.. _`Data Cube Service for Copernicus`: https://dcs4cop.eu/

==========
Viewer App
==========

The xcube viewer app is a simple, single-page web application to be used with the xcube server.

Demo
====

To test the viewer app, you can use the `xcube viewer demo`_.
When you open the page you will see a message "cannot reach server", this is normal as the demo is configured to
run with an xcube server started locally on default port 8080, see :doc:`webapi`. Hence, you can either run an xcube
server instance locally then reload the viewer page, or configure the viewer with an an existing xcube server.
To do so open the viewer's settings panels, select "Server". A "Select Server" panel is opened, click the "+"
button to add a new server. Here are two demo servers that you may add for testing:

* DCS4COP Demo Server (``https://xcube2.dcs4cop.eu/dcs4cop-dev/api/0.1.0.dev6/``) providing
  ocean color variables in the North Sea area for the `Data Cube Service for Copernicus`_ (DCS4COP) EU project;
* ESDL Server (``https://xcube.earthsystemdatalab.net``) providing global essential climate variables (ECVs)
  variables for the ESA `Earth System Data Lab`_.

Functionality
=============

Coming soon...


Build and Deploy
================

You can also build and deploy your own viewer instance. In the latter case, visit the `xcube-viewer`_ repository
on GitHub and follow the instructions provides in the related `README`_ file.


