.. _`xcube viewer demo`: https://xcube-viewer.s3.eu-central-1.amazonaws.com/index.html
.. _`xcube-viewer`: https://github.com/dcs4cop/xcube-viewer
.. _`DCS4COP Demo viewer`: https://dcs4cop-demo-viewer.obs-website.eu-de.otc.t-systems.com/
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
When you open the page a message "cannot reach server" will appear. This is normal as the demo is configured to
run with an xcube server started locally on default port 8080, see :doc:`webapi`. Hence, you can either run an xcube
server instance locally then reload the viewer page, or configure the viewer with an an existing xcube server.
To do so open the viewer's settings panels, select "Server". A "Select Server" panel is opened, click the "+"
button to add a new server. Here are two demo servers that you may add for testing:

* DCS4COP Demo Server (``https://xcube2.dcs4cop.eu/dcs4cop-dev/api/latest``) providing
  ocean color variables in the North Sea area for the `Data Cube Service for Copernicus`_ (DCS4COP) EU project;
* ESDL Server (``https://xcube.earthsystemdatalab.net``) providing global essential climate variables (ECVs)
  variables for the ESA `Earth System Data Lab`_.

Functionality
=============

The xcube viewer functionality is described exemplary using the `DCS4COP Demo viewer`_.
The xcube viewer includes a basemap layer with visualized data retrieved from the data cubes on top.
One can zoom in and out by using the zooming buttons in the top right corner of the map window or using
the zoom function of the computer mouse. A scale for the map is located in the lower right corner and in the
upper left corner a corresponding legend to the mapped data of the data cube is available.

.. image:: _static/screenshot_viewer_docu_1.png
  :width: 800

A xcube viewer may hold several xcube datasets which can be selected via the drop-down menu `Dataset`.
The viewer automatically adjusts the viewed area to the selected xcube dataset, meaning that if a newly selected
dataset is located in a different region, the map will display the correct region.


.. image:: _static/screenshot_viewer_docu_2.png
  :width: 800

If a selected xcube dataset contains more than one variable, the variable can be changed by using the drop-down menu
`Variable`.

.. image:: _static/screenshot_viewer_docu_3.png
  :width: 800

In order to view a time series for a certain location, one can click into the map and set a point. Next to
the drop-down menu for the variables is an icon with a graph, which generates a time series for the placed point.
The time series visualizes the data values available for the location through time. By clicking into one of the graph's points
the viewer displays the data corresponding to the newly selected date.

.. image:: _static/screenshot_viewer_docu_4.png
  :width: 800

Whenever switching the variable, the current date is preserved and the data for the selected variable is mapped.

.. image:: _static/screenshot_viewer_docu_5.png
  :width: 800

One can use the time series icon again and the time series will be now visualized for the current variable as well.

.. image:: _static/screenshot_viewer_docu_6.png
  :width: 800

Multiple points may be placed on the map and the time series can be generated for them. This may allow comparison between
two locations. The color of the points corresponds to the color of the graph of the time series. The coordinates of the points
used for the time series are visible beneath the graphs.

.. image:: _static/screenshot_viewer_docu_7.png
  :width: 800

By using the `remove`-icon the created locations can be deleted.
Not only point location may be selected via the viewer, the user can draw polygons and circular areas as well by
using the icons on the right-hand side of the `Place` drop-down menu. Time series for areas may be visualized as well.

.. image:: _static/screenshot_viewer_docu_8.png
  :width: 800

.. image:: _static/screenshot_viewer_docu_9.png
  :width: 800


The date for the data display can be changed by using the calendar or by stepping through the time line with the
arrows on the right-hand side of the calendar.

.. image:: _static/screenshot_viewer_docu_10.png
  :width: 800

When a time series is displayed there are two time-line tools visible, the upper one for selecting the date displayed
on the map of the viewer and the lower one may be used to narrow the time frame displayed in the time series graph.
Just above the graph of the time series on the right-hand side is an `x`-icon for removing the time series from the
view and to left of it is an icon which sets the time series back to the whole time extent.

.. image:: _static/screenshot_viewer_docu_11.png
  :width: 800

The user may change some of the default settings by selecting the `Settings`-icon on the very top right corner.
There the server url may be changed, in order to view data which is available via a different server.
The language - if available - may be changed and preferences of displaying data and graph of the time series.

On the very bottom of the `Settings` pop-up window information about the viewer and server version is included.

.. image:: _static/screenshot_viewer_docu_12.png
  :width: 800

The viewer allows the user to adjust the value ranges of the displayed variable. In order to change the value range
click into the legend where the value ticks are.

.. image:: _static/screenshot_viewer_docu_13.png
  :width: 800

Furthermore, the colormapping may be changed as well by clicking into the color range of the legend.

.. image:: _static/screenshot_viewer_docu_13.png
  :width: 800

The viewer app is constantly evolving and enhancements are added, therefore the above described features
may not always be completely up-to-date.

Build and Deploy
================

You can also build and deploy your own viewer instance. In the latter case, visit the `xcube-viewer`_ repository
on GitHub and follow the instructions provides in the related `README`_ file.


