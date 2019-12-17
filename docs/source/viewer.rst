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
Data from the xcube datasets are visualized on top of a basemap within the xcube viewer.
For zooming the buttons in the top right corner of the map window may be used or the zooming function of the
computer mouse. A scale for the map is located in the lower right corner and in the
upper left corner a corresponding legend to the mapped data of the data cube is available.

.. image:: _static/images/viewer/screenshot_overview.png
  :width: 800

A xcube viewer may hold several xcube datasets which can be selected via the drop-down menu `Dataset`.
The viewed area is automatically adjusted to a selected xcube dataset, meaning that if a newly selected
dataset is located in a different region, the correct region will be displayed on the map.

.. image:: _static/images/viewer/screenshot_datasets.png
  :width: 800

If more than one variable is available within a selected xcube dataset, the variable can be changed by using the drop-down menu
`Variable`.

.. image:: _static/images/viewer/screenshot_variables.png
  :width: 800

A time series may be obtained by setting a point marker on the map and then selecting the graph icon next to the `Variables` drop-down
menu. A different date can be selected by clicking into the time series graph on a value of interest. The data displayed
in the viewer will change accordingly to the newly selected date.

.. image:: _static/images/viewer/screenshot_timeseries.png
  :width: 800

The current date is preserved when a different variable is selected and the data of the variable is mapped for the date.

.. image:: _static/images/viewer/screenshot_change_variable.png
  :width: 800

A time series for the newly selected variable will be generated if the `time series`-icon is pressed again.

.. image:: _static/images/viewer/screenshot_timeseries_second_variable.png
  :width: 800

Multiple points may be placed on the map and the time series can be generated for them. This may allow comparison between
two locations. The color of the points corresponds to the color of the graph of the time series. The coordinates of the point
markers visualized the time series are displayed beneath the graphs.

.. image:: _static/images/viewer/screenshot_timeseries_second_location.png
  :width: 800

The created locations may be deleted by the `remove`-icon next to the `Place` drop-down menu.
Not only point location may be selected via the viewer, polygons and circular areas may be drawn by using the icons on
the right-hand side of the `Place` drop-down menu. Time series for areas may be visualized as well.

.. image:: _static/images/viewer/screenshot_polygon.png
  :width: 800

.. image:: _static/images/viewer/screenshot_circle.png
  :width: 800


The date for the data display can be changed by using the calendar or by stepping through the time line with the
arrows on the right-hand side of the calendar.

.. image:: _static/images/viewer/screenshot_calendar.png
  :width: 800

When a time series is displayed there are two time-line tools visible, the upper one for selecting the date displayed
on the map of the viewer and the lower one may be used to narrow the time frame displayed in the time series graph.
Just above the graph of the time series on the right-hand side is an `x`-icon for removing the time series from the
view and to left of it is an icon which sets the time series back to the whole time extent.

.. image:: _static/images/viewer/screenshot_timeline.png
  :width: 800

The default settings can be adjusted by the user by selecting the `Settings`-icon on the very top right corner.
There the server url may be changed, in order to view data which is available via a different server.
The language - if available - may be changed as well as preferences of displaying data and graph of the time series.

On the very bottom of the `Settings` pop-up window information about the viewer and server version is included.

.. image:: _static/images/viewer/screenshot_settings.png
  :width: 800

Furthermore, the value ranges of the displayed variable can be adjusted. This can be done by clicking into the area of the
legend where the value ticks are located.

.. image:: _static/images/viewer/screenshot_value_ranges.png
  :width: 800

The color mapping may be changed as well by clicking into the color range of the legend.

.. image:: _static/images/viewer/screenshot_colormap.png
  :width: 800

The viewer app is constantly evolving and enhancements are added, therefore the above described features
may not always be completely up-to-date.

Build and Deploy
================

You can also build and deploy your own viewer instance. In the latter case, visit the `xcube-viewer`_ repository
on GitHub and follow the instructions provides in the related `README`_ file.


