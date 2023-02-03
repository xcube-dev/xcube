.. _`xcube viewer demo`: https://bc-viewer.brockmann-consult.de/
.. _`xcube-viewer`: https://github.com/dcs4cop/xcube-viewer
.. _`README`: https://github.com/dcs4cop/xcube-viewer/blob/master/README.md
.. _`Euro Data Cube`: https://edc-viewer.brockmann-consult.de/

==========
Viewer App
==========

The xcube viewer app is a simple, single-page web application to be used with the xcube server.

Demo
====

To test the viewer app, you can use the `xcube viewer demo`_. This is our Brockmann Consult Demo xcube viewer.
Via the viewer's settings it is possible to change the xcube server url which is used for displaying data.
To do so open the viewer's settings panels, select "Server". A "Select Server" panel is opened, click the "+"
button to add a new server. Here is demo server that you may add for testing:

* Euro Data Cube Server (``https://edc-api.brockmann-consult.de/api``) has integrated amongst others a data cube with
  global essential climate variables (ECVs) variables from the ESA Earth System Data Lab Project.
  To access the Euro Data Cube viewer directly please visit https://edc-viewer.brockmann-consult.de .

Functionality
=============

The xcube viewer functionality is described exemplary using the `xcube viewer demo`_.
The viewer visualizes data from the xcube datasets on top of a basemap.
For zooming use the buttons in the top right corner of the map window or the zooming function of your
computer mouse. A scale for the map is located in the lower right corner and in the
upper left corner a corresponding legend to the mapped data of the data cube is available.

.. image:: _static/images/viewer/screenshot_overview.png
  :width: 800

A xcube viewer may hold several xcube datasets which you can select via the drop-down menu `Dataset`.
The viewed area automatically adjusts to a selected xcube dataset, meaning that if a newly selected
dataset is located in a different region, the correct region is displayed on the map.

.. image:: _static/images/viewer/screenshot_datasets.png
  :width: 800

If more than one variable is available within a selected xcube dataset, you may change the variable by using the drop-down menu
`Variable`.

.. image:: _static/images/viewer/screenshot_variables.png
  :width: 800

To see metadata for a dataset click on the `info`-icon on the right-hand side. Besides the dataset metadata you will
see the metadata for the selected variable.

.. image:: _static/images/viewer/screenshot_dataset_info.png
  :width: 800

To obtain a time series set a point marker on the map and then select the `graph`-icon next to the `Variables` drop-down
menu. You can select a different date by clicking into the time series graph on a value of interest. The data displayed
in the viewer changes accordingly to the newly selected date.

.. image:: _static/images/viewer/screenshot_timeseries.png
  :width: 800

The current date is preserved when you select a different variable and the data of the variable is mapped for the date.

.. image:: _static/images/viewer/screenshot_change_variable.png
  :width: 800

To generate a time series for the newly selected variable press the `time series`-icon again.

.. image:: _static/images/viewer/screenshot_timeseries_second_variable.png
  :width: 800

You may place multiple points on the map and you can generate time series for them. This allows a comparison between
two locations. The color of the points corresponds to the color of the graph in the time series. You can find the
coordinates of the point markers visualized in the time series beneath the graphs.

.. image:: _static/images/viewer/screenshot_timeseries_second_location.png
  :width: 800

To delete a created location use the `remove`-icon next to the `Place` drop-down menu.
Not only point location may be selected via the viewer, you can draw polygons and circular areas by using the icons on
the right-hand side of the `Place` drop-down menu as well. You can visualize time series for areas, too.

.. image:: _static/images/viewer/screenshot_polygon.png
  :width: 800

.. image:: _static/images/viewer/screenshot_circle.png
  :width: 800


In order to change the date for the data display use the calendar or step through the time line with the
arrows on the right-hand side of the calendar.

.. image:: _static/images/viewer/screenshot_calendar.png
  :width: 800

When a time series is displayed two time-line tools are visible, the upper one for selecting the date displayed
on the map of the viewer and the lower one may be used to narrow the time frame displayed in the time series graph.
Just above the graph of the time series on the right-hand side is an `x`-icon for removing the time series from the
view and to left of it is an icon which sets the time series back to the whole time extent.

.. image:: _static/images/viewer/screenshot_timeline.png
  :width: 800

To adjust the default settings select the `Settings`-icon on the very top right corner.
There you have the possibility to change the server url, in order to view data which is available via a different server.
You can choose a different language - if available - as well as set your preferences of displaying data and graph of
the time series.

.. image:: _static/images/viewer/screenshot_settings.png
  :width: 800

To see the map settings please scroll down in the settings window. There you can adjust the base map, switch the
displayed projection between `Geographic` and `Mercator`. You can also choose to turn image smoothing on and to
view the dataset boundaries.

On the very bottom of the `Settings` pop-up window you can see information about the viewer and server version.

.. image:: _static/images/viewer/screenshot_map_settings.png
  :width: 800

Back to the general view, if you would like to change the value ranges of the displayed variable you can do it by
clicking into the area of the legend where the value ticks are located or you can enter the desired values in the
`Minimum` and/or `Maximum` text field.

.. image:: _static/images/viewer/screenshot_value_ranges.png
  :width: 800

You can change the color mapping as well by clicking into the color range of the legend. There you can also decide to
hide lower values and it is possible to adjust the opacity.

.. image:: _static/images/viewer/screenshot_colormap.png
  :width: 800

The xcube viewer app is constantly evolving and enhancements are added, therefore please be aware that the above
described features may not always be completely up-to-date.

Build and Deploy
================

You can also build and deploy your own viewer instance. In the latter case, visit the `xcube-viewer`_ repository
on GitHub and follow the instructions provides in the related `README`_ file.


