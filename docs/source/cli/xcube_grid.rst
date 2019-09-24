==============
``xcube grid``
==============

.. attention:: This tool will likely change in the near future.

Synopsis
========

Find spatial xcube dataset resolutions and adjust bounding boxes.

::

    $ xcube grid --help

::

    Usage: xcube grid [OPTIONS] COMMAND [ARGS]...
    
      Find spatial xcube dataset resolutions and adjust bounding boxes.
    
      We find suitable resolutions with respect to a possibly regional fixed
      Earth grid and adjust regional spatial bounding boxes to that grid. We
      also try to select the resolutions such that they are taken from a certain
      level of a multi-resolution pyramid whose level resolutions increase by a
      factor of two.
    
      The graticule at a given resolution level L within the grid is given by
    
          RES(L) = COVERAGE * HEIGHT(L)
          HEIGHT(L) = HEIGHT_0 * 2 ^ L
          LON(L, I) = LON_MIN + I * HEIGHT_0 * RES(L)
          LAT(L, J) = LAT_MIN + J * HEIGHT_0 * RES(L)
    
      With
    
          RES:      Grid resolution in degrees.
          HEIGHT:   Number of vertical grid cells for given level
          HEIGHT_0: Number of vertical grid cells at lowest resolution level.
    
      Let WIDTH and HEIGHT be the number of horizontal and vertical grid cells
      of a global grid at a certain LEVEL with WIDTH * RES = 360 and HEIGHT *
      RES = 180, then we also force HEIGHT = TILE * 2 ^ LEVEL.
    
    Options:
      --help  Show this message and exit.
    
    Commands:
      abox    Adjust a bounding box to a fixed Earth grid.
      levels  List levels for a resolution or a tile size.
      res     List resolutions close to a target resolution.

    
Example: Find suitable target resolution for a ~300m (Sentinel 3 OLCI FR resolution) 
fixed Earth grid within a deviation of 5%.

::

    $ xcube grid res 300m -D 5%

::

    TILE    LEVEL   HEIGHT  INV_RES RES (deg)       RES (m), DELTA_RES (%)
    540     7       69120   384     0.0026041666666666665   289.9   -3.4
    4140    4       66240   368     0.002717391304347826    302.5   0.8
    8100    3       64800   360     0.002777777777777778    309.2   3.1
    ...
    
289.9m is close enough and provides 7 resolution levels, which is good. Its inverse resolution is 384,
which is the fixed Earth grid identifier.

We want to see if the resolution pyramid also supports a resolution close to 10m 
(Sentinel 2 MSI resolution).

::

    $ xcube grid levels 384 -m 6

::

    LEVEL   HEIGHT  INV_RES RES (deg)       RES (m)
    0       540     3       0.3333333333333333      37106.5
    1       1080    6       0.16666666666666666     18553.2
    2       2160    12      0.08333333333333333     9276.6
    ...
    11      1105920 6144    0.00016276041666666666  18.1
    12      2211840 12288   8.138020833333333e-05   9.1
    13      4423680 24576   4.0690104166666664e-05  4.5

This indicates we have a resolution of 9.1m at level 12.

Lets assume we have xcube dataset region with longitude from 0 to 5 degrees
and latitudes from 50 to 52.5 degrees. What is the adjusted bounding box 
on a fixed Earth grid with the inverse resolution 384?

::

    $ xcube grid abox  0,50,5,52.5  384

::

    Orig. box coord. = 0.0,50.0,5.0,52.5
    Adj. box coord.  = 0.0,49.21875,5.625,53.4375
    Orig. box WKT    = POLYGON ((0.0 50.0, 5.0 50.0, 5.0 52.5, 0.0 52.5, 0.0 50.0))
    Adj. box WKT     = POLYGON ((0.0 49.21875, 5.625 49.21875, 5.625 53.4375, 0.0 53.4375, 0.0 49.21875))
    Grid size  = 2160 x 1620 cells
    with
      TILE      = 540
      LEVEL     = 7
      INV_RES   = 384
      RES (deg) = 0.0026041666666666665
      RES (m)   = 289.89450727414993

    
Note, to check bounding box WKTs, you can use the 
handy `Wicket <https://arthur-e.github.io/Wicket/sandbox-gmaps3.html>`_ tool.
     
