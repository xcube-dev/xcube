# xcube server changes

## Changes in 0.1.0.dev6 (in dev)

* Respecting chunk sizes when computing tile sizes [#44](https://github.com/dcs4cop/xcube-server/issues/44)
* New CLI option "--traceperf" that allows switching on performance diagnostics.
* The RESTful tile operations now have a query parameter "debug=1" which also switches on tile 
  computation performance diagnostics.
* Can now associate place groups with datasets.
* Major revision of API. URLs are now more consistent.

## Changes in 0.1.0.dev5

* Request for obtaining a legend for a layer of given by a variable of a data set was added.
* Added a Dockerfile to build an xcube docker image and to run the demo

## Changes in 0.1.0.dev4

* The timeseries API now returns ISO-formatted UTC dates [#26](https://github.com/dcs4cop/xcube-server/issues/26)