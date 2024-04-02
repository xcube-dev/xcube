# Spatial Rectification Algorithm

This page describes the algorithm used in the function `rectify_dataset()` 
of module `xcube.core.resampling`. This function performs a rectification of 
2-dimensional satellite imagery from its original viewing geometry into a 
common spatial coordinate reference system (CRS) that uses a cartesic 
coordinate system. 
It geometrically transforms data variables of a dataset with a provided 
source image geometry (given as [grid mapping](TODO)) 
into new data variables for a given target geometry. The source geometry 
is assumed to be given in satellite perspective in form of two 2-dimensional 
arrays of spatial coordinates, one for each spatial dimension. 
We also assume that coordinates of the viewing geometry are already 
terrain corrected with respect to a digital elevation model (DEM) that 
represents the Earth's geoid. The following figure illustrates the situation:

<img src="rectify/geoid.png" alt="Geoid" style="width:40em;display:block;margin-left:auto;margin-right:auto"/>

Thus, for each source pixel we have a spatial coordinates pair *x,y* which
is assumed to refer to an image pixel's center at *i + ½,j + ½*. 
The combined *x,y* images represent the surface of the Earth geoid as seen from 
the satellite. While the deltas between the pixel coordinates, *Δx* and *Δy*, are 
generally not constant in the source image geometry, we demand a unique spatial 
pixel size *Δx = Δy = const.* in the target image geometry.

The transformation algorithm described here is fast and simple.  
Given that the spatial pixel size in the target geometry is equal or less the 
pixel size of the source geometry, the algorithm is also general and accurate:
we walk through all *P = x,y* in the source image, span triangles between adjacent 
*P1, P2, P3*, and "paint" the triangles into a new 
rectified target image that contains the fractional source pixel coordinates
*i + u,j + v*.

<img src="rectify/source-coords.png" alt="Source coordinates" style="width:40em;display:block;margin-left:auto;margin-right:auto"/>

The spatial surface described by the field of supporting points *P(i+½, j+½) = x,y*
is representing the Earth's rough geoid surface. As the true surface is a 2D-fractal, 
there is no defined "best guess" for any point *P(i,j)* in between the supporting 
points at *P(i+½, j+½)*. Hence, we use triangulation for its simplicity so that any 
in-between *P* is found by linear interpolation.



