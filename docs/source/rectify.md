# Spatial Rectification Algorithm

This chapter describes the algorithm used in the function [`rectify_dataset()`](api.html#xcube.core.resampling.rectify_dataset) 
of module `xcube.core.resampling`. The function geometrically transforms 
spatial [data variables](https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Variable) 
of a given [dataset](https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Dataset) 
with from an irregular source 
[grid mapping](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#grid-mappings-and-projections) 
into new data variables for a given regular target grid mapping and returns 
them as a new dataset. 

## Problem Description

The following figure shows a Sentinel-3 OLCI Level-1b scene in its original 
satellite perspective. In addition to the measured reflectances, the data 
product also provides the latitude and longitude for each pixel as two 
individual coordinate images. To the right, the absolute value of the gradient 
vectors of the latitude and longitude are shown:  

<!-- <img src="rectify/olci-input.png" alt="OLCI L1B Input" style="width:60em;display:block;margin-left:auto;margin-right:auto"/> -->
![OLCI L1B Input](rectify/olci-input.png)

The given coordinates, latitude and longitude, are _terrain corrected_ with 
respect to a digital elevation model (DEM) that represents the Earth's geoid.
Therefore, the gradient vectors are not monotonically varying over the scene, 
instead they represent the roughness of the geoid surface. This is depicted
by the following figure:

<!-- <img src="rectify/geoid.png" alt="Geoid" style="width:30em;display:block;margin-left:auto;margin-right:auto"/> -->
![Geoid](rectify/geoid.png)

A _rectification_ is the transformation of satellite imagery from its original 
viewing geometry into a target geometry that forms a regular grid in a defined 
coordinate reference system (CRS) with uniform spatial resolution for each 
pixel in each dimension. For the Sentinel-3 OLCI Level-1b scene above, the
rectified measurement image for the geographic projection (CRS EPSG:4326) 
is shown here:

<!-- <img src="rectify/olci-output.png" alt="OLCI L1C Output" style="width:60em;display:block;margin-left:auto;margin-right:auto"/> -->
![OLCI L1C Output](rectify/olci-output.png)


## Algorithm Description

The input to the rectification algorithm is satellite imagery in satellite 
viewing perspective. In addition, two images - one for each spatial dimension -
provide the terrain corrected spatial coordinates.
Thus, for each source pixel we have a given spatial coordinate pair *x,y* which
is assumed to refer to an image pixel's center at *i + ½, j + ½*. 

While the gradient of the coordinate images, *Δx* and *Δy* per pixel, is 
generally not constant in the source image geometry, we demand for the
output of the rectification algorithm *Δx = Δy = const* for all pixels in 
the generated target images.

A fast and simple algorithm to perform the rectification is to visit each
source pixel *i,j*, collect the spatial coordinates *P = x,y*, span
two triangles between four adjacent coordinates,
*(P1, P2, P3)* and *(P2, P4, P3)*, and "paint" fractional source pixel
coordinates into a new target lookup image. The lookup image can than be used 
to retrieve the source pixel values for a target image, either by nearest 
neighbor lookup or by interpolation.

<!-- <img src="rectify/source-coords.png" alt="Source coordinates" style="width:60em;display:block;margin-left:auto;margin-right:auto"/> -->
![Source coordinates](rectify/source-coords.png)

The true geoid surface is fractal, hence there is no defined "best guess" for 
any point *P(i+u,j+v)* with *0 ≤ u ≤ 1, 0 ≤ v ≤ 1* in between the given coordinates 
points at *P(i+½, j+½)*. Hence, we use triangulation for its simplicity so 
that any in-between *P* is found by linear interpolation.

From the coordinates *(P1, P2, P3)* of the first source triangle, the bounding
box in pixel coordinates in the target image can be exactly determined, 
because the target grid is regular, *P = x0 + i Δx, y0 - j Δy*, for each 
target pixel *i,j* and target pixel size *Δx,Δy*. Given *P* and the plane
given by *(P1, P2, P3)* the parameters *u,v* can be computed according to
*P = P1 + u (P2 – P1) + v (P3 – P1)*. 
If *0 ≤ u ≤ 1, 0 ≤ u ≤ 1, u+v ≤ 1*, then *P* is a point within the 
triangle. 

<!-- <img src="rectify/algo-1.png" alt="Algorithm #1" style="width:60em;display:block;margin-left:auto;margin-right:auto"/> -->
![Algorithm #1](rectify/algo-1.png)

At the same time, *u* and *v* are the fractions of source pixel
coordinate, *i + ½ + u* and *j + ½ + v*, which will be both stored in two 
target lookup images.

<!-- <img src="rectify/algo-2.png" alt="Algorithm #2" style="width:60em;display:block;margin-left:auto;margin-right:auto"/> -->
![Algorithm #2](rectify/algo-2.png)

After all source pixels have been processed, the resulting target lookup 
images containing the fractional source pixel indexes, *i + ½ + u* 
and *j + ½ + v*, can be used to efficiently map source measurements images 
into target measurements images. 

<!-- <img src="rectify/algo-3.png" alt="Algorithm #3" style="width:60em;display:block;margin-left:auto;margin-right:auto"/> -->
![Algorithm #3](rectify/algo-3.png)

In the simplest case, as shown above, a nearest neighbor lookup is performed 
according to:  
    *V = V2 if u > ½*  
    *V = V3 if v > ½*  
    *V = V1 else*  

The fractions *u,v* can also be used to perform a bilinear interpolation 
between the four adjacent measurements pixels:  
    *VA = V1 + u (V2 - V1)*  
    *VB = V3 + u (V4 - V3)*  
    *V = VA + v (VB - VA)*  

with *V1 = V(i,j)*, *V2 = V(i+1,j)*, *V3 = V(i,j+1)*, and *V4 = V(i+1,j+1)* 
being the pixel values of a measurement image *V*.
