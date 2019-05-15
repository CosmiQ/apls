#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:05:30 2019

@author: avanetten
"""

import numpy as np
from osgeo import gdal, ogr, osr
import scipy.spatial
import rasterio as rio
import affine as af
import shapely
import time
from math import sqrt, radians, cos, sin, asin
# import logging


###############################################################################
def pixelToGeoCoord(xPix, yPix, inputRaster, sourceSR='', geomTransform='',
                    targetSR=''):
    '''From spacenet geotools'''
    # If you want to gauruntee lon lat output, specify TargetSR  otherwise, geocoords will be in image geo reference
    # targetSR = osr.SpatialReference()
    # targetSR.ImportFromEPSG(4326)
    # Transform can be performed at the polygon level instead of pixel level

    if targetSR == '':
        performReprojection = False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection = True

    if geomTransform == '':
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())

    geom = ogr.Geometry(ogr.wkbPoint)
    xOrigin = geomTransform[0]
    yOrigin = geomTransform[3]
    pixelWidth = geomTransform[1]
    pixelHeight = geomTransform[5]

    xCoord = (xPix * pixelWidth) + xOrigin
    yCoord = (yPix * pixelHeight) + yOrigin
    geom.AddPoint(xCoord, yCoord)

    if performReprojection:
        if sourceSR == '':
            srcRaster = gdal.Open(inputRaster)
            sourceSR = osr.SpatialReference()
            sourceSR.ImportFromWkt(srcRaster.GetProjectionRef())
        coord_trans = osr.CoordinateTransformation(sourceSR, targetSR)
        geom.Transform(coord_trans)

    return (geom.GetX(), geom.GetY())


###############################################################################
def nodes_near_point(x, y, kdtree, kd_idx_dic, x_coord='x', y_coord='y',
                     n_neighbors=-1,
                     radius_m=150,
                     verbose=False):
    """
    Get nodes near the given point.

    Notes
    -----
    if n_neighbors < 0, query based on distance,
    else just return n nearest neighbors

    Arguments
    ---------
    x : float
        x coordinate of point
    y: float
        y coordinate of point
    kdtree : scipy.spatial.kdtree
        kdtree of nondes in graph
    kd_idx_dic : dict
        Dictionary mapping kdtree entry to node name
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.
    n_neighbors : int
        Neareast number of neighbors to return. If < 0, ignore.
        Defaults to ``-1``.
    radius_meters : float
        Radius to search for nearest neighbors
    Returns
    -------
    kd_idx_dic, kdtree, arr : tuple
        kd_idx_dic maps kdtree entry to node name
        kdree is the actual kdtree
        arr is the numpy array of node positions
    """

    point = [x, y]

    # query kd tree for nodes of interest
    if n_neighbors > 0:
        node_names, idxs_refine, dists_m_refine = _query_kd_nearest(
            kdtree, kd_idx_dic, point, n_neighbors=n_neighbors)
    else:
        node_names, idxs_refine, dists_m_refine = _query_kd_ball(
            kdtree, kd_idx_dic, point, radius_m)

    if verbose:
        print(("subgraph node_names:", node_names))

    # get subgraph
    # G_sub = G_.subgraph(node_names)

    return node_names, dists_m_refine  # G_sub


###############################################################################
def G_to_kdtree(G_, x_coord='x', y_coord='y'):
    """
    Create kd tree from node positions.

    Notes
    -----
    (x, y) = (lon, lat)
    kd_idx_dic maps kdtree entry to node name:
        kd_idx_dic[i] = n (n in G.nodes())
    x_coord can be in utm (meters), or longitude

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with nodes assumed to have a dictioary of
        properties that includes position
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.

    Returns
    -------
    kd_idx_dic, kdtree, arr : tuple
        kd_idx_dic maps kdtree entry to node name
        kdree is the actual kdtree
        arr is the numpy array of node positions
    """

    nrows = len(G_.nodes())
    ncols = 2
    kd_idx_dic = {}
    arr = np.zeros((nrows, ncols))
    # populate node array
    t1 = time.time()
    for i, n in enumerate(G_.nodes()):
        n_props = G_.node[n]
        lat, lon = n_props['lat'], n_props['lon']
        x0, y0 = n_props[x_coord], n_props[y_coord]

        if x_coord == 'lon':
            x, y = lon, lat
        else:
            x, y = x0, y0

        arr[i] = [x, y]
        kd_idx_dic[i] = n

    # now create kdtree from numpy array
    kdtree = scipy.spatial.KDTree(arr)
    print("Time to create k-d tree:", time.time() - t1, "seconds")
    return kd_idx_dic, kdtree, arr


###############################################################################
def _query_kd_nearest(kdtree, kd_idx_dic, point, n_neighbors=10,
                      keep_point=True):
    '''
    Query the kd-tree for neighbors
    Return nearest node names, distances, nearest node indexes
    If not keep_point, remove the origin point from the list
    '''

    dists_m, idxs = kdtree.query(point, k=n_neighbors)

    idxs_refine = list(np.asarray(idxs))
    dists_m_refine = list(dists_m)
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine


###############################################################################
def _query_kd_ball(kdtree, kd_idx_dic, point, r_meters, keep_point=True):
    '''
    Query the kd-tree for neighbors within a distance r of the point
    Return nearest node names, distances, nearest node indexes
    if not keep_point, remove the origin point from the list
    '''

    dists_m, idxs = kdtree.query(point, k=500, distance_upper_bound=r_meters)
    # keep only points within distance and greaater than 0?
    if not keep_point:
        f0 = np.where((dists_m <= r_meters) & (dists_m > 0))
    else:
        f0 = np.where((dists_m <= r_meters))
    idxs_refine = list(np.asarray(idxs)[f0])
    dists_m_refine = list(dists_m[f0])
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine


###############################################################################
def _get_graph_extent(G_):
    '''min and max x and y'''
    xall = [G_.node[n]['x'] for n in G_.nodes()]
    yall = [G_.node[n]['y'] for n in G_.nodes()]
    xmin, xmax = np.min(xall), np.max(xall)
    ymin, ymax = np.min(yall), np.max(yall)
    dx, dy = xmax-xmin, ymax-ymin
    return xmin, xmax, ymin, ymax, dx, dy


###############################################################################
def _latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):
    '''
    Convert latitude, longitude coords to pixexl coords.
    From spacenet geotools
    '''

    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(4326)

    geom = ogr.Geometry(ogr.wkbPoint)
    # geom.AddPoint(lon, lat)
    geom.AddPoint(lat, lon)

    if targetsr == '':
        src_raster = gdal.Open(input_raster)
        targetsr = osr.SpatialReference()
        targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)
    if geom_transform == '':
        src_raster = gdal.Open(input_raster)
        transform = src_raster.GetGeoTransform()
    else:
        transform = geom_transform

    x_origin = transform[0]
    # print(x_origin)
    y_origin = transform[3]
    # print(y_origin)
    pixel_width = transform[1]
    # print(pixel_width)
    pixel_height = transform[5]
    # print(pixel_height)
    geom.Transform(coord_trans)
    # print(geom.GetPoint())
    x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
    y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height

    return (x_pix, y_pix)


###############################################################################
def _wmp2pixel(x, y, input_raster='', targetsr='', geom_transform=''):
    '''
    Convert wmp coords to pixexl coords.
    '''

    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(3857)

    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(x, y)

    if targetsr == '':
        src_raster = gdal.Open(input_raster)
        targetsr = osr.SpatialReference()
        targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)
    if geom_transform == '':
        src_raster = gdal.Open(input_raster)
        transform = src_raster.GetGeoTransform()
    else:
        transform = geom_transform

    x_origin = transform[0]
    # print(x_origin)
    y_origin = transform[3]
    # print(y_origin)
    pixel_width = transform[1]
    # print(pixel_width)
    pixel_height = transform[5]
    # print(pixel_height)
    geom.Transform(coord_trans)
    # print(geom.GetPoint())
    x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
    y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height

    return (x_pix, y_pix)


###############################################################################
def _set_pix_coords(G_, im_test_file=''):
    '''Get pixel coords.  Update G_ and get control_points, and graph_coords'''

    if len(G_.nodes()) == 0:
        return G_, [], []

    control_points, cp_x, cp_y = [], [], []
    for n in G_.nodes():
        u_x, u_y = G_.nodes[n]['x'], G_.nodes[n]['y']
        control_points.append([n, u_x, u_y])
        lat, lon = G_.nodes[n]['lat'], G_.nodes[n]['lon']
        if len(im_test_file) > 0:
            pix_x, pix_y = _latlon2pixel(lat, lon, input_raster=im_test_file)
        else:
            print("set_pix_coords(): oops, no image file")
            pix_x, pix_y = 0, 0
        # update G_
        G_.nodes[n]['pix_col'] = pix_x
        G_.nodes[n]['pix_row'] = pix_y
        G_.nodes[n]['x_pix'] = pix_x
        G_.nodes[n]['y_pix'] = pix_y
        # add to arrays
        cp_x.append(pix_x)
        cp_y.append(pix_y)
    # get line segements in pixel coords
    seg_endpoints = []
    for (u, v) in G_.edges():
        ux, uy = G_.nodes[u]['pix_col'], G_.nodes[u]['pix_row']
        vx, vy = G_.nodes[v]['pix_col'], G_.nodes[v]['pix_row']
        seg_endpoints.append([(ux, uy), (vx, vy)])
    gt_graph_coords = (cp_x, cp_y, seg_endpoints)

    return G_, control_points, gt_graph_coords


### Helper Functions
###############################################################################
def geomGeo2geomPixel(geom, affineObject=[], input_raster='', gdal_geomTransform=[]):
    '''spacenet utilities v3 geotools.py'''
    # This function transforms a shapely geometry in geospatial coordinates into pixel coordinates
    # geom must be shapely geometry
    # affineObject = rasterio.open(input_raster).affine
    # gdal_geomTransform = gdal.Open(input_raster).GetGeoTransform()
    # input_raster is path to raster to gather georectifcation information
    if not affineObject:
        if input_raster != '':
            affineObject = rio.open(input_raster).transform
        elif gdal_geomTransform != []:
            affineObject = af.Affine.from_gdal(gdal_geomTransform)
        else:
            return geom

    affineObjectInv = ~affineObject

    geomTransform = shapely.affinity.affine_transform(geom,
                                      [affineObjectInv.a,
                                       affineObjectInv.b,
                                       affineObjectInv.d,
                                       affineObjectInv.e,
                                       affineObjectInv.xoff,
                                       affineObjectInv.yoff]
                                      )

    return geomTransform


###############################################################################
def geomPixel2geomGeo(geom, affineObject=[], input_raster='', gdal_geomTransform=[]):
    '''spacenet utilities v3 geotools.py'''
    # This function transforms a shapely geometry in pixel coordinates into geospatial coordinates
    # geom must be shapely geometry
    # affineObject = rasterio.open(input_raster).affine
    # gdal_geomTransform = gdal.Open(input_raster).GetGeoTransform()
    # input_raster is path to raster to gather georectifcation information
    if not affineObject:
        if input_raster != '':
            affineObject = rio.open(input_raster).transform
        elif gdal_geomTransform != []:
            affineObject = af.Affine.from_gdal(gdal_geomTransform)
        else:
            return geom

    geomTransform = shapely.affinity.affine_transform(geom,
                                                      [affineObject.a,
                                                       affineObject.b,
                                                       affineObject.d,
                                                       affineObject.e,
                                                       affineObject.xoff,
                                                       affineObject.yoff]
                                                      )

    return geomTransform


###############################################################################
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points in m
    on the earth (specified in decimal degrees)
    http://stackoverflow.com/questions/15736995/how-can-i-
        quickly-estimate-the-distance-between-two-latitude-longitude-points
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    m = 1000. * km
    return m


###############################################################################
def get_gsd(im_test_file):
    '''return gsd in meters'''
    srcImage = gdal.Open(im_test_file)
    geoTrans = srcImage.GetGeoTransform()
    ulX = geoTrans[0]
    ulY = geoTrans[3]
    # xDist = geoTrans[1]
    yDist = geoTrans[5]
    # rtnX = geoTrans[2]
    # rtnY = geoTrans[4]

    # get haversine distance
    # dx = _haversine(ulX, ulY, ulX+xDist, ulY) #haversine(lon1, lat1, lon2, lat2)
    dy = _haversine(ulX, ulY, ulX, ulY+yDist)   #haversine(lon1, lat1, lon2, lat2)

    return dy  # dx


###############################################################################
def get_extent(srcFileImage):
    gdata = gdal.Open(srcFileImage)
    geo = gdata.GetGeoTransform()
    # data = gdata.ReadAsArray()

    xres = geo[1]
    yres = geo[5]
    # xmin = geo[0]
    # xmax = geo[0] + (xres * gdata.RasterXSize)
    # ymin = geo[3] + (yres * gdata.RasterYSize)
    # ymax = geo[3]
    xmin = geo[0] + xres * 0.5
    xmax = geo[0] + (xres * gdata.RasterXSize) - xres * 0.5
    ymin = geo[3] + (yres * gdata.RasterYSize) + yres * 0.5
    ymax = geo[3] - yres * 0.5

    return xmin, ymin, xmax, ymax


###############################################################################
def get_pixel_dist_from_meters(im_test_file, len_meters):
    '''For the input image, we want a buffer or other distance in meters,
    this function determines the pixel distance by calculating the GSD'''
    gsd = get_gsd(im_test_file)
    pix_width = max(1, np.rint(len_meters/gsd))

    return gsd, pix_width


###############################################################################
def get_unique(seq, idfun=None):
    '''https://www.peterbe.com/plog/uniqifiers-benchmark'''
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result
