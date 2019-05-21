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
import os
import sys
import subprocess
from math import sqrt, radians, cos, sin, asin
# import logging

# add apls path and import apls_tools
path_apls_src = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_apls_src)
import osmnx_funcs


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
def _nodes_near_origin(G_, node, kdtree, kd_idx_dic,
                      x_coord='x', y_coord='y', radius_m=150, verbose=False):
    '''Get nodes a given radius from the desired node.  G_ should be the 
    maximally simplified graph'''

    # get node coordinates
    n_props = G_.node[node]
    x0, y0 = n_props[x_coord], n_props[y_coord]
    point = [x0, y0]

    # query kd tree for nodes of interest
    node_names, idxs_refine, dists_m_refine = _query_kd_ball(
        kdtree, kd_idx_dic, point, radius_m)
    if verbose:
        print(("subgraph node_names:", node_names))

    # get subgraph
    # G_sub = G_.subgraph(node_names)

    return node_names, dists_m_refine  # G_sub


###############################################################################
def G_to_kdtree(G_, x_coord='x', y_coord='y', verbose=False):
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
    if verbose:
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


###############################################################################
def convertTo8Bit(rasterImageName, outputRaster,
                  outputPixType='Byte',
                  outputFormat='GTiff',
                  rescale_type='rescale',
                  percentiles=[2, 98]):
    '''
    This does a relatively poor job of converting to 8bit, as opening in qgis
    the images look very different.
    rescale_type = [clip, rescale]
        if resceale, each band is rescaled to its own min and max
        if clip, scaling is done sctricly between 0 65535
    '''

    srcRaster = gdal.Open(rasterImageName)
    nbands = srcRaster.RasterCount
    if nbands == 3:
        cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat,
               '-co', '"PHOTOMETRIC=rgb"']
    else:
        cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat]

    for bandId in range(srcRaster.RasterCount):
        bandId = bandId+1
        band = srcRaster.GetRasterBand(bandId)
        if rescale_type == 'rescale':
            bmin = band.GetMinimum()
            bmax = band.GetMaximum()
            # if not exist minimum and maximum values
            if bmin is None or bmax is None:
                (bmin, bmax) = band.ComputeRasterMinMax(1)
            # else, rescale
            band_arr_tmp = band.ReadAsArray()
            bmin = np.percentile(band_arr_tmp.flatten(), percentiles[0])
            bmax = np.percentile(band_arr_tmp.flatten(), percentiles[1])

        else:
            bmin, bmax = 0, 65535

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(rasterImageName)
    cmd.append(outputRaster)
    print(cmd)
    subprocess.call(cmd)

    return


###############################################################################
def createBufferGeoPandas(inGDF, bufferDistanceMeters=5,
                          bufferRoundness=1, projectToUTM=True):
    '''Create a buffer around the lines of the geojson'''

    # inGDF = gpd.read_file(geoJsonFileName)
    if len(inGDF) == 0:
        return []

    # Transform gdf Roadlines into UTM so that Buffer makes sense
    if projectToUTM:
        tmpGDF = osmnx_funcs.project_gdf(inGDF)
    else:
        tmpGDF = inGDF

    gdf_utm_buffer = tmpGDF

    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = tmpGDF.buffer(bufferDistanceMeters,
                                               bufferRoundness)

    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by='class')
    gdf_utm_dissolve.crs = gdf_utm_buffer.crs

    if projectToUTM:
        gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)
    else:
        gdf_buffer = gdf_utm_dissolve

    return gdf_buffer


################################################################################
# def edit_node_props(props, new):
#    pass

###############################################################################
def get_road_buffer(geoJson, im_vis_file, output_raster,
                    buffer_meters=2, burnValue=1,
                    # max_mask_val=1,
                    bufferRoundness=6,
                    useSpacenetLabels=False,
                    plot_file='', figsize=(11, 3), fontsize=6,
                    dpi=800, show_plot=False,
                    valid_road_types=set([]), verbose=False):
    '''
    Get buffer around roads defined by geojson and image files
    valid_road_types serves as a filter of valid types (no filter if len==0)
    https://wiki.openstreetmap.org/wiki/Key:highway
    valid_road_types = set(['motorway', 'trunk', 'primary', 'secondary',
                            'tertiary',
                            'motorway_link', 'trunk_link', 'primary_link',
                            'secondary_link', 'tertiary_link',
                            'unclassified', 'residential', 'service' ])
    '''

    # get buffer

    # filter out roads of the wrong type
    try:
        inGDF_raw = gpd.read_file(geoJson)
    except:
        mask_gray = np.zeros(cv2.imread(im_vis_file, 0).shape)
        cv2.imwrite(output_raster, mask_gray)
        return [], []

    if useSpacenetLabels:
        inGDF = inGDF_raw
        # use try/except to handle empty label files
        try:
            inGDF['type'] = inGDF['road_type'].values
            inGDF['class'] = 'highway'
            inGDF['highway'] = 'highway'
        except:
            pass

    else:
        # filter out roads of the wrong type
        if (len(valid_road_types) > 0) and (len(inGDF_raw) > 0):
            if 'highway' in inGDF_raw.columns:
                inGDF = inGDF_raw[inGDF_raw['highway'].isin(valid_road_types)]
                # set type tag
                inGDF['type'] = inGDF['highway'].values
                inGDF['class'] = 'highway'
            else:
                inGDF = inGDF_raw[inGDF_raw['type'].isin(valid_road_types)]
                # set highway tag
                inGDF['highway'] = inGDF['type'].values

            if verbose:
                print("gdf.type:", inGDF['type'])
                if len(inGDF) != len(inGDF_raw):
                    print("len(inGDF), len(inGDF_raw)",
                          len(inGDF), len(inGDF_raw))
                    print("gdf['type']:", inGDF['type'])
        else:
            inGDF = inGDF_raw
            try:
                inGDF['type'] = inGDF['highway'].values
                inGDF['class'] = 'highway'
            except:
                pass

    gdf_buffer = createBufferGeoPandas(inGDF,
                                       bufferDistanceMeters=buffer_meters,
                                       bufferRoundness=bufferRoundness,
                                       projectToUTM=True)

    # make sure gdf is not null
    if len(gdf_buffer) == 0:
        mask_gray = np.zeros(cv2.imread(im_vis_file, 0).shape)
        cv2.imwrite(output_raster, mask_gray)
    # create label image
    else:
        gdf_to_array(gdf_buffer, im_vis_file, output_raster,
                     burnValue=burnValue)
    # load mask
    mask_gray = cv2.imread(output_raster, 0)
    #mask_gray = np.clip(mask_gray, 0, max_mask_val)

    if plot_file:

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=figsize)

        # road lines
        try:
            gdfRoadLines = gpd.read_file(geoJson)
            gdfRoadLines.plot(ax=ax0, marker='o', color='red')
        except:
            ax0.imshow(mask_gray)
        ax0.axis('off')
        ax0.set_aspect('equal')
        ax0.set_title('Unfiltered Roads from GeoJson', fontsize=fontsize)

        # first show raw image
        im_vis = cv2.imread(im_vis_file, 1)
        img_mpl = cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_mpl)
        ax1.axis('off')
        ax1.set_title('Raw Image', fontsize=fontsize)

        # plot mask
        ax2.imshow(mask_gray)
        ax2.axis('off')
        ax2.set_title('Roads Mask (' + str(np.round(buffer_meters))
                      + ' meter buffer)', fontsize=fontsize)

        # plot combined
        ax3.imshow(img_mpl)
        # overlay mask
        # set zeros to nan
        z = mask_gray.astype(float)
        z[z == 0] = np.nan
        # change palette to orange
        palette = plt.cm.gray
        palette.set_over('orange', 1.0)
        ax3.imshow(z, cmap=palette, alpha=0.4,
                   norm=matplotlib.colors.Normalize(vmin=0.5, vmax=0.9, clip=False))
        ax3.set_title('Raw Image + Buffered Roads', fontsize=fontsize)
        ax3.axis('off')

        #plt.axes().set_aspect('equal', 'datalim')

        # plt.tight_layout()
        plt.savefig(plot_file, dpi=dpi)
        if not show_plot:
            plt.close()

    return mask_gray, gdf_buffer


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
## Haversine formula example in Python
## Author: Wayne Dyck
#def distance_haversine(lat1, lon1, lat2, lon2, earth_radius_km=6371):
#    #lat1, lon1 = origin
#    #lat2, lon2 = destination
#
#    dlat = math.radians(lat2-lat1)
#    dlon = math.radians(lon2-lon1)
#    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
#        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
#    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
#    d = earth_radius_km * c
#
#    return d


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


###############################################################################
def _get_node_positions(G_, x_coord='x', y_coord='y'):
    '''Get position array for all nodes'''
    nrows = len(G_.nodes())
    ncols = 2
    arr = np.zeros((nrows, ncols))
    # populate node array
    for i, n in enumerate(G_.nodes()):
        n_props = G_.node[n]
        x, y = n_props[x_coord], n_props[y_coord]
        arr[i] = [x, y]
    return arr
