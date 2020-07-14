#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:05:30 2019

@author: avanetten
"""

import numpy as np
from osgeo import gdal, ogr, osr
import scipy.spatial
import geopandas as gpd
import rasterio as rio
import affine as af
import shapely
import time
import os
import sys
import cv2
import skimage
import subprocess
import matplotlib.pyplot as plt
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
        n_props = G_.nodes[n]
        if x_coord == 'lon':
            lat, lon = n_props['lat'], n_props['lon']
            x, y = lon, lat
        else:
            x, y = n_props[x_coord], n_props[y_coord]

        arr[i] = [x, y]
        kd_idx_dic[i] = n

    # now create kdtree from numpy array
    kdtree = scipy.spatial.KDTree(arr)
    if verbose:
        print("Time to create k-d tree:", time.time() - t1, "seconds")
    return kd_idx_dic, kdtree, arr


###############################################################################
def _query_kd_nearest(kdtree, kd_idx_dic, point, n_neighbors=10,
                      distance_upper_bound=1000, keep_point=True):
    '''
    Query the kd-tree for neighbors
    Return nearest node names, distances, nearest node indexes
    If not keep_point, remove the origin point from the list
    '''

    dists_m, idxs = kdtree.query(point, k=n_neighbors,
                                 distance_upper_bound=distance_upper_bound)

    idxs_refine = list(np.asarray(idxs))
    # print("apls_utils.query_kd_neareast - idxs_refilne:", idxs_refine)
    # print("apls_utils.query_kd_neareast - dists_m_refilne:", dists_m)
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
# def edit_node_props(props, new):
#    pass


###############################################################################
def create_buffer_geopandas(inGDF, buffer_distance_meters=2,
                            buffer_cap_style=1, dissolve_by='class',
                            projectToUTM=True, verbose=False):
    """
    Create a buffer around the lines of the geojson

    Arguments
    ---------
    inGDF : geodataframe
        Geodataframe from a SpaceNet geojson.
    buffer_distance_meters : float
        Width of buffer around geojson lines.  Formally, this is the distance
        to each geometric object.  Optional.  Defaults to ``2``.
    buffer_cap_style : int
        Cap_style of buffer, see: (https://shapely.readthedocs.io/en/stable/manual.html#constructive-methods)
        Defaults to ``1`` (round).
    dissolve_by : str
        Method for differentiating rows in geodataframe, and creating unique
        mask values.  Defaults to ``'class'``.
    projectToUTM : bool
        Switch to project gdf to UTM coordinates. Defaults to ``True``.
    verbose : bool
        Switch to print relevant values.  Defaults to ``False``.

    Returns
    -------
    gdf_buffer : geopandas dataframe
        Dataframe created from geojson

    """

    # inGDF = gpd.read_file(geoJsonFileName)
    if len(inGDF) == 0:
        return []

    # if we want a geojson instead of gdf for input
    # try:
    #    inGDF = gpd.read_file(geoJsonFileName)
    # except:
    #    return []

    # Transform gdf Roadlines into UTM so that Buffer makes sense
    if projectToUTM:
        tmpGDF = osmnx_funcs.project_gdf(inGDF, inGDF.crs)
    else:
        tmpGDF = inGDF

    if verbose:
        print("inGDF.columns:", tmpGDF.columns)
    gdf_utm_buffer = tmpGDF.copy()

    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = tmpGDF.buffer(buffer_distance_meters,
                                               cap_style=buffer_cap_style)

    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by=dissolve_by)
    gdf_utm_dissolve.crs = gdf_utm_buffer.crs
    if projectToUTM:
        gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)
    else:
        gdf_buffer = gdf_utm_dissolve
    if verbose:
        print("gdf_buffer['geometry'].values[0]:",
              gdf_buffer['geometry'].values[0])

    # add the dissolve_by column back into final gdf, since it's now the index
    gdf_buffer[dissolve_by] = gdf_buffer.index.values

    return gdf_buffer


###############################################################################
def _get_road_buffer(geoJson, im_vis_file, output_raster,
                     buffer_meters=2, burnValue=1,
                     # max_mask_val=1,
                     buffer_cap_style=6,
                     useSpacenetLabels=False,
                     plot_file='', figsize=(11, 3), fontsize=6,
                     dpi=800, show_plot=False,
                     valid_road_types=set([]), verbose=False):
    '''
    Wrapper around create_buffer_geopandas(), with plots
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

    gdf_buffer = create_buffer_geopandas(inGDF,
                                         buffer_distance_meters=buffer_meters,
                                         buffer_cap_style=buffer_cap_style,
                                         dissolve_by='class',
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
    # mask_gray = np.clip(mask_gray, 0, max_mask_val)

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


##############################################################################
def gdf_to_array(gdf, im_file, output_raster, burnValue=150,
                 mask_burn_val_key='', compress=True, NoData_value=0,
                 verbose=False):
    """
    Create buffer around geojson for desired geojson feature, save as mask

    Notes
    -----
    https://gis.stackexchange.com/questions/260736/how-to-burn-a-different-value-for-each-polygon-in-a-json-file-using-gdal-rasteri/260737


    Arguments
    ---------
    image_path : gdf
        Input geojson
    im_file : str
        Path to image file corresponding to gdf.
    output_raster : str
        Output path of saved mask (should end in .tif).
    burnValue : int
        Value to burn to mask. Superceded by mask_burn_val_key.
        Defaults to ``150``.
    mask_burn_val_key : str
        Column name in gdf to use for mask burning. Supercedes burnValue.
        Defaults to ``''`` (in which case burnValue is used).
    compress : bool
        Switch to compress output raster. Defaults to ``True``.
    NoData_value : int
        Value to assign array if no data exists. If this value is <0
        (e.g. -9999), a null value will show in the image. Defaults to ``0``.
    verbose : bool
        Switch to print relevant values.  Defaults to ``False``.

    Returns
    -------
    None
    """

    gdata = gdal.Open(im_file)

    # set target info
    if compress:
        target_ds = gdal.GetDriverByName('GTiff').Create(output_raster,
                                                         gdata.RasterXSize,
                                                         gdata.RasterYSize, 1,
                                                         gdal.GDT_Byte,
                                                         ['COMPRESS=LZW'])
    else:
        target_ds = gdal.GetDriverByName('GTiff').Create(output_raster,
                                                         gdata.RasterXSize,
                                                         gdata.RasterYSize, 1,
                                                         gdal.GDT_Byte)

    target_ds.SetGeoTransform(gdata.GetGeoTransform())
    if verbose:
        print("gdata.GetGeoTransform():", gdata.GetGeoTransform())

    

    # set raster info
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(gdata.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())
    if verbose:
        print ("target_ds:", target_ds)

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)

    outdriver = ogr.GetDriverByName('MEMORY')
    outDataSource = outdriver.CreateDataSource('memData')
    tmp = outdriver.Open('memData', 1)
    outLayer = outDataSource.CreateLayer("states_extent", raster_srs,
                                         geom_type=ogr.wkbMultiPolygon)
    # burn
    burnField = "burn"
    idField = ogr.FieldDefn(burnField, ogr.OFTInteger)
    outLayer.CreateField(idField)
    featureDefn = outLayer.GetLayerDefn()
    for j, geomShape in enumerate(gdf['geometry'].values):
        if verbose:
            print (j, "geomshape:", geomShape)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(ogr.CreateGeometryFromWkt(geomShape.wkt))
        if len(mask_burn_val_key) > 0:
            burnVal = int(gdf[mask_burn_val_key].values[j])
            if verbose:
                print("burnVal:", burnVal)
        else:
            burnVal = burnValue
        outFeature.SetField(burnField, burnVal)
        outLayer.CreateFeature(outFeature)
        # if verbose:
        #     print ("outFeature:", outFeature)
        outFeature = 0

    if len(mask_burn_val_key) > 0:
        gdal.RasterizeLayer(target_ds, [1], outLayer,
                            options=["ATTRIBUTE=%s" % burnField])
    else:
        gdal.RasterizeLayer(target_ds, [1], outLayer, burn_values=[burnVal])

    outLayer = 0
    outDatSource = 0
    tmp = 0
    return


###############################################################################
def geojson_to_arr(image_path, geojson_path, mask_path_out_gray,
                   buffer_distance_meters=2, buffer_cap_style=1,
                   dissolve_by='speed_mph', mask_burn_val_key='burnValue',
                   min_burn_val=0, max_burn_val=255,
                   verbose=False):

    """
    Create buffer around geojson for desired geojson feature, save as mask

    Arguments
    ---------
    image_path : str
        Path to input image corresponding to the geojson file.
    geojson_path : str
        Path to geojson file.
    mask_path_out_gray : str
        Output path of saved mask (should end in .tif).
    buffer_distance_meters : float
        Width of buffer around geojson lines.  Formally, this is the distance
        to each geometric object.  Optional.  Defaults to ``2``.
    buffer_cap_style : int
        Cap_style of buffer, see: (https://shapely.readthedocs.io/en/stable/manual.html#constructive-methods)
        Defaults to ``1`` (round).
    dissolve_by : str
        Method for differentiating rows in geodataframe, and creating unique
        mask values.  Defaults to ``'speed_m/s'``.
    mask_burn_value : str
        Column to name burn value in geodataframe. Defaults to ``'burnValue'``.
    min_burn_val : int
        Minimum value to burn to mask. Rescale all values linearly with this
        minimum value.  If <= 0, ignore.  Defaultst to ``0``.
    max_burn_val : int
        Maximum value to burn to mask. Rescale all values linearly with this
        maxiumum value.  If <= 0, ignore.  Defaultst to ``256``.
    verbose : bool
        Switch to print relevant values.  Defaults to ``False``.

    Returns
    -------
    gdf_buffer : geopandas dataframe
        Dataframe created from geojson
    """

    # get gdf_buffer
    try:
        inGDF = gpd.read_file(geojson_path)
    except TypeError:
        print("Empty mask for path:", geojson_path)
        # create emty mask
        h, w = cv2.imread(image_path, 0).shape[:2]
        mask_gray = np.zeros((h, w)).astype(np.uint8)
        skimage.io.imsave(mask_path_out_gray, mask_gray)
        # cv2.imwrite(mask_path_out, mask_gray)
        return []

    gdf_buffer = create_buffer_geopandas(
        inGDF, buffer_distance_meters=buffer_distance_meters,
        buffer_cap_style=buffer_cap_style, dissolve_by=dissolve_by,
        projectToUTM=False, verbose=verbose)

    if verbose:
        print("gdf_buffer.columns:", gdf_buffer.columns)
        print("gdf_buffer:", gdf_buffer)

    # set burn values
    burn_vals_raw = gdf_buffer[dissolve_by].values.astype(float)
    if verbose:
        print("burn_vals_raw:", burn_vals_raw)
    if (max_burn_val > 0) and (min_burn_val >= 0):
        scale_mult = (max_burn_val - min_burn_val) / np.max(burn_vals_raw)
        # scale_mult = max_burn_val / np.max(burn_vals_raw)
        burn_vals = min_burn_val + scale_mult * burn_vals_raw
    else:
        burn_vals = burn_vals_raw
    if verbose:
        print("np.unique burn_vals:", np.sort(np.unique(burn_vals)))
    gdf_buffer[mask_burn_val_key] = burn_vals

    # create mask
    gdf_to_array(gdf_buffer, image_path, mask_path_out_gray,
                 mask_burn_val_key=mask_burn_val_key,
                 verbose=verbose)

    return gdf_buffer


###############################################################################
def _create_speed_arr(image_path, geojson_path, mask_path_out_gray,
                      bin_conversion_func, mask_burn_val_key='burnValue',
                      buffer_distance_meters=2, buffer_cap_style=1,
                      dissolve_by='speed_m/s', bin_conversion_key='speed_mph',
                      verbose=False):

    '''
    Similar to create_arr_from_geojson()
    Create buffer around geojson for speeds, use bin_conversion_func to
    assign values to the mask
    '''

    # get gdf_buffer
    try:
        inGDF = gpd.read_file(geojson_path)
    except:
        print("Empty mask for path:", geojson_path)
        # create emty mask
        h, w = cv2.imread(image_path, 0).shape[:2]
        mask_gray = np.zeros((h, w)).astype(np.uint8)
        skimage.io.imsave(mask_path_out_gray, mask_gray)
        # cv2.imwrite(mask_path_out, mask_gray)
        return []

    gdf_buffer = create_buffer_geopandas(
        inGDF, buffer_distance_meters=buffer_distance_meters,
        buffer_cap_style=buffer_cap_style, dissolve_by=dissolve_by,
        projectToUTM=True, verbose=verbose)

    # set burn values
    speed_arr = gdf_buffer[bin_conversion_key].values
    burnVals = [bin_conversion_func(s) for s in speed_arr]
    gdf_buffer[mask_burn_val_key] = burnVals

    # create mask
    gdf_to_array(gdf_buffer, image_path, mask_path_out_gray,
                 mask_burn_val_key=mask_burn_val_key, verbose=verbose)

    return gdf_buffer


###############################################################################
def create_speed_gdf_v0(image_path, geojson_path, mask_path_out_gray,
                     bin_conversion_func, mask_burn_val_key='burnValue',
                     buffer_distance_meters=2, buffer_cap_style=1,
                     dissolve_by='speed_m/s', bin_conversion_key='speed_mph',
                     verbose=False):

    '''
    Create buffer around geojson for speeds, use bin_conversion_func to
    assign values to the mask
    '''

    # get gdf_buffer
    try:
        inGDF = gpd.read_file(geojson_path)
    except:
        print("Empty mask for path:", geojson_path)
        # create emty mask
        h, w = cv2.imread(image_path, 0).shape[:2]
        mask_gray = np.zeros((h, w)).astype(np.uint8)
        skimage.io.imsave(mask_path_out_gray, mask_gray)
        # cv2.imwrite(mask_path_out, mask_gray)
        return []

    # project
    projGDF = osmnx_funcs.project_gdf(inGDF)
    if verbose:
        print("inGDF.columns:", inGDF.columns)

    gdf_utm_buffer = projGDF.copy()
    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = gdf_utm_buffer.buffer(buffer_distance_meters,
                                                       buffer_cap_style)
    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by=dissolve_by)
    gdf_utm_dissolve.crs = gdf_utm_buffer.crs
    gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)
    if verbose:
        print("gdf_buffer['geometry'].values[0]:",
              gdf_buffer['geometry'].values[0])

    # set burn values
    speed_arr = gdf_buffer[bin_conversion_key].values
    burnVals = [bin_conversion_func(s) for s in speed_arr]
    gdf_buffer[mask_burn_val_key] = burnVals

    # create mask
    gdf_to_array(gdf_buffer, image_path, mask_path_out_gray,
                 mask_burn_val_key=mask_burn_val_key, verbose=verbose)

    return gdf_buffer


###############################################################################
def convert_array_to_multichannel(in_arr, n_channels=7, burnValue=255, 
                                  append_total_band=False, verbose=False):
    '''Take input array with multiple values, and make each value a unique
    channel.  Assume a zero value is background, while value of 1 is the 
    first channel, 2 the second channel, etc.'''
    
    h,w = in_arr.shape[:2]
    # scikit image wants it in this format by default
    out_arr = np.zeros((n_channels, h,w), dtype=np.uint8)
    #out_arr = np.zeros((h,w,n_channels), dtype=np.uint8)
    
    for band in range(n_channels):
        val = band + 1
        band_out = np.zeros((h, w), dtype=np.uint8)
        if verbose:
            print ("band:", band)
        band_arr_bool = np.where(in_arr == val)
        band_out[band_arr_bool] = burnValue
        out_arr[band,:,:] = band_out
        #out_arr[:,:,band] = band_out
 
    if append_total_band:
        tot_band = np.zeros((h,w), dtype=np.uint8)
        band_arr_bool = np.where(in_arr > 0)
        tot_band[band_arr_bool] = burnValue
        tot_band = tot_band.reshape(1,h,w)
        out_arr = np.concatenate((out_arr, tot_band), axis=0).astype(np.uint8)
    
    if verbose:
        print ("out_arr.shape:", out_arr.shape)
    return out_arr


### Helper Functions
###############################################################################
def CreateMultiBandGeoTiff(OutPath, Array):
    '''
    Author: Jake Shermeyer
    Array has shape:
        Channels, Y, X?
    '''
    driver = gdal.GetDriverByName('GTiff')
    DataSet = driver.Create(OutPath, Array.shape[2], Array.shape[1],
                            Array.shape[0], gdal.GDT_Byte,
                            ['COMPRESS=LZW'])
    for i, image in enumerate(Array, 1):
        DataSet.GetRasterBand(i).WriteArray(image)
    del DataSet

    return OutPath


###############################################################################
def geomGeo2geomPixel(geom, affineObject=[], input_raster='',
                      gdal_geomTransform=[]):
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
def _haversine(lon1, lat1, lon2, lat2):
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
