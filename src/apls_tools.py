#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:32:20 2017

@author: avanetten
"""

import osmnx as ox
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
from osgeo import gdal, ogr, osr
import cv2
import subprocess
import rasterio as rio
import affine as af
import shapely
from shapely.geometry import MultiLineString
from matplotlib.patches import PathPatch
from matplotlib import collections  as mpl_collections
import matplotlib.path
import skimage.io
import logging

###############################################################################
### Previsously in apls.py
###############################################################################
def plot_metric(C, diffs, routes_str=[], 
                figsize=(10,5), scatter_png='', hist_png='',
                scatter_alpha=0.3, scatter_size=2, scatter_cmap='jet', dpi=300):
    ''' Plot outpute of cost metric in both scatterplot and histogram format'''
    
    # plot diffs
    title = 'Path Length Similarity: ' + str(np.round(C,2)) 
    fig, (ax0) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))
    #ax0.plot(diffs)
    ax0.scatter(list(range(len(diffs))), diffs, s=scatter_size, c=diffs, 
                alpha=scatter_alpha, 
                cmap=scatter_cmap)
    if len(routes_str) > 0:
        xticklabel_pad = 0.1
        ax0.set_xticks(list(range(len(diffs))))
        ax0.set_xticklabels(routes_str, rotation=50, fontsize=4)
        ax0.tick_params(axis='x', which='major', pad=xticklabel_pad)

    ax0.set_ylabel('Length Diff (Normalized)')
    ax0.set_xlabel('Path ID')
    ax0.set_title(title)
    #plt.tight_layout()
    if scatter_png:
        plt.savefig(scatter_png, dpi=dpi)
    
    # plot and plot diffs histo
    bins = np.linspace(0, 1, 30)
    bin_centers = np.mean( list(zip(bins, bins[1:])), axis=1)
    #digitized = np.digitize(diffs, bins)
    #bin_means = [np.array(diffs)[digitized == i].mean() for i in range(1, len(bins))]
    hist, bin_edges = np.histogram(diffs, bins=bins)
    fig, ax1 = plt.subplots(nrows=1, ncols=1,  figsize=figsize)
    #ax1.plot(bins[1:],hist, type='bar')
    #ax1.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0] )
    ax1.bar(bin_centers, 1.*hist/len(diffs), width=bin_centers[1]-bin_centers[0] )
    ax1.set_xlim([0,1])
    #ax1.set_ylabel('Num Routes')
    ax1.set_ylabel('Frac Num Routes')
    ax1.set_xlabel('Length Diff (Normalized)')
    ax1.set_title('Length Diff Histogram - Score: ' + str(np.round(C,2)) )
    ax1.grid(True)
    #plt.tight_layout()
    if hist_png:
        plt.savefig(hist_png, dpi=dpi)
    
    return

###############################################################################
###############################################################################
# For plotting the buffer...
#https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
def ring_coding(ob):
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(ob.coords)
    codes = np.ones(n, dtype=matplotlib.path.Path.code_type) * matplotlib.path.Path.LINETO
    codes[0] = matplotlib.path.Path.MOVETO
    return codes

#https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
def pathify(polygon):
    # Convert coordinates to path vertices. Objects produced by Shapely's
    # analytic methods have the proper coordinate order, no need to sort.
    vertices = np.concatenate(
                    [np.asarray(polygon.exterior)]
                    + [np.asarray(r) for r in polygon.interiors])
    codes = np.concatenate(
                [ring_coding(polygon.exterior)]
                + [ring_coding(r) for r in polygon.interiors])
    return matplotlib.path.Path(vertices, codes)
###############################################################################

###############################################################################
def plot_buff(G_, ax, buff=20, color='yellow', alpha=0.3, 
              title='Proposal Snapping',
              title_fontsize=8, outfile='', 
              dpi=200,
              verbose=False):
    '''plot buffer around graph using shapely buffer'''
        
    # get lines
    line_list = []    
    for u, v, key, data in G_.edges(keys=True, data=True):
        if verbose:
            print(("u, v, key:", u, v, key))
            print(("  data:", data))
        geom = data['geometry']
        line_list.append(geom)
    
    mls = MultiLineString(line_list)
    mls_buff = mls.buffer(buff)

    if verbose: 
        print(("type(mls_buff) == MultiPolygon:", type(mls_buff) == shapely.geometry.MultiPolygon))
    
    if type(mls_buff) == shapely.geometry.Polygon:
        mls_buff_list = [mls_buff]
    else:
        mls_buff_list = mls_buff
    
    for poly in mls_buff_list:
        x,y = poly.exterior.xy
        coords = np.stack((x,y), axis=1)
        interiors = poly.interiors
        #coords_inner = np.stack((x_inner,y_inner), axis=1)
        
        if len(interiors) == 0:
            #ax.plot(x, y, color='#6699cc', alpha=0.0, linewidth=3, solid_capstyle='round', zorder=2)
            ax.add_patch(matplotlib.patches.Polygon(coords, alpha=alpha, color=color))
        else:
            path = pathify(poly)
            patch = PathPatch(path, facecolor=color, edgecolor=color, alpha=alpha)
            ax.add_patch(patch)
 
    ax.axis('off')
    if len(title) > 0:
        ax.set_title(title, fontsize=title_fontsize)
    #outfile = os.path.join(res_dir, 'gt_raw_buffer.png')
    if outfile:
        plt.savefig(outfile, dpi=dpi)
    return ax

###############################################################################
def plot_node_ids(G, ax, node_list=[], alpha=0.8, fontsize=8, 
                  plot_node=False, node_size=15,
                  node_color='orange'):
    '''
    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    '''
    Gnodes = set(G.nodes())

    if len(node_list) == 0:
        nodes = G.nodes()
    else:
        nodes = node_list
    for n in nodes: #G.nodes():
        if n not in Gnodes:
            continue
        x,y = G.node[n]['x'], G.node[n]['y']
        if plot_node:
            ax.scatter(x,y, s=node_size, color=node_color)
        ax.annotate(str(n), xy=(x, y), alpha=alpha, fontsize=fontsize)
        
    return ax



###############################################################################
def plot_graph_on_im(G_, im_test_file, figsize=(8,8), show_endnodes=False,
                     width_key='speed_m/s', width_mult=0.125, 
                     color='lime', title='', figname='', 
                     default_node_size=15,
                     max_speeds_per_line=12,  dpi=300, plt_save_quality=75, 
                     ax=None, verbose=False):

    '''
    Overlay graph on image,
    if width_key == int, use a constant width'''

    try:
        im_cv2 = cv2.imread(im_test_file, 1)
        img_mpl = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB)   
    except:
        img_sk = skimage.io.imread(im_test_file)
        # make sure image is h,w,channels (assume less than 20 channels)
        if (len(img_sk.shape) == 3) and (img_sk.shape[0] < 20): 
            img_mpl = np.moveaxis(img_sk, 0, -1)
        else:
            img_mpl = img_sk
    h,w = img_mpl.shape[:2]
            
    node_x, node_y, lines, widths, title_vals  = [], [], [], [], []
    # get edge data
    for i,(u, v, edge_data) in enumerate(G_.edges(data=True)):
        #if type(edge_data['geometry_pix'])
        coords = list(edge_data['geometry_pix'].coords)
        if verbose: #(i % 100) == 0:
            print ("\n", i, u , v, edge_data)
            print ("edge_data:", edge_data)
            #print ("  edge_data['geometry'].coords", edge_data['geometry'].coords)
            print ("  coords:", coords)
        lines.append( coords ) 
        node_x.append( coords[0][0] )
        node_x.append( coords[-1][0] )
        node_y.append( coords[0][1] )
        node_y.append( coords[-1][1] )
        if type(width_key) == str:
            if verbose:
                print ("edge_data[width_key]:", edge_data[width_key])
            width = int(np.rint(edge_data[width_key] * width_mult))
            title_vals.append(int(np.rint( edge_data[width_key] )))
        else:
            width = width_key
        widths.append(width)

    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)   
             
    ax.imshow(img_mpl)
    # plot nodes?
    if show_endnodes:
        ax.scatter(node_x, node_y, color=color, s=default_node_size, alpha=0.5)
    # plot segments
    #print (lines)
    lc = mpl_collections.LineCollection(lines, colors=color, 
                                    linewidths=widths, alpha=0.4,
                                    zorder=2)
    ax.add_collection(lc)
    ax.axis('off')

    # title
    if len(title_vals) > 0:
        if verbose:
            print ("title_vals:", title_vals)
        title_strs = np.sort(np.unique(title_vals)).astype(str)
        # split title str if it's too long
        if len(title_strs) > max_speeds_per_line:
            # construct new title str
            n, b = max_speeds_per_line, title_strs
            title_strs = np.insert(b, range(n, len(b), n), "\n")  
            #title_strs = '\n'.join(s[i:i+ds] for i in range(0, len(s), ds))        
        if verbose:
            print ("title_strs:", title_strs)
        title = title + '\n' \
                + width_key + " = " + " ".join(title_strs)
    if title:
        #plt.suptitle(title)
        ax.set_title(title)
        
    plt.tight_layout()
    print ("title:", title)
    if title:
        plt.subplots_adjust(top=0.96)
    #plt.subplots_adjust(left=1, bottom=1, right=1, top=1, wspace=5, hspace=5)

    # set dpi to approximate native resolution
    if verbose:
        print ("img_mpl.shape:", img_mpl.shape)
    desired_dpi = int( np.max(img_mpl.shape) / np.max(figsize)) 
    if verbose:
        print ("desired dpi:", desired_dpi)
    # max out dpi at 3500
    dpi = int(np.min([3500, desired_dpi ]))
    if verbose:
        print ("plot dpi:", dpi)


    if figname:
        plt.savefig(figname, dpi=dpi, quality=plt_save_quality)
    
    return ax


###############################################################################
def plot_gt_prop_graphs(G_gt, G_prop, im_test_file, 
                      figsize=(16,8), show_endnodes=False,
                      width_key='Inferred Speed (mph)', width_mult=0.125, 
                      gt_color='cyan', prop_color='lime',
                      default_node_size=15,
                      title='', figname='', adjust=True, verbose=False):
    '''Plot the ground truth, and prediction mask Overlay graph on image,
    if width_key == int, use a constant width'''
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=figsize)  

    print ("Plotting ground truth...")
    _ = plot_graph_on_im(G_gt, im_test_file, figsize=figsize, 
                              show_endnodes=show_endnodes,
                              width_key=width_key, width_mult=width_mult,
                              color=gt_color, 
                              default_node_size=default_node_size,
                              title='Ground Truth:  ' + title, 
                              figname='', 
                              ax=ax0, verbose=verbose)
    print ("Plotting proposal...")
    _ = plot_graph_on_im(G_prop, im_test_file, figsize=figsize, 
                              show_endnodes=show_endnodes,
                              width_key=width_key, width_mult=width_mult,
                              color=prop_color, 
                              default_node_size=default_node_size,
                              title='Proposal:  ' + title, figname='', 
                              ax=ax1, verbose=verbose)

    #if title:
    #    plt.suptitle(title)
    #ax1.set_title(im_test_root)
    plt.tight_layout()
    if adjust:
        plt.subplots_adjust(top=0.96)
    #plt.subplots_adjust(left=1, bottom=1, right=1, top=1, wspace=5, hspace=5)

    if figname:
        plt.savefig(figname, dpi=300)
    
    return fig

###############################################################################
def get_graph_extent(G_):
    '''min and max x and y'''
    xall = [G_.node[n]['x'] for n in G_.nodes()]
    yall = [G_.node[n]['y'] for n in G_.nodes()]
    xmin, xmax = np.min(xall), np.max(xall)
    ymin, ymax = np.min(yall), np.max(yall)
    dx, dy = xmax-xmin, ymax-ymin   
    return xmin, xmax, ymin, ymax, dx, dy

###############################################################################
###############################################################################

### Conversion and data formatting functions
###############################################################################
def convert_to_8Bit(inputRaster, outputRaster,
                           outputPixType='Byte',
                           outputFormat='GTiff',
                           rescale_type='rescale',
                           percentiles=[2, 98]):
    '''
    Convert 16bit image to 8bit
    rescale_type = [clip, rescale]
        if clip, scaling is done sctricly between 0 65535 
        if rescale, each band is rescaled to a min and max 
        set by percentiles
    '''

    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of', 
           outputFormat]
    
    # iterate through bands
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
            bmin = np.percentile(band_arr_tmp.flatten(), 
                                 percentiles[0])
            bmax= np.percentile(band_arr_tmp.flatten(), 
                                percentiles[1])
        elif isinstance(rescale_type, dict):
            bmin, bmax = rescale_type[bandId]
        else:
            bmin, bmax = 0, 65535

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(inputRaster)
    cmd.append(outputRaster)
    print(("Conversion command:", cmd))
    subprocess.call(cmd)
    
    return

###############################################################################
def load_multiband_im(image_loc, method='gdal'):
    '''
    Use gdal to laod multiband files.  If image is 1-band or 3-band and 8bit, 
    cv2 will be much faster, so set method='cv2'
    Return numpy array
    '''
    
    im_gdal = gdal.Open(image_loc)
    nbands = im_gdal.RasterCount
    
    # use gdal, necessary for 16 bit
    if method == 'gdal':
        bandlist = []
        for band in range(1, nbands+1):
            srcband = im_gdal.GetRasterBand(band)
            band_arr_tmp = srcband.ReadAsArray()
            bandlist.append(band_arr_tmp)
        img = np.stack(bandlist, axis=2)

    # use cv2, which is much faster if data is 8bit and 1-band or 3-band
    elif method == 'cv2':
        # check data type (must be 8bit)
        srcband = im_gdal.GetRasterBand(1)
        band_arr_tmp = srcband.ReadAsArray()
        if band_arr_tmp.dtype == 'uint16': 
            print("cv2 cannot open 16 bit images")
            return []
        # ingest
        if nbands == 1:
            img = cv2.imread(image_loc, 0)
        elif nbands == 3:
            img = cv2.imread(image_loc, 1)
        else:
            print(("cv2 cannot open images with", nbands, "bands"))
            return []

    return img


###############################################################################
def latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):
    '''
    Convert latitude, longitude coords to pixexl coords.
    From spacenet geotools
    '''

    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(4326)

    geom = ogr.Geometry(ogr.wkbPoint)
    #geom.AddPoint(lon, lat)
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
def wmp2pixel(x, y, input_raster='', targetsr='', geom_transform=''):
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
        else:
            affineObject = af.Affine.from_gdal(gdal_geomTransform)

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
        else:
            affineObject = af.Affine.from_gdal(gdal_geomTransform)


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
def create_buffer_geopandas(geoJsonFileName, bufferDistanceMeters=2, 
                          bufferRoundness=1, projectToUTM=True):
    '''
    Create a buffer around the lines of the geojson. 
    Return a geodataframe.
    '''
    
    try:
        inGDF = gpd.read_file(geoJsonFileName)
    except:
        return []
    
    # set a few columns that we will need later
    inGDF['type'] = inGDF['road_type'].values            
    inGDF['class'] = 'highway'  
    inGDF['highway'] = 'highway'  
    
    if len(inGDF) == 0:
        return []

    # Transform gdf Roadlines into UTM so that Buffer makes sense
    if projectToUTM:
        tmpGDF = ox.project_gdf(inGDF)
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

###############################################################################
def CreateMultiBandGeoTiff(OutPath, Array):
    '''
    Author: Jake Shermeyer
    Array has shape:
        Channels, Y, X? 
    '''
    driver=gdal.GetDriverByName('GTiff')
    DataSet = driver.Create(OutPath, Array.shape[2], Array.shape[1], 
                            Array.shape[0], gdal.GDT_Byte,
                            ['COMPRESS=LZW'])
    for i, image in enumerate(Array, 1):
        DataSet.GetRasterBand(i).WriteArray( image )
    del DataSet
    
    return OutPath

###############################################################################
def gdf_to_array(gdf, im_file, output_raster, burnValue=150,
                 mask_burn_val_key='', compress=True,
                 verbose=False):
    
    '''
    Turn geodataframe to array, save as image file with non-null pixels 
    set to burnValue
    if mask_bur_val_key != '', set the burnvalue from the key in gdf
    https://gis.stackexchange.com/questions/260736/how-to-burn-a-different-value-for-each-polygon-in-a-json-file-using-gdal-rasteri/260737
    '''

    NoData_value = 0      # -9999

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
    
    # set raster info
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(gdata.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())
    
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)
    
    outdriver=ogr.GetDriverByName('MEMORY')
    outDataSource=outdriver.CreateDataSource('memData')
    tmp=outdriver.Open('memData',1)
    outLayer = outDataSource.CreateLayer("states_extent", raster_srs, 
                                         geom_type=ogr.wkbMultiPolygon)
    # burn
    burnField = "burn"
    idField = ogr.FieldDefn(burnField, ogr.OFTInteger)
    outLayer.CreateField(idField)
    featureDefn = outLayer.GetLayerDefn()
    for j,geomShape in enumerate(gdf['geometry'].values):
        
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(ogr.CreateGeometryFromWkt(geomShape.wkt))
        if len(mask_burn_val_key) > 0:
            burnVal = int(gdf[mask_burn_val_key].values[j])
            if verbose:
                print ("burnVal:", burnVal)
        else:
            burnVal = burnValue
        outFeature.SetField(burnField, burnVal)
        outLayer.CreateFeature(outFeature)
        #if verbose:
        #    print ("outFeature:", outFeature)
        outFeature = 0
    
    if len(mask_burn_val_key) > 0:
        gdal.RasterizeLayer(target_ds, [1], outLayer, options=["ATTRIBUTE=%s" % burnField])
    else:
        gdal.RasterizeLayer(target_ds, [1], outLayer, burn_values=[burnVal])
    
    outLayer = 0
    outDatSource = 0
    tmp = 0
        
    return 

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
        

###############################################################################
def get_road_buffer(geoJson, im_vis_file, output_raster, 
                              buffer_meters=2, burnValue=1, 
                              bufferRoundness=6, 
                              plot_file='', figsize=(6,6), fontsize=6,
                              dpi=800, show_plot=False, 
                              verbose=False):    
    '''
    Get buffer around roads defined by geojson and image files.
    Calls create_buffer_geopandas() and gdf_to_array().
    Assumes in_vis_file is an 8-bit RGB file.
    Returns geodataframe and ouptut mask.
    '''
    
     
    gdf_buffer = create_buffer_geopandas(geoJson,
                                         bufferDistanceMeters=buffer_meters,
                                         bufferRoundness=bufferRoundness, 
                                         projectToUTM=True)    

        
    # create label image
    if len(gdf_buffer) == 0:
        mask_gray = np.zeros(cv2.imread(im_vis_file,0).shape)
        cv2.imwrite(output_raster, mask_gray)        
    else:
        gdf_to_array(gdf_buffer, im_vis_file, output_raster, 
                                          burnValue=burnValue)
    # load mask
    mask_gray = cv2.imread(output_raster, 0)
    
    # make plots
    if plot_file:
        # plot all in a line
        if (figsize[0] != figsize[1]):
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4, figsize=figsize)#(13,4))
        # else, plot a 2 x 2 grid
        else:
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=figsize)
    
        # road lines
        try:
            gdfRoadLines = gpd.read_file(geoJson)
            gdfRoadLines.plot(ax=ax0, marker='o', color='red')
        except:
            ax0.imshow(mask_gray)
        ax0.axis('off')
        ax0.set_aspect('equal')
        ax0.set_title('Roads from GeoJson', fontsize=fontsize)
                
        # first show raw image
        im_vis = cv2.imread(im_vis_file, 1)
        img_mpl = cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_mpl)
        ax1.axis('off')
        ax1.set_title('8-bit RGB Image', fontsize=fontsize)
        
        # plot mask
        ax2.imshow(mask_gray)
        ax2.axis('off')
        ax2.set_title('Roads Mask (' + str(np.round(buffer_meters)) \
                                   + ' meter buffer)', fontsize=fontsize)
     
        # plot combined
        ax3.imshow(img_mpl)    
        # overlay mask
        # set zeros to nan
        z = mask_gray.astype(float)
        z[z==0] = np.nan
        # change palette to orange
        palette = plt.cm.gray
        #palette.set_over('yellow', 0.9)
        palette.set_over('lime', 0.9)
        ax3.imshow(z, cmap=palette, alpha=0.66, 
                norm=matplotlib.colors.Normalize(vmin=0.5, vmax=0.9, clip=False))
        ax3.set_title('8-bit RGB Image + Buffered Roads', fontsize=fontsize) 
        ax3.axis('off')
        
        #plt.axes().set_aspect('equal', 'datalim')

        plt.tight_layout()
        plt.savefig(plot_file, dpi=dpi)
        if not show_plot:
            plt.close()
        
    return mask_gray, gdf_buffer

###############################################################################
def create_speed_gdf(image_path, geojson_path, mask_path_out_gray, 
                bin_conversion_func, mask_burn_val_key='burnValue',
                bufferDistanceMeters=2, bufferRoundness=1,
                dissolve_by='speed_m/s', bin_conversion_key='speed_mph',
                verbose=False):
    
    '''Create buffer around geojson for speeds, use bin_conversion_func to 
    assign values to the mask'''

    # get gdf_buffer
    try:
        inGDF = gpd.read_file(geojson_path)
    except:
        print ("Empty mask for path:", geojson_path)
        # create emty mask
        h, w = cv2.imread(image_path, 0).shape[:2]
        mask_gray = np.zeros((h,w)).astype(np.uint8)
        skimage.io.imsave(mask_path_out_gray, mask_gray)
        #cv2.imwrite(mask_path_out, mask_gray)    
        return []
    
    # project
    projGDF = ox.project_gdf(inGDF)
    if verbose:
        print ("inGDF.columns:", inGDF.columns)
    
    gdf_utm_buffer = projGDF.copy()
    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = gdf_utm_buffer.buffer(bufferDistanceMeters,
                                                bufferRoundness)
    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by=dissolve_by)
    gdf_utm_dissolve.crs = gdf_utm_buffer.crs
    gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)
    if verbose:
        print ("gdf_buffer['geometry'].values[0]:", gdf_buffer['geometry'].values[0])

    # set burn values
    speed_arr = gdf_buffer[bin_conversion_key].values
    burnVals = [bin_conversion_func(s) for s in speed_arr]
    gdf_buffer[mask_burn_val_key] = burnVals

    # create mask
    gdf_to_array(gdf_buffer, image_path, mask_path_out_gray,
                            mask_burn_val_key=mask_burn_val_key,
                            verbose=verbose)
    
    return gdf_buffer

