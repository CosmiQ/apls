#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:30:14 2017

@author: avanetten

Heavily modified from  spacenet utilities graphtools
"""

import shapely.geometry
import fiona
import numpy as np
import os
import sys
import time
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
# from osmnx.utils import log
# from osmnx import core

path_apls_src = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_apls_src)
import osmnx_funcs


###############################################################################
def parse_OGR_nodes_paths(vectorFileName, osmidx=0, osmNodeidx=0,
                          nodeListGpd=gpd.GeoDataFrame(),
                          valid_road_types=set([]),
                          roadTypeField='type',
                          verbose=True,
                          super_verbose=False):
    """
    Construct dicts of nodes and paths with key=osmid and value=dict of
    attributes.

    Notes
    -----
    valid_road_types is a set of road types to be allowed

    Parameters
    ----------
    vectorFileName : str
        Absolute path to vector file supported by OGR that has
        line segments JSON response from from the Overpass API.

    Returns
    -------
    nodes, paths : tuple
    """

    # ensure valid vectorFileName
    try:
        source = fiona.open(vectorFileName, 'r')
        doit = True
    except:
        doit = False
        return {}, {}

    #dataSource = ogr.Open(vectorFileName, 0)
    # with fiona.open(vectorFileName, 'r') as source:
    if doit:
        #layer = dataSource.GetLayer()
        nodes = {}
        paths = {}
        for i, feature in enumerate(source):

            geom = feature['geometry']
            properties = feature['properties']
            # todo create more adjustable filter
            if roadTypeField in properties:
                road_type = properties['type']
            elif 'highway' in properties:
                road_type = properties['highway']
            elif 'road_type' in properties:
                road_type = properties['road_type']
            else:
                road_type = 'None'

            if ((i % 100) == 0) and verbose:
                print("\n", i, "/", len(source))
                print("   geom:", geom)
                print("   properties:", properties)
                print("   road_type:", road_type)

            ##################
            # check if road type allowable, continue if not
            # first check if road
            if (len(valid_road_types) > 0) and \
                    (geom['type'] == 'LineString' or geom['type'] == 'MultiLineString'):
                if road_type not in valid_road_types:
                    if verbose:
                        print("Invalid road type, skipping...")
                    continue
            ###################

            # skip empty linestrings
            if 'LINESTRING EMPTY' in list(properties.values()):
                continue

            osmidx = int(osmidx + 1)
            # print ("geom:", geom)

            if geom['type'] == 'LineString':
                # print osmNodeidx
                lineString = shapely.geometry.shape(geom)
                if super_verbose:
                    print("lineString.wkt:", lineString.wkt)
                # if len(geom['coordinates']) == 0:
                #    continue

                path, nodeList, osmNodeidx, nodeListGpd = \
                    processLineStringFeature(lineString, osmidx, osmNodeidx,
                                             nodeListGpd, properties=properties)
                # print(nodeListGpd.head())
                osmNodeidx = osmNodeidx+1
                osmidx = osmidx+1
                #print ("nodeList:", nodeList)
                nodes.update(nodeList)
                paths[osmidx] = path
                # print(geom.GetGeometryName())

            elif geom['type'] == 'MultiLineString':
                for linestring in shapely.geometry.shape(geom):

                    path, nodeList, osmNodeidx, nodeListGpd = \
                        processLineStringFeature(linestring, osmidx, osmNodeidx,
                                                 nodeListGpd, properties=properties)
                    osmNodeidx = osmNodeidx + 1
                    osmidx = osmidx+1
                    # print(geom.GetGeometryName())
                    nodes.update(nodeList)
                    paths[osmidx] = path

        source.close()

    return nodes, paths


###############################################################################
def processLineStringFeature(lineString, keyEdge, osmNodeidx,
                             nodeListGpd=gpd.GeoDataFrame(), properties={},
                             roadTypeField='type'):
    """Iterage over points in LineString"""
    # print ("lineString:", lineString)

    osmNodeidx = osmNodeidx + 1
    path = {}
    nodes = {}
    # print ("keyEdge:", keyEdge)
    # print ("lineString:", lineString)
    path['osmid'] = keyEdge

    nodeList = []

    for point in lineString.coords:

        pointShp = shapely.geometry.shape(Point(point))
        if nodeListGpd.size == 0:
            nodeId = np.array([])
        else:
            # print(nodeListGpd.head())
            # print(point)
            nodeId = nodeListGpd[nodeListGpd.distance(
                pointShp) == 0.0]['osmid'].values

        if nodeId.size == 0:
            nodeId = osmNodeidx
            nodeListGpd = nodeListGpd.append({'geometry': pointShp,
                                              'osmid': osmNodeidx},
                                             ignore_index=True)
            osmNodeidx = osmNodeidx + 1

            node = {}
            # add properties
            node['x'] = point[0]
            node['y'] = point[1]
            node['osmid'] = nodeId

            # add properties
            for key, value in list(properties.items()):
                node[key] = value
            if roadTypeField in properties:
                node['highway'] = properties['type']
            else:
                node['highway'] = 'unclassified'

            nodes[nodeId] = node

        else:
            nodeId = nodeId[0]

        nodeList.append(nodeId)

    path['nodes'] = nodeList
    # add properties
    for key, value in list(properties.items()):
        path[key] = value
    # also set 'highway' flag
    if roadTypeField in properties:
        path['highway'] = properties['type']
    else:
        path['highway'] = 'unclassified'

    return path, nodes, osmNodeidx, nodeListGpd


###############################################################################
def create_graphGeoJson(geoJson, name='unnamed', retain_all=True,
                        network_type='all_private', valid_road_types=set([]),
                        roadTypeField='type',
                        osmidx=0, osmNodeidx=0,
                        verbose=True, super_verbose=False):
    """
    Create a networkx graph from OSM data.

    Parameters
    ----------
    geoJson : geoJsonFile Name
        will support any file format supported by OGR
    name : string
        the name of the graph
    retain_all : bool
        if True, return the entire graph even if it is not connected
    network_type : string
        what type of network to create

    Returns
    -------
    networkx multidigraph
    """

    print('Creating networkx graph from downloaded OSM data...')
    start_time = time.time()

    # make sure we got data back from the server requests

    # create the graph as a MultiDiGraph and set the original CRS to EPSG 4326
    G = nx.MultiDiGraph(name=name, crs={'init': 'epsg:4326'})

    # extract nodes and paths from the downloaded osm data
    nodes = {}
    paths = {}

    if verbose:
        print("Running parse_OGR_nodes_paths...")
    nodes_temp, paths_temp = parse_OGR_nodes_paths(geoJson,
                                                   valid_road_types=valid_road_types,
                                                   verbose=verbose,
                                                   super_verbose=super_verbose,
                                                   osmidx=osmidx, osmNodeidx=osmNodeidx,
                                                   roadTypeField=roadTypeField)

    if len(nodes_temp) == 0:
        return G
    if verbose:
        print(("len(nodes_temp):", len(nodes_temp)))
        print(("len(paths_temp):", len(paths_temp)))

    # add node props
    for key, value in list(nodes_temp.items()):
        nodes[key] = value
        if super_verbose:
            print(("node key:", key))
            print(("  node value:", value))

    # add edge props
    for key, value in list(paths_temp.items()):
        paths[key] = value
        if super_verbose:
            print(("path key:", key))
            print(("  path value:", value))

    # add each node to the graph
    for node, data in list(nodes.items()):
        G.add_node(node, **data)

    # add each way (aka, path) to the graph
    if super_verbose:
        print(("paths:", paths))
    G = osmnx_funcs.add_paths(G, paths, network_type)

    # retain only the largest connected component, if caller did not set retain_all=True
    if not retain_all:
        G = osmnx_funcs.get_largest_component(G)

    # add length (great circle distance between nodes) attribute to each edge to use as weight
    G = osmnx_funcs.add_edge_lengths(G)

    print('Created graph with {:,} nodes and {:,} edges in {:,.2f} seconds'.format(
        len(list(G.nodes())), len(list(G.edges())), time.time()-start_time))

    return G


###############################################################################
if __name__ == "__main__":

    # test
    truth_dir = '/raid/cosmiq/spacenet/data/spacenetv2/spacenetLabels/AOI_2_Vegas/400m/'
    out_pkl = '/raid/cosmiq/spacenet/data/spacenetv2/spacenetLabels/AOI_2_Vegas/spacenetroads_AOI_2_Vegas_img10_graphTools.pkl'

    geoJson = os.path.join(
        truth_dir, 'spacenetroads_AOI_2_Vegas_img10.geojson')
    G0 = create_graphGeoJson(geoJson, name='unnamed',
                             retain_all=True,
                             verbose=True)
    # pkl
    nx.write_gpickle(G0, out_pkl)
