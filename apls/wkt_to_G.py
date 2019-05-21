#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 00:10:40 2018

@author: avanetten

Read in a list of wkt linestrings, render to networkx graph

"""
from __future__ import print_function
import os
import sys
path_apls_src = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_apls_src)
import apls
import apls_utils
import osmnx_funcs
#from . import apls
#from . import apls_utils
import os
import utm
import shapely.wkt
import shapely.ops
from shapely.geometry import mapping, Point, LineString
import fiona
import networkx as nx
from osgeo import gdal, ogr, osr
import argparse
import json
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
# import cv2
# import osmnx as ox


###############################################################################
def wkt_list_to_nodes_edges(wkt_list, node_iter=0, edge_iter=0):
    '''Convert wkt list to nodes and edges
    Make an edge between each node in linestring. Since one linestring
    may contain multiple edges, this is the safest approach'''

    node_loc_set = set()    # set of edge locations
    node_loc_dic = {}       # key = node idx, val = location
    node_loc_dic_rev = {}   # key = location, val = node idx
    edge_loc_set = set()    # set of edge locations
    edge_dic = {}           # edge properties

    for i, lstring in enumerate(wkt_list):
        # get lstring properties
        shape = shapely.wkt.loads(lstring)
        xs, ys = shape.coords.xy
        length_orig = shape.length

        # iterate through coords in line to create edges between every point
        for j, (x, y) in enumerate(zip(xs, ys)):
            loc = (x, y)
            # for first item just make node, not edge
            if j == 0:
                # if not yet seen, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1

            # if not first node in edge, retrieve previous node and build edge
            else:
                prev_loc = (xs[j-1], ys[j-1])
                #print ("prev_loc:", prev_loc)
                prev_node = node_loc_dic_rev[prev_loc]

                # if new, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1
                # if seen before, retrieve node properties
                else:
                    node = node_loc_dic_rev[loc]

                # add edge, which is start_node to end_node
                edge_loc = (loc, prev_loc)
                edge_loc_rev = (prev_loc, loc)
                # shouldn't be duplicate edges, so break if we see one
                if (edge_loc in edge_loc_set) or (edge_loc_rev in edge_loc_set):
                    print("Oops, edge already seen, returning:", edge_loc)
                    return

                # get distance to prev_loc and current loc
                proj_prev = shape.project(Point(prev_loc))
                proj = shape.project(Point(loc))
                # edge length is the diffence of the two projected lengths
                #   along the linestring
                edge_length = abs(proj - proj_prev)
                # make linestring
                line_out = LineString([prev_loc, loc])
                line_out_wkt = line_out.wkt

                edge_props = {'start': prev_node,
                              'start_loc_pix': prev_loc,
                              'end': node,
                              'end_loc_pix': loc,
                              'length_pix': edge_length,
                              'wkt_pix': line_out_wkt,
                              'geometry_pix': line_out,
                              'osmid': i}
                #print ("edge_props", edge_props)

                edge_loc_set.add(edge_loc)
                edge_dic[edge_iter] = edge_props
                edge_iter += 1

    return node_loc_dic, edge_dic


###############################################################################
def nodes_edges_to_G(node_loc_dic, edge_dic, name='glurp'):
    '''Take output of wkt_list_to_nodes_edges(wkt_list) and create networkx 
    graph'''

    G = nx.MultiDiGraph()
    # set graph crs and name
    G.graph = {'name': name,
               'crs': {'init': 'epsg:4326'}
               }

    # add nodes
    # for key,val in node_loc_dic.iteritems():
    for key in node_loc_dic.keys():
        val = node_loc_dic[key]
        attr_dict = {'osmid': key,
                     'x_pix': val[0],
                     'y_pix': val[1]}
        G.add_node(key, **attr_dict)

    # add edges
    # for key,val in edge_dic.iteritems():
    for key in edge_dic.keys():
        val = edge_dic[key]
        attr_dict = val
        u = attr_dict['start']
        v = attr_dict['end']
        #attr_dict['osmid'] = str(i)

        #print ("nodes_edges_to_G:", u, v, "attr_dict:", attr_dict)
        if type(attr_dict['start_loc_pix']) == list:
            return

        G.add_edge(u, v, **attr_dict)

        # always set edge key to zero?  (for nx 1.X)
        # THIS SEEMS NECESSARY FOR OSMNX SIMPLIFY COMMAND
        #G.add_edge(u, v, key=0, attr_dict=attr_dict)
        ##G.add_edge(u, v, key=key, attr_dict=attr_dict)

    #G1 = osmnx_funcs.simplify_graph(G)

    G2 = G.to_undirected()

    return G2


###############################################################################
def wkt_to_shp(wkt_list, shp_file):
    '''Take output of build_graph_wkt() and render the list of linestrings
    into a shapefile
    # https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
    '''

    # Define a linestring feature geometry with one attribute
    schema = {
        'geometry': 'LineString',
        'properties': {'id': 'int'},
    }

    # Write a new shapefile
    with fiona.open(shp_file, 'w', 'ESRI Shapefile', schema) as c:
        for i, line in enumerate(wkt_list):
            shape = shapely.wkt.loads(line)
            c.write({
                    'geometry': mapping(shape),
                    'properties': {'id': i},
                    })

    return


###############################################################################
def get_node_geo_coords(G, im_file, verbose=False):

    nn = len(G.nodes())
    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
        if verbose:
            print("node:", n)
        if (i % 1000) == 0:
            print("node", i, "/", nn)
        x_pix, y_pix = attr_dict['x_pix'], attr_dict['y_pix']
        lon, lat = apls_utils.pixelToGeoCoord(x_pix, y_pix, im_file)
        [utm_east, utm_north, utm_zone, utm_letter] =\
            utm.from_latlon(lat, lon)
        attr_dict['lon'] = lon
        attr_dict['lat'] = lat
        attr_dict['utm_east'] = utm_east
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        attr_dict['utm_north'] = utm_north
        attr_dict['x'] = lon
        attr_dict['y'] = lat
        if verbose:
            print(" ", n, attr_dict)

    return G


###############################################################################
def convert_pix_lstring_to_geo(wkt_lstring, im_file):
    '''Convert linestring in pixel coords to geo coords'''
    shape = wkt_lstring  # shapely.wkt.loads(lstring)
    x_pixs, y_pixs = shape.coords.xy
    coords_latlon = []
    coords_utm = []
    for (x, y) in zip(x_pixs, y_pixs):
        lon, lat = apls_utils.pixelToGeoCoord(x, y, im_file)
        [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)
        coords_utm.append([utm_east, utm_north])
        coords_latlon.append([lon, lat])

    lstring_latlon = LineString([Point(z) for z in coords_latlon])
    lstring_utm = LineString([Point(z) for z in coords_utm])

    return lstring_latlon, lstring_utm, utm_zone, utm_letter


###############################################################################
def get_edge_geo_coords(G, im_file, remove_pix_geom=True,
                        verbose=False):

    ne = len(list(G.edges()))
    for i, (u, v, attr_dict) in enumerate(G.edges(data=True)):
        if verbose:
            print("edge:", u, v)
        if (i % 1000) == 0:
            print("edge", i, "/", ne)
        geom_pix = attr_dict['geometry_pix']
        lstring_latlon, lstring_utm, utm_zone, utm_letter = convert_pix_lstring_to_geo(
            geom_pix, im_file)
        attr_dict['geometry_latlon_wkt'] = lstring_latlon.wkt
        attr_dict['geometry_utm_wkt'] = lstring_utm.wkt
        attr_dict['length_latlon'] = lstring_latlon.length
        attr_dict['length_utm'] = lstring_utm.length
        attr_dict['length'] = lstring_utm.length
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        if verbose:
            print("  attr_dict:", attr_dict)

        # geometry screws up osmnx.simplify function
        if remove_pix_geom:
            #attr_dict['geometry_wkt'] = lstring_latlon.wkt
            attr_dict['geometry_pix'] = geom_pix.wkt

    return G


###############################################################################
def wkt_to_G(wkt_list, im_file=None,
             prop_subgraph_filter_weight='length_pix',
             min_subgraph_length=10,
             node_iter=0, edge_iter=0,
             simplify_graph=True, verbose=False):
    '''Execute all functions'''

    t0 = time.time()
    print("Running wkt_list_to_nodes_edges()...")
    node_loc_dic, edge_dic = wkt_list_to_nodes_edges(wkt_list,
                                                     node_iter=node_iter,
                                                     edge_iter=edge_iter)
    t1 = time.time()
    print("Time to run wkt_list_to_nodes_egdes():", t1 - t0, "seconds")

    #print ("node_loc_dic:", node_loc_dic)
    #print ("edge_dic:", edge_dic)

    print("Creating G...")
    G0 = nodes_edges_to_G(node_loc_dic, edge_dic)
    print("  len(G.nodes():", len(G0.nodes()))
    print("  len(G.edges():", len(G0.edges()))
    # for edge_tmp in G0.edges():
    #    print ("\n 0 wtk_to_G():", edge_tmp, G0.edge[edge_tmp[0]][edge_tmp[1]])

    t2 = time.time()
    print("Time to run nodes_edges_to_G():", t2-t1, "seconds")

    print("Clean out short subgraphs")
    G0 = apls._clean_sub_graphs(G0, min_length=min_subgraph_length,
                          weight=prop_subgraph_filter_weight,
                          max_nodes_to_skip=30,
                          verbose=True,
                          super_verbose=False)
    t3 = time.time()
    print("Time to run clean_sub_graphs():", t3-t2, "seconds")

#    print ("Simplifying graph")
#    G0 = osmnx_funcs.simplify_graph(G0.to_directed())
#    G0 = G0.to_undirected()
#    #G0 = osmnx_funcs.project_graph(G0)
#    #G_p_init = create_edge_linestrings(G_p_init, remove_redundant=True, verbose=False)
#    t3 = time.time()
#    print ("  len(G.nodes():", len(G0.nodes()))
#    print ("  len(G.edges():", len(G0.edges()))
#    print ("Time to run simplify graph:", t30 - t3, "seconds")

    # for edge_tmp in G0.edges():
    #    print ("\n 1 wtk_to_G():", edge_tmp, G0.edge[edge_tmp[0]][edge_tmp[1]])

    #edge_tmp = G0.edges()[5]
    #print (edge_tmp, "G0.edge props:", G0.edge[edge_tmp[0]][edge_tmp[1]])

    # geo coords
    if im_file:
        print("Running get_node_geo_coords()...")
        G1 = get_node_geo_coords(G0, im_file, verbose=verbose)
        t4 = time.time()
        print("Time to run get_node_geo_coords():", t4-t3, "seconds")

        print("Running get_edge_geo_coords()...")
        G1 = get_edge_geo_coords(G1, im_file, verbose=verbose)
        t5 = time.time()
        print("Time to run get_edge_geo_coords():", t5-t4, "seconds")

        print("projecting graph...")
        G_projected = osmnx_funcs.project_graph(G1)
        t6 = time.time()
        print("Time to project graph:", t6-t5, "seconds")

        # simplify
        #G_simp = osmnx_funcs.simplify_graph(G_projected.to_directed())
        # osmnx_funcs.plot_graph(G_projected)
        # G1.edge[19][22]

        Gout = G_projected  # G_simp

    else:
        Gout = G0

    if simplify_graph:
        print("Simplifying graph")
        t7 = time.time()
        G0 = osmnx_funcs.simplify_graph(Gout.to_directed())
        G0 = G0.to_undirected()
        Gout = osmnx_funcs.project_graph(G0)
        t8 = time.time()
        print("Time to run simplify graph:", t8-t7, "seconds")
        # When the simplify funciton combines edges, it concats multiple
        #  edge properties into a list.  This means that 'geometry_pix' is now
        #  a list of geoms.  Convert this to a linestring with
        #   shaply.ops.linemergeconcats
        print("Merge 'geometry' linestrings...")
        keys_tmp = ['geometry_pix', 'geometry_latlon_wkt', 'geometry_utm_wkt']
        for key_tmp in keys_tmp:
            print("Merge", key_tmp, "...")
            for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
                if (i % 10000) == 0:
                    print(i, u, v)
                geom_pix = attr_dict[key_tmp]
                # print (i, u, v, "geom_pix:", geom_pix)
                # print ("  type(geom_pix):", type(geom_pix))

                if type(geom_pix) == list:
                    # check if the list items are wkt strings, if so, create
                    #   linestrigs
                    # or (type(geom_pix[0]) == unicode):
                    if (type(geom_pix[0]) == str):
                        geom_pix = [shapely.wkt.loads(
                            ztmp) for ztmp in geom_pix]
                    # merge geoms
                    attr_dict[key_tmp] = shapely.ops.linemerge(geom_pix)

        # assign 'geometry' tag to geometry_utm_wkt
        for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
            if verbose:
                print("Create 'geometry' field in edges...")
            # geom_pix = attr_dict[key_tmp]
            line = attr_dict['geometry_utm_wkt']
            if type(line) == str:  # or type(line) == unicode:
                attr_dict['geometry'] = shapely.wkt.loads(line)
            else:
                attr_dict['geometry'] = attr_dict['geometry_utm_wkt']
            # attr_dict['geometry'] = attr_dict['geometry_utm_wkt']

        Gout = osmnx_funcs.project_graph(Gout)

    # get a few stats (and set to graph properties)
    print("Number of nodes:", len(Gout.nodes()))
    print("Number of edges:", len(Gout.edges()))
    Gout.graph['N_nodes'] = len(Gout.nodes())
    Gout.graph['N_edges'] = len(Gout.edges())

    # get total length of edges
    tot_meters = 0
    for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
        tot_meters += attr_dict['length']
    print("Length of edges (km):", tot_meters/1000)
    Gout.graph['Tot_edge_km'] = tot_meters/1000

    print("G.graph:", Gout.graph)

    t7 = time.time()
    print("Total time to run wkt_to_G():", t7-t0, "seconds")

    # for edge_tmp in Gout.edges():
    #   print ("\n 2 wtk_to_G():", edge_tmp, Gout.edge[edge_tmp[0]][edge_tmp[1]])

    return Gout


###############################################################################
if __name__ == "__main__":

    min_subgraph_length_pix = 300
    local = False  
    verbose = True
    super_verbose = False
    make_plots = False  
    pickle_protocol = 4     # 4 is most recent, python 2.7 can't read 4

    # local
    if local:
        prop_path = '/cosmiq/apls/inference_mod'
        path_images = '/cosmiq/spacenet/data/spacenetv2/AOI_2_Vegas_Test/400m/RGB-PanSharpen'
        res_root_dir = os.path.join(
            prop_path, 'results/AOI_2_Vegas_Test')
        csv_file = os.path.join(res_root_dir, 'wkt_submission.csv')
        graph_dir = os.path.join(res_root_dir, 'graphs')
        os.makedirs(graph_dir, exist_ok=True)

    # deployed on dev box
    else:
        from config import Config
        parser = argparse.ArgumentParser()
        parser.add_argument('config_path')
        args = parser.parse_args()
        with open(args.config_path, 'r') as f:
            cfg = json.load(f)
            config = Config(**cfg)

        # outut files
        res_root_dir = os.path.join(
            config.path_results_root, config.test_results_dir)
        path_images = os.path.join(
            config.path_data_root, config.test_data_refined_dir)
        csv_file = os.path.join(res_root_dir, config.wkt_submission)
        graph_dir = os.path.join(res_root_dir, config.graph_dir)
        os.makedirs(graph_dir, exist_ok=True)

#    csv_file = os.path.join(res_root_dir, 'merged_wkt_list.csv')
#    graph_dir = os.path.join(res_root_dir, 'graphs')
#    #os.makedirs(graph_dir, exist_ok=True)
#    try:
#        os.makedirs(graph_dir)
#    except:
#        pass

    # read in wkt list
    df_wkt = pd.read_csv(csv_file)
    # columns=['ImageId', 'WKT_Pix'])

    # iterate through image ids and create graphs
    t0 = time.time()
    image_ids = np.sort(np.unique(df_wkt['ImageId']))
    for i, image_id in enumerate(image_ids):

        # if image_id != 'AOI_2_Vegas_img586':
        #    continue

        print("\n")
        print(i, "/", len(image_ids), image_id)

        # for geo referencing, im_file should be the raw image
        if config.num_channels == 3:
            im_file = os.path.join(
                path_images, 'RGB-PanSharpen_' + image_id + '.tif')
        else:
            im_file = os.path.join(
                path_images, 'MUL-PanSharpen_' + image_id + '.tif')
        #im_file = os.path.join(path_images, image_id)
        if not os.path.exists(im_file):
            im_file = os.path.join(path_images, image_id + '.tif')

        # filter
        df_filt = df_wkt['WKT_Pix'][df_wkt['ImageId'] == image_id]
        wkt_list = df_filt.values
        #wkt_list = [z[1] for z in df_filt_vals]

        # print a few values
        print("\n", i, "/", len(image_ids), "num linestrings:", len(wkt_list))
        if verbose:
            print("image_file:", im_file, "wkt_list[:2]", wkt_list[:2])

        if (len(wkt_list) == 0) or (wkt_list[0] == 'LINESTRING EMPTY'):
            continue

        # create graph
        t1 = time.time()
        G = wkt_to_G(wkt_list, im_file=im_file,
                     min_subgraph_length_pix=min_subgraph_length_pix,
                     verbose=super_verbose)
        t2 = time.time()
        if verbose:
            print("Time to create graph:", t2-t1, "seconds")

        # print a node
        node = list(G.nodes())[-1]
        print(node, "random node props:", G.nodes[node])
        # print an edge
        edge_tmp = list(G.edges())[-1]
        # G.edge[edge_tmp[0]][edge_tmp[1]])
        print(edge_tmp, "random edge props:",
              G.edges([edge_tmp[0], edge_tmp[1]]))

        # save graph
        print("Saving graph to directory:", graph_dir)
        out_file = os.path.join(graph_dir, image_id.split('.')[0] + '.gpickle')
        nx.write_gpickle(G, out_file, protocol=pickle_protocol)
        # # save shapefile as well
        # ox.save_graph_shapefile(G, filename=image_id.split(
        #     '.')[0], folder=graph_dir, encoding='utf-8')

        #out_file2 = os.path.join(graph_dir, image_id.split('.')[0] + '.graphml')
        #ox.save_graphml(G, image_id.split('.')[0] + '.graphml', folder=graph_dir)

        # plot, if desired
        if make_plots:
            print("Plotting graph...")
            outfile_plot = os.path.join(graph_dir, image_id)
            print("outfile_plot:", outfile_plot)
            osmnx_funcs.plot_graph(G, fig_height=9, fig_width=9,
                          # save=True, filename=outfile_plot, margin=0.01)
                          )
            # plt.tight_layout()
            plt.savefig(outfile_plot, dpi=400)

        # if i > 30:
        #    break

    tf = time.time()
    print("Time to run wkt_to_G.py:", tf - t0, "seconds")

