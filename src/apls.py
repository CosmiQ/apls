#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:32:19 2017

@author: avanetten
"""

########################
path_apls = '/path/to/apls'
########################

import networkx as nx
import osmnx as ox   # https://github.com/gboeing/osmnx
import scipy.spatial
import scipy.stats
import numpy as np
import random
import utm           # pip install utm
import copy
from shapely.geometry import Point, LineString
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import sys
import os
import argparse

# add path
sys.path.extend([os.path.join(path_apls, 'src')])
import graphTools

###############################################################################
###############################################################################
def create_edge_linestrings(G):
    '''Ensure all edges have 'geometry' tag, use shapely linestrings'''
    
    G_ = G.copy()
    for u, v, key, data in G_.edges(keys=True, data=True):
        # create linestring if no geometry reported
        if 'geometry' not in data:
            sourcex, sourcey = G_.node[u]['x'],  G_.node[u]['y']
            targetx, targety = G_.node[v]['x'],  G_.node[v]['y']
            lstring = LineString([Point(sourcex, sourcey), 
                                  Point(targetx, targety)])
            data['geometry'] = lstring
            #G_.edge[u][v]['geometry'] = lstring
        else:            
            # check which direction linestring is travelling (it may be going from
            # v -> u, which means we need to reverse the linestring)
            # otherwise new edge is tangled
            line_geom = data['geometry']
            u_loc = [G_.node[u]['x'], G_.node[u]['y']]
            v_loc = [G_.node[v]['x'], G_.node[v]['y']]
            geom_p0 = list(line_geom.coords)[0]
            
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            #print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            if dist_to_u > dist_to_v:
                #data['geometry'].coords = list(line_geom.coords)[::-1]
                coords = list(data['geometry'].coords)[::-1]
                newL = LineString(coords)
                data['geometry'] = newL
            else:
                continue
        
    return G_

###############################################################################
def cut_linestring(line, distance):
    ''' 
    Cuts a line in two at a distance from its starting point
    http://toblerity.org/shapely/manual.html#linear-referencing-methods
    '''
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if pdl == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

###############################################################################
def get_closest_edge(G_, point):
    '''Return closest edge to point, and distance to said edge'''
    # get distances from point to lines
    dist_list = []
    edge_list = []
    geom_list = []
    p = point #Point(point_coords)
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        geom_list.append(line)
        dist_list.append(p.distance(line))
        edge_list.append([u,v])
    # get closest edge
    min_idx = np.argmin(dist_list)
    min_dist = dist_list[min_idx]
    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]
    
    return best_edge, min_dist, best_geom
    

###############################################################################
def insert_point(G_, point, node_id=100000, max_distance_meters=10,
                 allow_renaming=False,
                 verbose=False):
    '''
    Insert a new node in the graph closest to the given point, if it is
    within max_distance_meters.  Assume all edges have a linestring geometry
    http://toblerity.org/shapely/manual.html#object.simplify
    
    convert linestring to multipoint?
     https://github.com/Toblerity/Shapely/issues/190
     
     Sometimes the point to insert will have the same coordinates as an 
     existing point.  If allow_renaming == True, relabel the existing node 
    '''

    best_edge, min_dist, best_geom = get_closest_edge(G_, point)
    [u, v] = best_edge
 
    if verbose:
        print "best edge:", u,v
    
    if min_dist > max_distance_meters:
        if verbose:
            print "min_dist > max_distance_meters, skipping..."
        return G_, {}, 0, 0
    
    else:
        # update graph
        
        line_geom = best_geom #G_.edge[best_edge[0]][best_edge[1]][0]['geometry']        
        
        # Length along line that is closest to the point
        line_proj = line_geom.project(point)

        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(line_geom.project(point))
        x, y = new_point.x, new_point.y
        
        
        #################
        # create new node
        
        # first get zone, then convert to latlon
        _, _, zone_num, zone_letter = utm.from_latlon(G_.node[u]['lat'],
                                                          G_.node[u]['lon'])
        # convert utm to latlon
        lat, lon = utm.to_latlon(x, y, zone_num, zone_letter)
        
        # set properties
        #props = G_.node[u]
        node_props = {'highway': 'insertQ',
                 'lat':     lat,
                 'lon':     lon,
                 'osmid':   node_id,
                 'x':       x,
                 'y':       y}
        # add node
        G_.add_node(node_id, attr_dict=node_props)
        
        # assign, then update edge props for new edge
        edge_props_new = copy.deepcopy(G_.edge[u][v])
        # remove extraneous 0 key
        if edge_props_new.keys() == [0]:
            edge_props_new = edge_props_new[0]
 
         
        # cut line
        split_line = cut_linestring(line_geom, line_proj)
        #line1, line2, cp = cut_linestring(line_geom, line_proj)
        if split_line == None:
            print "type(split_line):", type(split_line)
            print "split_line:", split_line
            print "line_geom:", line_geom
            print "line_proj:", line_proj
            return G_, {}, 0, 0

        if verbose:
            print "split_line:", split_line
        
        #if cp.is_empty:        
        if len(split_line) == 1:
            if verbose:
                print "split line empty, min_dist:", min_dist

            # if the line cannot be split, that means that the new node 
            # is coincident with an existing node.  Relabel, if desired
            if allow_renaming:
                # get coincident node
                x_p, y_p = new_point.x, new_point.y
                x_u, y_u = G_.node[u]['x'], G_.node[u]['y']
                x_v, y_v = G_.node[v]['x'], G_.node[v]['y']
                if (x_p == x_u) and (y_p == y_u):
                    out_node = u
                elif (x_p == x_v) and (y_p == y_v):
                    out_node = v
                else:
                    print "Error in determining node coincident with" \
                    + str(node_id) + "along edge: " + str(best_edge)
                    print "x_p, y_p:", x_p, y_p
                    print "x_u, y_u:", x_u, y_u
                    print "x_v, y_v:", x_v, y_v
                    return
        
                node_props = G_.node[out_node]
                # A dictionary with the old labels as keys and new labels as values. A partial mapping is allowed.
                mapping = {out_node: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                if verbose:
                    print "Swapping out node ids:", mapping
                return Gout, node_props, x_p, y_p
            else:
                # if not renaming nodes, just return the original
                return G_, node_props, 0, 0
        
        else:
            # else, create new edges
            line1, line2 = split_line

            # get distances
            u_loc = [G_.node[u]['x'], G_.node[u]['y']]
            v_loc = [G_.node[v]['x'], G_.node[v]['y']]
            # compare to first point in linestring
            geom_p0 = list(line_geom.coords)[0]
            # or compare to inserted point? [this might fail if line is very
            #    curved!]
            #geom_p0 = (x,y)
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # reverse edge order if v closer than u
            if dist_to_v < dist_to_u:
                line2, line1 = split_line
                
            if verbose:
                print "Creating two edges from split..."
                print "   original_length:", line_geom.length
                print "   line1_length:", line1.length
                print "   line2_length:", line2.length
                print "   dist_u_to_point:", dist_to_u
                print "   dist_v_to_point:", dist_to_v

            # add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            # remove geometry?
            #edge_props_line1.pop('geometry', None) 
            # line2
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2
            # remove geometry?
            #edge_props_line1.pop('geometry', None) 

            # insert edge regardless of direction
            #G_.add_edge(u, node_id, attr_dict=edge_props_line1)
            #G_.add_edge(node_id, v, attr_dict=edge_props_line2)
            
            # check which direction linestring is travelling (it may be going from
            # v -> u, which means we need to reverse the linestring)
            # otherwise new edge is tangled
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            if verbose:
                print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, attr_dict=edge_props_line1)
                G_.add_edge(node_id, v, attr_dict=edge_props_line2)
            else:
                G_.add_edge(node_id, u, attr_dict=edge_props_line1)
                G_.add_edge(v, node_id, attr_dict=edge_props_line2)

            if verbose:
                print "insert edges:", u, '-',node_id, 'and', node_id, '-', v
                         
            
            # remove initial edge
            G_.remove_edge(u,v)
            
            return G_, node_props, x, y

###############################################################################
def insert_control_points(G_, control_points, max_distance_meters=10,
                          allow_renaming=False,
                          verbose=True):
    '''
    Wrapper around insert_point() for all control_points, assumed to be of
    the format:
        [[node_id, x, y], ... ]
    '''

    
    Gout = G_.copy()
    new_xs, new_ys = [], []
    if len(G_.nodes()) == 0:
        return Gout, new_xs, new_ys 
    
    for i, [node_id, x, y] in enumerate(control_points):
        if verbose:
            print "insert control point: i, node_id, x, y:", i, node_id, x, y
        point = Point(x, y)
        Gout, _, xnew, ynew = insert_point(Gout, point, node_id=node_id, 
                            max_distance_meters=max_distance_meters,
                            allow_renaming=allow_renaming,
                            verbose=verbose)
        if (x != 0) and (y != 0):
            new_xs.append(xnew)
            new_ys.append(ynew)
    
    return Gout, new_xs, new_ys
        

###############################################################################
def create_graph_midpoints(G_, linestring_delta=50, figsize=(0,0), 
                           is_curved_eps=0.03, n_id_add_val=1,
                           allow_renaming=False,
                           verbose=False):
    '''create midpoints along graph edges
    linestring_delta is the distance in meters between linsestring points
    n_id_add_val sets min midpoint id above existing nodes
        e.g.: G.nodes() = [1,2,4], if n_id_add_val = 5, midpoints will 
        be [9,10,11,...]
    if is_cuved_eps < 0, always inject points on line, regardless of 
    curvature'''
    
    #midpoint_loc = 0.5          # take the central midpoint for straight lines
    
    if len(G_.nodes()) == 0:
        return G_, [], []
    
    # midpoints
    xms, yms = [], []
    Gout = G_.copy()
    #midpoint_name_val, midpoint_name_inc = 0.01, 0.01
    midpoint_name_val, midpoint_name_inc = np.max(G_.nodes())+n_id_add_val, 1
    for u, v, key, data in G_.edges(keys=True, data=True):
        
        # curved line 
        if 'geometry' in data:
            
            # first edge props and  get utm zone and letter
            edge_props_init = G_.edge[u][v]
            _, _, zone_num, zone_letter = utm.from_latlon(G_.node[u]['lat'],
                                                          G_.node[u]['lon'])
            
            linelen = data['length']
            line = data['geometry']
            
            xs,ys = line.xy  # for plotting

            #################
            # check if curved or not
            minx, miny, maxx, maxy = line.bounds
            # get euclidean distance
            dst = scipy.spatial.distance.euclidean([minx, miny], [maxx, maxy])
            # ignore if almost straight 
            if np.abs(dst - linelen) / linelen < is_curved_eps:
                #print "Line straight, skipping..."
                continue
            #################

            #################
            # also ignore super short lines
            if linelen < 0.75*linestring_delta:
                #print "Line too short, skipping..."
                continue
            #################

            if verbose:
                print "u,v,key:", u,v,key
                print "data:", data
                print "edge_props_init:", edge_props_init

            # interpolate midpoints
            # if edge is short, use midpoint, else get evenly spaced points
            if linelen < linestring_delta:
                interp_dists = [0.5 * line.length]
            else:
                # get evenly spaced points
                npoints = len(np.arange(0, linelen, linestring_delta)) + 1
                interp_dists = np.linspace(0, linelen, npoints)[1:-1]
                if verbose:
                    print "interp_dists:", interp_dists
                
            # create nodes
            node_id_new_list = []
            xms_tmp , yms_tmp = [], []
            for j,d in enumerate(interp_dists):
                if verbose:
                    print "j,d", j,d

                midPoint = line.interpolate(d)
                xm0, ym0 = midPoint.xy
                xm = xm0[-1]
                ym = ym0[-1]
                point = Point(xm, ym)
                xms.append(xm)
                yms.append(ym)
                xms_tmp.append(xm)
                yms_tmp.append(ym)
                if verbose:
                    print "midpoint:", xm,ym
                    
                # add node to graph, with properties of u
                node_id = midpoint_name_val
                #node_id = np.round(u + midpoint_name_val,2)
                midpoint_name_val += midpoint_name_inc
                node_id_new_list.append(node_id)
                if verbose:
                    print "node_id:", node_id

                #if j > 3:
                #    continue
                
                # add to graph
                Gout, node_props, xn, yn = insert_point(Gout, point, 
                                                        node_id=node_id,
                                                        allow_renaming=allow_renaming,
                                                        verbose=verbose)
 

        # plot, if desired
        if figsize != (0,0):
            fig, (ax) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))        
            ax.plot(xs, ys, color='#6699cc', alpha=0.7,
                    linewidth=3, solid_capstyle='round', zorder=2)
            ax.scatter(xm,ym, color='red')
            ax.set_title('Line Midpoint')    
            plt.axis('equal')
            
        
    return Gout, xms, yms  
       
###############################################################################
def set_pix_coords(G_, im_test_file=''):
    '''get control points, and graph coords'''
    
    if len(G_.nodes()) == 0:
        return G_, [], []
    
    control_points, cp_x, cp_y = [], [], []
    for n in G_.nodes():
        u_x, u_y = G_.node[n]['x'], G_.node[n]['y']
        control_points.append([n, u_x, u_y])
        lat, lon = G_.node[n]['lat'], G_.node[n]['lon']
        if len(im_test_file) > 0:
            pix_x, pix_y = graphTools.latlon2pixel(lat, lon, 
                                               input_raster=im_test_file)
        else:
            pix_x, pix_y = 0, 0
        # update G_
        G_.node[n]['pix_col'] = pix_x
        G_.node[n]['pix_row'] = pix_y
        # add to arrays
        cp_x.append(pix_x)
        cp_y.append(pix_y)
    # get line segements in pixel coords
    seg_endpoints = []
    for (u,v) in G_.edges():
        ux, uy = G_.node[u]['pix_col'], G_.node[u]['pix_row']
        vx, vy = G_.node[v]['pix_col'], G_.node[v]['pix_row']
        seg_endpoints.append([(ux, uy), (vx, vy)])
    gt_graph_coords = (cp_x, cp_y, seg_endpoints)
    
    return G_, control_points, gt_graph_coords

###############################################################################
def clean_sub_graphs(G_, min_length=10, weight='length', verbose=False,
                     super_verbose=False):
    '''Remove subgraphs with a max path length less than min_length'''
    
    if len(G_.nodes()) == 0:
        return G_
    
    
    sub_graphs = list(nx.connected_component_subgraphs(G_))
    bad_nodes = []
    if verbose:
        print "len(G_.nodes()):", len(G_.nodes())
        print "len(G_.edges()):", len(G_.edges())
        print "G_.nodes:", G_.nodes()
    for G_sub in sub_graphs:
        all_lengths = nx.all_pairs_dijkstra_path_length(G_sub, weight=weight)
        if super_verbose:
                    print "\nGs.nodes:", G_sub.nodes()
                    print "all_lengths:", all_lengths
        # get all lenghts
        lens = []
        for u,v in all_lengths.iteritems():
            for uprime, vprime in v.iteritems():
                lens.append(vprime)
                if super_verbose:
                    print "u, v", u,v
                    print "  uprime, vprime:", uprime, vprime
        max_len = np.max(lens)
        if verbose:
            print "Max length of path:", max_len
        if max_len < min_length:
            bad_nodes.extend(G_sub.nodes())
            if verbose:
                print "appending to bad_nodes:", G_sub.nodes()

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)
    if verbose:
        print "bad_nodes:", bad_nodes
        print "len(G_.nodes()):", len(G_.nodes())
        print "len(G_.edges()):", len(G_.edges())
        print "G_.nodes:", G_.nodes()
        
    return G_

###############################################################################
def create_gt_graph(geoJson, im_test_file, network_type='all_private',
                 linestring_delta=50, is_curved_eps=0.012, 
                 valid_road_types=set([]),
                 osmidx=0, osmNodeidx=0,
                 min_subgraph_length=10, weight='length',
                 verbose=False):
    '''Ingest graph from geojson file and refine'''

    t0 = time.time()
    G0gt_init = graphTools.create_graphGeoJson(geoJson, name='unnamed', 
                                            retain_all=True, 
                                            network_type=network_type,
                                            valid_road_types=valid_road_types,
                                            osmidx=osmidx,
                                            osmNodeidx=osmNodeidx,
                                            verbose=verbose)
    t1 = time.time()
    if verbose:
        print "Time to create_graphGeoJson:", t1 - t0, "seconds"

    G0gt = ox.project_graph(G0gt_init)    
    #G1gt = ox.simplify_graph(G0gt)    
    #G2gt_init = G1gt.to_undirected()
    G2gt_init = ox.simplify_graph(G0gt).to_undirected()        
    # make sure all edges have a geometry assigned to them
    G2gt_init1 = create_edge_linestrings(G2gt_init)
    t2 = time.time()
    if verbose:
        print "Time to project, simplify, and create linestrings:", t2 - t1, "seconds"

    # clean up connected components
    G2gt_init2 = clean_sub_graphs(G2gt_init1.copy(), min_length=min_subgraph_length, 
                                  weight=weight, verbose=verbose,
                                  super_verbose=False)

    # add pixel coords
    G_gt, _, gt_graph_coords = set_pix_coords(G2gt_init2.copy(), 
                                                   im_test_file)
    t3 = time.time()
    if verbose:
        print "Time to set pixel coords:", t3 - t2, "seconds"
    
    # create graph with midpoints
    G_gt0, xms, yms = create_graph_midpoints(G_gt.copy(), 
                                                linestring_delta=linestring_delta, 
                                                figsize=(0,0), 
                                                is_curved_eps=is_curved_eps,
                                                verbose=False)
    midpoint_coords = (xms, yms)

    t4 = time.time()
    if verbose:
        print "Time to create graph midpoints:", t4 - t3, "seconds"
    
    # update with pixel coords
    G_gt_cp, control_points, gt_graph_coords = set_pix_coords(G_gt0.copy(), 
                                                           im_test_file)
    t5 = time.time()
    if verbose:
        print "Time to set pixel coords:", t5 - t4, "seconds"

    if len(G_gt_cp.nodes()) == 0:
        print "Proposal graph empty, skipping:",
        return G_gt, G_gt_cp, [], []


    t6 = time.time()
    if verbose:
        print "Time to set make plots:", t6 - t5, "seconds"
    
        
    return G_gt, G_gt_cp, control_points, gt_graph_coords, midpoint_coords


###############################################################################
def make_graphs(G_gt, G_p, 
                  weight='length', linestring_delta=35, 
                  is_curved_eps=0.012, max_snap_dist=8,
                  verbose=False):
    '''Make networkx graphs with midpoints'''
    
    t0 = time.time()

    # create graph with midpoints
    G_gt0 = create_edge_linestrings(G_gt.to_undirected())
    G_gt_cp, xms, yms = create_graph_midpoints(G_gt0.copy(), 
                                                linestring_delta=linestring_delta, 
                                                figsize=(0,0), 
                                                is_curved_eps=is_curved_eps,
                                                verbose=False)    
    # get control points
    control_points_gt = []
    for n in G_gt_cp.nodes():
        u_x, u_y = G_gt_cp.node[n]['x'], G_gt_cp.node[n]['y']
        control_points_gt.append([n, u_x, u_y])

    # get ground truth paths
    all_pairs_lengths_gt_native = nx.all_pairs_dijkstra_path_length(G_gt_cp, weight=weight)
    ###############

    # get proposal graph with native midpoints
    G_p = create_edge_linestrings(G_p.copy())    
    G_p_cp, xms_p, yms_p = create_graph_midpoints(G_p.copy(), 
                                                linestring_delta=linestring_delta, 
                                                figsize=(0,0), 
                                                is_curved_eps=is_curved_eps,
                                                verbose=verbose)
    # get control points
    control_points_p = []
    for n in G_p_cp.nodes():
        u_x, u_y = G_p_cp.node[n]['x'], G_p_cp.node[n]['y']
        control_points_p.append([n, u_x, u_y])

    # get paths
    all_pairs_lengths_prop_native = nx.all_pairs_dijkstra_path_length(G_p_cp, weight=weight)  


    # now insert control points
    if verbose:
        print "Inserting control points into G_gt..."
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime, xn_gt, yn_gt = insert_control_points(G_gt, control_points_p, 
                                        max_distance_meters=max_snap_dist,
                                        allow_renaming=True,
                                        verbose=verbose)

    if verbose:
        print "Inserting control points into G_p..."
        print "G_p.nodes():", G_p.nodes()
    G_p_cp_prime, xn_p, yn_p = insert_control_points(G_p, control_points_gt, 
                                        max_distance_meters=max_snap_dist,
                                        allow_renaming=True,
                                        verbose=verbose)

    # get paths
    all_pairs_lengths_gt_prime = nx.all_pairs_dijkstra_path_length(G_gt_cp_prime, weight=weight)
    all_pairs_lengths_prop_prime = nx.all_pairs_dijkstra_path_length(G_p_cp_prime, weight=weight)            

    tf = time.time()
    print "Time to run make_graphs in apls.py:", tf - t0, "seconds"
    
    return  G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime   


###############################################################################
def single_path_metric(len_gt, len_prop, diff_max=1):
    '''compute normalize path difference metric
    if len_prop < 0, return diff_max'''
    
    if len_gt <= 0:
        return 0
    elif len_prop < 0 and len_gt > 0:
        return diff_max    
    else:
        diff_raw = np.abs(len_gt - len_prop) / len_gt
        return np.min([diff_max, diff_raw])
        

###############################################################################
def path_sim_metric(all_pairs_lengths_gt, all_pairs_lengths_prop, 
                    control_nodes=[], min_path_length=10,
                    diff_max=1, missing_path_len=-1, verbose=False, 
                    normalize=True):
    '''compute metric, assume nodes in ground truth and proposed graph have
    the same names
    assume graph is undirected so don't evaluate routes in both directions
    control_nodes is the list of nodes to actually evaluate; if empty do all
        in all_pairs_lenghts_gt
    min_path_length is the minimum path length to evaluate'''
    diffs = []
    routes = []
    diff_dic = {}
    prop_start_nodes_set = set(all_pairs_lengths_prop.keys())
    t0 = time.time()
    
    # set nodes to inspect
    if len(control_nodes) == 0:
        good_nodes = set(all_pairs_lengths_gt.keys())
    else:
        good_nodes = set(control_nodes)
    
    # iterate overall start nodes
    #for start_node, paths in all_pairs_lengths.iteritems():
    for start_node in good_nodes:
        if verbose:
            print "start node:", start_node
        node_dic_tmp = {}
        
        paths = all_pairs_lengths_gt[start_node]

        # CASE 1
        # if the start node is missing from proposal, use maximum diff for 
        # all possible routes to the start node
        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.iteritems():
                diff = 0.25#diff_max
                diff = diff_max
                diffs.append(diff)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff
            diff_dic[start_node] = node_dic_tmp
            continue
            
        # else get proposed paths
        else:  
            paths_prop = all_pairs_lengths_prop[start_node]
            
            # get set of all nodes in paths_prop, and missing_nodes
            end_nodes_prop_set = set(paths_prop.keys())
            end_nodes_gt_set = set(paths.keys())
            missing_nodes = end_nodes_gt_set - end_nodes_prop_set
            if verbose:
                print "missing nodes:", missing_nodes
            
            # iterate over all paths from node
            #for end_node, len_gt in paths.iteritems():
            #for end_node in good_nodes - set([start_node]):
            for end_node in good_nodes:
                
                # skip self 
                if end_node == start_node:
                    continue
                
                # check if end_node not in paths.  If not and end_node is in 
                # paths_prop, assign diff_max
                elif (end_node not in end_nodes_gt_set):
                    
                    # CASE 2
                    # check if a path exists between start and end node in 
                    # the proposal graph but not the ground truth.
                    # if so, penalize maximally
                    if end_node in end_nodes_prop_set:
                        if paths_prop[end_node] >= min_path_length:
                            diff = 0.5#diff_max
                            diff = diff_max
                            diffs.append(diff)
                            routes.append([start_node, end_node])
                            node_dic_tmp[end_node] = diff
                        # if end_node not in either paths list, ignore it  
                        else:
                            continue
                    else:
                        continue
                    
                # if end_node in paths, get the path length difference 
                else:
                    len_gt = paths[end_node]
                    
                    # skip if too short
                    if len_gt < min_path_length:
                        continue
                    
                    # get proposed path
                    
                    # CASE 3, end_node in both paths and paths_prop, so  
                    # valid path exists
                    if end_node in end_nodes_prop_set:
                        len_prop = paths_prop[end_node]
                        # compute path difference metric
                        diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
                        diffs.append(diff)
                        routes.append([start_node, end_node])
                        node_dic_tmp[end_node] = diff
                        
                    # CASE 4: end_node in paths but not paths_prop, so assign
                    # length as diff_max 
                    else:
                        len_prop = missing_path_len
                        diff = 0.75#diff_max
                        diff = diff_max
                        diffs.append(diff)
                        routes.append([start_node, end_node])
                        node_dic_tmp[end_node] = diff
                        #print "start_node:", start_node
                        #print "end_node:", end_node
                    
                    # compute path difference metric (done slready in 
                    # case 3,4 above)
                    #diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
                    #diffs.append(diff)
                    #routes.append([start_node, end_node])
                    #node_dic_tmp[end_node] = diff
                    
                    if verbose:
                        print "start node:", start_node
                        print "  end_node:", end_node
                        print "   len_gt:", len_gt
                        print "   len_prop:", len_prop 
                        
            diff_dic[start_node] = node_dic_tmp
        
    print "Time to compute metric for ", len(diffs), "routes:", \
                    time.time() - t0, "seconds"
                    
    # compute Cost
    diff_tot = np.sum(diffs)
    if normalize:
        norm = len(diffs) 
        diff_norm = diff_tot / norm
        C = 1. - diff_norm
    else:
        C = diff_tot

    return C, diffs, routes, diff_dic


###############################################################################
def path_sim_metric_v0(all_pairs_lengths_gt, all_pairs_lengths_prop, 
                    control_nodes=[], min_path_length=10,
                    diff_max=1, 
                    missing_path_len=-1, verbose=False, normalize=True):
    '''compute metric, assume nodes in ground truth and proposed graph have
    the same names
    assume graph is undirected so don't evaluate routes in both directions
    control_nodes is the list of nodes to actually evaluate; if empty do all
        in all_pairs_lenghts_gt
    min_path_length is the minimum path length to evaluate'''
    diffs = []
    routes = []
    diff_dic = {}
    gt_start_nodes_set = set(all_pairs_lengths_gt.keys())
    prop_start_nodes_set = set(all_pairs_lengths_prop.keys())
    t0 = time.time()
    
    # set nodes to inspect
    if len(control_nodes) == 0:
        good_nodes = all_pairs_lengths_gt.keys()
    else:
        good_nodes = control_nodes
    
    # iterate overall start nodes
    #for start_node, paths in all_pairs_lengths.iteritems():
    for start_node in good_nodes:
        if verbose:
            print "start node:", start_node
        node_dic_tmp = {}
        
        # if we are not careful with control nodes, it's possible that the start_node
        # will not be in all_pairs_lengths_gt, in this case use max diff for 
        # all routes to that node 
        # if the start node is missing from proposal, use maximum diff for 
        # all possible routes to that node
        if start_node not in gt_start_nodes_set:
            print "for ss, node", start_node, "not in set"
            print "   skipping N paths:", len(all_pairs_lengths_prop[start_node].keys())
            for end_node, len_prop in all_pairs_lengths_prop[start_node].iteritems():
                diffs.append(diff_max)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff_max
            continue

        paths = all_pairs_lengths_gt[start_node]

        # if the start node is missing from proposal, use maximum diff for 
        # all possible routes to that node
        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.iteritems():
                diffs.append(diff_max)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff_max
            diff_dic[start_node] = node_dic_tmp
            continue
            
        # else get proposed paths
        else:  
            paths_prop = all_pairs_lengths_prop[start_node]
            
            # get set of all nodes in paths_prop, and missing_nodes
            end_nodes_prop_set = set(paths_prop.keys())
            end_nodes_gt_set = set(paths.keys())
            missing_nodes = end_nodes_gt_set - end_nodes_prop_set
            if verbose:
                print "missing nodes:", missing_nodes
            
            # iterate over all paths from node
            for end_node, len_gt in paths.iteritems():
                # sip if too short
                if len_gt < min_path_length:
                    continue
                # get proposed path
                if end_node in end_nodes_prop_set:
                    len_prop = paths_prop[end_node]
                else:
                    len_prop = missing_path_len
                
                if verbose:
                    print "end_node:", end_node
                    print "   len_gt:", len_gt
                    print "   len_prop:", len_prop
                
                # compute path difference metric
                diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
                diffs.append(diff)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff
            
            diff_dic[start_node] = node_dic_tmp
        
    print "Time to compute metric for ", len(diffs), "routes:", \
                    time.time() - t0, "seconds"
                    
    # compute Cost
    diff_tot = np.sum(diffs)
    if normalize:
        norm = len(diffs) 
        diff_norm = diff_tot / norm
        C = 1. - diff_norm
    else:
        C = diff_tot

    return C, diffs, routes, diff_dic

###############################################################################
def get_all_missing_paths(paths_gt, paths_prop):
    '''Function to find all paths that are in paths_gt and not paths_prop, and
    visa-versa.  This function could be called at the end of path_sim_metric
    if we want to clean up path_sim_metric()'''
    pass

###############################################################################
def compute_metric(all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, 
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
            res_dir='', min_path_length=10, 
            verbose=False):
    '''Compute metric and plot results'''
    
    t0 = time.time()
    ####################
    # compute metric (gt to prop)
    control_nodes = all_pairs_lengths_gt_native.keys()
    C00, diffs, routes, diff_dic = path_sim_metric(all_pairs_lengths_gt_native, 
                          all_pairs_lengths_prop_prime, 
                          control_nodes=control_nodes,
                          min_path_length=min_path_length,
                          diff_max=1, missing_path_len=-1, normalize=True,
                          verbose=verbose)
    dt1 = time.time() - t0
    if len(res_dir) > 0:
        scatter_png = os.path.join(res_dir, 'all_pairs_paths_diffs_gt_to_prop.png')
        hist_png =  os.path.join(res_dir, 'all_pairs_paths_diffs_hist_gt_to_prop.png')
        plot_metric(C00, diffs, figsize=(10,5), scatter_alpha=0.3,
                scatter_png=scatter_png, 
                hist_png=hist_png)
    ###################### 
     
    ####################
    # compute metric (prop to gt)
    t1 = time.time()
    control_nodes = all_pairs_lengths_prop_native.keys()
    C10, diffs, routes, diff_dic = path_sim_metric(all_pairs_lengths_prop_native, 
                          all_pairs_lengths_gt_prime, 
                          control_nodes=control_nodes,
                          min_path_length=min_path_length,
                          diff_max=1, missing_path_len=-1, normalize=True,
                          verbose=verbose)
    dt2 = time.time() - t1

    if len(res_dir) > 0:
        scatter_png = os.path.join(res_dir, 'all_pairs_paths_diffs_prop_to_gt.png')
        hist_png =  os.path.join(res_dir, 'all_pairs_paths_diffs_hist_prop_to_gt.png')
        plot_metric(C10, diffs, figsize=(10,5), scatter_alpha=0.3,
                scatter_png=scatter_png, 
                hist_png=hist_png)
        
    ####################

    ####################
    # Total
    
    print "C, C1:", C00, C10
    if (C00 <= 0) or (C10 <= 0) or (np.isnan(C00)) or (np.isnan(C10)):
        C_tot = 0
    else:
        C_tot = scipy.stats.hmean([C00, C10])
        if np.isnan(C_tot):
            C_tot = 0
    print "Total APLS Metric = Mean(", np.round(C00,2), "+", np.round(C10, 2), \
                ") =", np.round(C_tot, 2)
    print "Total time to compute metric:", str(dt1 + dt2), "seconds"

    return C_tot

###############################################################################
def plot_metric(C, diffs, figsize=(10,5), scatter_png='', hist_png='',
                scatter_alpha=0.3, scatter_cmap='jet', dpi=300):
    ''' Plot outpute of cost metric in both scatterplot and histogram format'''
    
    # plot diffs
    title = 'Path Length Similarity: ' + str(np.round(C,2)) 
    fig, (ax0) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))
    #ax0.plot(diffs)
    ax0.scatter(range(len(diffs)), diffs, s=2, c=diffs, alpha=scatter_alpha, 
                cmap=scatter_cmap)
    ax0.set_ylabel('Length Diff (Normalized)')
    ax0.set_xlabel('Path ID')
    ax0.set_title(title)
    #plt.tight_layout()
    if scatter_png:
        plt.savefig(scatter_png, dpi=dpi)
    
    # plot and plot diffs histo
    bins = np.linspace(0, 1, 30)
    bin_centers = np.mean( zip(bins, bins[1:]), axis=1)
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
def main():
    '''Explore'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_snap_dist', default=5, type=int,
        help='Buffer distance (meters) around graph')
    parser.add_argument('--linestring_delta', default=50, type=int,
        help='Distance between midpoints on edges')
    parser.add_argument('--min_path_length', default=1.0, type=float,
        help='Minimum path length to consider for metric')
    parser.add_argument('--is_curved_eps', default=10**3, type=int,
        help='Line curvature above which midpoints will be injected, (< 0 to inject midpoints on straight lines)')
    parser.add_argument('--use_geojson', default=True, type=bool, help='Use GeoJSON')
    args = parser.parse_args()

    # set proposal and ground truth files

    ###################
    if args.use_geojson:
        # use geojson and pkl files
        # This example is from the Paris AOI, image 1447
        # set graph_files to '' to download aa graph via osmnx and explore
        gt_file = os.path.join(path_apls, 'sample_data/OSMroads_img1447.geojson')
        # the proposal file can be created be exporting a networkx graph via:
        #        nx.write_gpickle(proposal_graph, outfile_pkl)
        prop_file = os.path.join(path_apls, 'sample_data/proposal_graph_1447.pkl')
        im_file = ''#os.path.join(path_apls, 'sample_data/RGB-PanSharpen_img1447.tif')
        outroot = 'RGB-PanSharpen_img1447'
    else:
        gt_file = ''
        outroot = 'seville'
    outdir_base = os.path.join(path_apls, 'example_output_ims/')
    outdir = os.path.join(outdir_base, outroot)

    ###################
    
    ###################
    # plotting and exploring settings
    verbose = False
    title_fontsize=8
    dpi=300
    show_plots = False
    #fig_height, fig_width = 6, 6
    # path settings
    route_linewidth=4
    weight = 'length'
    source_color = 'red'
    target_color = 'green'
    frac_edge_delete = 0.15
    # if using create_gt_graph, use the following road types
    #https://wiki.openstreetmap.org/wiki/Key:highway
    valid_road_types = set(['motorway', 'trunk', 'primary', 'secondary', 
                               'tertiary', 
                            'motorway_link', 'trunk_link', 'primary_link',
                               'secondary_link', 'tertiary_link',
                            'unclassified', 'residential', 'service'])
    ###################
        
    # make dirs
    d_list = [outdir_base, outdir]
    for p in d_list:
        if not os.path.exists(p):
            os.mkdir(p)

    # use create_gt_graph and inject a geojson
    if len(gt_file) > 0 and len(prop_file) > 0:# and len(im_file) > 0:
        # ground truth
        G_gt_init, G_gt_cp, control_points, gt_graph_coords, midpoints = \
            create_gt_graph(gt_file, im_file, network_type='all_private',
                 linestring_delta=args.linestring_delta, 
                 is_curved_eps=args.is_curved_eps, 
                 valid_road_types=valid_road_types,
                 weight=weight,
                 verbose=verbose)
        # proposal
        G_p_init = nx.read_gpickle(prop_file)

        
    else:
        # For this example, import a random city graph
        # medium
        #G0 = ox.graph_from_bbox(37.79, 37.78, -122.41, -122.43, network_type='drive', simplify=True, retain_all=False)
        # large
        #G0 = ox.graph_from_bbox(37.79, 37.77, -122.41, -122.43, network_type='drive', simplify=True, retain_all=False)
        # very small graph for plotting
        G0 = ox.graph_from_bbox(37.777, 37.77, -122.41, -122.417, network_type='drive')
 
        # huge 
        #ox.graph_from_place('Stresa, Italy')
        #G0 = ox.graph_from_place('Seville, Spain', simplify=True, retain_all=False)

        G_gt_init0 = ox.project_graph(G0)
        G_gt_init = create_edge_linestrings(G_gt_init0.to_undirected())
        print "Num G_gt_init.nodes():", len(G_gt_init.nodes())
        print "Num G_gt_init.edges():", len(G_gt_init.edges())
        
        ############
        # Proposal graph (take this as the ground truth graph with midpoints, then
        # remove some edges)
        G_p_init, _, _ = create_graph_midpoints(G_gt_init.copy(), 
                                                linestring_delta=args.linestring_delta, 
                                                is_curved_eps=args.is_curved_eps,
                                                verbose=verbose)
        
        # randomly remove edges
        n_edge_delete = int(frac_edge_delete * len(G_p_init.edges()))
        idxs_delete = np.random.choice(range(len(G_p_init.edges())), n_edge_delete)
        # for reproducibility, assign idxs_delete
        if len(gt_file) == 0:
            idxs_delete = [16, 45, 11,  9, 14, 53, 24, 11]
        print "idxs_delete:", idxs_delete
        ebunch = [G_p_init.edges()[idx_tmp] for idx_tmp in idxs_delete]
        # remove ebunch
        G_p_init.remove_edges_from(ebunch)
        print "New num Gp.edges():", len(G_p_init.edges())

    ### PLOTS
    # plot ground truth
    fig, ax = ox.plot_graph(G_gt_init, show=show_plots, close=False)
    ax.set_title('Ground Truth Graph', fontsize=title_fontsize)
    #plt.show()
    plt.savefig(os.path.join(outdir, 'gt_graph.png'), dpi=dpi)
    #plt.clf()
    #plt.cla()
    plt.close('all')

    # plot proposal
    fig, ax = ox.plot_graph(G_p_init, show=show_plots, close=False)
    ax.set_title('Proposal Graph', fontsize=title_fontsize)
    #plt.show()
    plt.savefig(os.path.join(outdir, 'prop_graph.png'), dpi=dpi)

    # get graphs with midpoints and paths
    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime  = \
        make_graphs(G_gt_init, G_p_init, 
                  weight='length', linestring_delta=args.linestring_delta, 
                  is_curved_eps=args.is_curved_eps,  max_snap_dist=args.max_snap_dist,
                  verbose=verbose)

    # midpoints
    fig0, ax0 = ox.plot_graph(G_gt_cp, show=show_plots, close=False)
    ax0.set_title('Ground Truth With Midpionts', fontsize=title_fontsize)
    #plt.show()
    plt.savefig(os.path.join(outdir, 'gt_graph_midpoints.png'), dpi=dpi)
    plt.close()



    ###################################
    # plot some paths...
    # get source and target nodes
    
    ## use idxs?
    #source_idx = np.random.randint(0,len(G_gt_cp.nodes()))
    #target_idx = np.random.randint(0,len(G_gt_cp.nodes()))
    ## specify source and target node, if desired
    ##if len(gt_file) == 0:
    ##    source_idx =  27 
    ##    target_idx =  36
    #print "source_idx:", source_idx
    #print "target_idx:", target_idx
    #source = G_gt_cp.nodes()[source_idx]
    #target = G_gt_cp.nodes()[target_idx]
    
    # get a random source and target that are in both ground truth and prop
    possible_sources = set(G_gt_cp.nodes()).intersection(set(G_p_cp_prime.nodes()))
    possible_targets = set(G_gt_cp.nodes()).intersection(set(G_p_cp_prime.nodes()))
    source = random.choice(list(possible_sources))
    target = random.choice(list(possible_targets))
    print "source, target:", source, target
    
    
    # compute paths to node of interest, and plot
    t0 = time.time()
    lengths, paths = nx.single_source_dijkstra(G_gt_cp, source=source, weight=weight) 
    print "Time to calculate:", len(lengths), "paths:", time.time() - t0, "seconds"
    
    # plot a single route
    fig, ax = ox.plot_graph_route(G_gt_cp, paths[target], 
                                        route_color='yellow',
                                        route_alpha=0.8,
                                        orig_dest_node_alpha=0.3,
                                        orig_dest_node_size=120,
                                        route_linewidth=route_linewidth,
                                        orig_dest_node_color=target_color,
                                        show=show_plots,
                                        close=False)
    plen = np.round(lengths[target],2)
    title= "Ground Truth Graph, L = " + str(plen)
    source_x = G_gt_cp.node[source]['x']
    source_y = G_gt_cp.node[source]['y']
    ax.scatter(source_x, source_y, color=source_color, s=75)
    t_x = G_gt_cp.node[target]['x']
    t_y = G_gt_cp.node[target]['y']
    ax.scatter(t_x, t_y, color=target_color, s=75)
    ax.set_title(title, fontsize=title_fontsize)
    #plt.show()
    plt.savefig(os.path.join(outdir, 'single_source_route_ground_truth.png'), dpi=dpi)
    plt.close()

    
      
    # get all paths from source for proposal graph
    lengths_prop, paths_prop = nx.single_source_dijkstra(G_p_cp_prime, source=source, weight=weight) 
    gt_set = set(lengths.keys())
    prop_set = set(lengths_prop.keys())
    missing_nodes = gt_set - prop_set
    print "missing nodes:", missing_nodes
    
    ##############
    # compute path to node of interest
    t0 = time.time()
    lengths_ptmp, paths_ptmp = nx.single_source_dijkstra(G_p_cp_prime, source=source, weight=weight) 
    print "Time to calculate:", len(lengths), "paths:", time.time() - t0, "seconds"
    
    # plot a single route
    fig, ax = ox.plot_graph_route(G_p_cp_prime, paths_ptmp[target], 
                                        route_color='yellow',
                                        route_alpha=0.8,
                                        orig_dest_node_alpha=0.3,
                                        orig_dest_node_size=120,
                                        route_linewidth=route_linewidth,
                                        orig_dest_node_color=target_color,
                                        show=show_plots,
                                        close=False)
    #title= "Source-" + source_color + " " + str(source) + " Target: " + str(target)
    plen = np.round(lengths_ptmp[target],2)
    title= "Proposal Graph, L = " + str(plen)
    source_x = G_p_cp_prime.node[source]['x']
    source_y = G_p_cp_prime.node[source]['y']
    ax.scatter(source_x, source_y, color=source_color, s=75)
    t_x = G_p_cp_prime.node[target]['x']
    t_y = G_p_cp_prime.node[target]['y']
    ax.scatter(t_x, t_y, color=target_color, s=75)
    ax.set_title(title, fontsize=title_fontsize)
    #plt.show()
    plt.savefig(os.path.join(outdir, 'single_source_route_prop.png'), dpi=dpi)
    plt.close()

    ##############

    #########################
    ### Metric
    C = compute_metric(all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, 
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
            min_path_length=args.min_path_length, 
            verbose=verbose,
            res_dir=outdir)

    print "APLS Metric = ", C
    
    return C


###############################################################################
if __name__ == "__main__":
    main()
