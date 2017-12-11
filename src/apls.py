#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:32:19 2017

@author: avanetten

"""

import networkx as nx
import osmnx as ox   # https://github.com/gboeing/osmnx
import scipy.spatial
import scipy.stats
import numpy as np
import random
import utm           # pip install utm
import copy
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import time
import sys
import os
import argparse

# get current path
path_apls_src = os.path.dirname(os.path.realpath(__file__))
path_apls = os.path.dirname(path_apls_src)
# add path and import graphTools
sys.path.extend([path_apls_src])
import graphTools, apls_tools
    

# get current path
path_apls_src = os.path.dirname(os.path.realpath(__file__))
# add path and import graphTools
sys.path.extend([path_apls_src])
import graphTools
reload(graphTools)

path_apls = os.path.dirname(path_apls_src)
print "path_apls:", path_apls


###############################################################################
def create_edge_linestrings(G, remove_redundant=True, verbose=False):
    '''Ensure all edges have 'geometry' tag, use shapely linestrings
    If identical edges exist, remove extras'''

    # clean out redundant edges with identical geometry
    edge_seen_set = set([])
    geom_seen = []
    bad_edges = []
    
    G_ = G.copy()
    for i,(u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        # create linestring if no geometry reported
        if 'geometry' not in data:
            sourcex, sourcey = G_.node[u]['x'],  G_.node[u]['y']
            targetx, targety = G_.node[v]['x'],  G_.node[v]['y']
            line_geom = LineString([Point(sourcex, sourcey), 
                                  Point(targetx, targety)])
            data['geometry'] = line_geom

            # get reversed line
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
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
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            if dist_to_u > dist_to_v:
                #data['geometry'].coords = list(line_geom.coords)[::-1]
                data['geometry'] = line_geom_rev
            #else:
            #    continue
            
        # flag redundant edges
        if remove_redundant:
            if i == 0:
                edge_seen_set = set([(u,v)])
                edge_seen_set.add((v,u))
                geom_seen.append(line_geom)
                
            else:
                if ((u,v) in edge_seen_set) or ((v,u) in edge_seen_set):
                    # test if geoms have already been seen
                    for geom_seen_tmp in geom_seen:
                        if (line_geom == geom_seen_tmp) \
                                or (line_geom_rev == geom_seen_tmp):
                            bad_edges.append((u, v, key))
                            if verbose:
                                print "\nRedundant edge:", u, v, key
                else:
                    edge_seen_set.add((u,v))  
                    geom_seen.append(line_geom)
                    geom_seen.append(line_geom_rev)
    
    if remove_redundant:
        if verbose:
            print "\nedge_seen_set:", edge_seen_set
            print "redundant edges:", bad_edges
        for (u,v,key) in bad_edges:
            try:
                G_.remove_edge(u, v, key)
            except:
                if verbose:
                    print "Edge DNE:", u,v,key
                pass
        
    return G_

###############################################################################
def cut_linestring(line, distance, verbose=False):
    ''' 
    Cuts a line in two at a distance from its starting point
    http://toblerity.org/shapely/manual.html#linear-referencing-methods
    '''
    if verbose:
        print "Cutting linestring at distance", distance, "..."
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
        
    # iterate through coorda and check if interpolated point has been passed
    # already or not
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if verbose:
            print i, p, "pdl:", pdl
        if pdl == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]
            
    # if we've reached here then that means we've encountered a self-loop and 
    # the interpolated point is between the final midpoint and the the original
    # node
    i = len(coords) - 1
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
        edge_list.append([u, v, key])
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
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())
 
    if verbose:
        print "Inserting point:", node_id
        print "best edge:", best_edge
        u_loc = [G_.node[u]['x'], G_.node[u]['y']]
        v_loc = [G_.node[v]['x'], G_.node[v]['y']]
        print "ploc:", (point.x, point.y)
        print "uloc:", u_loc
        print "vloc:", v_loc
    
    if min_dist > max_distance_meters:
        if verbose:
            print "min_dist > max_distance_meters, skipping..."
        return G_, {}, -1, -1
    
    else:
        # update graph
        
        # skip if node exists already
        if node_id in G_node_set:
            if verbose:
                print "Node ID:", node_id, "already exists, skipping..."
            return G_, {}, -1, -1

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
            print "Failure in cut_linestring()..."
            print "type(split_line):", type(split_line)
            print "split_line:", split_line
            print "line_geom:", line_geom
            print "line_geom.length:", line_geom.length
            print "line_proj:", line_proj
            print "min_dist:", min_dist
            return #G_, {}, 0, 0

        if verbose:
            print "split_line:", split_line
        
        #if cp.is_empty:        
        if len(split_line) == 1:
            if verbose:
                print "split line empty, min_dist:", min_dist
            # get coincident node
            outnode = ''
            outnode_x, outnode_y = -1, -1
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.node[u]['x'], G_.node[u]['y']
            x_v, y_v = G_.node[v]['x'], G_.node[v]['y']
            if (x_p == x_u) and (y_p == y_u):
                outnode = u
                outnode_x, outnode_y = x_u, y_u
            elif (x_p == x_v) and (y_p == y_v):
                outnode = v
                outnode_x, outnode_y = x_v, y_v
            else:
                print "Error in determining node coincident with node: " \
                + str(node_id) + " along edge: " + str(best_edge)
                print "x_p, y_p:", x_p, y_p
                print "x_u, y_u:", x_u, y_u
                print "x_v, y_v:", x_v, y_v
                return

            # if the line cannot be split, that means that the new node 
            # is coincident with an existing node.  Relabel, if desired
            if allow_renaming:
                node_props = G_.node[outnode]
                # A dictionary with the old labels as keys and new labels as values. A partial mapping is allowed.
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                if verbose:
                    print "Swapping out node ids:", mapping
                return Gout, node_props, x_p, y_p
            
            else:
                # new node is already added, presumably at the exact location
                # of an existing node.  So just remove the best edge and make
                # an edge from new node to existing node, length should be 0.0

                line1 = LineString([new_point, Point(outnode_x, outnode_y)])
                edge_props_line1 = edge_props_new.copy()         
                edge_props_line1['length'] = line1.length
                edge_props_line1['geometry'] = line1
                # make sure length is zero
                if line1.length > 0:
                    print "Nodes should be coincident and length 0!"
                    return
                
                # add edge of length 0 from new node to neareest existing node
                G_.add_edge(node_id, outnode, attr_dict=edge_props_line1)

#                # Another option that isn't fully functional: inject the point,
#                # and insert edges between [u, node_id and [node_id, v]
#                
#                # get distances
#                u_loc = [G_.node[u]['x'], G_.node[u]['y']]
#                v_loc = [G_.node[v]['x'], G_.node[v]['y']]
#                # compare to first point in linestring
#                geom_p0 = list(line_geom.coords)[0]
#                # or compare to inserted point? [this might fail if line is very
#                #    curved!]
#                #geom_p0 = (x,y)
#                dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
#                dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
#
#                line1 = LineString([new_point, Point(outnode_x, outnode_y)])
#                edge_props_line1 = edge_props_new.copy()         
#                edge_props_line1['length'] = line1.length
#                edge_props_line1['geometry'] = line1
#                # make sure length is zero
#                if line1.length > 0:
#                    print "Nodes should be coincident and length 0!"
#                    return
#                # add edges
#                edge_props_line2 = edge_props_new.copy()
#                if dist_to_u < dist_to_v:
#                    G_.add_edge(u, node_id, attr_dict=edge_props_line1)
#                    G_.add_edge(node_id, v, attr_dict=edge_props_line2)
#                else:
#                    G_.add_edge(node_id, u, attr_dict=edge_props_line1)
#                    G_.add_edge(v, node_id, attr_dict=edge_props_line2)
#                #G_.add_edge(u, node_id, attr_dict=edge_props_new.copy())
#                #G_.add_edge(node_id, v, attr_dict=edge_props_line2)
#                if verbose:
#                    print "insert edges:", u, '-',node_id, 'and', node_id, '-', v  
#                    print "G_.edge[u][node_id]:", G_.edge[u][node_id]
#                    print u, node_id, "length:", G_.edge[u][node_id][0]['geometry'].length
#                    print node_id, v, "length:", G_.edge[node_id][v][0]['geometry'].length
#
#                # remove initial edge
#                G_.remove_edge(u, v, key)
             
                return G_, node_props, x, y


                ## originally, if not renaming nodes, 
                ## just ignore this complication and return the orignal
                #return G_, node_props, 0, 0

        
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
                print "   u, dist_u_to_point:", u, dist_to_u
                print "   v, dist_v_to_point:", v, dist_to_v
                print "   min_dist:", min_dist

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
            #if verbose:
            #    print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, attr_dict=edge_props_line1)
                G_.add_edge(node_id, v, attr_dict=edge_props_line2)
            else:
                G_.add_edge(node_id, u, attr_dict=edge_props_line1)
                G_.add_edge(v, node_id, attr_dict=edge_props_line2)

            if verbose:
                print "insert edges:", u, '-',node_id, 'and', node_id, '-', v
                         
            
            # remove initial edge
            G_.remove_edge(u, v, key)
            
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
            print "\n", i, "Insert control point:", node_id, "x =", x, "y =", y 
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
            if linelen <= linestring_delta:
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
    '''Get pixel coords.  Update G_ and get control_points, and graph_coords'''
    
    if len(G_.nodes()) == 0:
        return G_, [], []
    
    control_points, cp_x, cp_y = [], [], []
    for n in G_.nodes():
        u_x, u_y = G_.node[n]['x'], G_.node[n]['y']
        control_points.append([n, u_x, u_y])
        lat, lon = G_.node[n]['lat'], G_.node[n]['lon']
        if len(im_test_file) > 0:
            pix_x, pix_y = apls_tools.latlon2pixel(lat, lon, 
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
        
    if len(G0gt_init.nodes()) == 0:
        return G0gt_init, G0gt_init#, [], [], [], []

    G0gt = ox.project_graph(G0gt_init) 
    if verbose:
        print "G0gt.nodes():", G0gt.nodes()
        print "G0gt.edges:", G0gt.edges()
    
    #G1gt = ox.simplify_graph(G0gt)    
    #G2gt_init = G1gt.to_undirected()
    
    G2gt_init = ox.simplify_graph(G0gt).to_undirected()        
    #G2gt_init = ox.simplify_graph(G0gt.to_undirected())  


    # make sure all edges have a geometry assigned to them
    G2gt_init1 = create_edge_linestrings(G2gt_init, remove_redundant=True)
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
    
#    # do this in make_graphs instead...
#    # create graph with midpoints
#    G_gt0, xms, yms = create_graph_midpoints(G_gt.copy(), 
#                                                linestring_delta=linestring_delta, 
#                                                figsize=(0,0), 
#                                                is_curved_eps=is_curved_eps,
#                                                verbose=False)
#    midpoint_coords = (xms, yms)
#
#    t4 = time.time()
#    if verbose:
#        print "Time to create graph midpoints:", t4 - t3, "seconds"
#    
#    # update with pixel coords
#    G_gt_cp, control_points, gt_graph_coords = set_pix_coords(G_gt0.copy(), 
#                                                           im_test_file)
#    t5 = time.time()
#    if verbose:
#        print "Time to set pixel coords:", t5 - t4, "seconds"
#
#    if len(G_gt_cp.nodes()) == 0:
#        print "Proposal graph empty, skipping:",
#        return G_gt, G_gt_cp, [], []
#
#
#    t6 = time.time()
#    if verbose:
#        print "Time to set make plots:", t6 - t5, "seconds"
    
        
    return G_gt, G0gt_init#, G_gt_cp, control_points, gt_graph_coords, midpoint_coords


###############################################################################
def make_graphs(G_gt, G_p, 
                  weight='length', linestring_delta=35, 
                  is_curved_eps=0.012, max_snap_dist=8,
                  allow_renaming=False,
                  verbose=False):
    '''Make networkx graphs with midpoints'''
    
    t0 = time.time()

    # create graph with midpoints
    G_gt0 = create_edge_linestrings(G_gt.to_undirected())
    if verbose:
        print "G_gt.nodes():", G_gt0.nodes()
        print "G_gt.edges():", G_gt0.edges()

    G_gt_cp, xms, yms = create_graph_midpoints(G_gt0.copy(), 
                                                linestring_delta=linestring_delta, 
                                                figsize=(0,0), 
                                                is_curved_eps=is_curved_eps,
                                                verbose=False)    
    # get ground truth control points
    control_points_gt = []
    for n in G_gt_cp.nodes():
        u_x, u_y = G_gt_cp.node[n]['x'], G_gt_cp.node[n]['y']
        control_points_gt.append([n, u_x, u_y])
    if verbose:
        print "control_points_gt:", control_points_gt

    # get ground truth paths
    all_pairs_lengths_gt_native = nx.all_pairs_dijkstra_path_length(G_gt_cp, weight=weight)
    ###############

    # get proposal graph with native midpoints
    G_p = create_edge_linestrings(G_p.to_undirected())    
    if verbose:
        print "G_p.nodes():", G_p.nodes()
        print "G_p.edges():", G_p.edges()
                
    G_p_cp, xms_p, yms_p = create_graph_midpoints(G_p.copy(), 
                                                linestring_delta=linestring_delta, 
                                                figsize=(0,0), 
                                                is_curved_eps=is_curved_eps,
                                                verbose=verbose)
    # insert gt control points into proposal
    if verbose:
        print "Inserting control points into G_p..."
        print "G_p.nodes():", G_p.nodes()
    G_p_cp_prime, xn_p, yn_p = insert_control_points(G_p, control_points_gt, 
                                        max_distance_meters=max_snap_dist,
                                        allow_renaming=allow_renaming,
                                        verbose=verbose)
    # set proposal control nodes, originally just all nodes in G_p_cp
    # original method sets proposal control points as all nodes in G_p_cp
    # get proposal control points
    control_points_prop = []
    for n in G_p_cp.nodes():
        u_x, u_y = G_p_cp.node[n]['x'], G_p_cp.node[n]['y']
        control_points_prop.append([n, u_x, u_y])

        
    G_p_cp, xn_p, yn_p = insert_control_points(G_p_cp, control_points_gt, 
                                        max_distance_meters=max_snap_dist,
                                        allow_renaming=allow_renaming,
                                        verbose=verbose)

    # get paths
    all_pairs_lengths_prop_native = nx.all_pairs_dijkstra_path_length(G_p_cp, weight=weight)  


    # now insert control points into ground truth
    if verbose:
        print "\nInserting control points into G_gt..."
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime, xn_gt, yn_gt = insert_control_points(G_gt, 
                                        control_points_prop,
                                        max_distance_meters=max_snap_dist,
                                        allow_renaming=allow_renaming,
                                        verbose=verbose)

    # get paths
    all_pairs_lengths_gt_prime = nx.all_pairs_dijkstra_path_length(G_gt_cp_prime, weight=weight)
    all_pairs_lengths_prop_prime = nx.all_pairs_dijkstra_path_length(G_p_cp_prime, weight=weight)            

    tf = time.time()
    print "Time to run make_graphs in apls.py:", tf - t0, "seconds"
    
    return  G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
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
    
    print 
    if len(gt_start_nodes_set) == 0:
        return 0, [], [], {}
    
    # set nodes to inspect
    if len(control_nodes) == 0:
        good_nodes = all_pairs_lengths_gt.keys()
    else:
        good_nodes = control_nodes
        
    if verbose:
        print "\nComputing path_sim_metric()..."
        print "good_nodes:", good_nodes
    
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
            return

        paths = all_pairs_lengths_gt[start_node]

        # CASE 1
        # if the start node is missing from proposal, use maximum diff for 
        # all possible routes to the start node
        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.iteritems():
                if (end_node != start_node) and (end_node in good_nodes):
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
                    node_dic_tmp[end_node] = diff_max
            diff_dic[start_node] = node_dic_tmp
            continue
            
        # else get proposed paths
        else:  
            paths_prop = all_pairs_lengths_prop[start_node]
            
            # get set of all nodes in paths_prop, and missing_nodes
            end_nodes_gt_set = set(paths.keys()).intersection(good_nodes)
            #end_nodes_gt_set = set(paths.keys())   # old version with all nodes
        
            end_nodes_prop_set = set(paths_prop.keys())
            missing_nodes = end_nodes_gt_set - end_nodes_prop_set
            if verbose:
                print "missing nodes:", missing_nodes
            
            # iterate over all paths from node
            for end_node in end_nodes_gt_set:
            #for end_node, len_gt in paths.iteritems():
            
                len_gt = paths[end_node]
                # skip if too short
                if len_gt < min_path_length:
                    continue

                # get proposed path
                if end_node in end_nodes_prop_set:
                    # CASE 2, end_node in both paths and paths_prop, so  
                    # valid path exists
                    len_prop = paths_prop[end_node]
                else:
                    # CASE 3: end_node in paths but not paths_prop, so assign
                    # length as diff_max 
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
               
                
    if len(diffs) == 0:
        return 0, [], [], {}

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
def compute_metric(all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, 
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
            control_points_gt, control_points_prop,
            res_dir='', min_path_length=10, 
            verbose=False):
    '''Compute metric and plot results'''
    
    t0 = time.time()
    
    # return 0 if no paths
    if (len(all_pairs_lengths_gt_native.keys()) == 0) \
                or (len(all_pairs_lengths_prop_native.keys()) == 0):
        return 0, 0, 0
    
    ####################
    # compute metric (gt to prop)
    if verbose:
        print "\nCompute metric (gt snapped onto prop)..."
    #control_nodes = all_pairs_lengths_gt_native.keys()
    control_nodes = [z[0] for z in control_points_gt]
    C_gt_onto_prop, diffs, routes, diff_dic = path_sim_metric(all_pairs_lengths_gt_native, 
                          all_pairs_lengths_prop_prime, 
                          control_nodes=control_nodes,
                          min_path_length=min_path_length,
                          diff_max=1, missing_path_len=-1, normalize=True,
                          verbose=verbose)
    dt1 = time.time() - t0
    if verbose:
        print "len(diffs):", len(diffs)
        if len(diffs) > 0:
            print "  max(diffs):", np.max(diffs)
            print "  min(diffs)", np.min(diffs)
    if len(res_dir) > 0:
        scatter_png = os.path.join(res_dir, 'all_pairs_paths_diffs_gt_to_prop.png')
        hist_png =  os.path.join(res_dir, 'all_pairs_paths_diffs_hist_gt_to_prop.png')
        # can't plot route names if there are too many...
        if len(routes) > 100:
            routes_str = []
        else:
            routes_str = [str(z[0]) + '-' + str(z[1]) for z in routes]
        apls_tools.plot_metric(C_gt_onto_prop, diffs, routes_str=routes_str,
                    figsize=(10,5), scatter_alpha=0.8, scatter_size=8,
                scatter_png=scatter_png, 
                hist_png=hist_png)
    ###################### 
     
    ####################
    # compute metric (prop to gt)
    if verbose:
        print "\nCompute metric (prop snapped onto gt)..."
    t1 = time.time()
    #control_nodes = all_pairs_lengths_prop_native.keys()
    control_nodes = [z[0] for z in control_points_prop]
    if verbose:
        print "control_nodes:", control_nodes
    C_prop_onto_gt, diffs, routes, diff_dic = path_sim_metric(all_pairs_lengths_prop_native, 
                          all_pairs_lengths_gt_prime, 
                          control_nodes=control_nodes,
                          min_path_length=min_path_length,
                          diff_max=1, missing_path_len=-1, normalize=True,
                          verbose=verbose)
    dt2 = time.time() - t1
    if verbose:
        print "len(diffs):", len(diffs)
        if len(diffs) > 0:
            print "  max(diffs):", np.max(diffs)
            print "  min(diffs)", np.min(diffs)
    if len(res_dir) > 0:
        scatter_png = os.path.join(res_dir, 'all_pairs_paths_diffs_prop_to_gt.png')
        hist_png =  os.path.join(res_dir, 'all_pairs_paths_diffs_hist_prop_to_gt.png')
        if len(routes) > 100:
            routes_str = []
        else:
            routes_str = [str(z[0]) + '-' + str(z[1]) for z in routes]
        apls_tools.plot_metric(C_prop_onto_gt, diffs, routes_str=routes_str, 
                    figsize=(10,5), scatter_alpha=0.8, scatter_size=8,
                scatter_png=scatter_png, 
                hist_png=hist_png)
        
    ####################

    ####################
    # Total
    
    print "C_gt_onto_prop, C_prop_onto_gt:", C_gt_onto_prop, C_prop_onto_gt
    if (C_gt_onto_prop <= 0) or (C_prop_onto_gt <= 0) \
                or (np.isnan(C_gt_onto_prop)) or (np.isnan(C_prop_onto_gt)):
        C_tot = 0
    else:
        C_tot = scipy.stats.hmean([C_gt_onto_prop, C_prop_onto_gt])
        if np.isnan(C_tot):
            C_tot = 0
    print "Total APLS Metric = Mean(", np.round(C_gt_onto_prop,2), "+", \
            np.round(C_prop_onto_gt, 2), \
                ") =", np.round(C_tot, 2)
    print "Total time to compute metric:", str(dt1 + dt2), "seconds"

    return C_tot, C_gt_onto_prop, C_prop_onto_gt
 
###############################################################################
###############################################################################
def main():
    '''Explore'''
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_snap_dist', default=5, type=int,
        help='Buffer distance (meters) around graph')
    parser.add_argument('--linestring_delta', default=50, type=int,
        help='Distance between midpoints on edges')
    parser.add_argument('--min_path_length', default=0.001, type=float,
        help='Minimum path length to consider for metric')
    parser.add_argument('--is_curved_eps', default=-10**3, type=int,
        help='Line curvature above which midpoints will be injected, (< 0 to inject midpoints on straight lines)')
    parser.add_argument('--test_method', default='pkl', type=str, 
            help="method for creating ground truth and propoal " \
            + "graphs.  ['test_geojson_multi', 'test_geojson',  'pkl', 'osmnx']" \
            + "'test_geojson_multi' assumes multiple geojsons in" \
            + "the given directories " \
            + "'test_geojson' assumes the user has created " \
            + "sample geojson networks" \
            + "'pkl' imports an example ground truth geojson " \
            + "and a proposal networkx pickle " \
            + "'osmnx' downloads an osm network over the " \
            + "desired region and removes a few edges for the " \
            + "proposal graph")
            
    args = parser.parse_args()
    
    ###################
    # plotting and exploring settings
    verbose = False#True
    title_fontsize=4
    dpi=200
    show_plots = False
    show_node_ids = True
    fig_height, fig_width = 6, 6
    # path settings
    route_linewidth=4
    weight = 'length'
    source_color = 'red'
    target_color = 'green'
    frac_edge_delete = 0.15
    outdir_root = args.test_method               
    # if using create_gt_graph, use the following road types
    #   (empty set uses all road types)
    #https://wiki.openstreetmap.org/wiki/Key:highway
    valid_road_types = set(['motorway', 'trunk', 'primary', 'secondary', 
                               'tertiary', 
                            'motorway_link', 'trunk_link', 'primary_link',
                               'secondary_link', 'tertiary_link',
                            'unclassified', 'residential', 'service'])
    ###################
                            

    ###################
    # Get ground truth and proposal graphs

    ###################
    # ingest multiple ground truth and propoal geojsons in a folder
    if args.test_method == 'test_geojson_multi':
        
        outdir_root = 'topcoder_training_tests'
        truth_dir = os.path.join(path_apls, 'topcoder_training_tests/to_geojson/truthGeoJson')
        prop_dir = os.path.join(path_apls, 'topcoder_training_tests/to_geojson/proposalGeoJson')
        
        gt_list, gt_raw_list, gp_list, root_list = [], [], [], []

        name_list = os.listdir(truth_dir)
        
        for f in name_list:
            # skip non-geojson files
            if not f.endswith('.geojson'):
                continue
            
            # define values
            outroot = f.split('.')[0]
            print "\n\noutroot:", outroot
            gt_file = os.path.join(truth_dir, f)
            prop_file = os.path.join(prop_dir, outroot + '.geojson')
            # Naming convention is inconsistent
            if not os.path.exists(prop_file):
                prop_file = os.path.join(prop_dir, outroot + 'prop.geojson')
            im_file = ''
            valid_road_types = set([])   # assume no road type in geojsons

            # ground truth
            osmidx, osmNodeidx = 0, 0
            #G_gt_init, G_gt_cp, control_points, gt_graph_coords, midpoints_gt = \
            G_gt_init, G_gt_raw = \
                create_gt_graph(gt_file, im_file, network_type='all_private',
                     linestring_delta=args.linestring_delta, 
                     is_curved_eps=args.is_curved_eps, 
                     valid_road_types=valid_road_types,
                     weight=weight,
                     osmidx=osmidx, osmNodeidx=osmNodeidx,
                     verbose=verbose)
            # skip empty ground truth graphs
            if len(G_gt_init.nodes()) == 0:
                continue
                
            # proposal
            osmidx, osmNodeidx = 500, 500
            G_p_init, G_p_raw = \
                create_gt_graph(prop_file, im_file, network_type='all_private',
                     linestring_delta=args.linestring_delta, 
                     is_curved_eps=args.is_curved_eps, 
                     valid_road_types=valid_road_types,
                     weight=weight,
                     osmidx=osmidx, osmNodeidx=osmNodeidx,
                     verbose=verbose)
            # append to lists
            gt_list.append(G_gt_init)
            gt_raw_list.append(G_gt_raw)
            gp_list.append(G_p_init)
            root_list.append(outroot)
        
    ###################
    # ingest ground truth and propoal geojsons created in qgis
    if args.test_method == 'test_geojson':
        outroot = 'test0'
        gt_file = os.path.join(path_apls, 'sample_data/sample_geojson/test0_gt.geojson')
        prop_file = os.path.join(path_apls, 'sample_data/sample_geojson/test0_prop.geojson')
        im_file = '' #os.path.join(path_apls, 'sample_data/sample_geojson/RGB-PanSharpen_AOI_2_Vegas_img49.tif')
        valid_road_types = set([])   # assume no road type

        # ground truth
        osmidx, osmNodeidx = 0, 0
        G_gt_init, _ = \
            create_gt_graph(gt_file, im_file, network_type='all_private',
                 linestring_delta=args.linestring_delta, 
                 is_curved_eps=args.is_curved_eps, 
                 valid_road_types=valid_road_types,
                 weight=weight,
                 osmidx=osmidx, osmNodeidx=osmNodeidx,
                 verbose=verbose)
            
        # proposal
        osmidx, osmNodeidx = 100, 100
        G_p_init, _ = \
            create_gt_graph(prop_file, im_file, network_type='all_private',
                 linestring_delta=args.linestring_delta, 
                 is_curved_eps=args.is_curved_eps, 
                 valid_road_types=valid_road_types,
                 weight=weight,
                 osmidx=osmidx, osmNodeidx=osmNodeidx,
                 verbose=verbose)
        gt_list = [G_gt_init]
        gp_list = [G_p_init]
        root_list = [outroot]
        
    
    # import a ground truth geojson and a pickled propoal graph
    elif args.test_method == 'pkl':
        # use geojson and pkl files
        # This example is from the Paris AOI, image 1447
        outroot = 'RGB-PanSharpen_img1447'
        # set graph_files to '' to download aa graph via osmnx and explore
        gt_file = os.path.join(path_apls, 'sample_data/pkl/OSMroads_img1447.geojson')
        # the proposal file can be created be exporting a networkx graph via:
        #        nx.write_gpickle(proposal_graph, outfile_pkl)
        prop_file = os.path.join(path_apls, 'sample_data/pkl/proposal_graph_1447.pkl')
        im_file = ''#os.path.join(path_apls, 'sample_data/RGB-PanSharpen_img1447.tif')
    
        # ground truth
        #G_gt_init, G_gt_cp, control_points, gt_graph_coords, midpoints = \
        G_gt_init, G_gt_raw = \
            create_gt_graph(gt_file, im_file, network_type='all_private',
                 linestring_delta=args.linestring_delta, 
                 is_curved_eps=args.is_curved_eps, 
                 valid_road_types=valid_road_types,
                 weight=weight,
                 verbose=verbose)
        # proposal
        G_p_init = nx.read_gpickle(prop_file)
        gt_list = [G_gt_init]
        gp_list = [G_p_init]
        root_list = [outroot]
        
    # downloada graph via osmnx and remove some edges for the proposal
    elif args.test_method == 'osmnx':
        outroot = 'seville'
        # For this example, import a random city graph
        # huge 
        #ox.graph_from_place('Stresa, Italy')
        # large
        #G0 = ox.graph_from_bbox(37.79, 37.77, -122.41, -122.43, network_type='drive', simplify=True, retain_all=False)
        #G0 = ox.graph_from_place('Seville, Spain', simplify=True, retain_all=False) 
        # medium
        #G0 = ox.graph_from_bbox(37.79, 37.78, -122.41, -122.43, network_type='drive', simplify=True, retain_all=False)
        # very small graph for plotting
        G0 = ox.graph_from_bbox(37.777, 37.77, -122.41, -122.417, network_type='drive')
 

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
        
        # OPTIONAL: for reproducibility, assign idxs_delete
        idxs_delete = [16, 45, 11,  9, 14, 53, 24, 11]
        
        print "idxs_delete:", idxs_delete
        ebunch = [G_p_init.edges()[idx_tmp] for idx_tmp in idxs_delete]
        # remove ebunch
        G_p_init.remove_edges_from(ebunch)
        print "New num Gp.edges():", len(G_p_init.edges())
        
        gt_list = [G_gt_init]
        gp_list = [G_p_init]
        root_list = [outroot]


    # now compute results
    print "\n\n\nCompute Results..."
        
    for i,[outroot, G_gt_init, G_p_init] in enumerate(zip(root_list, gt_list, gp_list)):
        
        print "\n\nComputing:", outroot
        t0 = time.time()
        ##################
        # make dirs
        outdir_base = os.path.join(path_apls, 'example_output_ims')
        outdir_base2 = os.path.join(outdir_base, outdir_root)
        outdir = os.path.join(outdir_base2, outroot)
        d_list = [outdir_base, outdir_base2, outdir]
        for p in d_list:
            if not os.path.exists(p):
                os.mkdir(p) 
                
        ##################
        # get graphs with midpoints and geometry
        G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
                control_points_gt, control_points_prop, \
                all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
                all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime  \
            = make_graphs(G_gt_init, G_p_init, 
                      weight=weight, linestring_delta=args.linestring_delta, 
                      is_curved_eps=args.is_curved_eps,  
                      max_snap_dist=args.max_snap_dist,
                      verbose=verbose)
        
        if verbose:
            print "control_points_gt:", control_points_gt
            print "G_gt_init.nodes():", G_gt_init.nodes()
            print "G_gt_init.edges():", G_gt_init.edges()
            for e in G_gt_init.edges():
                print "  G_gt_init edge:", e, G_gt_init.edge[e[0]][e[1]][0]['length']
            print "G_gt_cp.nodes():", G_gt_cp.nodes()
            print "G_gt_cp.edges():", G_gt_cp.edges()
            for e in G_gt_cp.edges():
                print "  G_gt_cp edge:", e, G_gt_cp.edge[e[0]][e[1]][0]['length']
            print "G_gt_cp_prime.nodes():", G_gt_cp_prime.nodes()
            print "G_gt_cp_prime.edges():", G_gt_cp_prime.edges()        
            
            
            print "control_points_prop:", control_points_prop
            print "G_p_init.nodes():", G_p_init.nodes()
            print "G_p_init.edges():", G_p_init.edges()
            print "G_p_cp.nodes():", G_p_cp.nodes()
            print "G_p_cp.edges():", G_p_cp.edges()
            print "G_p_cp_prime.nodes():", G_p_cp_prime.nodes()
            print "G_p_cp_prime.edges():", G_p_cp_prime.edges()

            
        #########################
        ### Metric
        C, C_gt_onto_prop, C_prop_onto_gt \
            = compute_metric(all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, 
                all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
                control_points_gt, control_points_prop,
                min_path_length=args.min_path_length, 
                verbose=verbose,
                res_dir=outdir)
        print "APLS Metric = ", C
        # save scores
        f = open(os.path.join(outdir, 'output.txt'), 'w')
        f.write("Ground Truth Nodes Snapped Onto Proposal Score: " + str(C_gt_onto_prop) + "\n")
        f.write("Proposal Nodes Snapped Onto Ground Truth Score: " + str(C_prop_onto_gt) + "\n")
        f.write("Total Score: " + str(C))
        f.close()
            
        t1 = time.time()
        print "Total time to create graphs and compute metric:", t1-t0, "seconds"
        
        # skip plots if no nodes
        if (len(G_gt_cp.nodes()) == 0) or (len(G_p_cp.nodes()) == 0):
            continue
        
        ##################
        ### PLOTS
         
        # set graph size
        max_extent = max(fig_height, fig_width)
        xmin, xmax, ymin, ymax, dx, dy = apls_tools.get_graph_extent(G_gt_cp)
        if dx <= dy:
            fig_height = max_extent
            fig_width = max(1, 1. * max_extent * dx / dy)
        else:
            fig_width = max_extent
            fig_height = max(1, 1. * max_extent * dy / dx) 
        if verbose:
            print "fig_width, fig_height:", fig_width, fig_height
            
        
        # plot ground truth
        fig, ax = ox.plot_graph(G_gt_init, show=show_plots, close=False,
                                fig_height=fig_height, fig_width=fig_width)
        if show_node_ids:
            ax = apls_tools.plot_node_ids(G_gt_init, ax, fontsize=4)  # node ids
        ax.set_title('Ground Truth Graph', fontsize=title_fontsize)
        #plt.show()
        plt.savefig(os.path.join(outdir, 'gt_graph.png'), dpi=dpi)
        #plt.clf()
        #plt.cla()
        plt.close('all')
    
        #  gt midpoints
        fig0, ax0 = ox.plot_graph(G_gt_cp, show=show_plots, close=False,
                                fig_height=fig_height, fig_width=fig_width)
        if show_node_ids:
            ax0 = apls_tools.plot_node_ids(G_gt_cp, ax0, fontsize=4)  # node ids
        ax0.set_title('Ground Truth With Midpionts', fontsize=title_fontsize)
        #plt.show()
        plt.savefig(os.path.join(outdir, 'gt_graph_midpoints.png'), dpi=dpi)
        plt.close('all')
    
        # plot ground truth nodes from prop
        fig, ax = ox.plot_graph(G_gt_cp_prime, show=show_plots, close=False,
                                fig_height=fig_height, fig_width=fig_width)
        if show_node_ids:
            ax = apls_tools.plot_node_ids(G_gt_cp_prime, ax, fontsize=4)  # node ids
        ax.set_title('Ground Truth Graph with Proposal Control Nodes', fontsize=title_fontsize)
        #plt.show()
        plt.savefig(os.path.join(outdir, 'gt_graph_prop_control_points.png'), dpi=dpi)
        #plt.clf()
        #plt.cla()
        plt.close('all')

        # remove geometry to test whether we correctly added midpoints and edges
        Gtmp = G_gt_cp.copy()  #G_gt_cp_prime.copy()
        for i, (u, v, key, data) in enumerate(Gtmp.edges(keys=True, data=True)):
            try:
                #line = data['geometry']
                data.pop('geometry', None)
            except:
                data[0].pop('geometry',None)
        fig, ax = ox.plot_graph(Gtmp, show=show_plots, close=False,
                                fig_height=fig_height, fig_width=fig_width)
        ax.set_title('Ground Truth Graph (cp) without any geometry', size='x-small')
        #plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'gt_without_geom.png'), dpi=dpi)
        plt.close('all')

       
        # plot proposal
        fig, ax = ox.plot_graph(G_p_init, show=show_plots, close=False,
                                fig_height=fig_height, fig_width=fig_width)
        if show_node_ids:
            ax = apls_tools.plot_node_ids(G_p_init, ax, fontsize=4)  # node ids
        ax.set_title('Proposal Graph', fontsize=title_fontsize)
        #plt.show()
        plt.savefig(os.path.join(outdir, 'prop_graph.png'), dpi=dpi)
        plt.close('all')
    
        #  proposal midpoints
        fig0, ax0 = ox.plot_graph(G_p_cp, show=show_plots, close=False,
                                fig_height=fig_height, fig_width=fig_width)
        if show_node_ids:
            ax = apls_tools.plot_node_ids(G_p_cp, ax0, fontsize=4)  # node ids
        ax0.set_title('Proposal With Midpionts', fontsize=title_fontsize)
        #plt.show()
        plt.savefig(os.path.join(outdir, 'prop_graph_midpoints.png'), dpi=dpi)
        plt.close('all')

        #  proposal midpoints
        fig0, ax0 = ox.plot_graph(G_p_cp_prime, show=show_plots, close=False,
                                fig_height=fig_height, fig_width=fig_width)
        if show_node_ids:
            ax = apls_tools.plot_node_ids(G_p_cp_prime, ax0, fontsize=4)  # node ids
        ax0.set_title('Proposal With Midpionts from GT', fontsize=title_fontsize)
        #plt.show()
        plt.savefig(os.path.join(outdir, 'prop_graph_midpoints_gt_control_points.png'), dpi=dpi)
        plt.close('all')
        
        # plot ground truth buffer and proposal graph
        fig, ax3 = ox.plot_graph(G_p_init, show=show_plots, close=False,
                                fig_height=fig_height, fig_width=fig_width)
        apls_tools.plot_buff(G_gt_init, ax3, buff=args.max_snap_dist, 
                  color='yellow', alpha=0.3, 
                  title='', 
                  title_fontsize=title_fontsize, outfile='',
                  verbose=False)
        ax3.set_title('Propoal Graph with Ground Truth Buffer', fontsize=title_fontsize)
        #plt.show()
        plt.savefig(os.path.join(outdir, 'prop_graph_plus_gt_buff.png'), dpi=dpi)
        #plt.clf()
        #plt.cla()
        plt.close('all')
        
        # plot proposal buffer and ground truth graph
        fig, ax4 = ox.plot_graph(G_gt_init, show=show_plots, close=False,
                                fig_height=fig_height, fig_width=fig_width)
        apls_tools.plot_buff(G_p_init, ax4, buff=args.max_snap_dist, color='yellow', alpha=0.3, 
                  title='',  
                  title_fontsize=title_fontsize, outfile='',
                  verbose=False)
        ax4.set_title('Ground Graph with Proposal Buffer', fontsize=title_fontsize)
        #plt.show()
        plt.savefig(os.path.join(outdir, 'gt_graph_plus_prop_buff.png'), dpi=dpi)
        #plt.clf()
        #plt.cla()
        plt.close('all')
    
    
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
        print "G_gt_cp.nodes():", G_gt_cp.nodes()
        print "G_p_cp_prime.nodes():", G_gt_cp_prime.nodes()
        possible_sources = set(G_gt_cp.nodes()).intersection(set(G_p_cp_prime.nodes()))
        if len(possible_sources) == 0:
            continue
        source = random.choice(list(possible_sources))
        possible_targets = set(G_gt_cp.nodes()).intersection(set(G_p_cp_prime.nodes())) - set([source])
        if len(possible_targets) == 0:
            continue
        target = random.choice(list(possible_targets))
        print "source, target:", source, target
        
        # compute paths to node of interest, and plot
        t0 = time.time()
        lengths, paths = nx.single_source_dijkstra(G_gt_cp, source=source, weight=weight) 
        print "Time to calculate:", len(lengths), "paths:", time.time() - t0, "seconds"
        
        # plot a single route
        try:
            fig, ax = ox.plot_graph_route(G_gt_cp, paths[target], 
                                            route_color='yellow',
                                            route_alpha=0.8,
                                            orig_dest_node_alpha=0.3,
                                            orig_dest_node_size=120,
                                            route_linewidth=route_linewidth,
                                            orig_dest_node_color=target_color,
                                            show=show_plots,
                                            close=False,
                                            fig_height=fig_height, 
                                            fig_width=fig_width)
            plen = np.round(lengths[target],2)
        except:
            print "Prpoosal route not possible"
            fig, ax = plt.subplots()
            plen = -1
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
        plt.close('all')
          
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
        try:
            fig, ax = ox.plot_graph_route(G_p_cp_prime, paths_ptmp[target], 
                                            route_color='yellow',
                                            route_alpha=0.8,
                                            orig_dest_node_alpha=0.3,
                                            orig_dest_node_size=120,
                                            route_linewidth=route_linewidth,
                                            orig_dest_node_color=target_color,
                                            show=show_plots,
                                            close=False,
                                            fig_height=fig_height, 
                                            fig_width=fig_width)
            #title= "Source-" + source_color + " " + str(source) + " Target: " + str(target)
            plen = np.round(lengths_ptmp[target],2)
        except:
            print "Prpoosal route not possible"
            fig, ax = plt.subplots()
            plen = -1
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
        plt.close('all')
    
        ##############
        t2 = time.time()
        print "Total time to create graphs, compute metric, and plot:", t2-t0, "seconds"

    
    return 


###############################################################################
if __name__ == "__main__":
    main()
