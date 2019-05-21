#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:32:19 2017

@author: avanetten

"""

# from __future__ import print_function
# from . import apls_utils
# from . import apls_plots
# from . import osmnx_funcs
# import sp_metric
# import topo_metric
# import apls_tools
# import graphTools
import networkx as nx
import scipy.spatial
import scipy.stats
import numpy as np
import random
import utm           # pip install utm
import copy
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import time
import os
import sys

import argparse
import pandas as pd
import shapely.wkt
# import osmnx as ox   # https://github.com/gboeing/osmnx
# import pickle
# import shutil

path_apls_src = os.path.dirname(os.path.realpath(__file__))
path_apls = os.path.dirname(path_apls_src)
print("path_apls:", path_apls)
# add path and import graphTools
sys.path.append(path_apls_src)
import apls_utils
import apls_plots
import osmnx_funcs
import graphTools
import wkt_to_G
import topo_metric
import sp_metric

# if in docker, the line below may be necessary
matplotlib.use('agg')


###############################################################################
def add_travel_time(G_, speed_key='speed_m/s', length_key='length',
                    travel_time_key='travel_time', default_speed=13.41,
                    verbose=False):
    """
    Compute and add travel time estimaes to each graph edge.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes speed.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'speed_m/s'``.
    length_key : str
        Key in the edge properties dictionary to use for the edge length.
        Defaults to ``'length'`` (asumed to be in meters).
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time'``.
    default_speed : float
        Default speed to use if speed_key is not found in edge properties
        Defaults to ``13.41`` (this is in m/s, and corresponds to 30 mph).
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    G_ : networkx graph
        Updated graph with travel time attached to each edge.
    """

    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        if speed_key in data:
            speed = data[speed_key]
        else:
            data['inferred_speed'] = default_speed
            data[speed_key] = default_speed
            speed = default_speed
        if verbose:
            print("data[length_key]:", data[length_key])
            print("speed:", speed)
        travel_time_seconds = data[length_key] / speed
        data[travel_time_key] = travel_time_seconds

    return G_


###############################################################################
def create_edge_linestrings(G_, remove_redundant=True, verbose=False):
    """
    Ensure all edges have the 'geometry' tag, use shapely linestrings.

    Notes
    -----
    If identical edges exist, remove extras.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that may or may not include 'geometry'.
    remove_redundant : boolean
        Switch to remove identical edges, if they exist.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    G_ : networkx graph
        Updated graph with every edge containing the 'geometry' tag.
    """

    # clean out redundant edges with identical geometry
    edge_seen_set = set([])
    geom_seen = []
    bad_edges = []

    # G_ = G_.copy()
    # for i,(u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        # create linestring if no geometry reported
        if 'geometry' not in data:
            sourcex, sourcey = G_.nodes[u]['x'],  G_.nodes[u]['y']
            targetx, targety = G_.nodes[v]['x'],  G_.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
            data['geometry'] = line_geom

            # get reversed line
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            # G_.edges[u][v]['geometry'] = lstring
        else:
            # check which direction linestring is travelling (it may be going
            #   from v -> u, which means we need to reverse the linestring)
            #   otherwise new edge is tangled
            line_geom = data['geometry']
            # print (u,v,key,"create_edge_linestrings() line_geom:", line_geom)
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            if dist_to_u > dist_to_v:
                # data['geometry'].coords = list(line_geom.coords)[::-1]
                data['geometry'] = line_geom_rev
            # else:
            #    continue

        # flag redundant edges
        if remove_redundant:
            if i == 0:
                edge_seen_set = set([(u, v)])
                edge_seen_set.add((v, u))
                geom_seen.append(line_geom)

            else:
                if ((u, v) in edge_seen_set) or ((v, u) in edge_seen_set):
                    # test if geoms have already been seen
                    for geom_seen_tmp in geom_seen:
                        if (line_geom == geom_seen_tmp) \
                                or (line_geom_rev == geom_seen_tmp):
                            bad_edges.append((u, v))  # , key))
                            if verbose:
                                print("\nRedundant edge:", u, v)  # , key)
                else:
                    edge_seen_set.add((u, v))
                    geom_seen.append(line_geom)
                    geom_seen.append(line_geom_rev)

    if remove_redundant:
        if verbose:
            print("\nedge_seen_set:", edge_seen_set)
            print("redundant edges:", bad_edges)
        for (u, v) in bad_edges:
            if G_.has_edge(u, v):
                G_.remove_edge(u, v)  # , key)
            # # for (u,v,key) in bad_edges:
            # try:
            #     G_.remove_edge(u, v)  # , key)
            # except:
            #     if verbose:
            #         print("Edge DNE:", u, v)  # ,key)
            #     pass

    return G_


###############################################################################
def cut_linestring(line, distance, verbose=False):
    '''
    Cuts a line in two at a distance from its starting point
    http://toblerity.org/shapely/manual.html#linear-referencing-methods
    '''
    """
    Cuts a shapely linestring at a specified distance from its starting point.

    Notes
    ----
    Return orignal linestring if distance <= 0 or greater than the length of
    the line.
    Reference:
        http://toblerity.org/shapely/manual.html#linear-referencing-methods

    Arguments
    ---------
    line : shapely linestring
        Input shapely linestring to cut.
    distanct : float
        Distance from start of line to cut it in two.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    [line1, line2] : list
        Cut linestrings.  If distance <= 0 or greater than the length of
        the line, return input line.
    """

    if verbose:
        print("Cutting linestring at distance", distance, "...")
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    # iterate through coorda and check if interpolated point has been passed
    # already or not
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if verbose:
            print(i, p, "line.project point:", pdl)
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
def get_closest_edge_from_G(G_, point, nearby_nodes_set=set([]),
                            verbose=False):
    """
    Return closest edge to point, and distance to said edge.

    Notes
    -----
    Just discovered a similar function:
        https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py#L501

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    best_edge, min_dist, best_geom : tuple
        best_edge is the closest edge to the point
        min_dist is the distance to that edge
        best_geom is the geometry of the ege
    """

    # get distances from point to lines
    dist_list = []
    edge_list = []
    geom_list = []
    p = point  # Point(point_coords)
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        # print((" in get_closest_edge(): u,v,key,data:", u,v,key,data))
        # print ("  in get_closest_edge(): data:", data)

        # skip if u,v not in nearby nodes
        if len(nearby_nodes_set) > 0:
            if (u not in nearby_nodes_set) and (v not in nearby_nodes_set):
                continue
        if verbose:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except KeyError:
            line = data['attr_dict']['geometry']
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
def insert_point_into_G(G_, point, node_id=100000, max_distance_meters=5,
                        nearby_nodes_set=set([]), allow_renaming=False,
                        verbose=False, super_verbose=False):
    """
    Insert a new node in the graph closest to the given point.

    Notes
    -----
    If the point is too far from the graph, don't insert a node.
    Assume all edges have a linestring geometry
    http://toblerity.org/shapely/manual.html#object.simplify
    Sometimes the point to insert will have the same coordinates as an
    existing point.  If allow_renaming == True, relabel the existing node.
    convert linestring to multipoint?
     https://github.com/Toblerity/Shapely/issues/190

    TODO : Implement a version without renaming that tracks which node is
        closest to the desired point.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates
    node_id : int
        Unique identifier of node to insert. Defaults to ``100000``.
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    G_, node_props, min_dist : tuple
        G_ is the updated graph
        node_props gives the properties of the inserted node
        min_dist is the distance from the point to the graph
    """

    # check if node_id already exists in G
    # if node_id in set(G_.nodes()):
    #    print ("node_id:", node_id, "already in G, cannot insert node!")
    #    return

    best_edge, min_dist, best_geom = get_closest_edge_from_G(
            G_, point, nearby_nodes_set=nearby_nodes_set,
            verbose=super_verbose)
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())

    if verbose:
        print("Inserting point:", node_id)
        print("best edge:", best_edge)
        print("  best edge dist:", min_dist)
        u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
        v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
        print("ploc:", (point.x, point.y))
        print("uloc:", u_loc)
        print("vloc:", v_loc)

    if min_dist > max_distance_meters:
        if verbose:
            print("min_dist > max_distance_meters, skipping...")
        return G_, {}, -1, -1

    else:
        # updated graph

        # skip if node exists already
        if node_id in G_node_set:
            if verbose:
                print("Node ID:", node_id, "already exists, skipping...")
            return G_, {}, -1, -1

        # G_.edges[best_edge[0]][best_edge[1]][0]['geometry']
        line_geom = best_geom

        # Length along line that is closest to the point
        line_proj = line_geom.project(point)

        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(line_geom.project(point))
        x, y = new_point.x, new_point.y

        #################
        # create new node
        
        try:
            # first get zone, then convert to latlon
            _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
                                                          G_.nodes[u]['lon'])
            # convert utm to latlon
            lat, lon = utm.to_latlon(x, y, zone_num, zone_letter)
        except:
            lat, lon = y, x

        # set properties
        # props = G_.nodes[u]
        node_props = {'highway': 'insertQ',
                      'lat':     lat,
                      'lon':     lon,
                      'osmid':   node_id,
                      'x':       x,
                      'y':       y}
        # add node
        G_.add_node(node_id, **node_props)

        # assign, then update edge props for new edge
        _, _, edge_props_new = copy.deepcopy(
            list(G_.edges([u, v], data=True))[0])
        # remove extraneous 0 key

        # print ("edge_props_new.keys():", edge_props_new)
        # if list(edge_props_new.keys()) == [0]:
        #    edge_props_new = edge_props_new[0]

        # cut line
        split_line = cut_linestring(line_geom, line_proj)
        # line1, line2, cp = cut_linestring(line_geom, line_proj)
        if split_line is None:
            print("Failure in cut_linestring()...")
            print("type(split_line):", type(split_line))
            print("split_line:", split_line)
            print("line_geom:", line_geom)
            print("line_geom.length:", line_geom.length)
            print("line_proj:", line_proj)
            print("min_dist:", min_dist)
            return G_, {}, 0, 0

        if verbose:
            print("split_line:", split_line)

        # if cp.is_empty:
        if len(split_line) == 1:
            if verbose:
                print("split line empty, min_dist:", min_dist)
            # get coincident node
            outnode = ''
            outnode_x, outnode_y = -1, -1
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
            x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']
            # if verbose:
            #    print "x_p, y_p:", x_p, y_p
            #    print "x_u, y_u:", x_u, y_u
            #    print "x_v, y_v:", x_v, y_v

            # sometimes it seems that the nodes aren't perfectly coincident,
            # so see if it's within a buffer
            buff = 0.05  # meters
            if (abs(x_p - x_u) <= buff) and (abs(y_p - y_u) <= buff):
                outnode = u
                outnode_x, outnode_y = x_u, y_u
            elif (abs(x_p - x_v) <= buff) and (abs(y_p - y_v) <= buff):
                outnode = v
                outnode_x, outnode_y = x_v, y_v
            # original method with exact matching
            # if (x_p == x_u) and (y_p == y_u):
            #    outnode = u
            #    outnode_x, outnode_y = x_u, y_u
            # elif (x_p == x_v) and (y_p == y_v):
            #    outnode = v
            #    outnode_x, outnode_y = x_v, y_v
            else:
                print("Error in determining node coincident with node: "
                      + str(node_id) + " along edge: " + str(best_edge))
                print("x_p, y_p:", x_p, y_p)
                print("x_u, y_u:", x_u, y_u)
                print("x_v, y_v:", x_v, y_v)
                # return
                return G_, {}, 0, 0

            # if the line cannot be split, that means that the new node
            # is coincident with an existing node.  Relabel, if desired
            if allow_renaming:
                node_props = G_.nodes[outnode]
                # A dictionary with the old labels as keys and new labels
                #  as values. A partial mapping is allowed.
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                if verbose:
                    print("Swapping out node ids:", mapping)
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
                if line1.length > buff:
                    print("Nodes should be coincident and length 0!")
                    print("  line1.length:", line1.length)
                    print("  x_u, y_u :", x_u, y_u)
                    print("  x_v, y_v :", x_v, y_v)
                    print("  x_p, y_p :", x_p, y_p)
                    print("  new_point:", new_point)
                    print("  Point(outnode_x, outnode_y):",
                          Point(outnode_x, outnode_y))
                    return

                # add edge of length 0 from new node to neareest existing node
                G_.add_edge(node_id, outnode, **edge_props_line1)
                return G_, node_props, x, y

                # originally, if not renaming nodes,
                # just ignore this complication and return the orignal
                # return G_, node_props, 0, 0

        else:
            # else, create new edges
            line1, line2 = split_line

            # get distances
            # print ("insert_point(), G_.nodes[v]:", G_.nodes[v])
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            # compare to first point in linestring
            geom_p0 = list(line_geom.coords)[0]
            # or compare to inserted point? [this might fail if line is very
            #    curved!]
            # geom_p0 = (x,y)
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # reverse edge order if v closer than u
            if dist_to_v < dist_to_u:
                line2, line1 = split_line

            if verbose:
                print("Creating two edges from split...")
                print("   original_length:", line_geom.length)
                print("   line1_length:", line1.length)
                print("   line2_length:", line2.length)
                print("   u, dist_u_to_point:", u, dist_to_u)
                print("   v, dist_v_to_point:", v, dist_to_v)
                print("   min_dist:", min_dist)

            # add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            # remove geometry?
            # edge_props_line1.pop('geometry', None)
            # line2
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2
            # remove geometry?
            # edge_props_line1.pop('geometry', None)

            # insert edge regardless of direction
            # G_.add_edge(u, node_id, **edge_props_line1)
            # G_.add_edge(node_id, v, **edge_props_line2)

            # check which direction linestring is travelling (it may be going
            # from v -> u, which means we need to reverse the linestring)
            # otherwise new edge is tangled
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # if verbose:
            #    print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, **edge_props_line1)
                G_.add_edge(node_id, v, **edge_props_line2)
            else:
                G_.add_edge(node_id, u, **edge_props_line1)
                G_.add_edge(v, node_id, **edge_props_line2)

            if verbose:
                print("insert edges:", u, '-', node_id, 'and', node_id, '-', v)

            # remove initial edge
            G_.remove_edge(u, v, key)

            return G_, node_props, x, y


###############################################################################
def insert_control_points(G_, control_points, max_distance_meters=10,
                          allow_renaming=False,
                          n_nodes_for_kd=1000, n_neighbors=20,
                          x_coord='x', y_coord='y',
                          verbose=True, super_verbose=False):
    """
    Wrapper around insert_point_into_G() for all control_points.

    Notes
    -----
    control_points are assumed to be of the format:
        [[node_id, x, y], ... ]

    TODO : Implement a version without renaming that tracks which node is
        closest to the desired point.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    control_points : array
        Points to insert in the graph, assumed to the of the format:
            [[node_id, x, y], ... ]
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    n_nodes_for_kd : int
        Minumu size of graph to render to kdtree to speed node placement.
        Defaults to ``1000``.
    n_neighbors : int
        Number of neigbors to return if building a kdtree. Defaults to ``20``.
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    Gout, new_xs, new_ys : tuple
        Gout is the updated graph
        new_xs, new_ys are coordinates of the inserted points
    """

    t0 = time.time()

    # insertion can be super slow so construct kdtree if a large graph
    if len(G_.nodes()) > n_nodes_for_kd:
        # construct kdtree of ground truth
        kd_idx_dic, kdtree, pos_arr = apls_utils.G_to_kdtree(G_)

    Gout = G_.copy()
    new_xs, new_ys = [], []
    if len(G_.nodes()) == 0:
        return Gout, new_xs, new_ys

    for i, [node_id, x, y] in enumerate(control_points):
        # if verbose:
        if (i % 20) == 0:
            print(i, "/", len(control_points),
                  "Insert control point:", node_id, "x =", x, "y =", y)
        point = Point(x, y)

        # if large graph, determine nearby nodes
        if len(G_.nodes()) > n_nodes_for_kd:
            # get closest nodes
            node_names, dists_m_refine = apls_utils.nodes_near_point(
                    x, y, kdtree, kd_idx_dic, x_coord=x_coord, y_coord=y_coord,
                    # radius_m=radius_m,
                    n_neighbors=n_neighbors,
                    verbose=False)
            nearby_nodes_set = set(node_names)
        else:
            nearby_nodes_set = set([])

        # insert point
        Gout, node_props, xnew, ynew = insert_point_into_G(
            Gout, point, node_id=node_id,
            max_distance_meters=max_distance_meters,
            nearby_nodes_set=nearby_nodes_set,
            allow_renaming=allow_renaming,
            verbose=super_verbose)
        # xnew = node_props['x']
        # ynew = node_props['y']
        if (x != 0) and (y != 0):
            new_xs.append(xnew)
            new_ys.append(ynew)

    t1 = time.time()
    print("Time to run insert_control_points():", t1-t0, "seconds")
    return Gout, new_xs, new_ys


###############################################################################
def create_graph_midpoints(G_, linestring_delta=50, is_curved_eps=0.03,
                           n_id_add_val=1, allow_renaming=False,
                           figsize=(0, 0),
                           verbose=False, super_verbose=False):
    """
    Insert midpoint nodes into long edges on the graph.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    linestring_delta : float
        Distance in meters between linestring midpoints. Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.  Defaults to ``0.3``.
    n_id_add_val : int
        Sets min midpoint id above existing nodes
        e.g.: G.nodes() = [1,2,4], if n_id_add_val = 5, midpoints will
        be [9,10,11,...]
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    figsize : tuple
        Figure size for optional plot. Defaults to ``(0,0)`` (no plot).
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    Gout, xms, yms : tuple
        Gout is the updated graph
        xms, yms are coordinates of the inserted points
    """

    # midpoint_loc = 0.5        # take the central midpoint for straight lines
    if len(G_.nodes()) == 0:
        return G_, [], []

    # midpoints
    xms, yms = [], []
    Gout = G_.copy()
    # midpoint_name_val, midpoint_name_inc = 0.01, 0.01
    midpoint_name_val, midpoint_name_inc = np.max(G_.nodes())+n_id_add_val, 1
    # for u, v, key, data in G_.edges(keys=True, data=True):
    for u, v, data in G_.edges(data=True):

        # curved line
        if 'geometry' in data:

            # first edge props and  get utm zone and letter
            edge_props_init = G_.edges([u, v])
            # _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
            #                                              G_.nodes[u]['lon'])

            linelen = data['length']
            line = data['geometry']

            xs, ys = line.xy  # for plotting

            #################
            # check if curved or not
            minx, miny, maxx, maxy = line.bounds
            # get euclidean distance
            dst = scipy.spatial.distance.euclidean([minx, miny], [maxx, maxy])
            # ignore if almost straight
            if np.abs(dst - linelen) / linelen < is_curved_eps:
                # print "Line straight, skipping..."
                continue
            #################

            #################
            # also ignore super short lines
            if linelen < 0.75*linestring_delta:
                # print "Line too short, skipping..."
                continue
            #################

            if verbose:
                print("create_graph_midpoints()...")
                print("  u,v:", u, v)
                print("  data:", data)
                print("  edge_props_init:", edge_props_init)

            # interpolate midpoints
            # if edge is short, use midpoint, else get evenly spaced points
            if linelen <= linestring_delta:
                interp_dists = [0.5 * line.length]
            else:
                # get evenly spaced points
                npoints = len(np.arange(0, linelen, linestring_delta)) + 1
                interp_dists = np.linspace(0, linelen, npoints)[1:-1]
                if verbose:
                    print("  interp_dists:", interp_dists)

            # create nodes
            node_id_new_list = []
            xms_tmp, yms_tmp = [], []
            for j, d in enumerate(interp_dists):
                if verbose:
                    print("    ", j, "interp_dist:", d)

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
                    print("    midpoint:", xm, ym)

                # add node to graph, with properties of u
                node_id = midpoint_name_val
                # node_id = np.round(u + midpoint_name_val,2)
                midpoint_name_val += midpoint_name_inc
                node_id_new_list.append(node_id)
                if verbose:
                    print("    node_id:", node_id)

                # if j > 3:
                #    continue

                # add to graph
                Gout, node_props, _, _ = insert_point_into_G(
                    Gout, point, node_id=node_id,
                    allow_renaming=allow_renaming,
                    verbose=super_verbose)

        # plot, if desired
        if figsize != (0, 0):
            fig, (ax) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))
            ax.plot(xs, ys, color='#6699cc', alpha=0.7,
                    linewidth=3, solid_capstyle='round', zorder=2)
            ax.scatter(xm, ym, color='red')
            ax.set_title('Line Midpoint')
            plt.axis('equal')

    return Gout, xms, yms


###############################################################################
def _clean_sub_graphs(G_, min_length=80, max_nodes_to_skip=100,
                      weight='length', verbose=True,
                      super_verbose=False):
    """
    Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length
       (this step great reduces processing time)
    """

    if len(G_.nodes()) == 0:
        return G_

    if verbose:
        print("Running clean_sub_graphs...")
    sub_graphs = list(nx.connected_component_subgraphs(G_))
    bad_nodes = []
    if verbose:
        print(" len(G_.nodes()):", len(G_.nodes()))
        print(" len(G_.edges()):", len(G_.edges()))
    if super_verbose:
        print("G_.nodes:", G_.nodes())
        edge_tmp = G_.edges()[np.random.randint(len(G_.edges()))]
        print(edge_tmp, "G.edge props:", G_.edges[edge_tmp[0]][edge_tmp[1]])

    for G_sub in sub_graphs:
        # don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue

        else:
            all_lengths = dict(
                nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            if super_verbose:
                print("  \nGs.nodes:", G_sub.nodes())
                print("  all_lengths:", all_lengths)
            # get all lenghts
            lens = []

            # for u,v in all_lengths.iteritems():
            for u in all_lengths.keys():
                v = all_lengths[u]
                # for uprime, vprime in v.iteritems():
                for uprime in v.keys():
                    vprime = v[uprime]
                    lens.append(vprime)
                    if super_verbose:
                        print("  u, v", u, v)
                        print("    uprime, vprime:", uprime, vprime)
            max_len = np.max(lens)
            if super_verbose:
                print("  Max length of path:", max_len)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())
                if super_verbose:
                    print(" appending to bad_nodes:", G_sub.nodes())

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)
    if verbose:
        print(" num bad_nodes:", len(bad_nodes))
        # print ("bad_nodes:", bad_nodes)
        print(" len(G'.nodes()):", len(G_.nodes()))
        print(" len(G'.edges()):", len(G_.edges()))
    if super_verbose:
        print("  G_.nodes:", G_.nodes())

    return G_


###############################################################################
def _create_gt_graph(geoJson, im_test_file, network_type='all_private',
                     # linestring_delta=50, is_curved_eps=0.012,
                     valid_road_types=set([]),
                     osmidx=0, osmNodeidx=0,
                     # weight='length',
                     subgraph_filter_weight='length',
                     min_subgraph_length=5,
                     travel_time_key='travel_time',
                     speed_key='speed_m/s',
                     use_pix_coords=False,
                     verbose=False,
                     super_verbose=False):
    '''Ingest graph from geojson file and refine'''

    t0 = time.time()
    if verbose:
        print("Executing graphTools.create_graphGeoJson()...")
    G0gt_init = graphTools.create_graphGeoJson(
        geoJson, name='unnamed', retain_all=True,
        network_type=network_type, valid_road_types=valid_road_types,
        osmidx=osmidx, osmNodeidx=osmNodeidx, verbose=verbose)
    t1 = time.time()
    if verbose:
        print("Time to run create_graphGeoJson():", t1 - t0, "seconds")

    # save latlon geometry (osmnx overwrites the 'geometry' tag)
    # also compute pixel geom
    for i, (u, v, key, data) in enumerate(G0gt_init.edges(keys=True, data=True)):
        # print ("apsl.create_gt_graph(): data:", data)
        if 'geometry' not in data:
            # print ("G0gt_init.nodes[u]:", G0gt_init.nodes[u])
            sourcex, sourcey = G0gt_init.nodes[u]['x'],  G0gt_init.nodes[u]['y']
            targetx, targety = G0gt_init.nodes[v]['x'],  G0gt_init.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
        else:
            line_geom = data['geometry']
        data['geometry_latlon'] = line_geom.wkt
        
        # print ("im_test_file:", im_test_file)
        if os.path.exists(im_test_file):
            # get pixel geom (do this after simplify so that we don't have to
            #   collapse the lines (see wkt_to_G.wkt_to_G)
            geom_pix = apls_utils.geomGeo2geomPixel(line_geom,
                                                    input_raster=im_test_file)
            data['geometry_pix'] = geom_pix.wkt
            data['length_pix'] = geom_pix.length

    # print coords
    # n = G0gt_init.nodes()[0]
    # print "n, G0gt_init.nodes[n]:", n, G0gt_init.nodes[n]

    if len(G0gt_init.nodes()) == 0:
        return G0gt_init, G0gt_init  # , [], [], [], []

    G0gt = osmnx_funcs.project_graph(G0gt_init)
    if verbose:
        print("len G0gt.nodes():", len(G0gt.nodes()))
        print("len G0gt.edges:", len(G0gt.edges()))
        # print an edge?
        # edge_tmp = list(G0gt.edges())[-1]
        # print (edge_tmp, "random edge props:", G0gt.edges[edge_tmp[0], edge_tmp[1]]) #G.edge[edge_tmp[0]][edge_tmp[1]])
        # print (edge_tmp, "random edge props:", G0gt.edges([edge_tmp[0], edge_tmp[1]])) #G.edge[edge_tmp[0]][edge_tmp[1]])

    # print coords
    # n = G0gt.nodes()[0]
    # print "n, G0gt.nodes[n]:", n, G0gt.nodes[n]

    # G1gt = osmnx_funcs.simplify_graph(G0gt)
    # G2gt_init = G1gt.to_undirected()

    if verbose:
        print("Simplifying graph...")
    G2gt_init0 = osmnx_funcs.simplify_graph(G0gt).to_undirected()
    # G2gt_init = osmnx_funcs.simplify_graph(G0gt.to_undirected())

    # make sure all edges have a geometry assigned to them
    G2gt_init1 = create_edge_linestrings(
        G2gt_init0.copy(), remove_redundant=True)
    t2 = time.time()
    if verbose:
        print("Time to project, simplify, and create linestrings:",
              t2 - t1, "seconds")

    # clean up connected components
    G2gt_init2 = _clean_sub_graphs(
        G2gt_init1.copy(), min_length=min_subgraph_length,
        weight=subgraph_filter_weight,
        verbose=verbose, super_verbose=super_verbose)

    # add pixel coords
    if os.path.exists(im_test_file):
        G_gt_almost, _, gt_graph_coords = apls_utils._set_pix_coords(
            G2gt_init2.copy(), im_test_file)
    else:
        G_gt_almost = G2gt_init2

    # update pixel and lat lon geometries that get turned into lists upon
    #   simplify() that produces a 'geometry' tag in wmp
    if verbose:
        print("Merge 'geometry' linestrings...")
    keys_tmp = ['geometry_pix', 'geometry_latlon']
    for i, (u, v, attr_dict) in enumerate(G_gt_almost.edges(data=True)):
        # if verbose and (i % 10000) == 0:
        #    print (i, u , v)
        for key_tmp in keys_tmp:

            if key_tmp not in attr_dict.keys():
                continue

            if super_verbose:
                print("Merge", key_tmp, "...")
            geom = attr_dict[key_tmp]

            if type(geom) == list:
                # check if the list items are wkt strings, if so, create
                #   linestrigs
                # or (type(geom_pix[0]) == unicode):
                if (type(geom[0]) == str):
                    geom = [shapely.wkt.loads(ztmp) for ztmp in geom]
                # merge geoms
                # geom = shapely.ops.linemerge(geom)
                # attr_dict[key_tmp] =  geom
                attr_dict[key_tmp] = shapely.ops.linemerge(geom)
            elif type(geom) == str:
                attr_dict[key_tmp] = shapely.wkt.loads(geom)
            else:
                pass

        # update wkt_pix?
        if 'wkt_pix' in attr_dict.keys():
            # print ("attr_dict['geometry_pix':", attr_dict['geometry_pix'])
            attr_dict['wkt_pix'] = attr_dict['geometry_pix'].wkt

        # update 'length_pix'
        if 'length_pix' in attr_dict.keys():
            attr_dict['length_pix'] = np.sum([attr_dict['length_pix']])

        # check if simplify created various speeds on an edge
        speed_keys = ['speed_mph', 'speed_m/s']
        for sk in speed_keys:
            if sk not in attr_dict.keys():
                continue
            if type(attr_dict[sk]) == list:
                print("  Taking mean of multiple speeds on edge:", u, v)
                attr_dict[sk] = np.mean(attr_dict[sk])
                if verbose:
                    print("u, v, speed_key, attr_dict)[speed_key]:",
                          u, v, sk, attr_dict[sk])

    # add travel time
    G_gt = add_travel_time(G_gt_almost.copy(),
                           speed_key=speed_key,
                           travel_time_key=travel_time_key)

    return G_gt, G0gt_init


###############################################################################
def make_graphs(G_gt, G_p,
                weight='length',
                speed_key='speed_m/s',
                travel_time_key='travel_time',
                max_nodes_for_midpoints=500,
                linestring_delta=50,
                is_curved_eps=0.012,
                max_snap_dist=4,
                allow_renaming=False,
                verbose=False,
                super_verbose=False):
    """
    Match nodes in ground truth and propsal graphs, and get paths.

    Notes
    -----
    The path length dictionaries returned by this function will be fed into
    compute_metric().

    Arguments
    ---------
    G_gt : networkx graph
        Ground truth graph.
    G_p : networkd graph
        Proposal graph over the same region.
    weight : str
        Key in the edge properties dictionary to use for the path length
        weight.  Defaults to ``'length'``.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'speed_m/s'``.
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time'``.
    max_nodes_for_midpoints : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    linestring_delta : float
        Distance in meters between linestring midpoints.
        If len gt nodes > max_nodes_for_midppoints this argument is ignored.
        Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.
        If len gt nodes > max_nodes_for_midppoints this argument is ignored.
        Defaults to ``0.012``.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Return
    ------
    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime : tuple
        G_gt_cp  is ground truth with control points inserted
        G_p_cp is proposal with control points inserted
        G_gt_cp_prime is ground truth with control points from prop inserted
        G_p_cp_prime is proposal with control points from gt inserted
        all_pairs_lengths_gt_native is path length dict corresponding to G_gt_cp
        all_pairs_lengths_prop_native is path length dict corresponding to G_p_cp
        all_pairs_lengths_gt_prime is path length dict corresponding to G_gt_cp_prime
        all_pairs_lenfgths_prop_prime is path length dict corresponding to G_p_cp_prime 
    """

    t0 = time.time()
    print("Executing make_graphs()...")

    print("Ensure G_gt 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_gt.edges(keys=True, data=True)):
        if i == 0:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except KeyError:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    # create graph with midpoints
    G_gt0 = create_edge_linestrings(G_gt.to_undirected())
    # create graph with linestrings?
    G_gt_cp = G_gt.to_undirected()
    # G_gt_cp = create_edge_linestrings(G_gt.to_undirected())

    if verbose:
        print("len G_gt.nodes():", len(list(G_gt0.nodes())))
        print("len G_gt.edges():", len(list(G_gt0.edges())))

    if verbose:
        print("Creating gt midpoints")
    G_gt_cp, xms, yms = create_graph_midpoints(
        G_gt0.copy(),
        linestring_delta=linestring_delta,
        figsize=(0, 0),
        is_curved_eps=is_curved_eps,
        verbose=False)
    # add travel time
    G_gt_cp = add_travel_time(G_gt_cp.copy(),
                              speed_key=speed_key,
                              travel_time_key=travel_time_key)

    # get ground truth control points
    control_points_gt = []
    for n in G_gt_cp.nodes():
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])
    if verbose:
        print("len control_points_gt:", len(control_points_gt))

    # get ground truth paths
    if verbose:
        print("Get ground truth paths...")
    all_pairs_lengths_gt_native = dict(
        nx.all_pairs_dijkstra_path_length(G_gt_cp, weight=weight))
    ###############

    ###############
    # get proposal graph with native midpoints
    print("Ensure G_p 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_p.edges(keys=True, data=True)):
        if i == 0:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    G_p = create_edge_linestrings(G_p.to_undirected())
    # add travel time
    G_p = add_travel_time(G_p.copy(),
                          speed_key=speed_key,
                          travel_time_key=travel_time_key)

    if verbose:
        print("len G_p.nodes():", len(G_p.nodes()))
        print("len G_p.edges():", len(G_p.edges()))

    if verbose:
        print("Creating proposal midpoints")
    G_p_cp, xms_p, yms_p = create_graph_midpoints(
        G_p.copy(),
        linestring_delta=linestring_delta,
        figsize=(0, 0),
        is_curved_eps=is_curved_eps,
        verbose=False)
    # add travel time
    G_p_cp = add_travel_time(G_p_cp.copy(),
                             speed_key=speed_key,
                             travel_time_key=travel_time_key)
    if verbose:
        print("len G_p_cp.nodes():", len(G_p_cp.nodes()))
        print("len G_p_cp.edges():", len(G_p_cp.edges()))

    # set proposal control nodes, originally just all nodes in G_p_cp
    # original method sets proposal control points as all nodes in G_p_cp
    # get proposal control points
    control_points_prop = []
    for n in G_p_cp.nodes():
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])

    # get paths
    all_pairs_lengths_prop_native = dict(
        nx.all_pairs_dijkstra_path_length(G_p_cp, weight=weight))

    ###############
    # insert gt control points into proposal
    if verbose:
        print("Inserting", len(control_points_gt),
              "control points into G_p...")
        print("G_p.nodes():", G_p.nodes())
    G_p_cp_prime, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt,
        max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming,
        verbose=super_verbose)
    # add travel time
    G_p_cp_prime = add_travel_time(G_p_cp_prime.copy(),
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key)
#    G_p_cp, xn_p, yn_p = insert_control_points(G_p_cp, control_points_gt,
#                                        max_distance_meters=max_snap_dist,
#                                        allow_renaming=allow_renaming,
#                                        verbose=verbose)

    ###############
    # now insert control points into ground truth
    if verbose:
        print("\nInserting", len(control_points_prop),
              "control points into G_gt...")
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime, xn_gt, yn_gt = insert_control_points(
        G_gt,
        control_points_prop,
        max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming,
        verbose=super_verbose)
    # add travel time
    G_gt_cp_prime = add_travel_time(G_gt_cp_prime.copy(),
                                    speed_key=speed_key,
                                    travel_time_key=travel_time_key)

    ###############
    # get paths
    all_pairs_lengths_gt_prime = dict(
        nx.all_pairs_dijkstra_path_length(G_gt_cp_prime, weight=weight))
    all_pairs_lengths_prop_prime = dict(
        nx.all_pairs_dijkstra_path_length(G_p_cp_prime, weight=weight))

    tf = time.time()
    print("Time to run make_graphs in apls.py:", tf - t0, "seconds")

    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
        control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime


###############################################################################
def make_graphs_yuge(G_gt, G_p,
                     weight='length',
                     speed_key='speed_m/s',
                     travel_time_key='travel_time',
                     max_nodes=500,
                     max_snap_dist=4,
                     allow_renaming=False,
                     verbose=True, super_verbose=False):
    """
    Match nodes in large ground truth and propsal graphs, and get paths.

    Notes
    -----
    Skip midpoint injection and only select a subset of routes to compare.
    The path length dictionaries returned by this function will be fed into
    compute_metric().

    Arguments
    ---------
    G_gt : networkx graph
        Ground truth graph.
    G_p : networkd graph
        Proposal graph over the same region.
    weight : str
        Key in the edge properties dictionary to use for the path length
        weight.  Defaults to ``'length'``.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'speed_m/s'``.
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time'``.
    max_nodess : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Return
    ------
    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime : tuple
        G_gt_cp  is ground truth with control points inserted
        G_p_cp is proposal with control points inserted
        G_gt_cp_prime is ground truth with control points from prop inserted
        G_p_cp_prime is proposal with control points from gt inserted
        all_pairs_lengths_gt_native is path length dict corresponding to G_gt_cp
        all_pairs_lengths_prop_native is path length dict corresponding to G_p_cp
        all_pairs_lengths_gt_prime is path length dict corresponding to G_gt_cp_prime
        all_pairs_lenfgths_prop_prime is path length dict corresponding to G_p_cp_prime 
    """
    
    t0 = time.time()
    print("Executing make_graphs_yuge()...")

    print("Ensure G_gt 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_gt.edges(keys=True, data=True)):
        if i == 0:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    print("Ensure G_p 'geometry' is a shapely geometry, not a linestring...")
    for i, (u, v, key, data) in enumerate(G_p.edges(keys=True, data=True)):
        if i == 0:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    # create graph with linestrings?
    G_gt_cp = G_gt.to_undirected()
    # G_gt_cp = create_edge_linestrings(G_gt.to_undirected())
    if verbose:
        print("len(G_gt.nodes()):", len(G_gt_cp.nodes()))
        print("len(G_gt.edges()):", len(G_gt_cp.edges()))
        # print("G_gt.nodes():", G_gt_cp.nodes())
        # print("G_gt.edges()):", G_gt_cp.edges())
        # gt node and edge props
        node = random.choice(list(G_gt.nodes()))
        print("node:", node, "G_gt random node props:", G_gt.nodes[node])
        edge_tmp = random.choice(list(G_gt.edges()))
        print("G_gt edge_tmp:", edge_tmp)
        try:
            print("edge:", edge_tmp, "G_gt random edge props:",
                  G_gt.edges[edge_tmp[0]][edge_tmp[1]])
        except:
            print("edge:", edge_tmp, "G_gt random edge props:",
                  G_gt.edges[edge_tmp[0], edge_tmp[1], 0])
        # prop node and edge props
        node = random.choice(list(G_p.nodes()))
        print("node:", node, "G_p random node props:", G_p.nodes[node])
        edge_tmp = random.choice(list(G_p.edges()))
        print("G_p edge_tmp:", edge_tmp)
        try:
            print("edge:", edge_tmp, "G_p random edge props:",
                  G_p.edges[edge_tmp[0]][edge_tmp[1]])
        except:
            print("edge:", edge_tmp, "G_p random edge props:",
                  G_p.edges[edge_tmp[0], edge_tmp[1], 0])

    # get ground truth control points, which will be a subset of nodes
    sample_size = min(max_nodes, len(G_gt_cp.nodes()))
    rand_nodes_gt = random.sample(G_gt_cp.nodes(), sample_size)
    rand_nodes_gt_set = set(rand_nodes_gt)
    control_points_gt = []
    for n in rand_nodes_gt:
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])
    if verbose:
        print("len control_points_gt:", len(control_points_gt))
    # add travel time
    G_gt_cp = add_travel_time(G_gt_cp,
                              speed_key=speed_key,
                              travel_time_key=travel_time_key)

    # get route lengths between all control points
    # gather all paths from nodes of interest, keep only routes to control nodes
    tt = time.time()
    if verbose:
        print("Computing all_pairs_lengths_gt_native...")
    all_pairs_lengths_gt_native = {}
    for itmp, source in enumerate(rand_nodes_gt):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        paths_tmp = nx.single_source_dijkstra_path_length(
            G_gt_cp, source, weight=weight)
        # delete items
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_gt_set:
                del paths_tmp[k]
        all_pairs_lengths_gt_native[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               sample_size, "nodes:", time.time() - tt, "seconds"))

    # get individual routes (super slow!)
    #t0 = time.time()
    #all_pairs_lengths_gt_native = {}
    # for source in rand_nodes_gt:
    #    print ("source:", source)
    #    source_dic = {}
    #    for target in rand_nodes_gt:
    #        print ("target:", target)
    #        p = nx.dijkstra_path_length(G_gt_init, source, target, weight=weight)
    #        source_dic[target] = p
    #    all_pairs_lengths_gt_native[source] = source_dic
    #print ("Time to compute all source routes:", time.time() - t0, "seconds")
    ## ('Time to compute all source routes:', 9.418055057525635, 'seconds')

    #all_pairs_lengths_gt_native = nx.all_pairs_dijkstra_path_length(G_gt_cp, weight=weight)
    ###############

    ###############
    # get proposal graph with native midpoints
    G_p_cp = G_p.to_undirected()
    #G_p_cp = create_edge_linestrings(G_p.to_undirected())
    if verbose:
        print("len G_p_cp.nodes():", len(G_p_cp.nodes()))
        print("G_p_cp.edges():", len(G_p_cp.edges()))

    # get control points, which will be a subset of nodes
    # (original method sets proposal control points as all nodes in G_p_cp)
    sample_size = min(max_nodes, len(G_p_cp.nodes()))
    rand_nodes_p = random.sample(G_p_cp.nodes(), sample_size)
    rand_nodes_p_set = set(rand_nodes_p)
    control_points_prop = []
    for n in rand_nodes_p:
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])
    # add travel time
    G_p_cp = add_travel_time(G_p_cp,
                             speed_key=speed_key,
                             travel_time_key=travel_time_key)

    # get paths
    # gather all paths from nodes of interest, keep only routes to control nodes
    tt = time.time()
    if verbose:
        print("Computing all_pairs_lengths_prop_native...")
    all_pairs_lengths_prop_native = {}
    for itmp, source in enumerate(rand_nodes_p):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        paths_tmp = nx.single_source_dijkstra_path_length(
            G_p_cp, source, weight=weight)
        # delete items
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_p_set:
                del paths_tmp[k]
        all_pairs_lengths_prop_native[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    ###############
    # insert gt control points into proposal
    if verbose:
        print("Inserting", len(control_points_gt),
              "control points into G_p...")
        print("len G_p.nodes():", len(G_p.nodes()))
    G_p_cp_prime, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt, max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming, verbose=super_verbose)
    # add travel time
    G_p_cp_prime = add_travel_time(G_p_cp_prime,
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key)

    ###############
    # now insert control points into ground truth
    if verbose:
        print("\nInserting", len(control_points_prop),
              "control points into G_gt...")
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime, xn_gt, yn_gt = insert_control_points(
        G_gt, control_points_prop, max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming, verbose=super_verbose)
    G_gt_cp_prime = add_travel_time(G_gt_cp_prime,
                                    speed_key=speed_key,
                                    travel_time_key=travel_time_key)

    ###############
    # get paths for graphs_prime
    # gather all paths from nodes of interest, keep only routes to control nodes
    # gt_prime
    tt = time.time()
    all_pairs_lengths_gt_prime = {}
    if verbose:
        print("Computing all_pairs_lengths_gt_prime...")
    G_gt_cp_prime_nodes_set = set(G_gt_cp_prime.nodes())
#    for source in G_gt_cp_prime_nodes_set:
#        if source in G_gt_cp_prime_nodes_set:
#            paths_tmp = nx.single_source_dijkstra_path_length(G_gt_cp_prime, source, weight=weight)
    for itmp, source in enumerate(rand_nodes_p_set):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        if source in G_gt_cp_prime_nodes_set:
            paths_tmp = nx.single_source_dijkstra_path_length(
                G_gt_cp_prime, source, weight=weight)
            # delete items
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_p_set:
                    del paths_tmp[k]
            all_pairs_lengths_gt_prime[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    # prop_prime
    tt = time.time()
    all_pairs_lengths_prop_prime = {}
    if verbose:
        print("Computing all_pairs_lengths_prop_prime...")
    G_p_cp_prime_nodes_set = set(G_p_cp_prime.nodes())
#    for source in G_p_cp_prime_nodes_set:
#        if source in G_p_cp_prime_nodes_set:
#            paths_tmp = nx.single_source_dijkstra_path_length(G_p_cp_prime, source, weight=weight)
    for itmp, source in enumerate(rand_nodes_gt_set):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        if source in G_p_cp_prime_nodes_set:
            paths_tmp = nx.single_source_dijkstra_path_length(
                G_p_cp_prime, source, weight=weight)
            # delete items
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_gt_set:
                    del paths_tmp[k]
            all_pairs_lengths_prop_prime[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    #all_pairs_lengths_gt_prime = nx.all_pairs_dijkstra_path_length(G_gt_cp_prime, weight=weight)
    #all_pairs_lengths_prop_prime = nx.all_pairs_dijkstra_path_length(G_p_cp_prime, weight=weight)

    ###############
    tf = time.time()
    print("Time to run make_graphs_yuge in apls.py:", tf - t0, "seconds")

    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
        control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime


###############################################################################
def single_path_metric(len_gt, len_prop, diff_max=1):
    """
    Compute APLS metric for single path.

    Notes
    -----
    Compute normalize path difference metric, if len_prop < 0, return diff_max

    Arguments
    ---------
    len_gt : float
        Length of ground truth edge.
    len_prop : float
        Length of proposal edge.
    diff_max : float
        Maximum value to return. Defaults to ``1``.

    Returns
    -------
    metric : float
        Normalized path difference.
    """

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
                    diff_max=1, missing_path_len=-1, normalize=True,
                    verbose=False):
    """
    Compute metric for multiple paths.

    Notes
    -----
    Assume nodes in ground truth and proposed graph have the same names.
    Assume graph is undirected so don't evaluate routes in both directions
    control_nodes is the list of nodes to actually evaluate; if empty do all
        in all_pairs_lenghts_gt
    min_path_length is the minimum path length to evaluate
    https://networkx.github.io/documentation/networkx-2.2/reference/algorithms/shortest_paths.html

    Parameters
    ----------
    all_pairs_lengths_gt : dict
        Dictionary of path lengths for ground truth graph.
    all_pairs_lengths_prop : dict
        Dictionary of path lengths for proposal graph.
    control_nodes : list
        List of control nodes to evaluate.
    min_path_length : float
        Minimum path length to evaluate.
    diff_max : float
        Maximum value to return. Defaults to ``1``.
    missing_path_len : float
        Value to assign a missing path.  Defaults to ``-1``.
    normalize : boolean
        Switch to normalize outputs. Defaults to ``True``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    C, diffs, routes, diff_dic
        C is the APLS score
        diffs is a list of the the route differences
        routes is a list of routes
        diff_dic is a dictionary of path differences
    """

    diffs = []
    routes = []
    diff_dic = {}
    gt_start_nodes_set = set(all_pairs_lengths_gt.keys())
    prop_start_nodes_set = set(all_pairs_lengths_prop.keys())
    t0 = time.time()

    print()
    if len(gt_start_nodes_set) == 0:
        return 0, [], [], {}

    # set nodes to inspect
    if len(control_nodes) == 0:
        good_nodes = list(all_pairs_lengths_gt.keys())
    else:
        good_nodes = control_nodes

    if verbose:
        print("\nComputing path_sim_metric()...")
        print("good_nodes:", good_nodes)

    # iterate overall start nodes
    # for start_node, paths in all_pairs_lengths.iteritems():
    for start_node in good_nodes:
        if verbose:
            print("start node:", start_node)
        node_dic_tmp = {}

        # if we are not careful with control nodes, it's possible that the
        # start node will not be in all_pairs_lengths_gt, in this case use max
        # diff for all routes to that node
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to that node
        if start_node not in gt_start_nodes_set:
            print("for ss, node", start_node, "not in set")
            print("   skipping N paths:", len(
                list(all_pairs_lengths_prop[start_node].keys())))
            for end_node, len_prop in all_pairs_lengths_prop[start_node].items():
                diffs.append(diff_max)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff_max
            return

        paths = all_pairs_lengths_gt[start_node]

        # CASE 1
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to the start node
        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.items():
                if (end_node != start_node) and (end_node in good_nodes):
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
                    node_dic_tmp[end_node] = diff_max
            diff_dic[start_node] = node_dic_tmp
            # print ("start_node missing:", start_node)
            continue

        # else get proposed paths
        else:
            paths_prop = all_pairs_lengths_prop[start_node]

            # get set of all nodes in paths_prop, and missing_nodes
            end_nodes_gt_set = set(paths.keys()).intersection(good_nodes)
            # end_nodes_gt_set = set(paths.keys()) # old version with all nodes

            end_nodes_prop_set = set(paths_prop.keys())
            missing_nodes = end_nodes_gt_set - end_nodes_prop_set
            if verbose:
                print("missing nodes:", missing_nodes)

            # iterate over all paths from node
            for end_node in end_nodes_gt_set:
                # for end_node, len_gt in paths.iteritems():

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
                    print("end_node:", end_node)
                    print("   len_gt:", len_gt)
                    print("   len_prop:", len_prop)

                # compute path difference metric
                diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
                diffs.append(diff)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff

            diff_dic[start_node] = node_dic_tmp

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

    print("Time to compute metric (score = ", C, ") for ", len(diffs),
          "routes:", time.time() - t0, "seconds")

    return C, diffs, routes, diff_dic


###############################################################################
def compute_apls_metric(all_pairs_lengths_gt_native,
                        all_pairs_lengths_prop_native,
                        all_pairs_lengths_gt_prime,
                        all_pairs_lengths_prop_prime,
                        control_points_gt, control_points_prop,
                        res_dir='', min_path_length=10,
                        verbose=False, super_verbose=False):
    """
    Compute APLS metric and plot results (optional)

    Notes
    -----
    Computes APLS and creates plots in res_dir (if it is not empty)

    Arguments
    ---------
    all_pairs_lengths_gt_native : dict
        Dict of paths for gt graph.
    all_pairs_lengths_prop_native : dict
        Dict of paths for prop graph.
    all_pairs_lengths_gt_prime : dict
        Dict of paths for gt graph with control points from prop.
    all_pairs_lengths_prop_prime : dict
        Dict of paths for prop graph with control points from gt.
    control_points_gt : list
        Array of control points.
    control_points_prop : list
        Array of control points.
    res_dir : str
        Output dir for plots.  Defaults to ``''`` (no plotting).
    min_path_length : float
        Minimum path length to evaluate.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    C_tot, C_gt_onto_prop, C_prop_onto_gt : tuple
        C_tot is the total APLS score
        C_gt_onto_prop is the score when inserting gt control nodes onto prop
        C_prop_onto_gt is the score when inserting prop control nodes onto gt
    """

    t0 = time.time()

    # return 0 if no paths
    if (len(list(all_pairs_lengths_gt_native.keys())) == 0) \
            or (len(list(all_pairs_lengths_prop_native.keys())) == 0):
        print("len(all_pairs_lengths_gt_native.keys()) == 0)")
        return 0, 0, 0

    ####################
    # compute metric (gt to prop)
    # if verbose:
    print("Compute metric (gt snapped onto prop)")
    # control_nodes = all_pairs_lengths_gt_native.keys()
    control_nodes = [z[0] for z in control_points_gt]
    if verbose:
        print(("control_nodes_gt:", control_nodes))
    C_gt_onto_prop, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_gt_native,
        all_pairs_lengths_prop_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)
    dt1 = time.time() - t0
    if verbose:
        print("len(diffs):", len(diffs))
        if len(diffs) > 0:
            print("  max(diffs):", np.max(diffs))
            print("  min(diffs)", np.min(diffs))
    if len(res_dir) > 0:
        scatter_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_gt_to_prop.png')
        hist_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_hist_gt_to_prop.png')
        # can't plot route names if there are too many...
        if len(routes) > 100:
            routes_str = []
        else:
            routes_str = [str(z[0]) + '-' + str(z[1]) for z in routes]
        apls_plots.plot_metric(
            C_gt_onto_prop, diffs, routes_str=routes_str,
            figsize=(10, 5), scatter_alpha=0.8, scatter_size=8,
            scatter_png=scatter_png,
            hist_png=hist_png)
    ######################

    ####################
    # compute metric (prop to gt)
    # if verbose:
    print("Compute metric (prop snapped onto gt)")
    t1 = time.time()
    # control_nodes = all_pairs_lengths_prop_native.keys()
    control_nodes = [z[0] for z in control_points_prop]
    if verbose:
        print("control_nodes:", control_nodes)
    C_prop_onto_gt, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)
    dt2 = time.time() - t1
    if verbose:
        print("len(diffs):", len(diffs))
        if len(diffs) > 0:
            print("  max(diffs):", np.max(diffs))
            print("  min(diffs)", np.min(diffs))
    if len(res_dir) > 0:
        scatter_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_prop_to_gt.png')
        hist_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_hist_prop_to_gt.png')
        if len(routes) > 100:
            routes_str = []
        else:
            routes_str = [str(z[0]) + '-' + str(z[1]) for z in routes]
        apls_plots.plot_metric(
            C_prop_onto_gt, diffs, routes_str=routes_str,
            figsize=(10, 5), scatter_alpha=0.8, scatter_size=8,
            scatter_png=scatter_png,
            hist_png=hist_png)

    ####################

    ####################
    # Total

    print("C_gt_onto_prop, C_prop_onto_gt:", C_gt_onto_prop, C_prop_onto_gt)
    if (C_gt_onto_prop <= 0) or (C_prop_onto_gt <= 0) \
            or (np.isnan(C_gt_onto_prop)) or (np.isnan(C_prop_onto_gt)):
        C_tot = 0
    else:
        C_tot = scipy.stats.hmean([C_gt_onto_prop, C_prop_onto_gt])
        if np.isnan(C_tot):
            C_tot = 0
    print("Total APLS Metric = Mean(", np.round(C_gt_onto_prop, 2), "+",
          np.round(C_prop_onto_gt, 2),
          ") =", np.round(C_tot, 2))
    print("Total time to compute metric:", str(dt1 + dt2), "seconds")

    return C_tot, C_gt_onto_prop, C_prop_onto_gt


###############################################################################
def gather_files(test_method, truth_dir, prop_dir,
                 im_dir='',
                 im_prefix='',
                 wkt_file='',
                 max_files=1000,
                 gt_subgraph_filter_weight='length',
                 gt_min_subgraph_length=5,
                 prop_subgraph_filter_weight='length_pix',
                 prop_min_subgraph_length=10,
                 use_pix_coords=True,
                 speed_key='speed_m/s',
                 travel_time_key='travel_time',
                 default_speed=13.41,
                 verbose=False, super_verbose=False):
    """
    Build lists of ground truth and proposal graphs

    Arguments
    ---------
    test_method : str
        Method for identifying and extraction graphs.
        Options:
            gt_pkl_prop_pkl   = ground truth pickle, proposal pickle
            gt_json_prop_pkl  = ground truth json, proposal pickle
            gt_json_prop_wkt  = ground truth json, proposal csv in wkt format
            gt_json_prop_json = ground truth json, proposal json
    truth_dir : str
        Location of ground truth graphs.
    prop_dir : str
        Location of proposal graphs.
    im_dir : str
        Location of image files.  Defaults to ``''``.
    im_prefix : str
        Prefix to prepend to image files. Defaults to ``''``.
    wkt_file : str
        Location of wkt file if analyzing wkt predictions. Defaults to ``''``.
    max_files : int
        Maximum number of files to analyze. Defaults to ``1000``.
    gt_subgraph_filter_weight : str
        Edge key for filtering ground truth edge length.
        Defaults to ``'length'``.
    gt_min_subgraph_length : float
        Minimum length of the edge. Defaults to ``5``.
    prop_subgraph_filter_weight : str
        Edge key for filtering proposal edge length.
        Defaults to ``'length_pix'``.
    prop_min_subgraph_length : float
        Minimum length of the edge. Defaults to ``10``.
    use_pix_coords : boolean
        Switch to use pixel coords for calculations. Defaults to ``True``.
    speed_key : str
        Edge key for speed. Defaults to ``'speed_m/s'``.
    travel_time_key : str
        Edge key for travel time. Defaults to ``'travel_time'``.
    default_speed : float
        Default speed for edge in m/s. Defaults to ``13.41`` (30 mph).
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    gt_list, gt_raw_list, gp_list, root_list, im_loc_list : tuple
        gt_list is a list of ground truth graphs.
        gp_list is a list of proposal graphs
        root_list is a list of names
        im_loc_list is the location of the images corresponding to root_list
    """

    ###################
    # Get ground truth and proposal graphs
    print("Gathering files...")

    ###################
    gt_list, gp_list, root_list, im_loc_list = [], [], [], []
    # gt_raw_list = []

    ###################
    # import a ground truth geojson and a pickled propoal graph
    if test_method == 'gt_pkl_prop_pkl':

        name_list = np.sort(os.listdir(truth_dir))
        for i, f in enumerate(name_list):
            if i >= max_files:
                break

            # skip non-pkl files
            if not f.endswith(('.gpickle', '.pkl')):
                continue

            # define values
            outroot = f.split('.')[0]
            print(("outroot:", outroot))
            gt_file = os.path.join(truth_dir, f)
            prop_file = os.path.join(prop_dir, outroot + '.gpickle')
            im_file = os.path.join(im_dir, outroot + '.tif')
            # im_file = os.path.join(im_dir, im_prefix + outroot + '.tif')
            # Naming convention is inconsistent
            if not os.path.exists(prop_file):
                continue

            # ground truth
            G_gt_init = nx.read_gpickle(gt_file)
            edge_tmp = list(G_gt_init.edges())[-1]
            print("gt random edge props for edge:", edge_tmp, " = ",
                  G_gt_init.edges[edge_tmp[0], edge_tmp[1], 0])
            # for i,(u,v,attr_dict) in enumerate(G_gt_init.edges(data=True)):
            #    print "\n\natrr_dictk.keys():", attr_dict.keys()
            #    print ("  attr_dict;", attr_dict)
            # return

            # proposal
            G_p_init = nx.read_gpickle(prop_file)
            edge_tmp = list(G_p_init.edges())[-1]
            print("prop random edge props for edge:", edge_tmp, " = ",
                  G_p_init.edges[edge_tmp[0], edge_tmp[1], 0])

#            #########################
#            # OPTIONAL - simplify propposal
#            G_p_init1 = osmnx_funcs.simplify_graph(G_p_init.to_directed())
#            G_p_init2 = G_p_init1.to_undirected()
#            G_p_init = osmnx_funcs.project_graph(G_p_init2)
#            G_p_init = create_edge_linestrings(
#                G_p_init, remove_redundant=True, verbose=False)
#            # for i,(u,v,attr_dict) in enumerate(G_p_init.edges(data=True)):
#            #    print "\n\n", u, v, "atrr_dictk.keys():", attr_dict.keys()
#            #    print ("  attr_dict;", attr_dict)
#            #    print ("  attr_dict[geometry];", attr_dict['geometry'])
#            #########################

            # append to lists
            gt_list.append(G_gt_init)
            # gt_raw_list.append('')
            gp_list.append(G_p_init)
            root_list.append(outroot)
            im_loc_list.append(im_file)

    # use ground truth spacenet geojsons, and submission pkl files
    elif test_method == 'gt_json_prop_pkl':

        valid_road_types = set([])   # assume no road type in geojsons
        name_list = sorted(os.listdir(truth_dir))
        for i, f in enumerate(name_list):
            # skip non-geojson files
            if not f.endswith('.geojson'):
                continue

            if i >= max_files:
                break

            # define values
            outroot = f.split('spacenetroads_')[-1].split('.')[0]
            # if verbose:
            print("\n", i, "outroot:", outroot)
            gt_file = os.path.join(truth_dir, f)
            im_file = os.path.join(im_dir, im_prefix + outroot + '.tif')
            prop_file = os.path.join(
                prop_dir, im_prefix + outroot + '.gpickle')
            if not os.path.exists(prop_file):
                print("prop file DNE, skipping:", prop_file)
                continue

            #########
            # ground truth
            osmidx, osmNodeidx = 10000, 10000
            # G_gt_init, G_gt_cp, control_points, gt_graph_coords, midpoints_gt = \
            G_gt_init, G_gt_raw = \
                _create_gt_graph(gt_file, im_file, network_type='all_private',
                                 valid_road_types=valid_road_types,
                                 subgraph_filter_weight=gt_subgraph_filter_weight,
                                 min_subgraph_length=gt_min_subgraph_length,
                                 use_pix_coords=use_pix_coords,
                                 osmidx=osmidx, osmNodeidx=osmNodeidx,
                                 speed_key=speed_key,
                                 travel_time_key=travel_time_key,
                                 verbose=verbose)
            # skip empty ground truth graphs
            if len(G_gt_init.nodes()) == 0:
                continue
            if verbose:
                # print a node
                node = list(G_gt_init.nodes())[-1]
                print(node, "gt random node props:", G_gt_init.nodes[node])
                # print an edge
                edge_tmp = list(G_gt_init.edges())[-1]
                # G.edge[edge_tmp[0]][edge_tmp[1]])
                print("gt random edge props for edge:", edge_tmp, " = ",
                      G_gt_init.edges[edge_tmp[0], edge_tmp[1], 0])

            # # # optional: save to pickle
            # outpickle = 'tmp.gpickle'
            # nx.write_gpickle(G_gt_init, outpickle)

            #########
            # proposal
            # print ("load proposal...")
            G_p_init = nx.read_gpickle(prop_file)
            # print a few values
            if verbose:
                # print a node
                try:
                    node = list(G_p_init.nodes())[-1]
                    print(node, "prop random node props:",
                          G_p_init.nodes[node])
                    # print an edge
                    edge_tmp = list(G_p_init.edges())[-1]
                    # G.edge[edge_tmp[0]][edge_tmp[1]])
                    print("prop random edge props for edge:", edge_tmp,
                          " = ", G_p_init.edges[edge_tmp[0], edge_tmp[1], 0])
                    # print (edge_tmp, "random edge props:", G_p_init.edges([edge_tmp[0], edge_tmp[1]])) #G.edge[edge_tmp[0]][edge_tmp[1]])
                except:
                    print("Empty proposal graph")

            # append to lists
            gt_list.append(G_gt_init)
            # gt_raw_list.append(G_gt_raw)
            gp_list.append(G_p_init)
            root_list.append(outroot)
            im_loc_list.append(im_file)

    ###################
    # use ground truth spacenet geojsons, and submission wkt files
    elif test_method == 'gt_json_prop_wkt':

        name_list = os.listdir(truth_dir)
        for i, f in enumerate(sorted(name_list)):
            # skip non-geojson files
            if not f.endswith('.geojson'):
                continue

            if i >= max_files:
                break

            # define values
            outroot = f.split('spacenetroads_')[-1].split('.')[0]
            if verbose:
                print("\n", i, "outroot:", outroot)
            gt_file = os.path.join(truth_dir, f)
            im_file = os.path.join(im_dir, im_prefix + outroot + '.tif')
            # print("im_dir:", im_dir)
            # print("im_prefix:", im_prefix)
            # print("outroot:", outroot)
            valid_road_types = set([])   # assume no road type in geojsons

            #########
            # ground truth
            osmidx, osmNodeidx = 0, 0
            G_gt_init, G_gt_raw = \
                _create_gt_graph(gt_file, im_file, network_type='all_private',
                                 # linestring_delta=args.linestring_delta,
                                 # is_curved_eps=args.is_curved_eps,
                                 valid_road_types=valid_road_types,
                                 subgraph_filter_weight=gt_subgraph_filter_weight,
                                 min_subgraph_length=gt_min_subgraph_length,
                                 use_pix_coords=use_pix_coords,
                                 osmidx=osmidx,
                                 osmNodeidx=osmNodeidx,
                                 speed_key=speed_key,
                                 travel_time_key=travel_time_key,
                                 verbose=verbose)
            # skip empty ground truth graphs
            if len(G_gt_init.nodes()) == 0:
                continue
            if verbose:
                # print a node
                node = list(G_gt_init.nodes())[-1]
                print(node, "random node props:", G_gt_init.nodes[node])
                # print an edge
                edge_tmp = list(G_gt_init.edges())[-1]
                # G.edge[edge_tmp[0]][edge_tmp[1]])
                print("random edge props for edge:", edge_tmp, " = ",
                      G_gt_init.edges[edge_tmp[0], edge_tmp[1], 0])
                
            #########
            print("Retrieve proposal graph...")
            # proposal
            # adapted from wkt_to_G.main()
            # read in wkt list
            df_wkt = pd.read_csv(wkt_file)
            # columns=['ImageId', 'WKT_Pix'])
            image_id = outroot

            # filter
            df_filt = df_wkt['WKT_Pix'][df_wkt['ImageId'] == image_id]
            wkt_list = df_filt.values

            # print a few values
            if verbose:
                #print ("\n", i, "/", len(image_ids), "num linestrings:", len(wkt_list))
                print("image_file:", im_file, "wkt_list[:2]", wkt_list[:2])

            if (len(wkt_list) == 0) or (wkt_list[0] == 'LINESTRING EMPTY'):
                continue

            t1 = time.time()
            node_iter, edge_iter = 1000, 1000
            G_p_init0 = wkt_to_G.wkt_to_G(
                wkt_list, im_file=im_file,
                prop_subgraph_filter_weight=prop_subgraph_filter_weight,
                min_subgraph_length=prop_min_subgraph_length,
                node_iter=node_iter,
                edge_iter=edge_iter,
                verbose=super_verbose)

            t2 = time.time()
            # add travel time
            G_p_init = add_travel_time(G_p_init0, speed_key=speed_key,
                                       travel_time_key=travel_time_key,
                                       default_speed=default_speed)
            if verbose:
                print("Time to create graph:", t2-t1, "seconds")
                # print a node
                node = list(G_p_init.nodes())[-1]
                print(node, "random node props:", G_p_init.nodes[node])
                # print an edge
                edge_tmp = list(G_p_init.edges())[-1]
                # G.edge[edge_tmp[0]][edge_tmp[1]])
                print("random edge props for edge:", edge_tmp, " = ",
                      G_p_init.edges[edge_tmp[0], edge_tmp[1], 0])

            # append to lists
            gt_list.append(G_gt_init)
            # gt_raw_list.append(G_gt_raw)
            gp_list.append(G_p_init)
            root_list.append(outroot)
            im_loc_list.append(im_file)

    ###################
    # ingest multiple ground truth and propoal geojsons in a folder
    if test_method == 'gt_json_prop_json':

        name_list = os.listdir(truth_dir)
        for f in name_list:
            # skip non-geojson files
            if not f.endswith('.geojson'):
                continue

            # define values
            outroot = f.split('.')[0]
            print("\n\noutroot:", outroot)
            gt_file = os.path.join(truth_dir, f)
            prop_file = os.path.join(prop_dir, outroot + '.geojson')
            # Naming convention is inconsistent
            if not os.path.exists(prop_file):
                prop_file = os.path.join(prop_dir, outroot + 'prop.geojson')
            im_file = ''
            valid_road_types = set([])   # assume no road type in geojsons

            # ground truth
            osmidx, osmNodeidx = 0, 0
            G_gt_init, G_gt_raw = \
                _create_gt_graph(gt_file, im_file, network_type='all_private',
                                 valid_road_types=valid_road_types,
                                 use_pix_coords=use_pix_coords,
                                 osmidx=osmidx,
                                 osmNodeidx=osmNodeidx,
                                 speed_key=speed_key,
                                 travel_time_key=travel_time_key,
                                 verbose=verbose)

            # skip empty ground truth graphs
            if len(G_gt_init.nodes()) == 0:
                continue

            # proposal
            osmidx, osmNodeidx = 500, 500
            G_p_init, G_p_raw = \
                _create_gt_graph(prop_file, im_file,
                                 network_type='all_private',
                                 # linestring_delta=args.linestring_delta,
                                 # is_curved_eps=args.is_curved_eps,
                                 valid_road_types=valid_road_types,
                                 use_pix_coords=use_pix_coords,
                                 osmidx=osmidx,
                                 osmNodeidx=osmNodeidx,
                                 verbose=verbose)
            # append to lists
            gt_list.append(G_gt_init)
            # gt_raw_list.append(G_gt_raw)
            gp_list.append(G_p_init)
            root_list.append(outroot)

    return gt_list, gp_list, root_list, im_loc_list


###############################################################################
def execute(output_name, gt_list, gp_list, root_list, im_loc_list=[],
            weight='length',
            speed_key='speed_m/s',
            travel_time_key='travel_time',
            max_files=1000,
            linestring_delta=50,
            is_curved_eps=10**3,
            max_snap_dist=4,
            max_nodes=500,
            n_plots=10,
            min_path_length=10,
            topo_hole_size=4,
            topo_subgraph_radius=150,
            topo_interval=30,
            sp_length_buffer=0.05,
            use_pix_coords=False,
            allow_renaming=True,
            verbose=True,
            super_verbose=False):
    """
    Compute APLS for the input data in gt_list, gp_list

    Arguments
    ---------
    output_name : str
        Output name in apls/outputs
    weight : str
        Edge key determining path length weights. Defaults to ``'length'``.
    speed_key : str
        Edge key for speed. Defaults to ``'speed_m/s'``.
    travel_time_key : str
        Edge key for travel time. Defaults to ``'travel_time'``.
    max_files : int
        Maximum number of files to analyze. Defaults to ``1000``.
    linestring_delta : float
        Distance in meters between linestring midpoints. Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.  Defaults to ``0.3``.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    max_nodes : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    n_plots : int
        Number of graphs to create plots for. Defaults to ``10``.
    min_path_length : float
        Mimumum path length to consider for APLS. Defaults to ``10``.
    topo_hole_size : float
        Hole size for TOPO in meters. Defaults to ``4``.
    topo_subgraph_radius : float
        Radius to search for TOPO, in meters. Defaults to ``150.``
    topo_interval : float
        Spacing of points in meters to inject for TOPO. Defaults to ``30``.
    sp_length_buffer : float
        Fractional difference in lengths for SP metric. Defaults to ``0.05``.
    use_pix_coords : boolean
        Switch to use pixel coords for colculating lengths.
        Defaults to ``False``.
    allow_renaming : boolean
        Switch to rename nodes when injecting nodes into graphs.
        Defaulst to ``True``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    None
    """

    # now compute results
    print("\n\n\nCompute Results...")
    C_arr = [["outroot", "APLS", "APLS_gt_onto_prop", "APLS_prop_onto_gt",
              "topo_tp_tot", "topo_fp_tot", "topo_fn_tot", "topo_precision",
              "topo_recall", "topo_f1",
              "sp_metric", "tot_meters_gt", "tot_meters_p"]]

    # plot settings
    title_fontsize = 4
    dpi = 200
    show_plots = False
    show_node_ids = True
    fig_height, fig_width = 6, 6
    # path settings
    route_linewidth = 4
    source_color = 'red'
    target_color = 'green'
    # if using create_gt_graph, use the following road types
    #   (empty set uses all road types)
    # https://wiki.openstreetmap.org/wiki/Key:highway
    valid_road_types = set([])
    # valid_road_types = set(['motorway', 'trunk', 'primary', 'secondary',
    #                           'tertiary',
    #                        'motorway_link', 'trunk_link', 'primary_link',
    #                           'secondary_link', 'tertiary_link',
    #                        'unclassified', 'residential', 'service'])
    ###################

    ##################
    # make dirs
    outdir_base = os.path.join(path_apls, 'outputs')
    print ("Outdir base:", outdir_base)
    outdir_base2 = os.path.join(outdir_base, str(output_name),
                                'weight=' + str(weight))
    print ("Outdir with weight:", outdir_base2)
    d_list = [outdir_base, outdir_base2]
    for p in d_list:
        if not os.path.exists(p):
            os.makedirs(p)

    ##################
    t0 = time.time()
    for i, [outroot, G_gt_init, G_p_init] in enumerate(zip(root_list, gt_list, gp_list)):

        if i >= max_files:
            break

        # copy image file to output dir, if desired
        if len(im_loc_list) > 0:
            # print ("im_loc_list:", im_loc_list)
            im_loc = im_loc_list[i]
        else:
            im_loc = ''

        print("\n\n\n", i+1, "/", len(root_list), "Computing:", outroot)
        t1 = time.time()

        # print a few properties
        print("len(G_gt_init.nodes():)", len(G_gt_init.nodes()))
        print("len(G_gt_init.edges():)", len(G_gt_init.edges()))
        print("len(G_p_init.nodes():)", len(G_p_init.nodes()))
        print("len(G_p_init.edges():)", len(G_p_init.edges()))

        ##################
        # make dirs
        outdir_base = os.path.join(path_apls, 'outputs')
        outdir_base2 = os.path.join(
            outdir_base, output_name, 'weight=' + weight)
        outdir = os.path.join(outdir_base2, outroot)
        print("output dir:", outdir)
        os.makedirs(outdir, exist_ok=True)
        # d_list = [outdir_base, outdir_base2, outdir]
        # for p in d_list:
        #    #if not os.path.exists(p) and make_plots:
        #    if not os.path.exists(p):
        #        os.makedirs(p)
        ##################

        # get graphs with midpoints and geometry (if small graph)
        print("\nMake gt, prop graphs...")
        if len(G_gt_init.nodes()) < 500:  # 2000:
            G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
                control_points_gt, control_points_prop, \
                all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
                all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime  \
                = make_graphs(G_gt_init, G_p_init,
                              weight=weight,
                              speed_key=speed_key,
                              travel_time_key=travel_time_key,
                              linestring_delta=linestring_delta,
                              is_curved_eps=is_curved_eps,
                              max_snap_dist=max_snap_dist,
                              allow_renaming=allow_renaming,
                              verbose=verbose)

        # get large graphs and paths
        else:
            G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
                control_points_gt, control_points_prop, \
                all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
                all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime  \
                = make_graphs_yuge(G_gt_init, G_p_init,
                                   weight=weight,
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key,
                                   max_nodes=max_nodes,
                                   max_snap_dist=max_snap_dist,
                                   allow_renaming=allow_renaming,
                                   verbose=verbose,
                                   super_verbose=super_verbose)

        if verbose:
            print("\nlen control_points_gt:", len(control_points_gt))
            # print "control_points_gt:", control_points_gt
            if len(G_gt_init.nodes()) < 200:
                print("G_gt_init.nodes():", G_gt_init.nodes())
            print("len G_gt_init.edges():", len(G_gt_init.edges()))
            # for e in G_gt_init.edges():
            #    print "  G_gt_init edge:", e, G_gt_init.edges[e[0]][e[1]][0]['length']
            if len(G_gt_cp.nodes()) < 200:
                print("G_gt_cp.nodes():", G_gt_cp.nodes())
            print("len G_gt_cp.nodes():", len(G_gt_cp.nodes()))
            print("len G_gt_cp.edges():", len(G_gt_cp.edges()))
            # for e in G_gt_cp.edges():
            #    print "  G_gt_cp edge:", e, G_gt_cp.edges[e[0]][e[1]][0]['length']
            print("len G_gt_cp_prime.nodes():", len(G_gt_cp_prime.nodes()))
            print("len G_gt_cp_prime.edges():", len(G_gt_cp_prime.edges()))

            print("\nlen control_points_prop:", len(control_points_prop))
            # print "control_points_prop:", control_points_prop
            if len(G_p_init.nodes()) < 200:
                print("G_p_init.nodes():", G_p_init.nodes())
            print("len G_p_init.edges():", len(G_p_init.edges()))
            if len(G_p_cp.nodes()) < 200:
                print("G_p_cp.nodes():", G_p_cp.nodes())
            print("len G_p_cp.nodes():", len(G_p_cp.nodes()))
            print("len G_p_cp.edges():", len(G_p_cp.edges()))

            print("len G_p_cp_prime.nodes():", len(G_p_cp_prime.nodes()))
            if len(G_p_cp_prime.nodes()) < 200:
                print("G_p_cp_prime.nodes():", G_p_cp_prime.nodes())
            print("len G_p_cp_prime.edges():", len(G_p_cp_prime.edges()))
            # print("G_p_cp_prime.edges():", G_p_cp_prime.edges())

            print("len all_pairs_lengths_gt_native:",
                  len(dict(all_pairs_lengths_gt_native)))
            # for ktmp,vtmp in all_pairs_lengths_gt_native.iteritems():
            #    print ("  key:", ktmp, "len(all_pairs_lengths_gt_native[key]):", len(vtmp))
            print("len all_pairs_lengths_gt_prime:",
                  len(dict(all_pairs_lengths_gt_prime)))
            # for ktmp,vtmp in all_pairs_lengths_gt_prime.iteritems():
            #    print ("  key:", ktmp, "len(all_pairs_lengths_gt_prime[key]):", len(vtmp))
            print("len all_pairs_lengths_prop_native",
                  len(dict(all_pairs_lengths_prop_native)))
            # for ktmp,vtmp in all_pairs_lengths_prop_native.iteritems():
            #    print ("  key:", ktmp, "len(all_pairs_lengths_prop_native[key]):", len(vtmp))
            print("len all_pairs_lengths_prop_prime",
                  len(dict(all_pairs_lengths_prop_prime)))
            # for ktmp,vtmp in all_pairs_lengths_prop_prime.iteritems():
            #    print ("  key:", ktmp, "len(all_pairs_lengths_prop_prime[key]):", len(vtmp))

        #########################
        # Metric
        if i < n_plots:
            res_dir = outdir
        else:
            res_dir = ''
        C, C_gt_onto_prop, C_prop_onto_gt = compute_apls_metric(
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
            control_points_gt, control_points_prop,
            min_path_length=min_path_length,
            verbose=verbose, res_dir=res_dir)
        print("APLS Metric = ", C)

        print("\nComputing TOPO Metric...")
        n_measurement_nodes = max_nodes
        topo_vals = topo_metric.compute_topo(
            G_gt_init, G_p_init,
            subgraph_radius=topo_subgraph_radius,
            interval=topo_interval,
            hole_size=topo_hole_size,
            n_measurement_nodes=n_measurement_nodes,
            x_coord='x', y_coord='y',
            allow_multi_hole=False,
            make_plots=False, verbose=False)
        topo_tp_tot, topo_fp_tot, topo_fn_tot, topo_precision, topo_recall, topo_f1 = topo_vals
        print("TOPO Metric subgraph_radius, interval:",
              topo_subgraph_radius, topo_interval)
        print("TOPO Metric =", topo_vals, "for", n_measurement_nodes,
              "nodes, subgraph_radius =", topo_subgraph_radius)

        print("\nComputing sp Metric...")
        sp_n_routes = max_nodes
        _, sp = sp_metric.compute_sp(
            G_gt_init, G_p_init,
            x_coord='x', y_coord='y',
            weight=weight, query_radius=max_snap_dist,
            length_buffer=sp_length_buffer, n_routes=sp_n_routes,
            verbose=False, make_plots=False)
        print("sp_length_buffer:", sp_length_buffer)
        print("sp Metric =", sp, "for", sp_n_routes,
              "routes, length buffer =", sp_length_buffer)

        # get total length of edges
        # ground truth
        tot_meters_gt = 0
        for itmp, (u, v, attr_dict) in enumerate(G_gt_init.edges(data=True)):
            tot_meters_gt += attr_dict['length']
        print("Ground truth total length of edges (km):", tot_meters_gt/1000)
        G_gt_init.graph['Tot_edge_km'] = tot_meters_gt/1000
        tot_meters_p = 0
        for itmp, (u, v, attr_dict) in enumerate(G_p_init.edges(data=True)):
            tot_meters_p += attr_dict['length']
        print("Proposal total length of edges (km):", tot_meters_p/1000)
        G_p_init.graph['Tot_edge_km'] = tot_meters_p/1000

        # save scores
        f = open(os.path.join(outdir, 'output__max_snap='
                              + str(np.round(max_snap_dist, 2)) + 'm'
                              + '_hole='
                              + str(np.round(topo_hole_size, 2)) + 'm'
                              + '.txt'), 'w')
        f.write("Ground Truth Nodes Snapped Onto Proposal Score: " +
                str(C_gt_onto_prop) + "\n")
        f.write("Proposal Nodes Snapped Onto Ground Truth Score: " +
                str(C_prop_onto_gt) + "\n")
        f.write("Total APLS Score: " + str(C) + "\n")
        f.write("TOPO vals - topo_tp_tot, topo_fp_tot, topo_fn_tot, topo_precision, topo_recall, topo_f1: " + str(topo_vals) + "\n")
        f.write("SP: " + str(sp))
        f.close()

        t2 = time.time()
        print("Total time to create graphs and compute metric:",
              t2-t1, "seconds")
        C_arr.append([outroot, C, C_gt_onto_prop, C_prop_onto_gt,
                      # 0, 0, 0, 0, 0, 0,
                      topo_tp_tot, topo_fp_tot, topo_fn_tot, topo_precision, topo_recall, topo_f1,
                      # 0,
                      sp,
                      tot_meters_gt, tot_meters_p])

        ##################
        # PLOTS
        if i < n_plots:

            # # plot init0
            # if test_method == 'osmnx':
            #    fig, ax = osmnx_funcs.plot_graph(G0, show=show_plots, close=False,
            #                            fig_height=fig_height,
            #                            fig_width=fig_width)
            #    ax.set_title('Raw Ground Truth Graph', fontsize=title_fontsize)
            #    plt.savefig(os.path.join(outdir, 'gt_graph_raw.png'), dpi=dpi)
            #    plt.close('all')

            # skip plots if no nodes
            if (len(G_gt_cp.nodes()) == 0) or (len(G_p_cp.nodes()) == 0):
                continue

            # set graph size
            max_extent = max(fig_height, fig_width)
            xmin, xmax, ymin, ymax, dx, dy = apls_utils._get_graph_extent(
                G_gt_cp)
            if dx <= dy:
                fig_height = max_extent
                fig_width = max(1, 1. * max_extent * dx / dy)
            else:
                fig_width = max_extent
                fig_height = max(1, 1. * max_extent * dy / dx)
            if verbose:
                print("fig_width, fig_height:", fig_width, fig_height)

            # plot ground truth
            fig, ax = osmnx_funcs.plot_graph(
                G_gt_init, show=show_plots, close=False,
                fig_height=fig_height, fig_width=fig_width)
            if show_node_ids:
                ax = apls_plots.plot_node_ids(
                    G_gt_init, ax, fontsize=4)  # node ids
            ax.set_title('Ground Truth Graph', fontsize=title_fontsize)
            # plt.show()
            plt.savefig(os.path.join(outdir, 'gt_graph.png'), dpi=dpi)
            # plt.clf()
            # plt.cla()
            plt.close('all')

            #  gt midpoints
            fig0, ax0 = osmnx_funcs.plot_graph(
                G_gt_cp, show=show_plots, close=False,
                fig_height=fig_height, fig_width=fig_width)
            if show_node_ids:
                ax0 = apls_plots.plot_node_ids(
                    G_gt_cp, ax0, fontsize=4)  # node ids
            ax0.set_title('Ground Truth With Midpionts',
                          fontsize=title_fontsize)
            # plt.show()
            plt.savefig(os.path.join(
                outdir, 'gt_graph_midpoints.png'), dpi=dpi)
            plt.close('all')

            # plot ground truth nodes from prop
            fig, ax = osmnx_funcs.plot_graph(
                G_gt_cp_prime, show=show_plots, close=False,
                fig_height=fig_height, fig_width=fig_width)
            if show_node_ids:
                ax = apls_plots.plot_node_ids(
                    G_gt_cp_prime, ax, fontsize=4)  # node ids
            ax.set_title(
                'Ground Truth Graph with Proposal Control Nodes',
                fontsize=title_fontsize)
            # plt.show()
            plt.savefig(os.path.join(
                outdir, 'gt_graph_prop_control_points.png'), dpi=dpi)
            # plt.clf()
            # plt.cla()
            plt.close('all')

            # remove geometry to test whether we correctly added midpoints and edges
            Gtmp = G_gt_cp.copy()  # G_gt_cp_prime.copy()
            for itmp, (u, v, key, data) in enumerate(Gtmp.edges(keys=True, data=True)):
                try:
                    #line = data['geometry']
                    data.pop('geometry', None)
                except:
                    data[0].pop('geometry', None)
            fig, ax = osmnx_funcs.plot_graph(Gtmp, show=show_plots, close=False,
                                    fig_height=fig_height, fig_width=fig_width)
            ax.set_title(
                'Ground Truth Graph (cp) without any geometry', size='x-small')
            # plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'gt_without_geom.png'), dpi=dpi)
            plt.close('all')

            # plot proposal
            fig, ax = osmnx_funcs.plot_graph(
                G_p_init, show=show_plots, close=False,
                fig_height=fig_height, fig_width=fig_width)
            if show_node_ids:
                ax = apls_plots.plot_node_ids(
                    G_p_init, ax, fontsize=4)  # node ids
            ax.set_title('Proposal Graph', fontsize=title_fontsize)
            # plt.show()
            plt.savefig(os.path.join(outdir, 'prop_graph.png'), dpi=dpi)
            plt.close('all')

            #  proposal midpoints
            fig0, ax0 = osmnx_funcs.plot_graph(
                G_p_cp, show=show_plots, close=False,
                fig_height=fig_height, fig_width=fig_width)
            if show_node_ids:
                ax = apls_plots.plot_node_ids(
                    G_p_cp, ax0, fontsize=4)  # node ids
            ax0.set_title('Proposal With Midpionts', fontsize=title_fontsize)
            # plt.show()
            plt.savefig(os.path.join(
                outdir, 'prop_graph_midpoints.png'), dpi=dpi)
            plt.close('all')

            #  proposal midpoints
            fig0, ax0 = osmnx_funcs.plot_graph(
                G_p_cp_prime, show=show_plots, close=False,
                fig_height=fig_height, fig_width=fig_width)
            if show_node_ids:
                ax = apls_plots.plot_node_ids(
                    G_p_cp_prime, ax0, fontsize=4)  # node ids
            ax0.set_title('Proposal With Midpionts from GT',
                          fontsize=title_fontsize)
            # plt.show()
            plt.savefig(os.path.join(
                outdir, 'prop_graph_midpoints_gt_control_points.png'), dpi=dpi)
            plt.close('all')

            # plot ground truth buffer and proposal graph
            # make sure geometry is in G_p_init
            G_tmp = G_p_init.copy()
            # for i,(u,v,attr_dict) in enumerate(G_tmp.edges(data=True)):
            #    if 'geometry_wkt' in attr_dict.keys():
            #        attr_dict['geometry'] = attr_dict['geometry_wkt']
            #        print "attr_dict:", attr_dict
            fig, ax3 = osmnx_funcs.plot_graph(
                G_tmp, show=show_plots, close=False,
                fig_height=fig_height, fig_width=fig_width)
            apls_plots._plot_buff(G_gt_init, ax3, buff=max_snap_dist,
                                 color='yellow', alpha=0.3,
                                 title='',
                                 title_fontsize=title_fontsize, outfile='',
                                 verbose=False)
            ax3.set_title('Propoal Graph with Ground Truth Buffer',
                          fontsize=title_fontsize)
            # plt.show()
            plt.savefig(os.path.join(
                outdir, 'prop_graph_plus_gt_buff.png'), dpi=dpi)
            # plt.clf()
            # plt.cla()
            plt.close('all')

            # plot proposal buffer and ground truth graph
            fig, ax4 = osmnx_funcs.plot_graph(
                G_gt_init, show=show_plots, close=False,
                fig_height=fig_height, fig_width=fig_width)
            apls_plots._plot_buff(G_p_init, ax4, buff=max_snap_dist,
                                  color='yellow', alpha=0.3,
                                  title='',
                                  title_fontsize=title_fontsize, outfile='',
                                  verbose=False)
            ax4.set_title('Ground Graph with Proposal Buffer',
                          fontsize=title_fontsize)
            # plt.show()
            plt.savefig(os.path.join(
                outdir, 'gt_graph_plus_prop_buff.png'), dpi=dpi)
            # plt.clf()
            # plt.cla()
            plt.close('all')

            # remove geometry to test whether we correctly added midpoints and edges
            Gtmp = G_p_cp.copy()  # G_gt_cp_prime.copy()
            for itmp, (u, v, key, data) in enumerate(Gtmp.edges(keys=True, data=True)):
                try:
                    #line = data['geometry']
                    data.pop('geometry', None)
                except:
                    data[0].pop('geometry', None)
            fig, ax = osmnx_funcs.plot_graph(
                Gtmp, show=show_plots, close=False,
                fig_height=fig_height, fig_width=fig_width)
            ax.set_title(
                'Proposal Graph (cp) without any geometry', size='x-small')
            # plt.tight_layout()
            plt.savefig(os.path.join(
                outdir, 'prop_cp_without_geom.png'), dpi=dpi)
            plt.close('all')

            # remove geometry to test whether we correctly added midpoints and edges
            Gtmp = G_p_init.copy()  # G_gt_cp_prime.copy()
            for itmp, (u, v, key, data) in enumerate(Gtmp.edges(keys=True, data=True)):
                try:
                    #line = data['geometry']
                    data.pop('geometry', None)
                except:
                    data[0].pop('geometry', None)
            fig, ax = osmnx_funcs.plot_graph(
                Gtmp, show=show_plots, close=False,
                fig_height=fig_height, fig_width=fig_width)
            ax.set_title('Proposal Graph without any geometry', size='x-small')
            # plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'prop_without_geom.png'), dpi=dpi)
            plt.close('all')

            ###################################
            # plot some paths...
            # get source and target nodes

            # use idxs?
            #source_idx = np.random.randint(0,len(G_gt_cp.nodes()))
            #target_idx = np.random.randint(0,len(G_gt_cp.nodes()))
            # specify source and target node, if desired
            # if len(gt_file) == 0:
            ##    source_idx =  27
            ##    target_idx =  36
            # print "source_idx:", source_idx
            # print "target_idx:", target_idx
            #source = G_gt_cp.nodes()[source_idx]
            #target = G_gt_cp.nodes()[target_idx]

            # get a random source and target that are in both ground truth and prop
            if len(G_gt_cp.nodes()) < 200:
                print("G_gt_cp.nodes():", G_gt_cp.nodes())
            if len(G_gt_cp_prime.nodes()) < 200:
                print("G_p_cp_prime.nodes():", G_gt_cp_prime.nodes())
            possible_sources = set(G_gt_cp.nodes()).intersection(
                set(G_p_cp_prime.nodes()))
            if len(possible_sources) == 0:
                continue
            source = random.choice(list(possible_sources))
            possible_targets = set(G_gt_cp.nodes()).intersection(
                set(G_p_cp_prime.nodes())) - set([source])
            if len(possible_targets) == 0:
                continue
            target = random.choice(list(possible_targets))
            print("source, target:", source, target)

            # compute paths to node of interest, and plot
            t0 = time.time()
            lengths, paths = nx.single_source_dijkstra(
                G_gt_cp, source=source, weight=weight)
            print("Time to calculate:", len(lengths),
                  "paths:", time.time() - t0, "seconds")

            # plot a single route
            try:
                fig, ax = osmnx_funcs.plot_graph_route(G_gt_cp, paths[target],
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
                plen = np.round(lengths[target], 2)
            except:
                print("Proposal route not possible")
                fig, ax = plt.subplots()
                plen = -1
            title = "Ground Truth Graph, L = " + str(plen)
            source_x = G_gt_cp.nodes[source]['x']
            source_y = G_gt_cp.nodes[source]['y']
            ax.scatter(source_x, source_y, color=source_color, s=75)
            t_x = G_gt_cp.nodes[target]['x']
            t_y = G_gt_cp.nodes[target]['y']
            ax.scatter(t_x, t_y, color=target_color, s=75)
            ax.set_title(title, fontsize=title_fontsize)
            # plt.show()
            plt.savefig(os.path.join(
                outdir, 'single_source_route_ground_truth.png'), dpi=dpi)
            plt.close('all')

            # get all paths from source for proposal graph
            lengths_prop, paths_prop = nx.single_source_dijkstra(
                G_p_cp_prime, source=source, weight=weight)
            gt_set = set(lengths.keys())
            prop_set = set(lengths_prop.keys())
            missing_nodes = gt_set - prop_set
            print("Proposal route missing nodes:", missing_nodes)

            ##############
            # compute path to node of interest
            t0 = time.time()
            lengths_ptmp, paths_ptmp = nx.single_source_dijkstra(
                G_p_cp_prime, source=source, weight=weight)
            print("Time to calculate:", len(lengths),
                  "paths:", time.time() - t0, "seconds")

            # plot a single route
            try:
                fig, ax = osmnx_funcs.plot_graph_route(
                    G_p_cp_prime, paths_ptmp[target],
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
                plen = np.round(lengths_ptmp[target], 2)
            except:
                print("Proposal route not possible")
                fig, ax = plt.subplots()
                plen = -1
            title = "Proposal Graph, L = " + str(plen)
            source_x = G_p_cp_prime.nodes[source]['x']
            source_y = G_p_cp_prime.nodes[source]['y']
            ax.scatter(source_x, source_y, color=source_color, s=75)
            t_x = G_p_cp_prime.nodes[target]['x']
            t_y = G_p_cp_prime.nodes[target]['y']
            ax.scatter(t_x, t_y, color=target_color, s=75)
            ax.set_title(title, fontsize=title_fontsize)
            # plt.show()
            plt.savefig(os.path.join(
                outdir, 'single_source_route_prop.png'), dpi=dpi)
            plt.close('all')

            # overlay plot on image
            if len(im_loc_list) > 0:
                # print ("im_loc_list:", im_loc_list)
                image_path = im_loc_list[i]
                if os.path.exists(image_path):

                    # copy image file to output dir, if desired
                    # shutil.copy(image_path, outdir)

                    # plot graphs overlaid on image
                    # width_key, width_mult = 'speed_mph', 0.3
                    gt_color, prop_color = 'cyan', 'lime'
                    image_name = outroot
                    figname = os.path.join(outdir, 'overlaid.png')
                    _ = apls_plots._plot_gt_prop_graphs(
                        G_gt_init, G_p_init, image_path,
                        figsize=(16, 8), show_endnodes=True,
                        width_key=2,  # width_key,
                        width_mult=1,
                        gt_color=gt_color, prop_color=prop_color,
                        default_node_size=20,
                        title=image_name, adjust=False,
                        figname=figname, verbose=super_verbose)

            #############
            t2 = time.time()
            print("Total time to create graphs, compute metric, and plot:",
                  t2-t0, "seconds")

    # print and save total cost
    print(("C_arr:", C_arr))
    tf = time.time()
    print(("Time to compute metric:", tf - t0, "seconds"))
    print(("N input images:", len(root_list)))
    #df = pd.DataFrame(C_arr[1:], columns=C_arr[0])
    #df.to_csv(os.path.join(outdir_base2, 'scores.csv'))
    #print(("len df:", len(df)))

    # Compute mean of C
    #print ("np.array(C_arr):", np.array(C_arr))
    means = np.mean(np.array(C_arr)[1:, 1:].astype(float), axis=0)
    C_arr.append(['means'] + list(means))

    # save to csv
    path_csv = os.path.join(outdir_base2,
                            'scores__max_snap='
                            + str(np.round(max_snap_dist, 2)) + 'm'
                            + '_hole='
                            + str(np.round(topo_hole_size, 2)) + 'm'
                            + '.csv')
    print("Save to csv:", path_csv)
    df = pd.DataFrame(C_arr[1:], columns=C_arr[0])
    print("len df:", len(df))
    df.to_csv(path_csv)

    print(("Tot APLS = np.mean(APLS_arr:", np.mean(df['APLS'].values)))

    return


###############################################################################
def main():
    '''Explore'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', default='apls_test0', type=str,
                        help='Output folder name in apls/outputs')
    parser.add_argument('--test_method', default='gt_pkl_prop_pkl', type=str,
                        help="Method for creating ground truth and proposal "
                        + "graphs. Options:"
                        + " gt_pkl_prop_pkl   = ground truth pickle, proposal pickle"
                        + " gt_json_prop_pkl  = ground truth json, proposal pickle"
                        + "gt_json_prop_wkt  = ground truth json, proposal csv in wkt format"
                        + "gt_json_prop_json = ground truth json, proposal json"
                        )
    parser.add_argument('--truth_dir', default='', type=str,
                        help='Location of ground truth graphs')
    parser.add_argument('--prop_dir', default='', type=str,
                        help='Location of proposal graphs')
    parser.add_argument('--im_dir', default='', type=str,
                        help='Location of images (optional)')
    parser.add_argument('--im_prefix', default='RGB-PanSharpen_', type=str,
                        help='Prefix of image files')
    parser.add_argument('--wkt_file', default='', type=str,
                        help='Location of prediction wkt file')
    parser.add_argument('--max_snap_dist', default=4, type=int,
                        help='Buffer distance (meters) around graph')
    parser.add_argument('--topo_hole_size', default=4, type=float,
                        help='hole size in meters for TOPO metric')
    parser.add_argument('--topo_subgraph_radius', default=150, type=float,
                        help='Subgraph radius for TOPO')
    parser.add_argument('--topo_interval', default=30, type=float,
                        help='Hole spacing for TOPO')
    parser.add_argument('--sp_length_buffer', default=0.05, type=float,
                        help='Frational length differnence for SP metric')
    parser.add_argument('--linestring_delta', default=200, type=int,
                        help='Distance between midpoints on edges')
    parser.add_argument('--is_curved_eps', default=0.12, type=float,
                        help='Line curvature above which midpoints will be'
                        ' injected, (< 0 to inject midpoints on straight'
                        ' lines)')
    parser.add_argument('--min_path_length', default=0.001, type=float,
                        help='Minimum path length to consider for metric')
    parser.add_argument('--max_nodes', default=500, type=int,
                        help='Maximum number of nodes to compare for APLS'
                        ' metric')
    parser.add_argument('--max_files', default=100, type=int,
                        help='Maximum number of graphs to analyze')
    parser.add_argument('--weight', default='length', type=str,
                        help='Weight for APLS metric [length, travel_time')
    parser.add_argument('--speed_key', default='speed_m/s', type=str,
                        help='Key in edge properties for speed')
    parser.add_argument('--travel_time_key', default='travel_time', type=str,
                        help='Key in edge properties for travel_time')
    parser.add_argument('--default_speed', default=13.41, type=float,
                        help='Default speed of edge in m/s'
                        ' (13.41 m/s = 30 mph)')
    parser.add_argument('--n_plots', default=12, type=int,
                        help='Number of plots to make when computing APLS')
    parser.add_argument('--use_pix_coords', default=1, type=int,
                        help='Switch to use pix coords, defaults to 1 (True)')
    parser.add_argument('--allow_renaming', default=1, type=int,
                        help='Switch to rename nodes. Defaults to 1 (True)')

    args = parser.parse_args()

    # Filtering parameters (shouldn't need changed)
    args.gt_subgraph_filter_weight = 'length'
    args.gt_min_subgraph_length = 5
    args.prop_subgraph_filter_weight = 'length_pix'
    args.prop_min_subgraph_length = 10  # GSD = 0.3

    # general settings
    verbose = True
    super_verbose = False

    # Gather files
    gt_list, gp_list, root_list, im_loc_list = gather_files(
        args.test_method,
        args.truth_dir,
        args.prop_dir,
        im_dir=args.im_dir,
        im_prefix=args.im_prefix,
        wkt_file=args.wkt_file,
        max_files=args.max_files,
        gt_subgraph_filter_weight=args.gt_subgraph_filter_weight,
        gt_min_subgraph_length=args.gt_min_subgraph_length,
        prop_subgraph_filter_weight=args.prop_subgraph_filter_weight,
        prop_min_subgraph_length=args.prop_min_subgraph_length,
        use_pix_coords=bool(args.use_pix_coords),
        speed_key=args.speed_key,
        travel_time_key=args.travel_time_key,
        default_speed=args.default_speed,
        verbose=verbose,
        super_verbose=super_verbose)

    # Compute
    execute(
        args.output_name, gt_list, gp_list, root_list,
        im_loc_list=im_loc_list,
        weight=args.weight,
        speed_key=args.speed_key,
        travel_time_key=args.travel_time_key,
        max_files=args.max_files,
        linestring_delta=args.linestring_delta,
        is_curved_eps=args.is_curved_eps,
        max_snap_dist=args.max_snap_dist,
        max_nodes=args.max_nodes,
        n_plots=args.n_plots,
        min_path_length=args.min_path_length,
        topo_hole_size=args.topo_hole_size,
        topo_subgraph_radius=args.topo_subgraph_radius,
        topo_interval=args.topo_interval,
        sp_length_buffer=args.sp_length_buffer,
        use_pix_coords=bool(args.use_pix_coords),
        allow_renaming=bool(args.allow_renaming),
        verbose=verbose,
        super_verbose=super_verbose)


###############################################################################
if __name__ == "__main__":
    main()
