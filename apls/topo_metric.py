#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 08:32:18 2018

@author: avanetten

Implemention of the TOPO metric
https://pdfs.semanticscholar.org/51b0/51eba4f58afc34021ae23641fc8e168fdf07.pdf

"""

import os
import sys
import time
import numpy as np
import networkx as nx
import scipy.spatial
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from shapely.geometry import Point, LineString
# import math
# import osmnx as ox

path_apls_src = os.path.dirname(os.path.realpath(__file__))
path_apls = os.path.dirname(path_apls_src)
sys.path.append(path_apls_src)
import apls
import apls_plots
import apls_utils
import osmnx_funcs


###############################################################################
def ensure_radial_linestrings(G_sub_, origin_node, x_coord='x', y_coord='y',
                              verbose=True):
    """
    Since we are injecting points on edges every X meters, make sure that
    the edge geometry is always pointing radially outward from the center
    node.  If geometries aren't always pointing the same direction we might
    inject points at different locations on the ground truth and proposal
    graphs.  Assume all edges have the geometry tag
    """

    # get location of graph center
    n_props = G_sub_.node[origin_node]
    origin_loc = [n_props[x_coord], n_props[y_coord]]

    # iterate through edges and check in linestring goes toward or away from
    #   center node
    for i, (u, v, key, data) in enumerate(G_sub_.edges(keys=True, data=True)):

        # now ensure linestring points away from origin
        #  assume that the start and end point arene't exactly the same
        #  distance from the origin
        line_geom = data['geometry']
        geom_p_start = list(line_geom.coords)[0]
        geom_p_end = list(line_geom.coords)[-1]
        dist_to_start = scipy.spatial.distance.euclidean(
            origin_loc, geom_p_start)
        dist_to_end = scipy.spatial.distance.euclidean(origin_loc, geom_p_end)
        # reverse the line if the end is closer to the origin than the start
        if dist_to_end < dist_to_start:
            if verbose:
                print(("Reverse linestring from", u, "to", v))
            coords_rev = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords_rev)
            data['geometry'] = line_geom_rev

    return G_sub_


###############################################################################
def insert_holes_or_marbles(G_, origin_node, interval=50, n_id_add_val=1,
                            verbose=False):
    """
    Insert points on the graph on the specified interval
    n_id_add_val sets min midpoint id above existing nodes
        e.g.: G.nodes() = [1,2,4], if n_id_add_val = 5, midpoints will
        be [9,10,11,...]
    Apapted from apls.py.create_graph(midpoints()
    """

    if len(G_.nodes()) == 0:
        return G_, [], []

    # midpoints
    xms, yms = [], []
    Gout = G_.copy()
    midpoint_name_val, midpoint_name_inc = np.max(
        list(G_.nodes())) + n_id_add_val, n_id_add_val
    # for u, v, key, data in G_.edges(keys=True, data=True):
    for u, v, data in G_.edges(data=True):

        # curved line
        if 'geometry' in data:

            edge_props_init = G_.edges([u, v])

            linelen = data['length']
            line = data['geometry']

            xs, ys = line.xy  # for plotting

            #################
            # ignore short lines
            if linelen < interval:
                # print "Line too short, skipping..."
                continue
            #################

            if verbose:
                print("u,v:", u, v)
                print("data:", data)
                print("edge_props_init:", edge_props_init)

            # interpolate injection points
            # get evenly spaced points (skip first point at 0)
            interp_dists = np.arange(0, linelen, interval)[1:]
            # evenly spaced midpoints (from apls.create_graph(midpoints()
            # npoints = len(np.arange(0, linelen, interval)) + 1
            # interp_dists = np.linspace(0, linelen, npoints)[1:-1]
            if verbose:
                print("interp_dists:", interp_dists)

            # create nodes
            node_id_new_list = []
            xms_tmp, yms_tmp = [], []
            for j, d in enumerate(interp_dists):
                if verbose:
                    print("j,d", j, d)

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
                    print("midpoint:", xm, ym)

                # add node to graph, with properties of u
                node_id = midpoint_name_val
                # node_id = np.round(u + midpoint_name_val,2)
                midpoint_name_val += midpoint_name_inc
                node_id_new_list.append(node_id)
                if verbose:
                    print("node_id:", node_id)

                # add to graph
                Gout, node_props, xn, yn = apls.insert_point_into_G(
                    Gout, point, node_id=node_id, allow_renaming=False,
                    verbose=verbose)

    return Gout, xms, yms


###############################################################################
def compute_single_topo(G_sub_gt_, G_sub_p_,
                        x_coord='x', y_coord='y', hole_size=5,
                        allow_multi_hole=False,
                        verbose=False, super_verbose=False):
    '''compute filled and empty holes for a single subgraph
    By default, Only allow one marble in each hole (allow_multi_hole=False)'''

    # get node positions
    pos_gt = apls_utils._get_node_positions(G_sub_gt_, x_coord=x_coord,
                                            y_coord=y_coord)
    pos_p = apls_utils._get_node_positions(G_sub_p_, x_coord=x_coord,
                                           y_coord=y_coord)

    # construct kdtree of ground truth
    kd_idx_dic0, kdtree0, pos_arr0 = apls_utils.G_to_kdtree(G_sub_gt_)

    prop_tp, prop_fp = [], []
    gt_tp, gt_fn = [], []
    gt_match_idxs_set = set()   # set of already matched gt idxs
    # iterate through marbles to see if it fall in a hole
    for i, prop_point in enumerate(pos_p):
        if verbose:
            print(("prop i:", i, "prop node:", list(G_sub_p_.nodes())[i]))
        # see if a marble is close to a hole
        # dists, idxs = kdtree.query(prop_point, k=100,
        #                           distance_upper_bount=hole_size)

        idxs_raw = kdtree0.query_ball_point(prop_point, r=hole_size)
        if not allow_multi_hole:
            # subtract previously matched items from indices
            idxs = list(set(idxs_raw) - gt_match_idxs_set)
        else:
            # allow multiple marbles to fall into the same hole
            idxs = idxs_raw

        if super_verbose:
            print(("idxs:", idxs))
            print(("idxs_raw:", idxs_raw))

        # if no results, add to false positive
        if len(idxs) == 0:
            prop_fp.append(i)

        # otherwise, check what's close
        else:

            # check distances of remaining items
            dists_m = np.asarray([scipy.spatial.distance.euclidean(
                prop_point, [kdtree0.data[itmp][0], kdtree0.data[itmp][1]])
                for itmp in idxs])
            # get min distance
            jmin = np.argmin(dists_m)
            idx_min = idxs[jmin]

            # add to true positive
            prop_tp.append(i)
            gt_tp.append(idx_min)
            gt_match_idxs_set.add(idx_min)

    # count up how many holes we've filled
    n_holes = len(pos_gt)
    n_marbles = len(pos_p)
    true_pos_count = len(prop_tp)
    # this should be the same as n_marbles_in_holes
    n_holes_filled = len(gt_tp)
    false_pos_count = len(prop_fp)
    gt_fn = list(set(range(len(pos_gt))) - gt_match_idxs_set)
    n_empty_holes = n_holes - true_pos_count
    false_neg_count = n_empty_holes
    try:
        precision = float(true_pos_count) / \
            float(true_pos_count + false_pos_count)
    except:
        precision = 0
    try:
        recall = float(true_pos_count) / \
            float(true_pos_count + false_neg_count)
    except:
        recall = 0
    try:
        f1 = 2. * precision * recall / (precision + recall)
    except:
        f1 = 0

    if verbose:
        print(("Num holes:", n_holes))
        print(("Num marbles:", n_marbles))
        print(("gt_match_idxs_set:", gt_match_idxs_set))
        print(("Num marbles in holes (true pos):", true_pos_count))
        print(("Num extra marbles:", false_pos_count))
        print(("Num holes filled:", n_holes_filled))
        print(("Num empty holes0:", false_neg_count))
        print(("Num empty holes1:", len(gt_fn)))
        print(("precision:", precision))
        print(("recall:", recall))
        print(("f1:", f1))

    # return  pos_gt, gt_tp, gt_fn, \
    #        pos_p, prop_tp, prop_fp

    return true_pos_count, false_pos_count, false_neg_count, \
        precision, recall, f1


###############################################################################
def compute_topo(G_gt_, G_p_, subgraph_radius=150, interval=30, hole_size=5,
                 n_measurement_nodes=20, x_coord='x', y_coord='y',
                 allow_multi_hole=False,
                 make_plots=False, verbose=False):
    '''Compute topo metric
     subgraph_radius = radius for topo computation
     interval is spacing of inserted points
     hole_size is the buffer within which proposals must fall
     '''

    t0 = time.time()
    if (len(G_gt_) == 0) or (len(G_p_) == 0):
        return 0, 0, 0, 0, 0, 0

    if verbose:
        print(("G_gt_.nodes():", G_gt_.nodes()))
    # define ground truth kdtree
    kd_idx_dic, kdtree, pos_arr = apls_utils.G_to_kdtree(G_gt_)
    # proposal graph kdtree
    kd_idx_dic_p, kdtree_p, pos_arr_p = apls_utils.G_to_kdtree(G_p_)

    true_pos_count_l, false_pos_count_l, false_neg_count_l = [], [], []
    # precision_l, recall_l, f1_l = [], [], []

    # make sure we don't pick more nodes than exist in the graph
    n_pick = min(n_measurement_nodes, len(G_gt_.nodes()))
    # pick a random node  to start
    origin_nodes = np.random.choice(G_gt_.nodes(), n_pick)
    # origin_nodes = [10030]

    for i, origin_node in enumerate(origin_nodes):

        if (i % 20) == 0:
            print(i, "Origin node:", origin_node)
        n_props = G_gt_.node[origin_node]
        x0, y0 = n_props[x_coord], n_props[y_coord]
        origin_point = [x0, y0]

        # get subgraph
        node_names, node_dists = apls_utils._nodes_near_origin(
            G_gt_, origin_node,
            kdtree, kd_idx_dic,
            x_coord=x_coord, y_coord=y_coord,
            radius_m=subgraph_radius,
            verbose=verbose)

        if verbose and len(node_names) == 0:
            print("subgraph empty")

        # get subgraph
        G_sub0 = G_gt_.subgraph(node_names)
        if verbose:
            print(("G_sub0.nodes():", G_sub0.nodes()))

        # make sure all nodes connect to origin
        node_names_conn = nx.node_connected_component(G_sub0, origin_node)
        G_sub1 = G_sub0.subgraph(node_names_conn)

        # ensure linestrings are radially out from origin point
        G_sub = ensure_radial_linestrings(G_sub1, origin_node,
                                          x_coord='x', y_coord='y',
                                          verbose=verbose)

        # insert points
        G_holes, xms, yms = insert_holes_or_marbles(
            G_sub, origin_node,
            interval=interval, n_id_add_val=1,
            verbose=False)

        #####
        # Proposal

        # determine nearast node to ground_truth origin_point
        # query kd tree for origin node
        node_names_p, idxs_refine_p, dists_m_refine_p = apls_utils._query_kd_ball(
            kdtree_p, kd_idx_dic_p, origin_point, hole_size)

        if len(node_names_p) == 0:
            if verbose:
                print(("Oops, no proposal node correspnding to", origin_node))
            # all nodes are false positives in this case
            true_pos_count_l.append(0)
            false_pos_count_l.append(0)
            false_neg_count_l.append(len(G_holes.nodes()))
            continue

        # get closest node
        origin_node_p = node_names_p[np.argmin(dists_m_refine_p)]
        # get coords of the closest point
        n_props = G_p_.node[origin_node_p]
        xp, yp = n_props['x'], n_props['y']
        if verbose:
            print(("origin_node_p:", origin_node_p))

        # get subgraph
        node_names_p, node_dists_p = apls_utils._nodes_near_origin(
            G_p_, origin_node_p,
            kdtree_p, kd_idx_dic_p,
            x_coord=x_coord, y_coord=y_coord,
            radius_m=subgraph_radius,
            verbose=verbose)

        # get subgraph
        G_sub0_p = G_p_.subgraph(node_names_p)
        if verbose:
            print(("G_sub0_p.nodes():", G_sub0_p.nodes()))

        # make sure all nodes connect to origin
        node_names_conn_p = nx.node_connected_component(
            G_sub0_p, origin_node_p)
        G_sub1_p = G_sub0_p.subgraph(node_names_conn_p)

        # ensure linestrings are radially out from origin point
        G_sub_p = ensure_radial_linestrings(G_sub1_p, origin_node_p,
                                            x_coord='x', y_coord='y',
                                            verbose=verbose)

        # insert points
        G_holes_p, xms, yms = insert_holes_or_marbles(
            G_sub_p, origin_node_p,
            interval=interval, n_id_add_val=1,
            verbose=False)

        ####################
        # compute topo metric
        true_pos_count, false_pos_count, false_neg_count, \
            precision, recall, f1 = compute_single_topo(
                G_holes, G_holes_p,
                x_coord=x_coord, y_coord=y_coord,
                allow_multi_hole=allow_multi_hole,
                hole_size=hole_size, verbose=verbose)
        true_pos_count_l.append(true_pos_count)
        false_pos_count_l.append(false_pos_count)
        false_neg_count_l.append(false_neg_count)

        # plot if i == 0:
        if i == 0 and make_plots:
            # plot G_gt_
            plt.close('all')
            # plot initial graph
            fig, ax = osmnx_funcs.plot_graph(G_gt_, axis_off=False)
            ax.set_title("Ground Truth Input plot")
            # draw a circle (this doesn't work unless it's a PatchCollection!)
            patches = [Circle((x0, y0), hole_size, alpha=0.3)]
            p = PatchCollection(patches, alpha=0.4)
            ax.add_collection(p)
            # also a simple point
            ax.scatter([x0], [y0], c='red')
            plt.close('all')

            # plot subgraph
            fig2, ax2 = osmnx_funcs.plot_graph(G_sub, axis_off=False)
            ax2.set_title("Subgraph")
            plt.close('all')

            # plot G_holes
            fig, ax = osmnx_funcs.plot_graph(G_holes, axis_off=False)
            ax.set_title("Subgraph with holes")
            # plot holes
            # draw a circle (this doesn't work unless it's a PatchCollection!)
            node_coords = []
            for i, n in enumerate(G_holes.nodes()):
                n_props = G_holes.node[n]
                node_coords.append([n_props[x_coord], n_props[y_coord]])
            patches = [Circle((coord), hole_size, alpha=0.3)
                       for coord in node_coords]
            p = PatchCollection(patches, alpha=0.4, color='yellow')
            ax.add_collection(p)
            plt.close('all')

            # plot initial graph
            fig, ax = osmnx_funcs.plot_graph(G_p_, axis_off=False)
            ax.set_title("Proposal Input plot")
            # draw a circle (this doesn't work unless it's a PatchCollection!)
            patches = [Circle((xp, yp), hole_size, alpha=0.3)]
            p = PatchCollection(patches, alpha=0.4)
            ax.add_collection(p)
            # also scatter plot a simple point
            ax.scatter([xp], [yp], c='red')
            plt.close('all')

            # plot subgraph
            fig2, ax2 = osmnx_funcs.plot_graph(G_sub_p, axis_off=False)
            ax2.set_title("Proposal Subgraph")
            plt.close('all')

            # plot G_holes
            fig, ax = osmnx_funcs.plot_graph(
                G_holes_p, axis_off=False, node_color='red')
            ax.set_title("Proposal Subgraph with marbles")
            # plot marbles?  (Since marbles have size 0 and only fall into holes, skip)
            # draw a circle (this doesn't work unless it's a PatchCollection!)
            #node_coords  = []
            # for i,n in enumerate(G_holes_p.nodes()):
            #    n_props = G_holes_p.node[n]
            #    node_coords.append([n_props[x_coord], n_props[y_coord]])
            #patches = [Circle((coord), hole_size, alpha=0.3) for coord in node_coords]
            #p = PatchCollection(patches, alpha=0.4, color='yellow')
            # ax.add_collection(p)
            plt.close('all')

            # plot marbles overlaid on holes
            # plot G_holes
            fig, ax = osmnx_funcs.plot_graph(G_holes, axis_off=False)
            ax.set_title("GT graph and holes (yellow), Prop marbles (red)")
            # plot holes
            # draw a circle (this doesn't work unless it's a PatchCollection!)
            node_coords = []
            for i, n in enumerate(G_holes.nodes()):
                n_props = G_holes.node[n]
                node_coords.append([n_props[x_coord], n_props[y_coord]])
            patches = [Circle((coord), hole_size, alpha=0.3)
                       for coord in node_coords]
            p = PatchCollection(patches, alpha=0.4, color='yellow')
            ax.add_collection(p)
            # scatter proposal nodes
            arr_marbles = apls_utils._get_node_positions(
                G_holes_p, x_coord='x', y_coord='y')
            ax.scatter(arr_marbles[:, 0], arr_marbles[:, 1], c='red')

            # show gt node ids
            ax = apls_plots._plot_node_ids(G_holes, ax, fontsize=9)  # node ids

            # show prop node ids
            ax = apls_plots.plot_node_ids(
                G_holes_p, ax, fontsize=9)  # node ids
            plt.close('all')

    # compute total score
    tp_tot = np.sum(true_pos_count_l)
    fp_tot = np.sum(false_pos_count_l)
    fn_tot = np.sum(false_neg_count_l)

    try:
        precision = float(tp_tot) / float(tp_tot + fp_tot)
    except:
        precision = 0
    try:
        recall = float(tp_tot) / float(tp_tot + fn_tot)
    except:
        recall = 0
    try:
        f1 = 2. * precision * recall / (precision + recall)
    except:
        f1 = 0

    print("TOPO metric:")
    print("  total time elapsed to compute TOPO and make plots:",
          time.time() - t0, "seconds")
    print("  total precison:", precision)
    print("  total recall:", recall)
    print("  total f1:", f1)

    return tp_tot, fp_tot, fn_tot, precision, recall, f1


###############################################################################
###############################################################################

###############################################################################
if __name__ == "__main__":

    # Test
    ##########################
    n_measurement_nodes = 10  # there are ~30 nodes per graph
    subgraph_radius = 150     # radius for topo computation
    interval = 30    # interval for holes/marbles
    hole_size = 5
    x_coord = 'x'
    y_coord = 'y'
    allow_multi_hole = False
    make_plots = True
    run_all = True
    verbose = True
    # pick_random_start_node = True

    truth_dir = '/raid/cosmiq/spacenet/data/spacenetv2/AOI_2_Vegas_Test/400m/gt_graph_pkls'
    prop_dir = 'raid/cosmiq/basiss/inference_mod_new/results/rgb_test_sn_vegas/graphs'
    ##########################

    name_list = os.listdir(truth_dir)
    f = name_list[np.random.randint(len(name_list))]
    f = 'AOI_2_Vegas_img150.pkl'
    # f = 'AOI_2_Vegas_img1532.pkl'  # empty proposal

    print(("f:", f))
    t0 = time.time()

    # get original graph
    outroot = f.split('.')[0]
    print("\noutroot:", outroot)
    gt_file = os.path.join(truth_dir, f)
    prop_file = os.path.join(prop_dir, outroot + '.gpickle')

    # ground truth graph
    G_gt_init = nx.read_gpickle(gt_file)
    # G_gt_init1 = osmnx_funcs.simplify_graph(G_gt_init0.to_directed()).to_undirected()
    # G_gt_init = osmnx_funcs.project_graph(G_gt_init1)
    G_gt_init = apls.create_edge_linestrings(
        G_gt_init, remove_redundant=True, verbose=False)

    print(("G_gt_init.nodes():", G_gt_init.nodes()))
    # define ground truth kdtree
    # kd_idx_dic, kdtree, pos_arr = G_to_kdtree(G_gt_init)

    # proposal graph
    # G_p_ = nx.read_gpickle(prop_file)
    G_p_init0 = nx.read_gpickle(prop_file)
    G_p_init1 = osmnx_funcs.simplify_graph(G_p_init0.to_directed()).to_undirected()
    G_p_init = osmnx_funcs.project_graph(G_p_init1)
    G_p_init = apls.create_edge_linestrings(
        G_p_init, remove_redundant=True, verbose=False)
    # kd_idx_dic_p, kdtree_p, pos_arr_p = G_to_kdtree(G_p_init)

    tp_tot, fp_tot, fn_tot, precision, recall, f1 = \
        compute_topo(G_gt_init, G_p_init, subgraph_radius=subgraph_radius,
                     interval=interval, hole_size=hole_size,
                     n_measurement_nodes=n_measurement_nodes,
                     x_coord=x_coord, y_coord=y_coord,
                     allow_multi_hole=allow_multi_hole,
                     make_plots=make_plots, verbose=verbose)

    ############
    # also compute total topo metric for entire folder
    if run_all:
        t0 = time.time()
        tp_tot_list, fp_tot_list, fn_tot_list = [], [], []
        for i, f in enumerate(name_list):

            # get original graph
            outroot = f.split('.')[0]
            print("\n", i, "/", len(name_list), "outroot:", outroot)
            #print ("\n", i, "/", len(name_list), "outroot:", outroot)
            gt_file = os.path.join(truth_dir, f)

            # ground truth graph
            G_gt_init = nx.read_gpickle(gt_file)
            # G_gt_init1 = osmnx_funcs.simplify_graph(G_gt_init0.to_directed()).to_undirected()
            # G_gt_init = osmnx_funcs.project_graph(G_gt_init1)
            G_gt_init = apls.create_edge_linestrings(
                G_gt_init, remove_redundant=True, verbose=False)
            if len(G_gt_init.nodes()) == 0:
                continue

            # proposal graph
            prop_file = os.path.join(prop_dir, outroot + '.gpickle')
            if not os.path.exists(prop_file):
                tp_tot_list.append(0)
                fp_tot_list.append(0)
                fn_tot_list.append(len(G_gt_init.nodes()))
                continue

            # G_p_init = nx.read_gpickle(prop_file)
            G_p_init0 = nx.read_gpickle(prop_file)
            G_p_init1 = osmnx_funcs.simplify_graph(
                G_p_init0.to_directed()).to_undirected()
            G_p_init = osmnx_funcs.project_graph(G_p_init1)
            G_p_init = apls.create_edge_linestrings(
                G_p_init, remove_redundant=True, verbose=False)

            tp, fp, fn, precision, recall, f1 = \
                compute_topo(G_gt_init, G_p_init,
                             subgraph_radius=subgraph_radius,
                             interval=interval, hole_size=hole_size,
                             n_measurement_nodes=n_measurement_nodes,
                             x_coord=x_coord, y_coord=y_coord,
                             allow_multi_hole=allow_multi_hole,
                             make_plots=False, verbose=False)

            tp_tot_list.append(tp)
            fp_tot_list.append(fp)
            fn_tot_list.append(fn)

        # compute total score
        tp_tot = np.sum(tp_tot_list)
        fp_tot = np.sum(fp_tot_list)
        fn_tot = np.sum(fn_tot_list)
        precision = float(tp_tot) / float(tp_tot + fp_tot)
        recall = float(tp_tot) / float(tp_tot + fn_tot)
        f1 = 2. * precision * recall / (precision + recall)

        print(("Total TOPO metric for", len(name_list), "files:"))
        print(("  hole_size:", hole_size, "interval:", interval))
        print(("  subgraph_radius:", subgraph_radius,
               "allow_multi_hole?", allow_multi_hole))
        print(("  total time elapsed to compute TOPO and make plots:",
               time.time() - t0, "seconds"))
        print(("  total precison:", precision))
        print(("  total recall:", recall))
        print(("  total f1:", f1))
