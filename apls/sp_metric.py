#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 08:32:18 2018

@author: avanetten

Implement SP Metric
https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Wegner_A_Higher-Order_CRF_2013_CVPR_paper.pdf

"""

import apls_utils
import apls
import os
import sys
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
# import osmnx as ox

path_apls_src = os.path.dirname(os.path.realpath(__file__))
path_apls = os.path.dirname(path_apls_src)
sys.path.append(path_apls_src)
import osmnx_funcs

###############################################################################
def compute_single_sp(G_gt_, G_prop_, kd_idx_dic_prop, kdtree_prop,
                      x_coord='x', y_coord='y',
                      weight='length', query_radius=5,
                      length_buffer=0.05, make_plots=False, verbose=False):
    '''Single SP metric
    return 1 if within length_buffer
    return 0 if path is outside length_buffer or DNE for either gt or prop
    return -1 if path between randomly chosen nodes DNE for both graphs'''

    # choose random ground truth source and target nodes
    [source_gt, target_gt] = np.random.choice(
        G_gt_.nodes(), size=2, replace=False)
    if verbose:
        print("source_gt:", source_gt, "target_gt:", target_gt)
    # source_gt, target_gt = 10002, 10039
    x_s_gt, y_s_gt = G_gt_.node[source_gt][x_coord], G_gt_.node[source_gt][y_coord]
    x_t_gt, y_t_gt = G_gt_.node[target_gt][x_coord], G_gt_.node[target_gt][y_coord]

    # if verbose:
    #    print ("x_s_gt:", x_s_gt)
    #    print ("y_s_gt:", y_s_gt)

    # get route.  If it does not exists, set len = -1
    if not nx.has_path(G_gt_, source_gt, target_gt):
        len_gt = -1
    else:
        len_gt = nx.dijkstra_path_length(
            G_gt_, source_gt, target_gt, weight=weight)

    # get nodes in prop graph
    # see if source, target node exists in proposal
    source_p_l, _ = apls_utils.nodes_near_point(x_s_gt, y_s_gt,
                                                kdtree_prop, kd_idx_dic_prop,
                                                x_coord=x_coord, y_coord=y_coord,
                                                radius_m=query_radius)
    target_p_l, _ = apls_utils.nodes_near_point(x_t_gt, y_t_gt,
                                                kdtree_prop, kd_idx_dic_prop,
                                                x_coord=x_coord, y_coord=y_coord,
                                                radius_m=query_radius)

    # if either source or target does not exists, set prop_len as -1
    if (len(source_p_l) == 0) or (len(target_p_l) == 0):
        len_prop = -1

    else:
        source_p, target_p = source_p_l[0], target_p_l[0]
        x_s_p, y_s_p = G_prop_.node[source_p][x_coord], G_prop_.node[source_p][y_coord]
        x_t_p, y_t_p = G_prop_.node[target_p][x_coord], G_prop_.node[target_p][y_coord]

        # get route
        if not nx.has_path(G_prop_, source_p, target_p):
            len_prop = -1
        else:
            len_prop = nx.dijkstra_path_length(
                G_prop_, source_p, target_p, weight=weight)

    # path length difference, as a percentage
    perc_diff = np.abs((len_gt - len_prop) / len_gt)
    # check path lengths
    # if both paths do not exist, skip
    if (len_gt == -1) and (len_prop == -1):
        match = -1
    # if one is positive and one negative, return 0
    elif (np.sign(len_gt) != np.sign(len_prop)):
        match = 0
    # else, campare lengths
    elif perc_diff > length_buffer:
        match = 0
    else:
        match = 1

    if verbose:
        # print ("source_gt:", source_gt, "target_gt:", target_gt)
        print("len_gt:", len_gt)
        print("len_prop:", len_prop)
        print("perc_diff:", perc_diff)

    if make_plots:

        # plot G_gt_init
        plt.close('all')
        # plot initial graph
        if len_gt != -1:
            fig, ax = osmnx_funcs.plot_graph_route(G_gt_, nx.shortest_path(
                G_gt_, source=source_gt, target=target_gt, weight=weight))
        else:
            fig, ax = osmnx_funcs.plot_graph(G_gt_, axis_off=True)
        ax.set_title("Ground Truth, L = " + str(np.round(len_gt, 2)))
        # draw a circle (this doesn't work unless it's a PatchCollection!)
        patches = [Circle((x_s_gt, y_s_gt), query_radius, alpha=0.3),
                   Circle((x_t_gt, y_t_gt), query_radius, alpha=0.3)]
        p = PatchCollection(patches, alpha=0.4, color='orange')
        ax.add_collection(p)
        # also a simple point
        ax.scatter([x_s_gt], [y_s_gt], c='green', s=6)
        ax.scatter([x_t_gt], [y_t_gt], c='red', s=6)

        # plot proposal graph
        if len_prop != -1:
            fig, ax1 = osmnx_funcs.plot_graph_route(G_prop_, nx.shortest_path(
                G_prop_, source=source_p, target=target_p, weight=weight))
        else:
            fig, ax1 = osmnx_funcs.plot_graph(G_prop_, axis_off=True)
        ax1.set_title("Proposal, L = " + str(np.round(len_prop, 2)))
        # draw patches from ground truth!
        patches = [Circle((x_s_gt, y_s_gt), query_radius, alpha=0.3),
                   Circle((x_t_gt, y_t_gt), query_radius, alpha=0.3)]
        p = PatchCollection(patches, alpha=0.4, color='orange')
        ax1.add_collection(p)
        if len_prop != -1:
            # also a simple point
            ax1.scatter([x_s_p], [y_s_p], c='green', s=6)
            ax1.scatter([x_t_p], [y_t_p], c='red', s=6)

    return match


###############################################################################
def compute_sp(G_gt_, G_prop_,
               x_coord='x', y_coord='y',
               weight='length', query_radius=5,
               length_buffer=0.05, n_routes=10, verbose=False,
               make_plots=True):
    '''Compute SP metric'''

    t0 = time.time()
    if len(G_prop_.nodes()) == 0:
        return [], 0

    kd_idx_dic_p, kdtree_p, pos_arr_p = apls_utils.G_to_kdtree(G_prop_)

    match_l = []
    for i in range(n_routes):
        if i == 0 and make_plots:
            make_plots_tmp = True
        else:
            make_plots_tmp = False

        if (i % 100) == 0:
            print((i, "/", n_routes))

        match_val = compute_single_sp(G_gt_, G_prop_, kd_idx_dic_p, kdtree_p,
                                      x_coord=x_coord, y_coord=y_coord,
                                      weight=weight, query_radius=query_radius,
                                      length_buffer=length_buffer, make_plots=make_plots_tmp,
                                      verbose=verbose)
        if match_val != -1:
            match_l.append(match_val)

    # total score is fraction of routes that match
    sp_tot = 1.0 * np.sum(match_l) / len(match_l)

    if verbose:
        print(("match_arr:", np.array(match_l)))
        # print ("  sp_tot:", sp_tot)

    print("sp metric:")
    print(("  total time elapsed to compute sp:",
           time.time() - t0, "seconds"))

    return match_l, sp_tot


###############################################################################
###############################################################################
###############################################################################
if __name__ == "__main__":

    # Test
    ##########################
    n_measurement_nodes = 10
    x_coord = 'x'
    y_coord = 'y'
    weight = 'length'
    query_radius = 5
    length_buffer = 0.05
    n_routes = 500
    verbose = False  # True
    run_all = True
    #pick_random_start_node = True

    truth_dir = '/raid/cosmiq/spacenet/data/spacenetv2/AOI_2_Vegas_Test/400m/gt_graph_pkls'
    prop_dir = 'raid/cosmiq/basiss/inference_mod_new/results/rgb_test_sn_vegas/graphs'
    ##########################

    name_list = os.listdir(truth_dir)
    f = name_list[np.random.randint(len(name_list))]
    #f = 'AOI_2_Vegas_img150.pkl'
    print(("f:", f))
    t0 = time.time()

    # get original graph
    outroot = f.split('.')[0]
    print("\noutroot:", outroot)
    gt_file = os.path.join(truth_dir, f)
    prop_file = os.path.join(prop_dir, outroot + '.gpickle')

    # ground truth graph
    G_gt_init = nx.read_gpickle(gt_file)
    G_gt_init1 = osmnx_funcs.simplify_graph(G_gt_init.to_directed()).to_undirected()
    G_gt_init = osmnx_funcs.project_graph(G_gt_init1)
    G_gt_init = apls.create_edge_linestrings(
        G_gt_init, remove_redundant=True, verbose=False)

    print(("G_gt_init.nodes():", G_gt_init.nodes()))
    (u, v) = G_gt_init.edges()[0]
    print(("random edge props:", G_gt_init.edge[u][v]))

    # proposal graph
    G_p_init = nx.read_gpickle(prop_file)
    #G_p_init0 = nx.read_gpickle(prop_file)
    #G_p_init1 = osmnx_funcs.simplify_graph(G_p_init0.to_directed()).to_undirected()
    #G_p_init = osmnx_funcs.project_graph(G_p_init1)
    G_p_init = apls.create_edge_linestrings(
        G_p_init, remove_redundant=True, verbose=False)

    t0 = time.time()
    print("\nComputing score...")
    match_list, score = compute_sp(G_gt_init, G_p_init,
                                   x_coord=x_coord, y_coord=y_coord,
                                   weight=weight, query_radius=query_radius,
                                   length_buffer=length_buffer, n_routes=n_routes,
                                   make_plots=True,
                                   verbose=verbose)
    print(("score:", score))
    print(("Time to compute score:", time.time() - t0, "seconds"))

    ############
    # also compute total topo metric for entire folder
    if run_all:
        t0 = time.time()
        plt.close('all')
        score_list = []
        match_list = []
        for i, f in enumerate(name_list):

            if i == 0:
                make_plots = True
            else:
                make_plots = False

            # get original graph
            outroot = f.split('.')[0]
            print("\n", i, "/", len(name_list), "outroot:", outroot)
            #print ("\n", i, "/", len(name_list), "outroot:", outroot)
            gt_file = os.path.join(truth_dir, f)

            # ground truth graph
            G_gt_init = nx.read_gpickle(gt_file)
            #G_gt_init1 = osmnx_funcs.simplify_graph(G_gt_init0.to_directed()).to_undirected()
            #G_gt_init = osmnx_funcs.project_graph(G_gt_init1)
            G_gt_init = apls.create_edge_linestrings(
                G_gt_init, remove_redundant=True, verbose=False)
            if len(G_gt_init.nodes()) == 0:
                continue

            # proposal graph
            prop_file = os.path.join(prop_dir, outroot + '.gpickle')
            if not os.path.exists(prop_file):
                score_list.append(0)
                continue

            G_p_init0 = nx.read_gpickle(prop_file)
            G_p_init1 = osmnx_funcs.simplify_graph(
                G_p_init0.to_directed()).to_undirected()
            G_p_init = osmnx_funcs.project_graph(G_p_init1)
            G_p_init = apls.create_edge_linestrings(
                G_p_init, remove_redundant=True, verbose=False)

            match_list_tmp, score = compute_sp(G_gt_init, G_p_init,
                                               x_coord=x_coord, y_coord=y_coord,
                                               weight=weight, query_radius=query_radius,
                                               length_buffer=length_buffer, n_routes=n_routes,
                                               make_plots=make_plots,
                                               verbose=verbose)
            score_list.append(score)
            match_list.extend(match_list_tmp)

        # compute total score
        # total score is fraction of routes that match
        sp_tot = 1.0 * np.sum(match_list) / len(match_list)
        #score_tot = np.sum(score_list)

        print(("Total sp metric for", len(name_list), "files:"))
        print(("  query_radius:", query_radius, "length_buffer:", length_buffer))
        print(("  n_measurement_nodes:", n_measurement_nodes, "n_routes:", n_routes))
        print(("  total time elapsed to compute sp and make plots:",
               time.time() - t0, "seconds"))
        print(("  total sp:", sp_tot))
