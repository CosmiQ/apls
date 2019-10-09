#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:30:33 2019

@author: avanetten

Execution example:
    python /apls/apls/gt_graph_to_wkt.py  \
    --root_dir=/spacenet/competitions/SN5_roads/train/AOI_7_Moscow/

"""

import os
import apls
import argparse
import osmnx_funcs
import pandas as pd
# import shapely.wkt


###############################################################################
def gt_geojson_to_wkt(geojson_path, im_path,
                      weight_keys=['length', 'travel_time_s'],
                      subgraph_filter_weight='length', min_subgraph_length=5,
                      travel_time_key='travel_time_s',
                      speed_key='inferred_speed_mps',
                      use_pix_coords=False,
                      verbose=False,
                      simplify=False,
                      refine=True,
                      super_verbose=False):
    '''
    Create wkt list of pixel coords in ground truth graph, for use in SpaceNet
    TopCoder competition.
    im_path has name like: ...SN5_roads_train_AOI_7_Moscow_PS-RGB_chip996.tif
    if weight == [], output list of [name_root, geom_pix_wkt]
    else:  output list of [name_root, geom_pix_wkt, weight1, weight2]
    '''

    # linestring = "LINESTRING {}"
    im_name = os.path.basename(im_path)

    # get name_root of image file
    name_root = im_name.split('.')[0].replace('PS-RGB_', '').replace('PS-MS_', '')
    # # v0
    # AOI_root = 'AOI' + im_name.split('AOI')[-1]
    # name_root = AOI_root.split('.')[0].replace('PS-RGB_', '').replace('PS-MS_', '')
    print("name_root:", name_root)
    print("im_path:", im_path)
    
    G_gt, _ = apls._create_gt_graph(geojson_path, im_path,
                     subgraph_filter_weight=subgraph_filter_weight,
                     min_subgraph_length=min_subgraph_length,
                     travel_time_key=travel_time_key,
                     speed_key=speed_key,
                     use_pix_coords=use_pix_coords,
                     refine_graph=refine,
                     simplify_graph=simplify,
                     verbose=verbose,
                     super_verbose=super_verbose)
    
    # simplify and turn to undirected
    if simplify:
        try:
            G_gt = osmnx_funcs.simplify_graph(G_gt).to_undirected()
        except:
            G_gt = G_gt.to_undirected()
    else:
        G_gt = G_gt.to_undirected()

    # return  [name_root, "LINESTRING EMPTY"] if no edges
    if (len(G_gt.nodes()) == 0) or (len(G_gt.edges()) == 0):
        print("  Empty graph")
        if len(weight_keys) > 0:
            return [[name_root, "LINESTRING EMPTY"] + [0] * len(weight_keys)]
        else:
            return [[name_root, "LINESTRING EMPTY"]]

    # extract geometry pix wkt, save to list
    wkt_list = []
    for i, (u, v, attr_dict) in enumerate(G_gt.edges(data=True)):
        print("attr_dict:", attr_dict)
        geom_pix_wkt = attr_dict['geometry_pix'].wkt
        if verbose:
            print(i, "/", len(G_gt.edges()), "u, v:", u, v)
            # print("  attr_dict:", attr_dict)
            print("  geom_pix_wkt:", geom_pix_wkt)

        wkt_item_root = [name_root, geom_pix_wkt]
        if len(weight_keys) > 0:
            weights = [attr_dict[w] for w in weight_keys]
            if verbose:
                print("  weights:", weights)
            wkt_list.append(wkt_item_root + weights)  # [name_root, geom_pix_wkt, weight])
        else:
            wkt_list.append(wkt_item_root)

    if verbose:
        print("wkt_list:", wkt_list)

    return wkt_list


###############################################################################
def gt_geojson_dir_to_wkt(geojson_dir, im_dir, output_csv_path,
                     weight_keys=['length', 'travel_time_s'],
                     subgraph_filter_weight='length', min_subgraph_length=5,
                     travel_time_key='travel_time_s',
                     speed_key='inferred_speed_mps',
                     use_pix_coords=False,
                     simplify=False,
                     verbose=False,
                     super_verbose=False):

    # make dict of image chip id to file name
    chipper = ''
    im_chip_dict = {}
    for im_name in [z for z in os.listdir(im_dir) if z.endswith('.tif')]:
        chip_id = im_name.split('chip')[-1].split('.')[0]
        if 'chip' in im_name:
            chipper = 'chip'
            chip_id = im_name.split(chipper)[-1].split('.')[0]
        elif 'img' in im_name:
            chipper = 'img'
            chip_id = im_name.split(chipper)[-1].split('.')[0]
        im_chip_dict[chip_id] = im_name
    if verbose:
        print("im_chip_dict:", im_chip_dict)
        
    # iterate through geojsons
    wkt_list_tot = []
    geojson_paths = sorted([z for z in os.listdir(geojson_dir) 
                                          if z.endswith('.geojson')])
    for i, geojson_name in enumerate(geojson_paths):
        
        # get image name
        chip_id =  geojson_name.split(chipper)[-1].split('.')[0]
        try:
            im_name = im_chip_dict[chip_id]
        except:
            print("im_name not in im_chip_dict:", im_name)
            return
            # continue

        geojson_path = os.path.join(geojson_dir, geojson_name)
        im_path = os.path.join(im_dir, im_name)

        if verbose:
            print(i, "/", len(geojson_paths), "geojson:", geojson_path,
                  "im_path:", im_path)

        wkt_list = gt_geojson_to_wkt(geojson_path, im_path,
                     weight_keys=weight_keys,
                     subgraph_filter_weight=subgraph_filter_weight,
                     min_subgraph_length=min_subgraph_length,
                     travel_time_key=travel_time_key,
                     speed_key=speed_key,
                     use_pix_coords=use_pix_coords,
                     simplify=simplify,
                     verbose=verbose,
                     super_verbose=super_verbose)

        wkt_list_tot.extend(wkt_list)

    # create dataframe
    if len(weight_keys) > 0:
        cols = ['ImageId', 'WKT_Pix'] + weight_keys
    else:
        cols = ['ImageId', 'WKT_Pix']

    # use 'length_m' instead?
    cols = [z.replace('length', 'length_m') for z in cols]

    # print("wkt_list_tot:", wkt_list_tot)
    # print("\n")
    # for i in wkt_list_tot:
    #     print(len(i))
    print("cols:", cols)

    df = pd.DataFrame(wkt_list_tot, columns=cols)
    print("df:", df)
    # save
    df.to_csv(output_csv_path, index=False)

    return df


###############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='', type=str,
                        help='Root directory of geojson data')
    parser.add_argument('--PSRGB_dir', default='', type=str,
                        help='PS-RGB dir, if '', assume in root_dir')    
    parser.add_argument('--travel_time_key', default='travel_time_s', type=str,
                        help='key for travel time')    
    parser.add_argument('--speed_key', default='inferred_speed_mps', type=str,
                        help='key for road speed')    
    parser.add_argument('--out_file_name',
                        default='geojson_roads_speed_wkt_weighted_v0.csv',
                        type=str,
                        help='name for output file')    
    parser.add_argument('--simplify_graph', default=True, type=bool,
                        help='switch to simplify graph prior to saving')    
    args = parser.parse_args()

    weight_keys = ['length', 'travel_time_s']
    verbose = True

    root_dir = args.root_dir  # '/raid/cosmiq/spacenet/competitions/SN5_roads/gt_to_topcoder_mumbai'
    if len(args.PSRGB_dir) > 0:
        im_dir = args.PSRGB_dir
    else:
        im_dir = os.path.join(root_dir, 'PS-RGB')
    geojson_dir = os.path.join(root_dir, 'geojson_roads_speed')
    # get name
    out_prefix = '_'.join(root_dir.split('/')[-3:])
    output_csv_path = os.path.join(root_dir, out_prefix + args.out_file_name)

    print("output_csv_path:", output_csv_path)
    df = gt_geojson_dir_to_wkt(geojson_dir, im_dir,
                               output_csv_path=output_csv_path,
                               travel_time_key=args.travel_time_key,
                               speed_key=args.speed_key,
                               simplify=args.simplify_graph,
                               weight_keys=weight_keys,
                               verbose=verbose)
