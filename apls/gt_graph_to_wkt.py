#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:30:33 2019

@author: avanetten
"""

import os
import apls
import argparse
import pandas as pd
# import shapely.wkt


###############################################################################
def gt_geojson_to_wkt(geojson_path, im_path,
                      weight_keys=['length', 'travel_time_s'],
                      subgraph_filter_weight='length', min_subgraph_length=5,
                      travel_time_key='travel_time_s',
                      speed_key='speed_m/s',
                      use_pix_coords=False,
                      verbose=False,
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
    AOI_root = 'AOI' + im_name.split('AOI')[-1]
    name_root = AOI_root.split('.')[0].replace('PS-RGB_', '')
    print("name_root:", name_root)

    G_gt, _ = apls._create_gt_graph(geojson_path, im_path,
                     subgraph_filter_weight=subgraph_filter_weight,
                     min_subgraph_length=min_subgraph_length,
                     travel_time_key=travel_time_key,
                     speed_key=speed_key,
                     use_pix_coords=use_pix_coords,
                     verbose=verbose,
                     super_verbose=super_verbose)
    
    # extract geometry pix wkt, save to list
    wkt_list = []
    for i, (u, v, attr_dict) in enumerate(G_gt.edges(data=True)):
        geom_pix_wkt = attr_dict['geometry_pix'].wkt
        if verbose:
            print(i, "/", len(G_gt.edges()), "u, v:", u, v)
            print("  attr_dict:", attr_dict)
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
                     speed_key='speed_m/s',
                     use_pix_coords=False,
                     verbose=False,
                     super_verbose=False):

    # make dict of image chip id to file name
    im_chip_dict = {}
    for im_name in [z for z in os.listdir(im_dir) if z.endswith('.tif')]:
        chip_id = im_name.split('chip')[-1].split('.')[0]
        im_chip_dict[chip_id] = im_name
    if verbose:
        print("im_chip_dict:", im_chip_dict)
        
    # iterate through geojsons
    wkt_list_tot = []
    geojson_paths = [z for z in os.listdir(geojson_dir) 
                                          if z.endswith('.geojson')]
    for i, geojson_name in enumerate(geojson_paths):
        # get image name
        chip_id =  geojson_name.split('chip')[-1].split('.')[0]
        try:
            im_name = im_chip_dict[chip_id]
        except:
            print("im_name not in im_chip_dict:", im_name)
            return
            continue

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
                     verbose=verbose,
                     super_verbose=super_verbose)

        wkt_list_tot.extend(wkt_list)

    # create dataframe
    if len(weight_keys) > 0:
        cols = ['ImageId', 'WKT_Pix'] + weight_keys
    else:
        cols = ['ImageId', 'WKT_Pix']

    print("cols:", cols)
    # use 'length_m' instead?
    cols = [z.replace('length', 'length_m') for z in cols]

    df = pd.DataFrame(wkt_list_tot, columns=cols)
    print("df:", df)
    # save
    df.to_csv(output_csv_path, index=False)

    return df


###############################################################################
if __name__ == "__main__":

    # Single chip
    #im_path = '/raid/cosmiq/spacenet/competitions/SN5_roads/train/AOI_7_Moscow/PS-RGB/SN5_roads_train_AOI_7_Moscow_PS-RGB_chip996.tif'
    #geojson_path = '/raid/cosmiq/spacenet/competitions/SN5_roads/train/AOI_7_Moscow/geojson_roads_speed/SN5_roads_train_AOI_7_Moscow_geojson_roads_speed_chip996.geojson'
    #weight_key = 'length'
    #verbose = True
    #
    #gt_geojson_to_wkt(geojson_path, im_path, weight_key=weight_key,
    #                  verbose=verbose)

    # Entire directory
#    im_dir = '/raid/cosmiq/spacenet/competitions/SN5_roads/train/AOI_7_Moscow/PS-RGB'
#    geojson_dir = '/raid/cosmiq/spacenet/competitions/SN5_roads/train/AOI_7_Moscow/geojson_roads_speed'
#    verbose = True
#    
#    weight_key = 'travel_time_s'
#    output_csv_path = '/raid/cosmiq/spacenet/competitions/SN5_roads/train/AOI_7_Moscow/' \
#        + 'geojson_roads_speed_wkt_' + weight_key + '.csv'


    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='', type=str,
                        help='Root directory of geojson data')
    parser.add_argument('--PSRGB_dir', default='', type=str,
                        help='PS-RGB dir, if '', assume in root_dir')    
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
    output_csv_path = os.path.join(
        root_dir, out_prefix + 'geojson_roads_speed_wkt_weighted.csv')

    df = gt_geojson_dir_to_wkt(geojson_dir, im_dir,
                               output_csv_path=output_csv_path,
                               weight_keys=weight_keys, verbose=verbose)

    
    '''
    Execute
    
    scp -r /raid/cosmiq/apls/apls/ 10.123.1.70:/raid/local/src/apls/
    
    python /raid/local/src/apls/apls/gt_graph_to_wkt.py  \
        --root_dir=/nfs/data/cosmiq/spacenet/competitions/SN5_roads/tiles_upload/train/AOI_7_Moscow/
    
    python /raid/local/src/apls/apls/gt_graph_to_wkt.py  \
        --root_dir=/nfs/data/cosmiq/spacenet/competitions/SN5_roads/tiles_upload/train/AOI_8_Mumbai/

   python /raid/local/src/apls/apls/gt_graph_to_wkt.py  \
        --root_dir=/nfs/data/cosmiq/spacenet/competitions/SN5_roads/tiles_upload/train/AOI_8_Mumbai/

    
        
    '''
