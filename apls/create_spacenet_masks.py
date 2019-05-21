#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:56:31 2017

@author: avanetten
"""


import os
import sys
import time
import argparse
import pandas as pd

# add apls path and import apls_tools
path_apls_src = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_apls_src)
import apls_utils


###############################################################################
def create_masks(path_data, buffer_meters=2, n_bands=3,
                 burnValue=150, make_plots=True, overwrite_ims=False,
                 output_df_file='',
                 header=['name', 'im_file', 'im_vis_file', 'mask_file',
                         'mask_vis_file']):
    '''
    Create masks from files in path_data.
    Write 8bit images and masks to file.
    Return a dataframe of file locations with the following columns:
        ['name', 'im_file', 'im_vis_file', 'mask_file', 'mask_vis_file']
    We record locations of im_vis_file and mask_vis_file in case im_file
      or mask_file is not 8-bit or has n_channels != [1,3]
    if using 8band data, the RGB-PanSharpen_8bit should already exist, so
        3band should be run prior to 8band
    '''

    t0 = time.time()
    # set paths
    path_labels = os.path.join(path_data, 'geojson/spacenetroads')
    # output directories
    path_masks = os.path.join(path_data, 'masks_' + str(buffer_meters) + 'm')
    path_masks_plot = os.path.join(
        path_data, 'masks_' + str(buffer_meters) + 'm_plots')
    # image directories
    path_images_vis = os.path.join(path_data, 'RGB-PanSharpen_8bit')
    if n_bands == 3:
        path_images_raw = os.path.join(path_data, 'RGB-PanSharpen')
        path_images_8bit = os.path.join(path_data, 'RGB-PanSharpen_8bit')
    else:
        path_images_raw = os.path.join(path_data, 'MUL-PanSharpen')
        path_images_8bit = os.path.join(path_data, 'MUL-PanSharpen_8bit')
        if not os.path.exists(path_images_vis):
            print("Need to run 3band prior to 8band!")
            return

    # create directories
    for d in [path_images_8bit, path_masks, path_masks_plot]:
        if not os.path.exists(d):
            os.mkdir(d)

    # iterate through images, convert to 8-bit, and create masks
    outfile_list = []
    im_files = os.listdir(path_images_raw)
    nfiles = len(im_files)
    for i, im_name in enumerate(im_files):
        if not im_name.endswith('.tif'):
            continue

        # define files
        name_root = 'AOI' + im_name.split('AOI')[1].split('.')[0]
        im_file_raw = os.path.join(path_images_raw, im_name)
        im_file_out = os.path.join(path_images_8bit, im_name)
        im_file_out_vis = im_file_out.replace('MUL', 'RGB')
        # get visible file (if using 8band imagery we want the 3band file
        # for plotting purposes)
        # if n_bands == 3:
        #    im_file_out_vis = im_file_out
        # else:
        #    name_vis = im_name.replace('MUL', 'RGB')
        #    im_file_out_vis = os.path.join(path_images_vis, name_vis)

        # convert to 8bit, if desired
        if not os.path.exists(im_file_out) or overwrite_ims:
            apls_utils.convert_to_8Bit(im_file_raw, im_file_out,
                                       outputPixType='Byte',
                                       outputFormat='GTiff',
                                       rescale_type='rescale',
                                       percentiles=[2, 98])

        # determine output files
        # label_file = os.path.join(path_labels, 'spacenetroads_AOI_2_Vegas_' \
        #                             + name_root + '.geojson')
        label_file = os.path.join(path_labels, 'spacenetroads_' + name_root
                                  + '.geojson')
        label_file_tot = os.path.join(path_labels, label_file)
        mask_file = os.path.join(path_masks,  name_root + '.png')
        if make_plots:
            plot_file = os.path.join(path_masks_plot,  name_root + '.png')
        else:
            plot_file = ''

        print("\n", i+1, "/", nfiles)
        print("  im_name:", im_name)
        print("  name_root:", name_root)
        print("  im_file_out:", im_file_out)
        print("  mask_file:", mask_file)
        print("  output_plot_file:", plot_file)

        # create masks
        if not os.path.exists(mask_file) or overwrite_ims:
            mask, gdf_buffer = apls_utils.get_road_buffer(label_file_tot,
                                                          im_file_out_vis,
                                                          mask_file,
                                                          buffer_meters=buffer_meters,
                                                          burnValue=burnValue,
                                                          bufferRoundness=6,
                                                          plot_file=plot_file,
                                                          figsize=(6, 6),
                                                          fontsize=8,
                                                          dpi=500,
                                                          show_plot=False,
                                                          verbose=False)

        # resize in ingest so we don't have to save the very large arrays
        outfile_list.append([im_name, im_file_out, im_file_out_vis,
                             mask_file, mask_file])

    # make dataframe and save
    df = pd.DataFrame(outfile_list, columns=header)
    if len(output_df_file) > 0:
        df.to_csv(output_df_file, index=False)
    print("\ndf.ix[0]:", df.ix[0])
    print("\nTotal data length:", len(df))
    t4 = time.time()
    print("Time to run create_masks():", t4 - t0, "seconds")
    return df


###############################################################################
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', default='/spacenet_data/sample_data/AOI_2_Vegas_Train', type=str,
                        help='Folder containing imagery and geojson labels')
    parser.add_argument('--output_df_path', default='/spacenet_data/sample_data', type=str,
                        help='csv of dataframe containing image and mask locations')
    parser.add_argument('--buffer_meters', default=2, type=float,
                        help='Buffer distance (meters) around graph')
    parser.add_argument('--n_bands', default=3, type=int,
                        help='Number of bands to use [3,8]')
    parser.add_argument('--burnValue', default=150, type=int,
                        help='Value of road pixels (for plotting)')
    parser.add_argument('--make_plots', default=1, type=int,
                        help='Switch to create gridded plots of geojson, image, and mask')
    parser.add_argument('--overwrite_ims', default=1, type=int,
                        help='Switch to overwrite 8bit images and masks')

    args = parser.parse_args()

    data_root = 'AOI' + args.path_data.split('AOI')[-1].replace('/', '_')
    output_df_file = os.path.join(args.output_df_path, data_root + '_'
                                  + 'files_loc_'
                                  + str(args.buffer_meters) + 'm.csv')

    path_masks = create_masks(args.path_data,
                              buffer_meters=args.buffer_meters,
                              n_bands=args.n_bands,
                              burnValue=args.burnValue,
                              output_df_file=output_df_file,
                              make_plots=bool(args.make_plots),
                              overwrite_ims=bool(args.overwrite_ims))
    print("Output_df_file:", output_df_file)

    return path_masks


###############################################################################
if __name__ == "__main__":
    main()
