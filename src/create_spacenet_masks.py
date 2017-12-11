#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:56:31 2017

@author: avanetten
"""

import os
import sys
import argparse

path_apls_src = os.path.dirname(os.path.realpath(__file__))
# add path and import apls_tools
sys.path.append(path_apls_src)
import apls_tools
reload(apls_tools)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--buffer_meters', default=2, type=float,
        help='Buffer distance (meters) around graph')
    parser.add_argument('--burnValue', default=150, type=int,
        help='Value of road pixels (for plotting)')
    parser.add_argument('--test_data_loc', default='AOI_2_Vegas_Train', type=str,
        help='Folder within sample_data directory of test data')      
    args = parser.parse_args()


    # set paths
    path_apls = os.path.dirname(path_apls_src)
    path_data = os.path.join(path_apls, 'sample_data/' + args.test_data_loc)
    path_outputs = os.path.join(path_apls, 'example_output_ims/' + args.test_data_loc)
    path_images_raw = os.path.join(path_data, 'RGB-PanSharpen')
    path_images_8bit = os.path.join(path_data, 'RGB-PanSharpen_8bit')
    path_labels = os.path.join(path_data, 'geojson/spacenetroads')
    # output directories
    path_masks = os.path.join(path_outputs, 'masks_' + str(args.buffer_meters) + 'm')
    path_masks_plot = os.path.join(path_outputs, 'masks_' + str(args.buffer_meters) + 'm_plots')
    # create directories
    for d in [path_outputs, path_images_8bit, path_masks, path_masks_plot]:
        if not os.path.exists(d):
            os.mkdir(d)
            
    
    # iterate through images, convert to 8-bit, and create masks
    im_files = os.listdir(path_images_raw)
    for im_file in im_files:
        if not im_file.endswith('.tif'):
            continue
        
        name_root = im_file.split('_')[-1].split('.')[0]    
    
        # create 8-bit image
        im_file_raw = os.path.join(path_images_raw, im_file)
        im_file_out = os.path.join(path_images_8bit, im_file)
        # convert to 8bit
        apls_tools.convert_to_8Bit(im_file_raw, im_file_out,
                               outputPixType='Byte',
                               outputFormat='GTiff',
                               rescale_type='rescale',
                               percentiles=[2,98])

        # determine output files
        label_file = os.path.join(path_labels, 'spacenetroads_AOI_2_Vegas_' \
                                     + name_root + '.geojson')
        label_file_tot = os.path.join(path_labels, label_file)
        output_raster = os.path.join(path_masks, 'mask_' + name_root + '.png')
        plot_file = os.path.join(path_masks_plot, 'mask_' + name_root + '.png')
        
        print "\nname_root:", name_root
        print "  output_raster:", output_raster
        print "  output_plot_file:", plot_file
        
        # create masks
        mask, gdf_buffer = apls_tools.get_road_buffer(label_file_tot, im_file_out, 
                                                      output_raster, 
                                                      buffer_meters=args.buffer_meters, 
                                                      burnValue=args.burnValue, 
                                                      bufferRoundness=6, 
                                                      plot_file=plot_file, 
                                                      figsize= (6,6), #(13,4), 
                                                      fontsize=8,
                                                      dpi=200, show_plot=False, 
                                                      verbose=False)  
    return
        
###############################################################################
if __name__ == "__main__":
    main()
