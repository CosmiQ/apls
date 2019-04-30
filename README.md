# APLS
Python code to evaluate the APLS metric

README

==================

Code Overview

This code evaluates the Average Path Length Similarity (APLS) metric to measure the difference between ground truth and proposal graphs.  The metric sums the differences in optimal path lengths between all nodes in the ground truth graph G and the proposal graph G’.   For further details, see [Blog1](https://medium.com/the-downlinq/spacenet-road-detection-and-routing-challenge-part-i-d4f59d55bfce) and [Blog2](https://medium.com/the-downlinq/spacenet-road-detection-and-routing-challenge-part-ii-apls-implementation-92acd86f4094).  For details on how to prepare data for the SpaceNet Road Detection and Routing Challenge, see [Blog3](https://medium.com/the-downlinq/creating-training-datasets-for-the-spacenet-road-detection-and-routing-challenge-6f970d413e2f).

==================

1.0:	Packages required (included in apls_environment.yml):


	Networkx
	Osmnx
	Scipy
	Numpy
	Utm
	Shapely
	Fiona
	Osgeo (gdal, osr, ogr)
	Geopandas
	Matplotlib
==================

2.0: graphTools.py

This script parses geojson labels into osmnx format for analysis.  The valid_road_types field filters out only those road types desired (e.g. motorway, primary, etc.).  This function requires the osgeo packages: gdal, osr, ogr.

==================

2.1: apls.py

Primary function for comparing ground truth and proposal graphs.  The actual metric takes up a relatively small portion of this code (functions: single_path_metric, path_sim_metric, and compute_metric).  Much of the code (functions: cut_linestring, get_closest_edge, insert_point, insert_control_points, great_graph_midpoints) is concerned with injecting nodes into graphs. We inject nodes (i.e. control points) into the graph at a predetermined distance along edges, and at the location nearest proposal nodes.  Injecting nodes is essential to properly compare graphs, though it unfortunately requires quite a bit of code.  

If graphTools.py is not installed (or more likely, if gdal is difficult to install), simply comment out the following functions in apls.py: set_pix_coords and create_gt_graph.  

The example in main() is self-contained, and demonstrates the process to insert midpoints into ground truth and proposal graphs, get shortest paths, and then compute the metric.  

Instructions for downloading SpaceNet data can be found [here](https://github.com/SpaceNetChallenge/utilities/tree/master/content/download_instructions).

==================

3.0:	apls.py Execution

Use conda to install all packages https://conda.io/miniconda.html (currently tested with OSX and python 2)

	cd /path/to/apls/src
	conda env create -f apls_environment.yml   # to deactivate environment: source deactivate
	source activate apls_environment
	python apls.py 
	# for further details: python apls.py --help
	#     primary option will 'test_method', experiment with: 
	#		 python apls.py test_method=pkl
	#		 python apls.py test_method=test_geojson
	#		 python apls.py test_method=osmnx
			 


==================

4.0:	apls.py Outputs

Running apls.py yields a number of plots in the example_output_ims directory, along with the APLS score

![Alt text](/example_output_ims/pkl/RGB-PanSharpen_img1447/single_source_route_ground_truth.png?raw=true "Figure 1")

![Alt text](/example_output_ims/pkl/RGB-PanSharpen_img1447/all_pairs_paths_diffs_prop_to_gt.png?raw=true "Figure 2")


==================

5.0:	create_spacenet_masks.py

Run this script to create training masks with spacenet geojsons

	cd /path/to/apls/src
	python create_spacenet_masks.py
	# results are saved in example_output_ims/AOI_2_Vegas_Train

