<h1 align="center">APLS Metric</h1>
<p align="center">
<a href="http://www.cosmiqworks.org"><img src="http://www.cosmiqworks.org/wp-content/uploads/2016/02/cropped-CosmiQ-Works-Logo_R_RGB.png" width="350" alt="CosmiQ Works"></a>
<br>
<br>
</p>

- [Installation Instructions](#installation-instructions)
- [Dependencies](#dependencies)
- [License](#license)
- [Useage](#usage)

____

This package evaluates the Average Path Length Similarity (APLS) metric to measure the difference between ground truth and proposal graphs.  The metric sums the differences in optimal path lengths between all nodes in the ground truth graph G and the proposal graph G’.   This metric was used to score the SpaceNet 3 challenge.  For further details, see [Blog1](https://medium.com/the-downlinq/spacenet-road-detection-and-routing-challenge-part-i-d4f59d55bfce) and [Blog2](https://medium.com/the-downlinq/spacenet-road-detection-and-routing-challenge-part-ii-apls-implementation-92acd86f4094).  

____

## Installation Instructions

#### pip

```
pip install apls
```

____

## Dependencies
All dependencies can be found in [environment.yml](./environment.yml)

____

## License
See [LICENSE](./LICENSE.txt).

____

## Usage

apls.py compares ground truth and proposal graphs.  The actual metric takes up a relatively small portion of this code (functions: _single\_path\_metric_, _path\_sim\_metric__, and _compute\_metric_).  Much of the code (e.g. _cut\_linestring_, _get\_closest_edge_, _insert\_point_into\_G_, _insert\_control\_points_, _create\_graph\_midpoints_) is concerned with injecting nodes into graphs. We inject nodes (i.e. control points) into the graph at a predetermined distance along edges, and at the location nearest proposal nodes.  Injecting nodes is essential to properly compare graphs, though it unfortunately requires quite a bit of code.  graphTools.py parses geojson labels into networkx graphs for analysis.  Examples for running apls.py are shown below with the attached sample data

	# for further details: python apls.py --help
	
	# 1. Compare a ground truth SpaceNet geojson with a submission csv
	python apls.py --test_method=gt_json_prop_wkt --output_name=gt_json_prop_wkt \
		--max_snap_dist=4 --is_curved_eps=0.12 \
		--truth_dir=data/gt_json_prop_wkt/ground_truth_randomized \
		--wkt_file=data/gt_json_prop_wkt/proposal/sn3_sample_submission_albu.csv \
		--im_dir=data/images
	
	# 2. Compare a ground truth geojson with a proposal json
	python apls.py --test_method=gt_json_prop_json --output_name=gt_json_prop_json \
		--max_snap_dist=4 --is_curved_eps=0.12 \
		--truth_dir=data/gt_json_prop_json/AOI_2_Vegas_Train/spacenetroads \
		--prop_dir=data/gt_json_prop_json/AOI_2_Vegas_Train/osm 
			
	# 3. Compare a ground truth geojson with a pickled proposal graph 
	python apls.py --test_method=gt_json_prop_pkl --output_name=gt_json_prop_pkl \
		--truth_dir=data/gt_json_prop_pkl/ground_truth_randomized \
		--prop_dir=data/gt_json_prop_pkl/proposal \
		--im_dir=data/images
	
	# 4. Compare a pickled ground truth graph with a pickled proposal graph 
	python apls.py --test_method=gt_pkl_prop_pkl --output_name=gt_pkl_prop_pkl \
		--max_snap_dist=4 --is_curved_eps=0.12 \
		--truth_dir=data/gt_pkl_prop_pkl/ground_truth_randomized \
		--prop_dir=data/gt_pkl_prop_pkl/proposal \
		--im_dir=data/images	


### Outputs

Running apls.py yields a number of plots in the _outputs_ directory, along with the APLS score

![Alt text](/apls/data/_sample_outputs/single_source_route_ground_truth.png?raw=true "Figure 1")

![Alt text](/apls/data/_sample_outputs/all_pairs_paths_diffs_gt_to_prop.png?raw=true "Figure 2")

____

### SpaceNet Training Masks

Run the _create\_spacenet\_masks.py_ script to create training masks with spacenet geojsons
