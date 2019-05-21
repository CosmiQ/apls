#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:30:30 2019

@author: avanetten

https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17036&pm=14735
https://arxiv.org/pdf/1807.01232.pdf


Appendix A: The SpaceNet Roads Dataset Labeling Guidelines
The SpaceNet Roads Dataset labeling guidelines:

1.	Road vectors must be drawn as a center line within 2m (7 pixels) of observed road
a.	The centerline of a road is defined as the centerline of the roadway. If a road has an even number of lanes, the centerline shall be drawn on the line separating lanes.  If the road has an odd number of lanes then the centerline should be drawn down the center of the middle lane.
b.	Divided highways should have two centerlines, a centerline for each direction of traffic.  See below for the definition of a divided highway.
2.	Road vectors must be represented as a connected network to support routing.  Roads that intersect each other should share points as an intersection like instructed through OSM.  Roads that cross each other that are not connected such as while using an overpass should not share a point of connection.
3.	Roads must not bisect building footprints.
4.	Sections of a road that are a bridge or overpass must be labeled as a bridge via a Boolean flag.
5.	Divided highways must be represented as two lines with traffic direction indicated when possible.
6.	 Surface type must be classified as:  paved, unpaved, or unknown.
7.	Road will be identified by type: (Motorway, Primary, Secondary, Tertiary, Residential, Unclassified, Cart Track)
8.	Number of lanes will be listed as number of lanes for each centerline as defined in rule 1.  If road has two lanes in each direction, the number of lanes shall equal 4.  If a road has 3 lanes in one direction and 2 directions in another the total number of lanes shall be 5.


Definition of Divided Highway:

A divided highway is a road that has a median or barrier that physically prevents turns across traffic.
A median can be:
●	Concrete
●	Asphalt
●	Green Space
●	Dirt/unpaved road
A median is not:
●	Yellow hatched lines on pavement.


Road Type Guidelines:
All road types were defined using the Open Street Maps taxonomy for key=highway. The below descriptions were taken from the Open Street Maps tagging guidelines for highway and the East Africa Tagging Guidelines.
1.	motorway - A restricted access major divided highway, normally with 2 or more running lanes plus emergency hard shoulder. Access onto a motorway comes exclusively through ramps (controlled access).  Equivalent to the Freeway, Autobahn, etc.
2.	primary - National roads connect the most important cities/towns in a country. In most countries, these roads will usually be tarmacked and show center markings. (In South Sudan, however, primary roads might also be unpaved.)
3.	secondary – Secondary roads are the second most important roads in a country's transport system. They typically link medium-sized places. They may be paved but in in some countries they are not.
4.	tertiary - Tertiary roads are busy through roads that link smaller towns and larger villages. More often than not, these roads will be unpaved. However, this tag should be used only on roads wide enough to allow two cars to pass safely.
5.	residential - Roads which serve as an access to housing, without function of connecting settlements. Often lined with housing.
6.	unclassified -The least important through roads in a country's system – i.e. minor roads of a lower classification than tertiary, but which serve a purpose other than access to properties. Often link villages and hamlets. (The word 'unclassified' is a historical artifact of the UK road system and does not mean that the classification is unknown; you can use highway=road for that.)
7.	Cart track – This is a dirt path that shows vehicle traffic that is less defined than a residential

Additional information and rules for road identification come from following sources:
1: http://wiki.openstreetmap.org/wiki/Highway_Tag_Africa
2: http://wiki.openstreetmap.org/wiki/East_Africa_Tagging_Guidelines
3: http://wiki.openstreetmap.org/wiki/Key:highway


GeoJSON Schema
Attributes:
1)	“geometry”: Linestring
2)	“road_id”: int
Identifier Index
3)	“road_type”: int
1: Motorway
2: Primary
3: Secondary
4: Tertiary
5: Residential
6: Unclassified
7: Cart track
4)	“paved”: int
1: Paved
2: Unpaved
3: Unknown
5)	“bridge_typ”: int
1: Bridge
2: Not a bridge
3: Unknown
6)	“lane_number”: int
1: one lane
2: two lanes
3: three lanes
etc.


# geojson example
{ "type": "Feature", "properties": { "gid": 15806, "road_id": 24791, "road_type": 5, "paved": 1, "bridge": 2, "one_way": 2, "heading": 0.0, "lane_numbe": 2, "ingest_tim": "2017\/09\/24 20:36:06.436+00", "edit_date": "2017\/09\/24 20:36:06.436+00", "edit_user": "ian_kitchen", "production": "0", "imagery_so": "0", "imagery_da": "0", "partialBuilding": 1.0, "partialDec": 0.0 },
    "geometry": { "type": "LineString", "coordinates": [ [ -115.305975139809291, 36.179169421086783, 0.0 ], [ -115.305540626738249, 36.179686396492464, 0.0 ], [ -115.305150516462803, 36.180003559318038, 0.0 ], [ -115.304760406187356, 36.18037781145221, 0.0 ], [ -115.304287833577249, 36.180932846396956, 0.0 ], [ -115.304305558679488, 36.18094769983459, 0.0 ] ] } }
"""

import os
import json
import fiona
import random
random.seed(2018)


###############################################################################
def speed_func(geojson_row):
    '''
    Infer road speed limit based on SpaceNet properties
    # geojson example
    { "type": "Feature", "properties": { "gid": 15806, "road_id": 24791,
            "road_type": 5, "paved": 1, "bridge": 2, "one_way": 2,
            "heading": 0.0, "lane_numbe": 2,
            "ingest_tim": "2017\/09\/24 20:36:06.436+00",
            "edit_date": "2017\/09\/24 20:36:06.436+00",
            "edit_user": "ian_kitchen", "production": "0", "imagery_so": "0",
            "imagery_da": "0", "partialBuilding": 1.0, "partialDec": 0.0 },
            "geometry": { "type": "LineString", "coordinates": [ [ -115.305975139809291, 36.179169421086783, 0.0 ], [ -115.305540626738249, 36.179686396492464, 0.0 ], [ -115.305150516462803, 36.180003559318038, 0.0 ], [ -115.304760406187356, 36.18037781145221, 0.0 ], [ -115.304287833577249, 36.180932846396956, 0.0 ], [ -115.304305558679488, 36.18094769983459, 0.0 ] ] }
    }
    '''

    road_type = int(geojson_row['properties']['road_type'])
    # lane number was incorrectly labeled in initial geojsons
    try:
        num_lanes = int(geojson_row['properties']['lane_numbe'])
    except:
        num_lanes = int(geojson_row['properties']['lane_number'])
    surface = int(geojson_row['properties']['paved'])
    try:
        bridge = int(geojson_row['properties']['bridge'])
    except:
        bridge = int(geojson_row['properties']['bridge_type'])

    # road type (int)
    '''
    1: Motorway
    2: Primary
    3: Secondary
    4: Tertiary
    5: Residential
    6: Unclassified
    7: Cart track
    '''
    road_type_dict = {
        1: 60,
        2: 45,
        3: 35,
        4: 25,
        5: 25,
        6: 20,
        7: 15
    }

    # feed in [road_type][num_lanes]
    nested_speed_dict = {
        1: {1: 45, 2: 50, 3: 55, 4: 65, 5: 65, 6: 65, 7: 65, 8: 65},
        2: {1: 35, 2: 40, 3: 45, 4: 45, 5: 45, 6: 45, 7: 45, 8: 45},
        3: {1: 30, 2: 30, 3: 30, 4: 30, 5: 30, 6: 30, 7: 30, 8: 30},
        4: {1: 25, 2: 25, 3: 25, 4: 25, 5: 25, 6: 25, 7: 25, 8: 25},
        5: {1: 25, 2: 25, 3: 25, 4: 25, 5: 25, 6: 25, 7: 25, 8: 25},
        6: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20},
        7: {1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 15},
    }

    # multiply speed by this factor based on surface
    road_surface_dict = {
        1: 1,
        2: 0.5
    }

    bridge_dict = {
        1: 0.8,
        2: 1}

    # default speed in miles per hour
    speed_init_mph = nested_speed_dict[road_type][num_lanes]
    # reduce speed for unpaved or bridge
    speed_final_mph = speed_init_mph * road_surface_dict[surface] \
        * bridge_dict[bridge]
    # get speed in meters per second
    speed_final_mps = 0.44704 * speed_final_mph

    return speed_final_mph, speed_final_mps


###############################################################################
def update_feature_name(geojson_row, name_bad, name_good):
    # optional: also update "ingest_tim" tag
    if name_bad in geojson_row['properties'].keys():
        x_tmp = geojson_row['properties'][name_bad]
        del geojson_row['properties'][name_bad]
        geojson_row['properties'][name_good] = x_tmp
    return


###############################################################################
def add_speed_to_geojson(geojson_path_in, geojson_path_out,
                         randomize_coords=False,
                         verbose=True):
    '''Update geojson data to add inferred speed information'''

    with open(geojson_path_in, 'r+') as f:
        geojson_data = json.load(f)

        for i, geojson_row in enumerate(geojson_data['features']):
            if verbose:
                print("\ngeojson_row:", geojson_row)

            # optional: also update "ingest_tim" tag
            if 'ingest_tim' in geojson_row['properties'].keys():
                x_tmp = geojson_row['properties']['ingest_tim']
                del geojson_row['properties']['ingest_tim']
                geojson_row['properties']['ingest_time'] = x_tmp

            # optional: also update "bridge_typ" tag
            if 'bridge_typ' in geojson_row['properties'].keys():
                x_tmp = geojson_row['properties']['bridge_typ']
                del geojson_row['properties']['bridge_typ']
                geojson_row['properties']['bridge_type'] = x_tmp

            # optional: also update "lane_numbe" tag
            if 'lane_numbe' in geojson_row['properties'].keys():
                x_tmp = geojson_row['properties']['lane_numbe']
                del geojson_row['properties']['lane_numbe']
                geojson_row['properties']['lane_number'] = x_tmp

            # infer route speed limit
            speed_mph, speed_mps = speed_func(geojson_row)
            if verbose:
                print("  speed_mph, speed_mps:", speed_mph, speed_mps)
            # update properties
            geojson_row['properties']['speed_mph'] = speed_mph
            geojson_row['properties']['speed_m/s'] = speed_mps

            if randomize_coords:
                rand_mag = 3.14*10**(-5)
                coords = geojson_row['geometry']['coordinates']
                print("coords:", coords)
                for jtmp in range(0, len(coords)):
                    # add random value to coords
                    coords[jtmp][0] +=  -1.0*rand_mag/2 + rand_mag * random.random()
                    coords[jtmp][1] +=  -1.0*rand_mag/2 + rand_mag * random.random()
                print("coords:", coords)
                geojson_row['geometry']['coordinates'] = coords

    # save file
    with open(geojson_path_out, 'w') as f:
        f.write(json.dumps(geojson_data))

#    # older version that doesn't print correctly
#    geojson_data = fiona.open(geojson_path, 'r')
#    out = []
#    for i,geojson_row in enumerate(geojson_data):
#        if verbose:
#            print ("\ngeojson_row:", geojson_row)
#        # infer route speed limit
#        speed_mph, speed_mps = speed_func(geojson_row)
#        if verbose:
#            print ("  speed_mph, speed_mps:", speed_mph, speed_mps)
#        # update properties
#        geojson_row['properties']['speed_mph'] = speed_mph
#        geojson_row['properties']['speed_m/s'] = speed_mps
#        #out.append(geojson_row) 
#    # save file
#    with open(geojson_path_out, 'w') as f:
#        json.dump(out, f, ensure_ascii=False)

    return


###############################################################################
def update_geojson_dir(geosjon_dir_in, geojson_dir_out, 
                       randomize_coords=False, verbose=True):
    '''Update geojson data to add inferred speed information for entire
    directory'''

    os.makedirs(geojson_dir_out, exist_ok=True)

    json_files = [j for j in os.listdir(geojson_dir_in) if j.endswith('.geojson')]
    if verbose:
        print("json_files:", json_files)

    for i, json_file in enumerate(json_files):
        if verbose:
            print(i, "/", len(json_files), json_file)
        geojson_path_in = os.path.join(geojson_dir_in, json_file)
        geojson_path_out = os.path.join(geojson_dir_out, json_file)
        add_speed_to_geojson(geojson_path_in, geojson_path_out,
                             randomize_coords=randomize_coords,
                             verbose=verbose)

    return


###############################################################################
if __name__ == "__main__":

    # Example
    randomize_coords = True
    geojson_dir_in = '/raid/cosmiq/apls/sample_data/gt_json_prop_pkl/ground_truth'
    geojson_dir_out = '/raid/cosmiq/apls/sample_data/gt_json_prop_pkl/ground_truth_randomized'
#    randomize_coords = False
#    geojson_dir_in = '/raid/cosmiq/spacenet/data/spacenetv2/spacenetLabels/AOI_2_Vegas/400m/'
#    geojson_dir_out = '/raid/cosmiq/spacenet/data/spacenetv2/spacenetLabels/AOI_2_Vegas/400m_noveau/'
    run_single = False
    run_dir = True
    verbose = True

    # single example
    if run_single:
        json_file = 'spacenetroads_AOI_2_Vegas_img16.geojson'
        os.makedirs(geojson_dir_out, exist_ok=True)
        geojson_path_in = os.path.join(geojson_dir_in, json_file)
        geojson_path_out = os.path.join(geojson_dir_out, json_file)

        with open(geojson_path_in, 'r+') as f:
            json_data = json.load(f)
        json_data['features']

        source = fiona.open(geojson_path_in, 'r')
        for i, geojson_row in enumerate(source):
            print("\ngeojson_row:", geojson_row)
            speed_mph, speed_mps = speed_func(geojson_row)
            print("  speed_mph, speed_mps:", speed_mph, speed_mps)

            # update properties?
            geojson_row['properties']['speed_mph'] = speed_mph
            geojson_row['properties']['speed_m/s'] = speed_mps
            # source[i] = geojson_row

        # create new geojson
        add_speed_to_geojson(geojson_path_in, geojson_path_out,
                             randomize_coords=randomize_coords)

    # full directory
    if run_dir:
        update_geojson_dir(geojson_dir_in, geojson_dir_out,
                           randomize_coords=randomize_coords, verbose=verbose)
