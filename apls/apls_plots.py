#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:34:07 2019

@author: avanetten
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import shapely
from shapely.geometry import MultiLineString
from matplotlib.patches import PathPatch
from matplotlib import collections as mpl_collections
import matplotlib.path
import skimage.io


###############################################################################
def plot_metric(C, diffs, routes_str=[],
                figsize=(10, 5), scatter_png='', hist_png='',
                scatter_alpha=0.3, scatter_size=2, scatter_cmap='jet',
                dpi=300):
    ''' Plot output of cost metric in both scatterplot and histogram format'''

    # plot diffs
    title = 'Path Length Similarity: ' + str(np.round(C, 2))
    fig, (ax0) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))
    # ax0.plot(diffs)
    ax0.scatter(list(range(len(diffs))), diffs, s=scatter_size, c=diffs,
                alpha=scatter_alpha,
                cmap=scatter_cmap)
    if len(routes_str) > 0:
        xticklabel_pad = 0.1
        ax0.set_xticks(list(range(len(diffs))))
        ax0.set_xticklabels(routes_str, rotation=50, fontsize=4)
        ax0.tick_params(axis='x', which='major', pad=xticklabel_pad)

    ax0.set_ylabel('Length Diff (Normalized)')
    ax0.set_xlabel('Path ID')
    ax0.set_title(title)
    # plt.tight_layout()
    if scatter_png:
        plt.savefig(scatter_png, dpi=dpi)

    # plot and plot diffs histo
    bins = np.linspace(0, 1, 30)
    bin_centers = np.mean(list(zip(bins, bins[1:])), axis=1)
    # digitized = np.digitize(diffs, bins)
    # bin_means = [np.array(diffs)[digitized == i].mean() for i in range(1, len(bins))]
    hist, bin_edges = np.histogram(diffs, bins=bins)
    fig, ax1 = plt.subplots(nrows=1, ncols=1,  figsize=figsize)
    # ax1.plot(bins[1:],hist, type='bar')
    # ax1.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0] )
    ax1.bar(bin_centers, 1.*hist/len(diffs),
            width=bin_centers[1]-bin_centers[0])
    ax1.set_xlim([0, 1])
    # ax1.set_ylabel('Num Routes')
    ax1.set_ylabel('Frac Num Routes')
    ax1.set_xlabel('Length Diff (Normalized)')
    ax1.set_title('Length Diff Histogram - Score: ' + str(np.round(C, 2)))
    ax1.grid(True)
    # plt.tight_layout()
    if hist_png:
        plt.savefig(hist_png, dpi=dpi)

    return


###############################################################################
###############################################################################
# For plotting the buffer...
# https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
def _ring_coding(ob):
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(ob.coords)
    codes = np.ones(n, dtype=matplotlib.path.Path.code_type) * \
        matplotlib.path.Path.LINETO
    codes[0] = matplotlib.path.Path.MOVETO
    return codes


###############################################################################
# https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
def _pathify(polygon):
    # Convert coordinates to path vertices. Objects produced by Shapely's
    # analytic methods have the proper coordinate order, no need to sort.
    vertices = np.concatenate(
        [np.asarray(polygon.exterior)]
        + [np.asarray(r) for r in polygon.interiors])
    codes = np.concatenate(
        [_ring_coding(polygon.exterior)]
        + [_ring_coding(r) for r in polygon.interiors])
    return matplotlib.path.Path(vertices, codes)
###############################################################################


###############################################################################
def _plot_buff(G_, ax, buff=20, color='yellow', alpha=0.3,
               title='Proposal Snapping',
               title_fontsize=8, outfile='',
               dpi=200,
               verbose=False):
    '''plot buffer around graph using shapely buffer'''

    # get lines
    line_list = []
    for u, v, key, data in G_.edges(keys=True, data=True):
        if verbose:
            print(("u, v, key:", u, v, key))
            print(("  data:", data))
        geom = data['geometry']
        line_list.append(geom)

    mls = MultiLineString(line_list)
    mls_buff = mls.buffer(buff)

    if verbose:
        print(("type(mls_buff) == MultiPolygon:", type(
            mls_buff) == shapely.geometry.MultiPolygon))

    if type(mls_buff) == shapely.geometry.Polygon:
        mls_buff_list = [mls_buff]
    else:
        mls_buff_list = mls_buff

    for poly in mls_buff_list:
        x, y = poly.exterior.xy
        coords = np.stack((x, y), axis=1)
        interiors = poly.interiors
        # coords_inner = np.stack((x_inner,y_inner), axis=1)

        if len(interiors) == 0:
            # ax.plot(x, y, color='#6699cc', alpha=0.0, linewidth=3,
            #     solid_capstyle='round', zorder=2)
            ax.add_patch(matplotlib.patches.Polygon(
                coords, alpha=alpha, color=color))
        else:
            path = _pathify(poly)
            patch = PathPatch(path, facecolor=color,
                              edgecolor=color, alpha=alpha)
            ax.add_patch(patch)

    ax.axis('off')
    if len(title) > 0:
        ax.set_title(title, fontsize=title_fontsize)
    if outfile:
        plt.savefig(outfile, dpi=dpi)
    return ax


###############################################################################
def _plot_node_ids(G, ax, node_list=[], alpha=0.8, fontsize=8,
                   plot_node=False, node_size=15,
                   node_color='orange'):
    '''
    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    '''
    Gnodes = set(G.nodes())

    if len(node_list) == 0:
        nodes = G.nodes()
    else:
        nodes = node_list
    for n in nodes:  # G.nodes():
        if n not in Gnodes:
            continue
        x, y = G.node[n]['x'], G.node[n]['y']
        if plot_node:
            ax.scatter(x, y, s=node_size, color=node_color)
        ax.annotate(str(n), xy=(x, y), alpha=alpha, fontsize=fontsize)

    return ax


###############################################################################
def plot_graph_on_im(G_, im_test_file, figsize=(8, 8), show_endnodes=False,
                     width_key='speed_m/s', width_mult=0.125,
                     color='lime', title='', figname='',
                     default_node_size=15,
                     max_speeds_per_line=12,  dpi=300, plt_save_quality=75,
                     ax=None, verbose=False):
    '''
    Overlay graph on image,
    if width_key == int, use a constant width'''

    try:
        im_cv2 = cv2.imread(im_test_file, 1)
        img_mpl = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB)
    except:
        img_sk = skimage.io.imread(im_test_file)
        # make sure image is h,w,channels (assume less than 20 channels)
        if (len(img_sk.shape) == 3) and (img_sk.shape[0] < 20):
            img_mpl = np.moveaxis(img_sk, 0, -1)
        else:
            img_mpl = img_sk
    h, w = img_mpl.shape[:2]

    node_x, node_y, lines, widths, title_vals = [], [], [], [], []
    # get edge data
    for i, (u, v, edge_data) in enumerate(G_.edges(data=True)):
        # if type(edge_data['geometry_pix'])
        if type(edge_data['geometry_pix']) == str:
            coords = list(shapely.wkt.loads(edge_data['geometry_pix']).coords)
        else:
            coords = list(edge_data['geometry_pix'].coords)
        if verbose:  # (i % 100) == 0:
            print("\n", i, u, v, edge_data)
            print("edge_data:", edge_data)
            print("  coords:", coords)
        lines.append(coords)
        node_x.append(coords[0][0])
        node_x.append(coords[-1][0])
        node_y.append(coords[0][1])
        node_y.append(coords[-1][1])
        if type(width_key) == str:
            if verbose:
                print("edge_data[width_key]:", edge_data[width_key])
            width = int(np.rint(edge_data[width_key] * width_mult))
            title_vals.append(int(np.rint(edge_data[width_key])))
        else:
            width = width_key
        widths.append(width)

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(img_mpl)
    # plot nodes?
    if show_endnodes:
        ax.scatter(node_x, node_y, color=color, s=default_node_size, alpha=0.5)
    # plot segments
    # print (lines)
    lc = mpl_collections.LineCollection(lines, colors=color,
                                        linewidths=widths, alpha=0.4,
                                        zorder=2)
    ax.add_collection(lc)
    ax.axis('off')

    # title
    if len(title_vals) > 0:
        if verbose:
            print("title_vals:", title_vals)
        title_strs = np.sort(np.unique(title_vals)).astype(str)
        # split title str if it's too long
        if len(title_strs) > max_speeds_per_line:
            # construct new title str
            n, b = max_speeds_per_line, title_strs
            title_strs = np.insert(b, range(n, len(b), n), "\n")
            # title_strs = '\n'.join(s[i:i+ds] for i in range(0, len(s), ds))
        if verbose:
            print("title_strs:", title_strs)
        title = title + '\n' \
            + width_key + " = " + " ".join(title_strs)
    if title:
        # plt.suptitle(title)
        ax.set_title(title)

    plt.tight_layout()
    print("title:", title)
    if title:
        plt.subplots_adjust(top=0.96)
    # plt.subplots_adjust(left=1, bottom=1, right=1, top=1, wspace=5, hspace=5)

    # set dpi to approximate native resolution
    if verbose:
        print("img_mpl.shape:", img_mpl.shape)
    desired_dpi = int(np.max(img_mpl.shape) / np.max(figsize))
    if verbose:
        print("desired dpi:", desired_dpi)
    # max out dpi at 3500
    dpi = int(np.min([3500, desired_dpi]))
    if verbose:
        print("plot dpi:", dpi)

    if figname:
        plt.savefig(figname, dpi=dpi, quality=plt_save_quality)

    return ax


###############################################################################
def _plot_gt_prop_graphs(G_gt, G_prop, im_test_file,
                         figsize=(16, 8), show_endnodes=False,
                         width_key='Inferred Speed (mph)', width_mult=0.125,
                         gt_color='cyan', prop_color='lime',
                         default_node_size=15,
                         title='', figname='', adjust=True, verbose=False):
    '''Plot the ground truth, and prediction mask Overlay graph on image,
    if width_key == int, use a constant width'''

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

    print("Plotting ground truth...")
    _ = plot_graph_on_im(G_gt, im_test_file, figsize=figsize,
                         show_endnodes=show_endnodes,
                         width_key=width_key, width_mult=width_mult,
                         color=gt_color,
                         default_node_size=default_node_size,
                         title='Ground Truth:  ' + title,
                         figname='',
                         ax=ax0, verbose=verbose)
    print("Plotting proposal...")
    _ = plot_graph_on_im(G_prop, im_test_file, figsize=figsize,
                         show_endnodes=show_endnodes,
                         width_key=width_key, width_mult=width_mult,
                         color=prop_color,
                         default_node_size=default_node_size,
                         title='Proposal:  ' + title, figname='',
                         ax=ax1, verbose=verbose)

    # if title:
    #    plt.suptitle(title)
    # ax1.set_title(im_test_root)
    plt.tight_layout()
    if adjust:
        plt.subplots_adjust(top=0.96)
    # plt.subplots_adjust(left=1, bottom=1, right=1, top=1, wspace=5, hspace=5)

    if figname:
        plt.savefig(figname, dpi=300)

    return fig


###############################################################################
def plot_node_ids(G, ax, node_list=[], alpha=0.8, fontsize=8,
                  plot_node=False, node_size=15,
                  node_color='orange'):
    '''
    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    '''
    Gnodes = set(G.nodes())

    if len(node_list) == 0:
        nodes = G.nodes()
    else:
        nodes = node_list
    for n in nodes:
        if n not in Gnodes:
            continue
        x, y = G.node[n]['x'], G.node[n]['y']
        if plot_node:
            ax.scatter(x, y, s=node_size, color=node_color)
        ax.annotate(str(n), xy=(x, y), alpha=alpha, fontsize=fontsize)

    return ax
