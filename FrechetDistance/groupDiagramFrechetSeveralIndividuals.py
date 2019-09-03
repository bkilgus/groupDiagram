import math
import numpy as np
import csv
import gmplot
from sortedcontainers import SortedList
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import split, linemerge, snap, transform
import matplotlib.pyplot as plt
import pyproj
from functools import partial
import geopy.distance
srcProj = pyproj.Proj(init='EPSG:4326')
dstProj = pyproj.Proj(init="EPSG:3857")
from LatLon import LatLon
import simplekml
import collections
from SetCoverPy import setcover
import time

def get_unique_words(all_words):
    unique_words = collections.Counter()
    for word in all_words:
        unique_words[word] += 1
    return unique_words.keys()


# Initialize Data 
migration_months = ["03", "03", "04", "05", "06"]
distance_between_points = 10000
distance_between_points_max = 400000  #meters
index_of_lat = 4
index_of_long = 3
index_of_date = 2
index_of_year_start = 0
index_of_year_end = 4
index_of_month_start = 5
index_of_month_end = 7
dist = 100000
dist_tol_1 = 10200
dist_tol = 105000
resolution = 1
segmentation_1 = True
segmentation_2 = True

data_set = "ICARUS_2014.csv"
reader = csv.reader(open(data_set, 'rb'), delimiter=',')
all_tracks = list(zip(*reader))[32]
#tracks = [name for name in get_unique_words(all_tracks) if name != 'individual-local-identifier']
tracks = ["700", "701", "707"] # , "720", "727", "728", "730", "742"]
# tracks = ["GWFG_2015_408", "GWFG_2015_409", "GWFG_2015_410"]
years = ["2014"]  # tracks = ["711"]


project = partial(
    pyproj.transform,
    srcProj, # source coordinate system
    dstProj) # destination coordinate system

g = pyproj.Geod(ellps='WGS84')


def distance(point1, point2):
    # #return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    # dst = g.inv(pyproj.transform(dstProj, srcProj, point1[0], point1[1])[1],
    #       pyproj.transform(dstProj, srcProj, point1[0], point1[1])[0],
    #       pyproj.transform(dstProj, srcProj, point2[0], point2[1])[1],
    #       pyproj.transform(dstProj, srcProj, point2[0], point2[1])[0])[2]
    # # return g.inv(point1[1], point1[0], point2[1], point2[0])[2]
    # return dst
    return geopy.distance.vincenty(point1, point2).meters


def cut_linestring_at_points(linestring, points):
    return split(linestring, MultiPoint(points))


def free_space(trajectory, leash):
    # Compute the Free Space Diagram of two trajectories.
    # Calculating the length of the trajectories
    length = len(trajectory)
    # initialize lists to store distances
    dist1 = []
    dist12 = np.zeros((length, length))
    # TODO: Warnings for incorrect input or if one trajectory consists of only one point

    # Computing the distances between each point and the next.
    for p1, p2 in zip(trajectory[:-1], trajectory[1:]):
        dist1.append(distance(p1, p2))

    # Todo: check endpoint distances
    # Computing the squares of all the distances between points of two copies of the trajectory
    for index_p, p in enumerate(trajectory):
        for index_q, q in enumerate(trajectory):
            dist12[index_p, index_q] = (distance(p, q)) * (distance(p, q))

    # setting up the Free Space
    leash_sq = leash * leash
    left = np.ones((2, length, length - 1)) * -1
    bottom = np.ones((2, length - 1, length)) * -1

    # Calculating the Free Space of the trajectory with respect to a copy of the trajectory
    for index1, p1, p2 in zip(range(0, len(trajectory) - 1), trajectory[:-1], trajectory[1:]):
        # Creating a unit vector in the direction of the next point from point1.
        unit_v1 = np.zeros(2)
        if dist1[index1] != 0:
            east_west = distance(p1, (p1[0], p2[1]))
            north_south = distance(p1, (p2[0], p1[1]))
            dist_test = distance(p1, p2)
            if p2[0] < p1[0]:
                north_south = -1*north_south
            if p2[1] < p1[1]:
                east_west = -1*east_west


            #unit_v1 = [1.0 * (x1 - x2) / dist1[index1] for (x1, x2) in zip(p2, p1)]
            unit_v1 = [east_west/dist1[index1], north_south/dist1[index1]]
        for index2, q1 in enumerate(trajectory):
            # Creating a vector from point1 to point 2.
            east_west = distance(p1, (p1[0], q1[1]))
            north_south = distance(p1, (q1[0], p1[1]))
            if q1[0] < p1[0]:
                north_south = -1*north_south
            if q1[1] < p1[1]:
                east_west = -1*east_west
            #v2 = [q - p for (q, p) in zip(q1, p1)]
            v2 = [east_west, north_south]
            # Dot product finds how far from point1 the closest point on the line is.
            point_dist = np.dot(unit_v1, v2)
            point_dist_sq = point_dist ** 2
            # The square of the distance between the line segment and the point.
            short_dist = dist12[index1, index2] - point_dist_sq
            # If some part of the current line can be used by the leash.
            if short_dist <= leash_sq:
                # Calculating the envelope along the line.
                env_size = math.sqrt(leash_sq - short_dist)
                env_low = point_dist - env_size
                env_high = point_dist + env_size
                if env_high >= dist1[index1] and env_low <= 0:
                    # If the whole line is within the envelope.
                    bottom[0, index1, index2] = 0.0
                    bottom[1, index1, index2] = 1.0
                elif env_low <= 0 <= env_high:
                    # If the start of the line is within the envelope.
                    bottom[0, index1, index2] = 0.0
                    bottom[1, index1, index2] = 1.0 * env_high / dist1[index1]
                elif env_low <= dist1[index1] <= env_high:
                    # If the end of the line is within the envelope.
                    bottom[0, index1, index2] = 1.0 * env_low / dist1[index1]
                    bottom[1, index1, index2] = 1.0
                elif env_high >= 0 and env_low <= dist1[index1]:
                    # If the envelope is completely within the line.
                    bottom[0, index1, index2] = 1.0 * env_low / dist1[index1]
                    bottom[1, index1, index2] = 1.0 * env_high / dist1[index1]

    # Calculating the Free Space of the second trajectory with respect to the first. Here, compute the transposed
    # matrix of the bottom matrix
    left[0] = bottom[0].transpose()
    left[1] = bottom[1].transpose()
    return bottom, left


def compute_grid_faster(left):
    length = len(left[0])
    # compute list of indices of cells with a positive free space boundary
    positive_free_space = []
    for row in left[0]:
        positive_free_space.append([i for i, entry in enumerate(row) if entry != -1])
    positive_free_space_swapped = []
    for index_row in range(0, length-1):
        positive_free_space_swapped.append([i for i in range(0, length) if left[0, i, index_row] != -1])
    # Compute the horizontal propagation of the vertical cell boundaries of the free space. Set up the grid:
    # Each entry stores a sorted list of all (propagated) vertices within cell boundary
    grid = [[] for x in xrange(length)]
    for i, row in enumerate(grid):
        grid[i] = [SortedList() for x in xrange(length - 1)]
    # Propagation:
    # loop through all positive entries of the free space boundary encoded in left
    for i, free_space_boundary in enumerate(left):
        for j, column in enumerate(free_space_boundary):
            #print "done"
            # just positive values
            for positive_entry in positive_free_space[j]:
                entry = left[i, j, positive_entry]
                [grid[l][positive_entry].add(entry) for l in positive_free_space_swapped[positive_entry]
                 if left[0, l, positive_entry] <= entry <= left[1, l, positive_entry]
                 and entry not in grid[l][positive_entry]]
    # add -1 values for each empty list
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell.__len__() == 0:
                grid[i][j].add(-1)
    return grid


def initialize_graph(grid):
    """Compute the graph encoding the reachability information.
    One entry for each point of the grid. An entry is a list of three values, the first value stores the
    position of the (propagated) point inside the cell boundary (or -1 if the cell boundary is empty),
    the second value is the index of the leftmost reachable column following the lefward edge and the third value 
    stores the index of the leftmost reachable column following the downward edge. If there is no downward or leftwarded
    edge the values are set to None"""

    # initialize first column
    graph = []
    row_list = []
    for index_cell, cell in enumerate(grid[0]):
        cell_list = []
        for index_entry, entry in enumerate(cell):
            if entry == -1:
                cell_list.insert(index_entry, [entry, None, [None, None]])
            else:
                cell_list.insert(index_entry, [entry, 0, [None, None]])
        row_list.insert(index_cell, cell_list)
    graph.insert(0, row_list)
    return graph


def add_column(graph, grid, bottom, left, index):
    # update graph when moving right sweep line one column to the right
    # iterate over all cells from top to bottom, set all leftward edges and their labels
    row_list = []
    for index_cell, cell in enumerate(grid[index]):
        cell_list = []
        for index_entry, entry in enumerate(cell):
            # if the value was propagated to the left cell boundary:
            if entry == -1:
                cell_list.insert(index_entry, [entry, None, []])
            else:
                try:
                    # does there exist a leftward edge, i.e. is the current entry propagated to the cell to the right?
                    edge_number = grid[index - 1][index_cell].index(entry)
                    # minim of the two labels stored in the vertex the new leftward edge is pointing to
                    if graph[index - 1][index_cell][edge_number][1] is None \
                            and graph[index - 1][index_cell][edge_number][2][0] is not None:
                        label = graph[index - 1][index_cell][edge_number][2][0]
                    elif graph[index - 1][index_cell][edge_number][2][0] is None \
                            and graph[index - 1][index_cell][edge_number][1] is not None:
                        label = graph[index - 1][index_cell][edge_number][1]
                    # it the labels of the vertex the leftward edge is pointing to are both None, set the leftmost
                    # reachable colunmn index to the current index - 1.
                    elif graph[index - 1][index_cell][edge_number][2][0] is None \
                            and graph[index - 1][index_cell][edge_number][1] is None:
                        label = index - 1
                    # only if both labels are not None one can compute and use the minimum label to set the new label
                    else:
                        label = min([graph[index - 1][index_cell][edge_number][1],
                                     graph[index - 1][index_cell][edge_number][2][0]])
                    # set edge with label
                    cell_list.insert(index_entry, [entry, label, []])
                # if the entry is not in the list of points inside the cell boundary of the cell to the left of the same
                # row, there is no leftward edge and the therefore a column index further left can not be reached.
                # Set the label here to the current index, otherwise the None value would be progagated to reachable
                # cells on the right when adding the next column of the free space.
                except ValueError:
                    cell_list.insert(index_entry, [entry, index, []])
        row_list.insert(index_cell, cell_list)
    graph.insert(index, row_list)

    # for the bottom most cell, check if the left cell boundary is reachable and set label to the minimum of a
    # reachable vertex
    if grid[index][0][0] == -1:
        graph[index][0][0][2] = [None, 0]
    else:
        # no need to compute anything if the intersection of the free space of index and index-1 is non-empt. In
        # this case the correct label is already computed by the left edge.
        if left[0, index, 0] >= left[1, index-1, 0]:
            left_l = graph[index-1][0][-1][1]
            down = graph[index-1][0][-1][2][0]
            if left_l is None:
                graph[index][0][0][2] = [down, 0]
            elif down is None:
                graph[index][0][0][2] = [left_l, 0]
            else:
                graph[index][0][0][2] = [min([down, left_l]), 0]
        else:
            graph[index][0][0][2] = [None, 0]
    for index_cell in range(len(grid[index]) - 1, 0, -1):
        if grid[index][index_cell][0] == -1:
            graph[index][index_cell][0][2] = [None, None]
        else:
            row_number = index_cell
            min_right_boundary = bottom[1, index - 1, row_number]
            left_boundary = bottom[0, index - 1, row_number]
            min_labels = []
            # minimal label of a vertex reachable in the same row, but only with a diagonal edge
            if left[0, index, row_number] >= left[1, index - 1, row_number] >= 0:
                left_l = graph[index - 1][row_number][-1][1]
                down = graph[index - 1][row_number][-1][2][0]
                min_labels.append(left_l)
                min_labels.append(down)
            # check how which lower cells in the same column are reachable
            while left_boundary <= min_right_boundary and row_number > 0:
                row_number = row_number - 1
                # get label of topmost vertex of the right cell boundaries for all reachable lower cells
                left_label = graph[index-1][row_number][-1][1]
                down_label = graph[index - 1][row_number][-1][2][0]
                if left_label is not None:
                    min_labels.append(left_label)
                if down_label is not None:
                    min_labels.append(down_label)
                # update the minimum right boundary
                if bottom[1, index - 1, row_number] < min_right_boundary:
                    min_right_boundary = bottom[1, index - 1, row_number]
                left_boundary = bottom[0, index - 1, row_number]
                if left_boundary == -1:
                    break
            min_labels = [entry for entry in min_labels if entry is not None]
            try:
                min_label = min(min_labels)
            except ValueError:
                min_label = None
            # label diagonal edge
            graph[index][index_cell][0][2] = [min_label,  row_number]

    for row_index, cell in enumerate(graph[index]):
        for entry_index in range(1, len(cell)):
            if graph[index][row_index][entry_index - 1][1] is None:
                graph[index][row_index][entry_index][2] = [graph[index][row_index][entry_index - 1][2][0], None]
            elif graph[index][row_index][entry_index - 1][2][0] is None:
                graph[index][row_index][entry_index][2] = [graph[index][row_index][entry_index - 1][1], None]
            else:
                minimal_label = min(graph[index][row_index][entry_index - 1][1],
                                    graph[index][row_index][entry_index - 1][2][0])
                graph[index][row_index][entry_index][2] = [minimal_label, None]
    return graph


def find_clusters(start, end, graph, grid, left, start_end_points, vertex_check=True, seg_return=True):
    current_curve_not_yet_added = True
    endpoints = [endpoint[-1] for endpoint in start_end_points]
    # index of the current curve (where start and end lies on)
    l = SortedList()
    count_cluster_curves = 0
    segments_of_cluster = []
    try:
        index_curve = [start_end_points.index(i) for i in start_end_points if
                      start in i and end in i][0]
        if seg_return:
            for seg in range(start, end):
                segments_of_cluster.append(seg)
        else:
            segments_of_cluster.append([start, end])
        count_cluster_curves = count_cluster_curves + 1
        l.add(index_curve)
    except IndexError:
        print "start- and endpoint not on the same trajectory", start, end
    # find maximal value on index end with label at most start
    try:
        highest_index = [len(graph[end]) - 1, 1]
    except IndexError:
        print "Wrong input: End of curve higher value than size of free space", end
    while highest_index[0] >= 0:
        current_index = end
        found_at_index = False
        # find edge with highest possible y value that is labeled with a value smaller or equal to start
        # if curve must start at a vertex one only need to check the topmost vertex of cell
        if vertex_check:
            for row_index in range(highest_index[0], -1, -1):
                value = graph[current_index][row_index][-1][0]
                left_label = graph[current_index][row_index][-1][1]
                down_label = graph[current_index][row_index][-1][2][0]
                if (left_label is not None and left_label <= start
                    and value == 1 and row_index not in endpoints
                    ) or (down_label is not None and down_label <= start
                          and value == 1 and row_index not in endpoints):
                    found_at_index = True
                    start_of_path = row_index + 1  #[row_index +1 , 1]
                    cell_index = len(graph[current_index][row_index]) - 1
                    break
        else:
            for row_index in range(highest_index[0], -1, -1):
                for cell_index in range(len(graph[current_index][row_index]) - 1, -1, -1):
                    value = graph[current_index][row_index][cell_index][0]
                    left_label = graph[current_index][row_index][cell_index][1]
                    down_label = graph[current_index][row_index][cell_index][2][0]
                    if (left_label is not None and left_label <= start
                        and value <= highest_index[1]
                        ) or (down_label is not None and down_label <= start
                              and value <= highest_index[1]):
                        found_at_index = True
                        if value == 1:
                            start_of_path = row_index + 1
                        else:
                            start_of_path = row_index
                        break
                if found_at_index:
                    break
        # follow the path if such a point exists
        if found_at_index:
            # on which input curve lies start_of_path. If it lies on the current curve, skip the computation
            # of the cluster and add the segments between start and end to the cluster. Then, set the highest index
            # below start.
            index_traj_of_representative = [start_end_points.index(i) for i in start_end_points
                                            if start_of_path in i][0]
            if index_traj_of_representative == index_curve:

                if index_curve > 0:
                    highest_index = [start_end_points[index_curve - 1][-1], 1]
                else:
                    highest_index = [-1, 0]
            else:
                while current_index > start:
                    try:
                        if graph[current_index][row_index][cell_index][1] <= start:
                            current_index = current_index - 1
                            cell_index = grid[current_index][row_index].index(value)
                        else:
                            # if a bottom most point is reached where the value of the left arrow is higher than start
                            if grid[current_index][row_index][cell_index] == left[0, current_index, row_index]:
                                current_index = current_index - 1
                                # find topmost point with edge label smaller or equal start
                                if not (left[1, current_index, row_index] <= left[0, current_index + 1, row_index] and left[
                                    1, current_index, row_index] != -1 and
                                            (graph[current_index][row_index][-1][1] <= start
                                             or graph[current_index][row_index][-1][2][0] <= start)):
                                    for i in range(row_index - 1, graph[current_index + 1][row_index][0][2][1] - 1, -1):
                                        if graph[current_index][i][-1][1] <= start\
                                                and graph[current_index][i][-1][1] is not None \
                                                or graph[current_index][i][-1][2][0] <= start \
                                                        and graph[current_index][i][-1][2][
                                                    0] is not None:
                                            row_index = i
                                            break
                                cell_index = len(grid[current_index][row_index]) - 1
                                value = graph[current_index][row_index][cell_index][0]
                            # not a bottom most point. Go to lower value in same cell
                            else:
                                cell_index = cell_index - 1
                                value = grid[current_index][row_index][cell_index]
                    except IndexError:
                        print current_index, row_index, cell_index
                if vertex_check:
                    ends_at_vertex = False
                    # from the current j-value, check if one can reach a vertex of the trajectory within
                    # the free space of the current boundary
                    #     if grid[index][row_index][cell_index] == 1:
                    #         ends_at_vertex = True
                    #     else:

                    # To get subtrajectory of maximum length, check how far one can walk along the current trajectory
                    # until no vertex of the trajectory is within distance d to start or until one exceeds distance d
                    # to start while walking along the trajectory. Therefore: check which lowest index can be reached
                    # in the free space while keeping position at start.
                    if grid[current_index][row_index][0] == 0:
                        ends_at_vertex = True
                        lowest_reachable_vertex = row_index
                    for search_index_lowest_vertex in range(row_index-1, -1, -1):
                        if grid[current_index][search_index_lowest_vertex][0] == 0 \
                                and grid[current_index][search_index_lowest_vertex][-1] == 1:
                            lowest_reachable_vertex = search_index_lowest_vertex
                        else:
                            break
                    if ends_at_vertex:
                        # update the list of trajectories which are part of the cluster
                        index_traj = [start_end_points.index(i) for i in start_end_points if
                                      lowest_reachable_vertex in i and start_of_path in i]
                        if len(index_traj) > 0:
                            if index_traj[0] not in l:
                                l.add(index_traj[0])
                        if seg_return:
                            for segment in range(lowest_reachable_vertex, start_of_path):
                                segments_of_cluster.append(segment)
                                # segments_of_cluster.append([row_index, start_of_path])
                        else:
                            segments_of_cluster.append([lowest_reachable_vertex, start_of_path])
                        count_cluster_curves = count_cluster_curves + 1
                        highest_index = [lowest_reachable_vertex - 1, 1]
                    else:
                        if row_index == highest_index[0]:
                            highest_index = [highest_index[0] - 1, 1]
                        else:
                            highest_index = [row_index, value]
                else:
                    index_traj = [start_end_points.index(i) for i in start_end_points if
                                  row_index in i and start_of_path in i]
                    if len(index_traj) > 0:
                        if index_traj[0] not in l:
                            l.add(index_traj[0])
                    segments_of_cluster.append([row_index, start_of_path])
                    count_cluster_curves = count_cluster_curves + 1
                    if row_index == highest_index[0]:
                        highest_index = [highest_index[0] - 1, 1]
                    else:
                        highest_index = [row_index, value]
        if not found_at_index:
            break
    return count_cluster_curves, segments_of_cluster, l


def compute_set_cover_instance(traj, starts, ends,  distance):
    print "length of trajectory:", len(traj)
    time_clustering_set_up_start = time.time()
    bottom, left = free_space(traj, distance)
    print "free space built"
    grid = compute_grid_faster(left)
    print "grid  built"
    graph = initialize_graph(grid)
    relevant_representatives = []
    subsets = []
    weights = []
    start_end_points = [range(starts[i], ends[i]) for i in range(0, len(starts))]
    for i in range(1, len(traj)):
        add_column(graph, grid, bottom, left, i)
    print "graph built"
    time_clustering_set_up_end = time.time()
    print "Time to set up graph for clustering:", time_clustering_set_up_end-time_clustering_set_up_start
    time_compute_relevant_clusters_start = time.time()
    # print "cluster", find_clusters(450, 483, graph, grid, left, start_end_points, seg_return=True)
    for curve in start_end_points:
        a = curve[0]
        b = a + 1
        while b < curve[-1]:
            is_irrelevant = True
            while is_irrelevant and b < curve[-1]:
                # c_1 = find_clusters(a, b, graph, grid, left, start_end_points, vertex_check=False)
                is_irrelevant = check_relevance(a, b, graph, grid, left, start_end_points, curve)
                # print a, b, is_irrelevant
                b = b + 1
            relevant_representatives.append([a, b-1])
            # create LineSring and compute length. Add this length as weight of the set.
            points_of_representative = [Point(p[1], p[0]) for p in traj[a:b]]
            line_of_representative = LineString(points_of_representative)
            line2 = transform(project, line_of_representative)
            # subsets.append(set(find_clusters(a, b-1, graph, grid, left, start_end_points, vertex_check=True)[1]))
            subsets.append(find_clusters(a, b-1, graph, grid, left, start_end_points, vertex_check=True)[1])
            weights.append(line2.length)
            a_old = a
            a = b - 1
        if is_irrelevant:
            relevant_representatives[-1][1] = relevant_representatives[-1][1] + 1
            # subsets[-1] = set(find_clusters(a_old, b, graph, grid, left, start_end_points, vertex_check=True)[1])
            points_of_representative = [Point(p[1], p[0]) for p in traj[a_old:b + 1]]
            line_of_representative = LineString(points_of_representative)
            line2 = transform(project, line_of_representative)
            subsets[-1] = (find_clusters(a_old, b, graph, grid, left, start_end_points, vertex_check=True)[1])
            weights[-1] = line2.length
        else:
            relevant_representatives.append([a, b])
            # subsets.append(set(find_clusters(a, b, graph, grid, left, start_end_points, vertex_check=True)[1]))
            points_of_representative = [Point(p[1], p[0]) for p in traj[a:b+1]]
            line_of_representative = LineString(points_of_representative)
            line2 = transform(project, line_of_representative)
            subsets.append(find_clusters(a, b, graph, grid, left, start_end_points, vertex_check=True)[1])
            weights.append(line2.length)
    total_weight = sum(weights)
    weights = [1.0*weight/total_weight for weight in weights]
    time_compute_relevant_clusters_end = time.time()
    print "Time to compute relevant clusters:", time_compute_relevant_clusters_end-time_compute_relevant_clusters_start
    return relevant_representatives, subsets, weights


def check_relevance(a, b, graph, grid, left, start_end_points, current_curve):
    # cluster_open = find_clusters(a, b, graph, grid, left, start_end_points, vertex_check=False)
    cluster_closed = find_clusters(a, b, graph, grid, left, start_end_points, vertex_check=True)
    if True: # cluster_open[2] == cluster_closed[2]:
        cluster_extended = find_clusters(a, b+1, graph, grid, left, start_end_points, vertex_check=True)
        cluster_extended_last_segment = find_clusters(b, b + 1, graph, grid, left, start_end_points)
        monitoring = True
        for segment in cluster_closed[1]:
            if segment not in cluster_extended[1]:
                monitoring = False
                break
        if cluster_extended[2] == cluster_closed[2] and monitoring:
            # compute cluster for last segment of extended cluster
            # compute G* of last segment
            G_last_segment = SortedList()
            cluster_last_segment = find_clusters(b-1, b, graph, grid, left, start_end_points,
                                                           vertex_check=True, seg_return=False)
            cluster_extended_last_segment_segs = find_clusters(b, b+1, graph, grid, left, start_end_points,
                                                 vertex_check=True, seg_return=True)
            monitoring_2 = True
            for segment in cluster_extended_last_segment_segs[1]:
                if segment not in cluster_extended[1]:
                    # print "last segment within distance d to new curve"
                    monitoring_2 = False

            for rep in cluster_last_segment[1]:
                list_of_trajs = find_clusters(rep[0], rep[1], graph, grid, left, start_end_points,
                                                           vertex_check=True, seg_return=False)[2]
                for entry in list_of_trajs:
                    if entry not in G_last_segment:
                        G_last_segment.add(entry)

            G_last_segment_extended = SortedList()
            cluster_extended_last_segement = find_clusters(b, b+1, graph, grid, left, start_end_points,
                                                           vertex_check=True, seg_return=False)
            for rep in cluster_extended_last_segement[1]:
                list_of_trajs = find_clusters(rep[0], rep[1], graph, grid, left, start_end_points,
                                              vertex_check=True, seg_return=False)[2]
                for entry in list_of_trajs:
                    if entry not in G_last_segment_extended:
                        G_last_segment_extended.add(entry)

            if G_last_segment != G_last_segment_extended or not monitoring_2:
                # print "Reason for being relevant: some event in 2d distance", a, b
                return False
            else:
                return True
        else:
            # print "Reson for being relevant: extended cluster smaller or bigger", a, b
            return False
    else:
        return True


def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    group_diagram_representatives = []
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        group_diagram_representatives.append(subsets.index(subset))
        covered |= subset

    return group_diagram_representatives


def read_data(data_set, year, distance_between_points, distance_between_points_max, index_of_lat, index_of_long,
                 index_of_date, index_of_year_start, index_of_year_end, index_of_month_start,
                 index_of_month_end, migration_months):
    trajectory = []
    trajectory_with_name = []
    with open(data_set) as csvFile:
        data = csv.reader(csvFile)
        next(data)
        first_row = next(data)
        for row in data:
            if row[index_of_date][index_of_year_start:index_of_year_end] == year: # and float(row[14]) < 15:
                trajectory.append([float(first_row[index_of_lat]), float(first_row[index_of_long])])
                trajectory_with_name.append([float(first_row[index_of_lat]), float(first_row[index_of_long]),
                                             first_row[32]])
                count = 0
                break
        for row in data:
            try:
                if row[index_of_date][index_of_year_start:index_of_year_end] == year: # and float(row[14]) < 15:  # and not (float(row[index_of_long]) < 35 and float(row[index_of_lat]) > 62)
                    last_point = trajectory[count]
                    new_point = [float(row[index_of_lat]), float(row[index_of_long])]
                    dist_travelled = g.inv(last_point[1], last_point[0], new_point[1], new_point[0])[2]
                    if dist_travelled > distance_between_points \
                            and row[index_of_date][index_of_month_start:index_of_month_end] in migration_months \
                            and new_point[0] > 25 and new_point[1] > 0:
                        trajectory.append([float(row[index_of_lat]), float(row[index_of_long])])
                        trajectory_with_name.append([float(row[index_of_lat]), float(row[index_of_long]),
                                                     row[32]])
                        count = count + 1
            except ValueError:
                "Incorrect Data Format"
    return trajectory, trajectory_with_name


for year in years:
    global_time_start = time.time()
    time_read_start = time.time()
    data = read_data(data_set, year, distance_between_points, distance_between_points_max, index_of_lat, index_of_long,
                     index_of_date, index_of_year_start, index_of_year_end, index_of_month_start,
                     index_of_month_end, migration_months)
    animal = data[0]
    animal_additional = data[1]

    animal_sample = [animal[k] for k in range(0, len(animal), resolution)]
    animal_sample_additional = [animal_additional[k] for k in range(0, len(animal_additional), resolution)]

    all_data = []
    for track in tracks:
        all_data.append([animal_sample_additional[date] for date in range(0, len(animal_sample_additional))
                          if animal_sample_additional[date][2] == track])

    track_with_almost_no_data = [track for track in all_data if len(track) < 10]
    all_data = [track for track in all_data if len(track) > 10]
    if True:#len(track_with_almost_no_data) == 0:
        print "Year:", year
        lines = []
        for track in all_data:
            points = []
            for point in track:
                points.append(Point(point[0], point[1]))
            line = LineString(points)
            lines.append(line)
        print "Size of input data", sum(len(line.coords) for line in lines)
        time_read_end = time.time()
        print "Time to read_data:", time_read_end - time_read_start
        # seg_start_lat_long = LatLon(23.23, 51.2)
        shifted_north_coordinates_lat = []
        shifted_north_coordinates_lon = []
        shifted_north_seg_lats =[]
        shifted_north_seg_longs = []
        time_segmentation_start = time.time()
        if segmentation_1:
            seg_1_start = time.time()
            for index_outer, line_outer in enumerate(lines):
                for index, line_inner in enumerate(lines):
                    if index != index_outer:
                        for seg_start, seg_end in zip(line_inner.coords[0:-1], line_inner.coords[1:]):
                            # create two LatLong instances
                            seg_start_lat_long = LatLon(seg_start[0], seg_start[1])
                            seg_end_lat_long = LatLon(seg_end[0], seg_end[1])
                            # compute heading from start of segment to end of segment
                            heading = seg_start_lat_long.heading_initial(seg_end_lat_long)
                            # offset startpoint perpendicular to heading with given distance
                            # fist towards north, handle negative headings correctly
                            if heading - 90 < 0:
                                heading_north = 270 + heading
                            else:
                                heading_north = heading - 90
                            offset_north_start = seg_start_lat_long.offset(heading_north, 1.0*dist/1000)
                            offset_north_end = seg_end_lat_long.offset(heading_north, 1.0*dist/1000)

                            shifted_north_coordinates_lat.append(offset_north_start.lat)
                            shifted_north_coordinates_lon.append(offset_north_start.lon)
                            shifted_north_seg_lats.append([offset_north_start.lat, offset_north_end.lat])
                            shifted_north_seg_longs.append([offset_north_start.lon, offset_north_end.lon])
                            # create new shapely LineString
                            shifted_line = LineString([Point(offset_north_start.lat, offset_north_start.lon),
                                                      Point(offset_north_end.lat, offset_north_end.lon)])
                            intersect = shifted_line.intersection(line_outer)
                            if not intersect.is_empty:
                                if intersect.geom_type == "MultiPoint":
                                    for point in intersect.geoms:
                                        p = Point(point.coords[0])
                                        lines[index_outer] = snap(lines[index_outer], p, 0.000001)
                                else:
                                    lines[index_outer] = snap(lines[index_outer], intersect, 0.000001)
                            # same for a shift towards south
                            if heading + 90 > 360:
                                heading_south = heading - 270
                            else:
                                heading_south = heading + 90
                            offset_south_start = seg_start_lat_long.offset(heading_south, 1.0 * dist/1000)
                            offset_south_end = seg_end_lat_long.offset(heading_south, 1.0 * dist/1000)
                            # create new shapely LineString
                            shifted_line = LineString([Point(offset_south_start.lat, offset_south_start.lon),
                                                       Point(offset_south_end.lat, offset_south_end.lon)])
                            intersect = shifted_line.intersection(line_outer)
                            if not intersect.is_empty:
                                if intersect.geom_type == "MultiPoint":
                                    for point in intersect.geoms:
                                        p = Point(point.coords[0])
                                        lines[index_outer] = snap(lines[index_outer], p, 0.0000001)
                                else:
                                    lines[index_outer] = snap(lines[index_outer], intersect, 0.0000001)
            seg_1_end = time.time()
            print "Time for segmentation 1:", seg_1_end - seg_1_start
        if segmentation_2:
            seg_2_start = time.time()
            for i in range(0, 2):
                lines_old = lines[:]
                for index_outer, line_outer in enumerate(lines_old):
                    for point in line_outer.coords:
                        p = Point(point[0], point[1])
                        x = np.array(p.coords[0])
                        for index, line_inner in enumerate(lines_old):
                            if index != index_outer:
                                for seg_start, seg_end in zip(line_inner.coords[0:-1], line_inner.coords[1:]):
                                    u = np.array(seg_start)
                                    v = np.array(seg_end)
                                    n = v - u
                                    n /= np.linalg.norm(n, 2)
                                    q = u + n * np.dot(x - u, n)
                                    new_point = Point(q[0], q[1])
                                    try:
                                        if geopy.distance.vincenty(point, q).meters <= dist_tol:
                                        #g.inv(point[1], point[0], q[1], q[0])[2] <= dist:
                                            if i == 0:
                                                lines[index] = snap(lines[index], new_point, 0.00000001)
                                            if i == 1:
                                                lines[index] = snap(lines[index], new_point, 0.00000001)
                                    except ValueError:
                                        print "projection to far away"
        seg_2_end = time.time()
        # print "Time for segmentation 2:", seg_2_end - seg_2_start
        starts = []
        ends = []
        new_trajectory = []
        for line in lines:
            trajectory = []
            for point in line.coords:
                trajectory.append([point[0], point[1]])
            starts.append(len(new_trajectory))
            ends.append(starts[-1]+len(trajectory))
            new_trajectory = new_trajectory + trajectory
        animal_sample = new_trajectory
        ends[-1] = ends[-1] - 1
        print "segmentation done"
        time_segmentation_end = time.time()
        print "Segmentation time", time_segmentation_end - time_segmentation_start
        time_construct_set_cover_start = time.time()
        set_cover_instance_animal = compute_set_cover_instance(animal_sample, starts, ends, dist_tol)
        print "set cover instance built" #, set_cover_instance_animal[1]
        time_construct_set_cover_end = time.time()
        print "Time to construct set cover instance", time_construct_set_cover_end - time_construct_set_cover_start
        time_solve_set_cover_start = time.time()
        universe = set([e for s in set_cover_instance_animal[1] for e in s])
        #universe_complete = set(range(0, len(animal_sample)))
        # build covering matrix
        matrix = []
        for computed_set in set_cover_instance_animal[1]:
            row = [entry in computed_set for entry in universe]
            matrix.append(row)
        covering_matrix = np.matrix(matrix).transpose()
        costs = np.array(set_cover_instance_animal[2])

        solution_animal = setcover.SetCover(covering_matrix, costs, maxiters=5)
        solution_animal.SolveSCP()
        group_diagram_animal = [set_cover_instance_animal[0][i] for i in range(0, len(solution_animal.s)) if
                                solution_animal.s[i]]
        group_diagram_animal = SortedList(group_diagram_animal)
        time_solve_set_cover_end = time.time()
        print "Time to solve set cover", time_solve_set_cover_end - time_solve_set_cover_start
        print "group diagram computed", group_diagram_animal

        #gmap = gmplot.GoogleMapPlotter(55, 9, 4)
        kml = simplekml.Kml()
        #
        for start, end, color_index in zip(starts, ends, range(0, len(starts))):
            colors = ["blue", "black", "green"]
            # if end == ends[-1][-1]:
            #     end_plot = end
            # else:
            #     end_plot = end + 1
            end_plot = end
            lats = [animal_sample[i][0] for i in range(start, end_plot)]
            longs = [animal_sample[j][1] for j in range(start, end_plot)]

            for lat_1, lon_1, lat_2, lon_2, s in zip(lats[:-1], longs[:-1],
                                                  lats[1:], longs[1:], range(start, end_plot)):
                lin = kml.newlinestring(name=str(s), coords=[(lon_1, lat_1), (lon_2, lat_2)])
                lin.style.linestyle.width = 3
                # lin.style.linestyle.color = colors[color_index]


            #gmap.plot(lats, longs, color="cornflowerblue", edge_width=2)

        lats = [animal_sample[i][0] for i in range(0, len(animal_sample))]
        longs = [animal_sample[j][1] for j in range(0, len(animal_sample))]

        for index, part in enumerate(group_diagram_animal):

            lats_group_diagram = [animal_sample[i][0] for i in range(part[0], part[1]+1)]
            longs_group_diagram = [animal_sample[i][1] for i in range(part[0], part[1]+1)]
            points = [(lon, lat) for lon, lat in zip(longs_group_diagram, lats_group_diagram)]
            # points = []
            # for lat_1, lon_1, lat_2, lon_2 in zip(lats_group_diagram[:-1], longs_group_diagram[:-1],
            #                                       lats_group_diagram[1:], longs_group_diagram[1:]):
            lin = kml.newlinestring(name=str(part), coords=points)
            lin.style.linestyle.color = 'ff0000ff'

            lin.style.linestyle.width = 6

            # gmap.plot(lats_group_diagram, longs_group_diagram, color="red", edge_width=4)

        if segmentation_1 and segmentation_2:
            html = year + "_migration_" + "segmentation_first_and_second_type_" + str(dist_tol) + "_" + str(resolution)

        elif segmentation_1 and not segmentation_2:
            html = year + "_migration_" + "segmentation_first_type_" + str(dist_tol) + "_" + str(resolution)

        elif not segmentation_1 and segmentation_2:
            html = year + "_migration_" + "segmentation_second_type_" + str(dist_tol) + "_" + str(resolution)

        else:
            html = year + "_migration_" + "no_segmentation" + str(dist_tol) + "_" + str(resolution)

        kml.save(html + " ICARUS_big" + ".kml")
        # gmap.draw(html + " ICARUS_big" + ".html")
        global_time_end = time.time()
        print "Overall time", global_time_end - global_time_start
