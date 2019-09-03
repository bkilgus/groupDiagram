from sortedcontainers import SortedList
from datetime import datetime
import time
import csv
import math
import simplejson
import gmplot
import collections
import os
import numpy as np
import simplekml

name='groupDiagramEqualSimilarTime'

def version():
    print "version 0.1"


# get all names
def get_unique_words(all_words):
    unique_words = collections.Counter()
    for word in all_words:
        unique_words[word] += 1
    return unique_words.keys()


# converting longitude, latitude to cartesian coordinates
def gps_to_cartesian(lon, lat):
    R = 6371000
    return [R * math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
            R * math.cos(math.radians(lat)) * math.sin(math.radians(lon))]


# distance in m for two earth coordinates given
def distance(longitude1, latitude1, longitude2, latitude2):
    x1 = gps_to_cartesian(longitude1, latitude1)[0]
    y1 = gps_to_cartesian(longitude1, latitude1)[1]
    x2 = gps_to_cartesian(longitude2,latitude2)[0]
    y2 = gps_to_cartesian(longitude2, latitude2)[1]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# computes the event time between 0 and 1
def compute_event_time(x_a, x_b, x_c, x_d, y_a, y_b, y_c, y_d, d):
    t = (math.sqrt(
        (-2 * y_c ** 2 - 2 * y_c * y_b + 2 * y_c * y_d + 4 * y_c * y_a + 2 * y_b * y_a - 2 * y_d * y_a - 2 * x_d * x_a +
         2 * x_d * x_c + 2 * x_b * x_a - 2 * x_b * x_c - 2 * x_a ** 2 + 4 * x_a * x_c - 2 * y_a ** 2 - 2 * x_c ** 2) ** 2 -
        4 * (-d ** 2 + y_c ** 2 - 2 * y_c * y_a + x_a ** 2 - 2 * x_a * x_c + y_a ** 2 + x_c ** 2)
        * (y_c ** 2 + 2 * y_c * y_b - 2 * y_c * y_d - 2 * y_c * y_a + y_b ** 2 - 2 * y_b * y_d - 2 * y_b * y_a + y_d ** 2
        + 2 * y_d * y_a + x_d ** 2 - 2 * x_d * x_b + 2 * x_d * x_a - 2 * x_d * x_c + x_b ** 2 - 2 * x_b * x_a +
        2 * x_b * x_c + x_a ** 2 - 2 * x_a * x_c + y_a ** 2 + x_c ** 2)) + 2 * y_c ** 2 + 2 * y_c * y_b - 2 * y_c * y_d
         - 4 * y_c * y_a - 2 * y_b * y_a + 2 * y_d * y_a + 2 * x_d * x_a - 2 * x_d * x_c - 2 * x_b * x_a + 2 * x_b * x_c +
         2 * x_a ** 2 - 4 * x_a * x_c + 2 * y_a ** 2 + 2 * x_c ** 2) / (1.0*(2 * (y_c ** 2 + 2 * y_c * y_b - 2 * y_c * y_d -
         2 * y_c * y_a + y_b ** 2 - 2 * y_b * y_d - 2 * y_b * y_a + y_d ** 2 + 2 * y_d * y_a + x_d ** 2 - 2 * x_d * x_b +
         2 * x_d * x_a - 2 * x_d * x_c + x_b ** 2 - 2 * x_b * x_a + 2 * x_b * x_c + x_a ** 2 - 2 * x_a * x_c + y_a ** 2
         + x_c ** 2)))
    return t


# returns t between 0 and 1 for which the distance between the segments is minimal
def minimum_distance(x_a, x_b, x_c, x_d, y_a, y_b, y_c, y_d):
    t = (
        y_c ** 2 + y_c * y_b - y_c * y_d - 2 * y_c * y_a - y_b * y_a + y_d * y_a + x_d * x_a - x_d * x_c - x_b * x_a + x_b * x_c + x_a ** 2
        - 2 * x_a * x_c + y_a ** 2 + x_c ** 2) / (1.0*(
        y_c ** 2 + 2 * y_c * y_b - 2 * y_c * y_d - 2 * y_c * y_a + y_b ** 2 - 2 * y_b * y_d
        - 2 * y_b * y_a + y_d ** 2 + 2 * y_d * y_a + x_d ** 2 - 2 * x_d * x_b + 2 * x_d * x_a - 2 * x_d * x_c + x_b ** 2 - 2 * x_b * x_a
        + 2 * x_b * x_c + x_a ** 2 - 2 * x_a * x_c + y_a ** 2 + x_c ** 2))
    return t


# distance of two segments for a time given between 0 and 1
def dist_at_time_t(x_a, x_b, x_c, x_d, y_a, y_b, y_c, y_d, t):
    return math.sqrt((((1-t)*x_a + t*x_b) - ((1-t)*x_c + t*x_d))**2 + (((1-t)*y_a + t*y_b) - ((1-t)*y_c + t*y_d))**2)


# distance from a point to a segment
def dist_point_to_segment(start_seg_x, start_seg_y, end_seg_x, end_seg_y, point_x, point_y, ):
    v = np.array([end_seg_x, end_seg_y]) - np.array([start_seg_x, start_seg_y])
    w = np.array([point_x, point_y]) - np.array([start_seg_x, start_seg_y])
    c1 = np.dot(v, w)
    if c1 <= 0:
        return math.sqrt((point_x - start_seg_x) ** 2 + (point_y - start_seg_y) ** 2)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return math.sqrt((point_x - end_seg_x) ** 2 + (point_y - end_seg_y) ** 2)
    b = c1 / c2
    p = np.array([start_seg_x, start_seg_y]) + b * v
    return math.sqrt((point_x - p[0]) ** 2 + (point_y - p[1]) ** 2)


def compare_two_segments(x_a1, x_b1, x_c1, x_d1, y_a1, y_b1, y_c1, y_d1, start_time, time_list, d):
    # converting long, lat to cartesian coordinates
    x_a = gps_to_cartesian(x_a1, y_a1)[0]
    x_b = gps_to_cartesian(x_b1, y_b1)[0]
    x_c = gps_to_cartesian(x_c1, y_c1)[0]
    x_d = gps_to_cartesian(x_d1, y_d1)[0]

    y_a = gps_to_cartesian(x_a1, y_a1)[1]
    y_b = gps_to_cartesian(x_b1, y_b1)[1]
    y_c = gps_to_cartesian(x_c1, y_c1)[1]
    y_d = gps_to_cartesian(x_d1, y_d1)[1]
    # if segments are within equal-time distance at most d return empty list, no event
    if max(distance(x_a1, y_a1, x_c1, y_c1), distance(x_b1, y_b1, x_d1, y_d1)) <= d:
        return []
    # if either starting points or endpoints have distance greater d compute the merge or the split event
    elif min(distance(x_a1, y_a1, x_c1, y_c1), distance(x_b1, y_b1, x_d1, y_d1)) <= d:
        try:
            t = compute_event_time(x_a, x_b, x_c, x_d, y_a, y_b, y_c, y_d, d)
        except ZeroDivisionError:
            return []
        if t < 0 or t > 1:
            try:
                minimal = minimum_distance(x_a, x_b, x_c, x_d, y_a, y_b, y_c, y_d)
            except ZeroDivisionError:
                return []
            t2 = t - 2 * (t - minimal)
            return [(1 - t2) * time_list[start_time] + t2 * time_list[start_time + 1]]
        else:
            return [(1 - t) * time_list[start_time] + t * time_list[start_time + 1]]
    # if distance of starting points and distance of endpoints are is greater d check whether there is a merge and
    # a split event in between
    else:
        try:
            minimal = minimum_distance(x_a, x_b, x_c, x_d, y_a, y_b, y_c, y_d)
        except ValueError:
            return []
        except ZeroDivisionError:
            return []
        minimal_distance = dist_at_time_t(x_a, x_b, x_c, x_d, y_a, y_b, y_c, y_d, minimal)
        if 0.0 < minimal < 1 and minimal_distance <= d:
            try:
                t = compute_event_time(x_a, x_b, x_c, x_d, y_a, y_b, y_c, y_d, d)
            except ValueError:
                return []
            except ZeroDivisionError:
                return []
            if t < 0 or t > 1:
                return []
            else:
                t2 = t - 2 * (t - minimal)
                times = [(1 - t) * time_list[start_time] + t * time_list[start_time + 1],
                         (1 - t2) * time_list[start_time] + t2 * time_list[start_time + 1]]
                return [t for t in times if t > 0]
        else:
            return []


# interpolate coordinates for a new timestamp
def interpolate(sorted_coordinate_list, sorted_time_list, timestamp):
    index_a = sorted_time_list.bisect(timestamp) - 1
    index_b = index_a + 1
    a = sorted_coordinate_list[index_a][0]
    b = sorted_coordinate_list[index_b][0]
    x_a = sorted_coordinate_list[index_a][1]
    y_a = sorted_coordinate_list[index_a][2]
    x_b = sorted_coordinate_list[index_b][1]
    y_b = sorted_coordinate_list[index_b][2]

    if math.isnan(x_a) or math.isnan(x_b) or math.isnan(y_a) or math.isnan(y_b):
        return [timestamp, float('NaN'), float('NaN')]
    else:
        x = (b - timestamp) / (b - a) * x_a + (timestamp - a) / (b - a) * x_b
        y = (b - timestamp) / (b - a) * y_a + (timestamp - a) / (b - a) * y_b

        return [timestamp, x, y]


# interpolate coordinates for a new timestamp (adding type of surface for new location: water or land)
def interpolate_surface(sorted_coordinate_list, sorted_time_list, timestamp):
    index_a = sorted_time_list.bisect(timestamp) - 1
    index_b = index_a + 1
    a = sorted_coordinate_list[index_a][0]
    b = sorted_coordinate_list[index_b][0]
    x_a = sorted_coordinate_list[index_a][1]
    y_a = sorted_coordinate_list[index_a][2]
    x_b = sorted_coordinate_list[index_b][1]
    y_b = sorted_coordinate_list[index_b][2]
    surface_a = sorted_coordinate_list[index_a][3]
    surface_b = sorted_coordinate_list[index_b][3]

    if math.isnan(x_a) or math.isnan(x_b) or math.isnan(y_a) or math.isnan(y_b):
        return [timestamp, float('NaN'), float('NaN')]
    else:
        x = (b - timestamp) / (b - a) * x_a + (timestamp - a) / (b - a) * x_b
        y = (b - timestamp) / (b - a) * y_a + (timestamp - a) / (b - a) * y_b
        if (b - timestamp) >= (timestamp - a):
            surface = surface_a
        else:
            surface = surface_b
        return [timestamp, x, y, surface]


# checks if an entity was close to a given coordinate of another entity within a bounded difference of time
# (was close to that coordinate slightly before or after the other entity was located at this coordinates
def within_alpha_similar_time_distance(segment, time_stamp_start, time_stamp_end, trajectory, time_list, time_shift, d,
                                       minimal_duration):
    if time_shift > 0:
        # add timestamps with time_shift
        to_be_added = interpolate(trajectory, time_list, time_stamp_start - time_shift)
        trajectory.add(to_be_added)
        time_list.add(time_stamp_start - time_shift)
        start = time_list.index(time_stamp_start - time_shift)

        to_be_added_2 = interpolate(trajectory, time_list, time_stamp_end + time_shift)
        trajectory.add(to_be_added_2)
        time_list.add(time_stamp_end + time_shift)
        end = time_list.index(time_stamp_end + time_shift)

        start_seg_x = gps_to_cartesian(segment[0][0], segment[0][1])[0]
        start_seg_y = gps_to_cartesian(segment[0][0], segment[0][1])[1]
        end_seg_x = gps_to_cartesian(segment[1][0], segment[1][1])[0]
        end_seg_y = gps_to_cartesian(segment[1][0], segment[1][1])[1]

        condition = False
        v = [[0, 0, 0]]
        duration_start = time_stamp_start
        for i, j in enumerate(range(start, end + 1)):
            # print timestamp, time_list[i]
            vertex_x = gps_to_cartesian(trajectory[j][1], trajectory[j][2])[0]
            vertex_y = gps_to_cartesian(trajectory[j][1], trajectory[j][2])[1]
            vertex_1_x = gps_to_cartesian(trajectory[j + 1][1], trajectory[j + 1][2])[0]
            vertex_1_y = gps_to_cartesian(trajectory[j + 1][1], trajectory[j + 1][2])[1]
            v.append([0, 0, 0])
            # vertex is inside the d-Ball around the start of the segment
            if dist_point_to_segment(vertex_x, vertex_y, vertex_1_x, vertex_1_y,
                                     start_seg_x, start_seg_y) <= d:
                v[i + 1][0] = 1
                v[i + 1][1] = 1
            # vertex has distance at most d to the segment
            if dist_point_to_segment(start_seg_x, start_seg_y, end_seg_x, end_seg_y,
                                     vertex_x, vertex_y) <= d:
                v[i + 1][1] = 1
            # vertex is inside the d-Ball around the end of the segment
            if dist_point_to_segment(vertex_x, vertex_y, vertex_1_x, vertex_1_y,
                                     end_seg_x, end_seg_y) <= d:
                v[i + 1][2] = 1
            # trajectory 'enters' the d-Ball around the start of the segment for the first time or after it
            # left the d-tube around the segment before
            if v[i][0] == 0 and v[i + 1][0] == 1:
                condition = True
                duration_start = time_list[j]
            # trajectory leaves d-Tube around the segment
            if v[i][1] == 1 and v[i + 1][1] == 0:
                condition = False
            # trajectory 'enters' the d-Ball around the end of the segment and the trajectory stepped into
            # the d-Ball around the start of the segment and stayed in the d-tube around the segment
            duration = time_list[j] - duration_start
            if v[i + 1][2] == 1 and condition and duration >= minimal_duration:
                trajectory.remove(to_be_added)
                time_list.remove(time_stamp_start - time_shift)
                trajectory.remove(to_be_added_2)
                time_list.remove(time_stamp_end + time_shift)
                return True
            # stop traversing the trajectory if "condition" is False and the time frame for entering the
            # d-ball around the start of the segment has been exceeded.
            if not condition and time_list[j] > time_stamp_start + time_shift:
                trajectory.remove(to_be_added)
                time_list.remove(time_stamp_start - time_shift)
                trajectory.remove(to_be_added_2)
                time_list.remove(time_stamp_end + time_shift)
                return False
        trajectory.remove(to_be_added)
        time_list.remove(time_stamp_start - time_shift)
        trajectory.remove(to_be_added_2)
        time_list.remove(time_stamp_end + time_shift)
        return False
    else:
        dist = max(distance(segment[0][0], segment[0][1], trajectory[time_list.index(time_stamp_start)][1],
                            trajectory[time_list.index(time_stamp_start)][2]),
                   distance(segment[1][0], segment[1][1], trajectory[time_list.index(time_stamp_end)][1],
                            trajectory[time_list.index(time_stamp_end)][2]))
        if dist <= d:
            return True
        else:
            return False


# Set Cover implementation
def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    cover_number = []
    sizes = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover_number.append(subsets.index(subset))
        sizes.append(len(subset))
        cover.append(subset)
        covered |= subset
    return cover, cover_number, sizes


# read data from csv file
def read_data(csv_file):
    counter = 0
    # read the input data and set up trajectories and lists of time stamps
    reader = csv.reader(open(csv_file, 'rb'), delimiter=',')
    all_names = list(zip(*reader))[3]
    entities_names = [name for name in get_unique_words(all_names) if name != 'name']
    all_time_stamps = SortedList()
    entities = [SortedList() for i in xrange(len(entities_names))]
    entities_times = [SortedList() for j in xrange(len(entities_names))]
    # print entities_names
    for name in entities_names:
        with open(csv_file) as csvFile:
            data = csv.reader(csvFile)
            next(data)
            for row in data:
                if row[3] == name:  # and len(row[0]) == 23 and float(row[4]) < 10:
                    # compute datetime and convert to unix time in seconds since the 1970 epoch.
                    year = int(row[0][0:4])
                    month = int(row[0][5:7])
                    day = int(row[0][8:10])
                    hour = int(row[0][11:13])
                    minute = int(row[0][14:16])
                    second = int(row[0][17:19])
                    new_date_time = datetime(year, month, day, hour, minute, second)
                    from1970 = time.mktime(new_date_time.timetuple())
                    if not (from1970 in all_time_stamps):
                        all_time_stamps.add(from1970)
                    if row[1] != '' and row[2] != '':
                        x = float(row[1])
                        y = float(row[2])
                        update = [from1970, x, y]
                        if x < 180 and y < 180 and from1970 not in entities_times[counter]:
                            entities[counter].add(update)
                            entities_times[counter].add(from1970)
        counter += 1
    # simulate equal time sampling
    for timestamp in all_time_stamps:
        for index, entity in enumerate(entities):
            length = len(entities_times[index])
            # check if the current entity has this timestamp, if not add it and interpolate the coordinates but only if
            # it is in the range of the trajectory. Otherwise interpolation is not possible
            if timestamp not in entities_times[index] and entities_times[index].bisect(timestamp) > 0 and entities_times[
                index].bisect(timestamp) < length:
                entity.add(interpolate(entity, entities_times[index], timestamp))
                entities_times[index].add(timestamp)
    return entities, entities_times, all_time_stamps


def read_data_surface(csv_file):
    counter = 0
    # read the input data and set up trajectories and lists of time stamps
    reader = csv.reader(open(csv_file, 'rb'), delimiter=',')
    all_names = list(zip(*reader))[3]
    entities_names = [name for name in get_unique_words(all_names) if name != 'name']
    all_time_stamps = SortedList()
    entities = [SortedList() for i in xrange(len(entities_names))]
    entities_times = [SortedList() for j in xrange(len(entities_names))]

    for name in entities_names:
        with open(csv_file) as csvFile:
            data = csv.reader(csvFile)
            next(data)
            for row in data:
                if row[3] == name:  # and len(row[0]) == 23 and float(row[4]) < 10:
                    # compute datetime and convert to unix time in seconds since the 1970 epoch.
                    year = int(row[0][0:4])
                    month = int(row[0][5:7])
                    day = int(row[0][8:10])
                    hour = int(row[0][11:13])
                    minute = int(row[0][14:16])
                    second = int(row[0][17:19])
                    new_date_time = datetime(year, month, day, hour, minute, second)
                    from1970 = time.mktime(new_date_time.timetuple())
                    if not (from1970 in all_time_stamps):
                        all_time_stamps.add(from1970)
                    if row[1] != '' and row[2] != '':
                        x = float(row[1])
                        y = float(row[2])
                        surface = float(row[5])
                        update = [from1970, x, y, surface]
                        if x < 180 and y < 180 and from1970 not in entities_times[counter] and float(row[4]) < 15:
                            entities[counter].add(update)
                            entities_times[counter].add(from1970)
        counter += 1
    # simulate equal time sampling
    for timestamp in all_time_stamps:
        for index, entity in enumerate(entities):
            length = len(entities_times[index])
            # check if the current entity has this timestamp, if not add it and interpolate the coordinates but only if
            # it is in the range of the trajectory. Otherwise interpolation is not possible
            if timestamp not in entities_times[index] and entities_times[index].bisect(timestamp) > 0 and entities_times[
                index].bisect(timestamp) < length:
                entity.add(interpolate_surface(entity, entities_times[index], timestamp))
                entities_times[index].add(timestamp)
    return entities, entities_times, all_time_stamps


def create_group_diagram(entities, entities_times, all_time_stamps, d, time_shift, first, last, birds, save_path):
    print "this code is used"
    # Initialize kml-file to store the GD representatives
    kml = simplekml.Kml()
    # find common start and end time stamps of the trajectories
    start_times = SortedList()
    end_times = SortedList()
    for entityTime in entities_times:
        start_times.add(entityTime[0])
        end_times.add(entityTime[-1])
    common_start_time = start_times[-1]
    common_end_time = end_times[0]
    all_common_time_stamps = SortedList(
        all_time_stamps[all_time_stamps.index(common_start_time): all_time_stamps.index(common_end_time)])
    # COMPUTING ALL SPLIT AND MERGE EVENTS
    event_count = 0
    event_times = SortedList()
    print "number of all common time stamps:", len(all_common_time_stamps)
    # loop through all time stamps and compute the events
    for time_index, timestamp in enumerate(all_common_time_stamps[:-1]):
        for index1 in range(0, len(entities)):
            start_segment_1 = entities_times[index1].index(timestamp)
            end_segment_1 = start_segment_1 + 1
            for index2 in range(0, len(entities)):
                start_segment_2 = entities_times[index2].index(timestamp)
                end_segment_2 = start_segment_2 + 1
                new_events = compare_two_segments(entities[index1][start_segment_1][1],
                                                  entities[index1][end_segment_1][1],
                                                  entities[index2][start_segment_2][1],
                                                  entities[index2][end_segment_2][1],
                                                  entities[index1][start_segment_1][2],
                                                  entities[index1][end_segment_1][2],
                                                  entities[index2][start_segment_2][2],
                                                  entities[index2][end_segment_2][2],
                                                  time_index + first, all_common_time_stamps, d)
                # print new_events
                if len(new_events) > 0:
                    for event in new_events:
                        event_times.add(event)
                    event_count += 1
    # print event_count
    # propagate event time stamps to all trajectories
    adding_count = 0
    print "Number of events:", len(event_times)
    for timestamp in event_times:
        if timestamp not in all_common_time_stamps:
            all_common_time_stamps.add(timestamp)
            adding_count += 1
        for index, entity in enumerate(entities):
            length = len(entities_times[index])
            # check if the current entity has this timestamp, if not add it and interpolate the
            # coordinates but only if it is in the range of the trajectory. Otherwise interpolation is not possible
            if timestamp not in entities_times[index] and entities_times[index].bisect(timestamp) > 0 and entities_times[
                index].bisect(timestamp) < length:
                entity.add(interpolate(entity, entities_times[index], timestamp))
                entities_times[index].add(timestamp)

    # solve set cover instance for every time stamp, return the segments of the solution and draw them
    universe = set(range(0, len(entities)))
    cover_size = []
    cover_numbers = []

    # for each entity initialize
    all_together = 0
    all_together_time = 0
    number_of_representations = 0
    start = time.time()
    for time_index, timestamp in enumerate(all_common_time_stamps[first:last]):
        if timestamp in event_times or timestamp == all_common_time_stamps[first]:
            set_list = []
            start_segment_common = all_common_time_stamps.index(timestamp)
            end_segment_common = start_segment_common + 1
            for index in range(0, len(entities)):
                current_set = set()
                start_segment = entities_times[index].index(timestamp)
                end_segment = start_segment + 1
                segment = [[entities[index][start_segment][1], entities[index][start_segment][2]],
                           [entities[index][end_segment][1], entities[index][end_segment][2]]]
                for number, entity in enumerate(entities):
                    if within_alpha_similar_time_distance(segment, timestamp, all_common_time_stamps[end_segment_common],
                                                          entity, entities_times[number], time_shift, d, 0):
                        current_set.add(number)
                set_list.append(current_set)
            result = set_cover(universe, set_list)
            # print time_index, result
        cover_size.append(len(result[0]))
        number_of_representations += len(result[0])
        if len(result[0]) == 1:
            all_together += 1
            all_together_time += all_common_time_stamps[time_index+1] - timestamp
        cover_numbers.append(result[1])
        for indexSet, entity in enumerate(result[1]):
            points = [(entities[entity][entities_times[entity].index(timestamp)][1],
                       entities[entity][entities_times[entity].index(timestamp)][2]),
                      (entities[entity][entities_times[entity].index(timestamp) + 1][1],
                       entities[entity][entities_times[entity].index(timestamp) + 1][2])]
            lin = kml.newlinestring(coords=points)
            # needs to be adjusted for groups with more than 4 entities
            if entity == 0:
                lin.style.linestyle.color = simplekml.Color.navy
            elif entity == 1:
                lin.style.linestyle.color = simplekml.Color.darkgreen
            elif entity == 2:
                lin.style.linestyle.color = simplekml.Color.darkorange
            elif entity == 3:
                lin.style.linestyle.color = simplekml.Color.darkred
            lin.style.linestyle.width = result[2][indexSet]*2
    end = time.time()
    print "time to construct and solve Set Cover", d, time_shift, end-start
    all_together = 1.0 * all_together / (len(all_common_time_stamps))
    number_of_representations = 1.0 * number_of_representations / (len(all_common_time_stamps))

    size = os.path.join(save_path, "outputSize_" + str(d) + '_' + str(time_shift) + ".txt")
    numbers = os.path.join(save_path, "outputNumber_" + birds + str(d) + '_' + str(time_shift) + ".txt")
    kml_save = os.path.join(save_path, 'groupDiagram2D' + birds + str(d) + '_' + str(time_shift) + '.kml')

    file1 = open(size, "w")
    simplejson.dump(cover_size, file1)
    file1.close()

    file2 = open(numbers, 'w')
    simplejson.dump(cover_numbers, file2)
    file2.close()

    kml.save(kml_save)
    return all_together, number_of_representations


# water/land distinction
def create_group_diagram_water_land(entities, entities_times, all_time_stamps, d, time_shift, first, last, birds, save_path):
    gmap = gmplot.GoogleMapPlotter(47, 9, 6)
    over_water = 0
    over_land = 0
    over_water_rep = 0
    over_land_rep = 0
    all_together_water = 0
    all_together_land = 0
    # start and end with same time
    start_times = SortedList()
    end_times = SortedList()
    for entityTime in entities_times:
        start_times.add(entityTime[0])
        end_times.add(entityTime[-1])
    common_start_time = start_times[-1]
    common_end_time = end_times[0]
    all_common_time_stamps = SortedList(
        all_time_stamps[all_time_stamps.index(common_start_time): all_time_stamps.index(common_end_time)])

    # computing all events
    event_count = 0
    event_times = SortedList()
    for time_index, timestamp in enumerate(all_common_time_stamps[:-1]):
        coordinates = [list() for i in xrange(len(entities))]
        for index, entity in enumerate(entities):
            start_segment = entities_times[index].index(timestamp)
            end_segment = start_segment + 1
            coordinates[index] = [entities[index][start_segment][1], entities[index][start_segment][2],
                                  entities[index][end_segment][1], entities[index][end_segment][2]]
        for index1 in range(0, len(coordinates)):  # , coordinates1 in enumerate(coordinates):
            for index2 in range(0, len(coordinates)):
                new_events = compare_two_segments(coordinates[index1][0], coordinates[index1][2], coordinates[index2][0],
                                                  coordinates[index2][2], coordinates[index1][1], coordinates[index1][3],
                                                  coordinates[index2][1], coordinates[index2][3],
                                                  time_index + first, all_common_time_stamps, d)
                if len(new_events) > 0:
                    for event in new_events:
                        event_times.add(event)
                    event_count += 1
    print event_count
    # propagate event time stamps to all trajectories
    adding_count = 0
    for timestamp in event_times:
        if timestamp not in all_common_time_stamps:
            all_common_time_stamps.add(timestamp)
            adding_count += 1
        for index, entity in enumerate(entities):
            length = len(entities_times[index])
            # check if the current entity has this timestamp, if not add it and interpolate the
            # coordinates but only if it is in the range of the trajectory. Otherwise interpolation is not possible
            if timestamp not in entities_times[index] and entities_times[index].bisect(timestamp) > 0 and entities_times[
                index].bisect(timestamp) < length:
                entity.add(interpolate_surface(entity, entities_times[index], timestamp))
                entities_times[index].add(timestamp)

    # solve set cover instance for every time stamp, return the segments of the solution and draw them
    universe = set(range(0, len(entities)))
    cover_size = []
    cover_numbers = []
    color_of_entities = ['red', 'blue', 'green', 'orange', 'black']

    # for each entity initialize
    average_representation_strength = [0 for i in xrange(len(entities))]

    all_together = 0
    number_of_representations = 0
    for time_index, timestamp in enumerate(all_common_time_stamps[:-200]):
        set_list = []
        start_segment_common = all_common_time_stamps.index(timestamp)
        end_segment_common = start_segment_common + 1
        for index in range(0, len(entities)):
            current_set = set()
            start_segment = entities_times[index].index(timestamp)
            end_segment = start_segment + 1
            segment = [[entities[index][start_segment][1], entities[index][start_segment][2]],
                       [entities[index][end_segment][1], entities[index][end_segment][2]]]
            for number, entity in enumerate(entities):
                if within_alpha_similar_time_distance(segment, timestamp, all_common_time_stamps[end_segment_common],
                                                      entity, entities_times[number], time_shift, d, 0):
                    current_set.add(number)
            set_list.append(current_set)
            average_representation_strength[index] += len(current_set)
        result = set_cover(universe, set_list)
        number_of_representations += len(result[0])
        if len(result[0]) == 1:
            all_together += 1
        cover_numbers.append(result[1])

        landmarks_start = [entities[i][entities_times[i].index(timestamp)][3] for i in range(0, len(entities))]
        landmarks_end = [entities[i][entities_times[i].index(timestamp)][3] for i in range(0, len(entities))]
        if 210 in landmarks_start or 210 in landmarks_end:
            water = True
        else:
            water = False

        cover_size.append([(len(result[0])), water])

        # get

        if water:
            over_water += 1
            over_water_rep += len(result[0])
            if len(result[0]) == 1:
                all_together_water += 1
        else:
            over_land += 1
            over_land_rep += len(result[0])
            if len(result[0]) == 1:
                all_together_land += 1

        for indexSet, entity in enumerate(result[1]):
            # print entities[3][entities_times[3].index(timestamp)][2], entities[3][entities_times[3].index(timestamp)][1]
            # print entities[3][entities_times[3].index(timestamp)][3]
            # print color
            # print "\n"
            color = color_of_entities[entity]
            gmap.plot([entities[entity][entities_times[entity].index(timestamp)][2],
                       entities[entity][entities_times[entity].index(timestamp) + 1][2]],
                      [entities[entity][entities_times[entity].index(timestamp)][1],
                       entities[entity][entities_times[entity].index(timestamp) + 1][1]],
                      color=color, edge_width=result[2][indexSet])
    all_together = 1.0 * all_together / (len(all_common_time_stamps)-200)  # ((last + adding_count) - first)
    number_of_representations = 1.0 * number_of_representations / (len(all_common_time_stamps)-200)  # ((last + adding_count) - first)
    print "Together Test:", d, time_shift, all_together
    print "Average Number Representatives:", d, time_shift, number_of_representations
    # average_representation_strength = [1.0 * entry / ((last + len(event_times)) - first) - 1 for entry in
    #                                   average_representation_strength]
    average_representation_strength = [1.0 * entry / (len(all_common_time_stamps)-200) - 1 for entry in
                                       average_representation_strength]

    size = os.path.join(save_path, "outputSize_" + str(d) + '_' + str(time_shift) + ".csv")
    numbers = os.path.join(save_path, "outputNumber_" + birds + str(d) + '_' + str(time_shift) + ".txt")
    representation = os.path.join(save_path, "averageRepresentationStrength" + birds + str(d) + '_' + str(
                                                     time_shift) + ".txt")
    html = os.path.join(save_path, 'groupDiagram_water_land' + birds + str(d) + '_' + str(time_shift) + '.html')

    # file1 = open(size, "w")
    # simplejson.dump(cover_size, file1)
    # file1.close()
    with open(size, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(cover_size)

    file2 = open(numbers, 'w')
    simplejson.dump(cover_numbers, file2)
    file2.close()

    file3 = open(representation, 'w')
    simplejson.dump(average_representation_strength, file3)
    file3.close()

    gmap.draw(html)
    print "average representatives over land:", 1.0*over_land_rep/over_land
    print "average representatives over water:", 1.0*over_water_rep/over_water
    print "time together over land:", 1.0*all_together_land/over_land
    print "time together over water:", 1.0*all_together_water/over_water

    return {d: [all_together, number_of_representations,  1.0*all_together_land/over_land, 1.0*over_land_rep/over_land,
                1.0 * all_together_water/over_water, 1.0*over_water_rep/over_water]}
