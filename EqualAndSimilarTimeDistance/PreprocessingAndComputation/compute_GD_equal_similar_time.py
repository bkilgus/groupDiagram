import groupDiagramEqualSimilarTime
import time

csv_file_name_with_path = ""
start_read = time.time()
[entities, entities_times, all_time_stamps] = groupDiagram.read_data(csv_file_name_with_path)
end_read = time.time()
print "data read"
start = time.time()

# distance in m
distance_threshold = 10
# timeshift in seconds
timeshift = 0
# fist time stamp
first = 0
# last time stamp (counted backwards) use -1 to loop till end
last = -1
name_of_group = "WFamily"
path_to_save_computations = ""


groupDiagram.create_group_diagram(entities, entities_times, all_time_stamps, distance_threshold, timeshift,
                                  first, last, name_of_group, path_to_save_computations)

print("\n")
end = time.time()

print "time for read:", end_read - start_read
print "time for GD computation", (end - start)
