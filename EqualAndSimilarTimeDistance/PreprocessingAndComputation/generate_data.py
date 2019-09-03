import csv
import random

# type 1 or 2 of data generation
type1 = False

file_to_store_generated_data = ""

r_1 = random.uniform(-0.0001, 0.0001)
r_2 = random.uniform(-0.0001, 0.0001)

blown_up_data = []
number_of_copies = 2
csv_file = ""
with open(csv_file) as csvFile:
    data = csv.reader(csvFile)
    next(data)
    blown_up_data.append(
        ["timestamp", "location-long", "location-lat", "name", "location-error-numerical", "surface"])

    for row in data:
        blown_up_data.append(row)
        if type1:
            for i in range(0, number_of_copies):
                new_row = [row[0], float(row[1]) + r_1, float(row[2]) + r_2, row[3] + "_" + str(i), row[4], row[5]]
                blown_up_data.append(new_row)
        else:
            for i in range(0, number_of_copies):
                r_1 = random.uniform(-0.0001, 0.0001)
                r_2 = random.uniform(-0.0001, 0.0001)
                new_row = [row[0], float(row[1]) + r_1, float(row[2]) + r_2, row[3] + "_" + str(i), row[4], row[5]]
                blown_up_data.append(new_row)

with open(file_to_store_generated_data, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(blown_up_data)