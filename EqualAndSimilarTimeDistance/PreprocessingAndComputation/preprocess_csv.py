import csv

csv_file = ""
store_preprocessed_file = ""
preprocessed_file = []
columns_needed = [2, 3, 4, 32]
with open(csv_file) as csvFile:
    data = csv.reader(csvFile)
    next(data)
    for row in data:
        try:
            if float(row[23]) < 10:
                preprocessed_file.append([row[i] for i in columns_needed])
        except ValueError:
            print "invalid input, row is ignored"

with open(store_preprocessed_file, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(preprocessed_file)