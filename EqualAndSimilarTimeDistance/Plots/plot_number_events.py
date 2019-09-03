import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'heavy'
number_of_events = [2245, 2654, 2298, 1434, 946, 730, 538, 296, 144, 60]
distance = [3, 5, 10, 20, 40, 80, 160, 320, 640, 1280]

plt.plot(distance, number_of_events, 'ro-', label="equal time", linewidth=10,  markersize=15)
plt.ylabel("number of events", fontdict=dict(weight='bold'))
plt.xlabel("distance in meters", fontdict=dict(weight='bold'))
plt.xscale("log")
plt.show()


