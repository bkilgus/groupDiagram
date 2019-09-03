import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'heavy'

time_for_events = [2.92499995232, 2.90000009537, 2.50999999046, 2.74500012398, 2.14599990845, 1.9960000515, 2.00900006294, 1.84899997711, 1.85399985313, 1.75999999046
]
time_for_GD_computation = [3.9430000782, 4.18299984932, 3.90799999237, 3.27699995041, 3.07700014114, 2.66999983788, 2.5529999733, 2.34600019455, 2.34600019455, 2.30700016022
]

# 4 entities

event_time_4 = [1.84400010109, 2.09400010109, 1.9849998951, 1.79700016975, 1.6099998951, 1.51600003242, 1.46900010109, 1.43799996376, 1.29699993134, 1.28200006485, 1.25]
solution_time_4 = [0.0, 0.56299996376, 0.68700003624, 0.577999830246, 0.343000173569, 0.218999862671, 0.171999931335, 0.125, 0.063000202179, 0.0310001373291, 0.0149998664856]
number_4 = [2245, 2654, 2298, 1434, 946, 730, 538, 296, 144, 60]


# 8 entities
event_time_8_1 = [8.12600016594, 8.87600016594, 8.63499999046, 8.01600003242, 6.78200006485, 6.07899999619, 5.75100016594, 5.40999984741, 5.12600016594, 4.79800009727, 4.64300012589]
solution_time_8_1 = [0.0, 7.34399986267, 10.1410000324, 9.4390001297, 5.51600003242, 3.57799983025, 2.71899986267, 1.95300006866, 1.03099989891, 0.5, 0.219000101089]
number_8_1 = [7507, 10102, 9462, 5770, 3788, 2924, 2162, 1180, 584, 240]

event_time_8_2 = [7.53200006485, 8.57899999619, 8.79800009727, 8.22000002861, 6.81299996376, 6.14100003242, 5.76600003242, 5.42300009727, 5.07899999619, 4.89100003242, 4.76600003242]
solution_time_8_2 = [0.0, 5.72000002861, 7.62599992752, 9.42200016975, 5.79800009727, 3.71900010109, 2.7349998951, 2.7349998951, 1.04699993134, 0.530999898911, 0.219000101089]
number_8_2 = [5908, 7765, 9470, 6035, 3836, 2928, 2138, 1180, 592, 240]

# 12 entities
event_time_12_1 = [16.3919999599, 18.7229998112, 18.2009999752, 16.7679998875, 14.3650000095, 12.8919999599, 12.3129999638, 11.6570000648, 10.8450000286, 10.3759999275, 12.1979999542]
solution_time_12_1 = [0.0, 30.5729999542, 41.0659999847, 46.8050000668, 27.7569999695, 17.736000061, 13.2669999599, 9.54799985886, 5.03199982643, 2.47399997711, 1.07100009918]
number_12_1 = [14573, 19202, 21708, 13286, 8610, 6586, 4850, 2656, 1320, 540]

event_time_12_2 = [17.9839999676, 19.3389999866, 20.0020000935, 17.3669998646, 14.896999836, 13.5820000172, 12.9279999733, 12.2430000305, 14.9170000553, 11.0050001144, 10.7829999924]
solution_time_12_2 = [0.0, 38.0910000801, 50.7999999523, 47.4350001812, 29.859000206, 18.7139999866, 13.7279999256, 13.3659999371, 7.117000103, 2.55599999428, 1.09200000763]
number_12_2 = [16981, 22140, 20946, 13094, 8562, 6546, 4846, 2656, 1312, 540]

# 16 entities
event_time_16_1 = [44.1520001888, 43.2049999237, 41.2739999294, 34.8339998722, 25.2720000744, 23.2969999313, 23.4849998951, 22.3989999294, 20.8340001106, 19.3169999123, 18.1510000229]
solution_time_16_1 = [0.0, 140.980999947, 204.269999981, 162.802000046, 101.174000025, 66.734000206, 44.6790001392, 32.1349999905, 18.4339997768, 8.11800003052, 3.31100010872]
number_16_1 = [28981, 38570, 37233, 23292, 15232, 11680, 8602, 4712, 2340, 960]

event_time_16_2 = [29.4249999523, 31.6549999714, 34.0130000114, 28.5390000343, 27.0020000935, 24.9279999733, 23.254999876, 22.4029998779, 21.1019999981, 21.5250000954, 20.3370001316]
solution_time_16_2 = [0, 135.888000011, 179.481999874, 161.129999876, 89.6480000019, 59.9400000572, 45.4880001545, 32.4479999542, 16.138999939, 8.94199991226, 3.32799983025]
number_16_2 = [34267, 41609, 36987, 22854, 15136, 11668, 8614, 4712, 2316, 960]


event_time_8 = [(x+y)/2 for x,y in zip(event_time_8_1, event_time_8_2)]
solution_time_8 = [(x+y)/2 for x,y in zip(solution_time_8_1, solution_time_8_2)]
number_8 = [(x+y)/2 for x,y in zip(number_8_1, number_8_2)]

event_time_12 = [(x+y)/2 for x,y in zip(event_time_12_1, event_time_12_2)]
solution_time_12 = [(x+y)/2 for x,y in zip(solution_time_12_1, solution_time_12_2)]
number_12 = [(x+y)/2 for x,y in zip(number_12_1, number_12_2)]

event_time_16 = [(x+y)/2 for x,y in zip(event_time_16_1, event_time_16_2)]
solution_time_16 = [(x+y)/2 for x,y in zip(solution_time_16_1, solution_time_16_2)]
number_16 = [(x+y)/2 for x,y in zip(number_16_1, number_16_2)]

distance = [0, 3, 5, 10, 20, 40, 80, 160, 320, 640, 1280]
time_shift = [0, 10, 20, 40, 80]
# plt.plot(distance, time_for_GD_computation, 'ro-', label="construct and solve Set-Cover", linewidth=7,  markersize=15)
# plt.plot(distance, time_for_events, 'go-', label="event computation and vertex propagation", linewidth=7,  markersize=15)
# plt.legend()
# plt.ylabel("computation time in seconds", fontdict=dict(weight='bold'))
# plt.xlabel("distance in meters", fontdict=dict(weight='bold'))
# plt.xscale("log")
# plt.show()
distance_5 = [1.70300006866, 18.2670001984, 24.7680001259, 28.2369999886, 34.1289999485]
distance_10 = [1.45399999619, 16.2680001259, 18.8389999866, 22.4070000648, 31.2949998379 ]
distance_40 = [0.563999891281, 3.8900001049, 4.74499988556, 7.45899987221, 9.60300016403 ]
distance_320 = [0.234000205994, 0.811000108719, 1.17899990082, 1.10100007057, 1.84300017357]


time_4 = [(x+y) for x, y in zip(event_time_4, solution_time_4)]
time_8_approach_1 = [(x+y) for x, y in zip(event_time_8, solution_time_8)]
time_12_approach_1 = [(x+y) for x, y in zip(event_time_12, solution_time_12)]
time_16_approach_1 = [(x+y) for x, y in zip(event_time_16, solution_time_16)]

time_8_approach_2 = [17.96, 146.8, 161.86, 89.29, 36.15, 24.81, 20.53, 17.05, 13.804, 11.304, 10.88]
time_12_approach_2 = [32.56, 735.02, 822.43, 570.04, 203.39, 81.72, 62.36, 51.20, 36.49, 28.44, 33.24]
time_16_approach_2 = [51.7960000038, 2161.04699993, 11326.0609999, 4283.773, 938.77699995, 419.799999952, 178.05599999, 119.37800002,  84.52399993, 58.17599988, 45.46700001]


number_8_1_approach_2 = [33460, 43658, 26422, 8290, 4292, 3168, 2262, 1236, 604, 240]
number_8_2_approach_2 = [34238, 43708, 26332, 8352, 4296, 3128, 2234, 1244, 604, 240]

number_12_1_approach_2 = [87435, 115704, 79036, 21400, 9918, 7098, 5130, 2832, 1344, 540]
number_12_2_approach_2 = [87937, 116153, 77183, 21550, 9862, 7070, 5138, 2780, 1364, 540]

number_16_1_approach_2 = [164028, 217545, 156460, 39214, 18004, 12860, 9166, 4932, 2488, 960]
number_16_2_approach_2 = [165935, 219627, 157295, 40265, 18168, 12808, 9212, 5020, 2424, 960]

number_8_approach_2 = [(x+y)/2 for x,y in zip(number_8_1_approach_2, number_8_2_approach_2)]
number_12_approach_2 = [(x+y)/2 for x,y in zip(number_12_1_approach_2, number_12_2_approach_2)]
number_16_approach_2 = [(x+y)/2 for x,y in zip(number_16_1_approach_2, number_16_2_approach_2)]

construction_time = [35.893999815, 51.9960010052, 28.6079993248, 10.1629989147, 4.87599992752, 3.14199924469, 2.00999999046, 1.03499984741,  0.542000055313, 0.205999851227]
solving_time = [0.862000226974, 1.02399897575, 0.427000999451, 0.218000888824, 0.0670003890991, 0.090000629425, 0.0429999828339, 0.0210001468658, 0.00499987602234, 0.00200009346008]

# plt.plot(distance, construction_time, 'ro-', label="construction time", linewidth=7,  markersize=15)
# plt.plot(distance, solving_time, 'go-', label="solving time", linewidth=7,  markersize=15)
# plt.legend()
# plt.ylabel("computation time in seconds", fontdict=dict(weight='bold'))
# plt.xlabel("distance in meters", fontdict=dict(weight='bold'))
# plt.xscale("log")
# plt.show()


# plt.plot(time_shift, distance_5, 'ro-', label="5 m", linewidth=7,  markersize=15)
# plt.plot(time_shift, distance_10, 'go-', label="10 m", linewidth=7,  markersize=15)
# plt.plot(time_shift, distance_40, 'bo-', label="40 m", linewidth=7,  markersize=15)
# plt.plot(time_shift, distance_320, 'o-', color="black", label="320 m", linewidth=7,  markersize=15)
# plt.legend()
# plt.ylabel("computation time in seconds", fontdict=dict(weight='bold'))
# plt.xlabel("allowed time shift in seconds", fontdict=dict(weight='bold'))
# plt.show()

# plt.subplot(121)
# plt.plot(distance, number_4, 'ro-', label="4 entities", linewidth=7,  markersize=15)
# plt.plot(distance, number_8, 'go-', label="8 entities", linewidth=7,  markersize=15)
# plt.plot(distance, number_12, 'bo-', label="12 entities", linewidth=7,  markersize=15)
# plt.plot(distance, number_16, 'o-', color="black", label="16 entities", linewidth=7,  markersize=15)
# plt.legend()
# plt.ylabel("number of events", fontdict=dict(weight='bold'))
# plt.xlabel("distance in meters", fontdict=dict(weight='bold'))
# plt.xscale("log")
# plt.subplots_adjust(wspace=0.4)
# plt.subplot(122)
# plt.plot(distance, number_4, 'ro-', label="4 entities", linewidth=7,  markersize=15)
# plt.plot(distance, number_8_approach_2, 'go-', label="8 entities", linewidth=7,  markersize=15)
# plt.plot(distance, number_12_approach_2, 'bo-', label="12 entities", linewidth=7,  markersize=15)
# plt.plot(distance, number_16_approach_2, 'o-', color="black", label="16 entities", linewidth=7,  markersize=15)
# plt.legend()
# plt.ylabel("number of events", fontdict=dict(weight='bold'))
# plt.xlabel("distance in meters", fontdict=dict(weight='bold'))
# plt.xscale("log")
# plt.show()

# comparison total runtime for different data generation strategies
plt.subplot(121)
plt.plot(distance, time_4, 'ro-', label="4 entities", linewidth=7,  markersize=15)
plt.plot(distance, time_8_approach_1, 'go-', label="8 entities", linewidth=7,  markersize=15)
plt.plot(distance, time_12_approach_1, 'bo-', label="12 entities", linewidth=7,  markersize=15)
plt.plot(distance, time_16_approach_1, 'o-', color="black", label="16 entities", linewidth=7,  markersize=15)
# plt.legend()
plt.ylabel("computation time in seconds",  fontdict=dict(weight='bold'))
plt.xlabel("distance in meters", fontdict=dict(weight='bold'))
plt.yscale("log")
plt.ylim(0, 15000)
plt.xscale("symlog")
plt.subplots_adjust(wspace=0.4)
plt.subplot(122)
plt.plot(distance, time_4, 'ro-', label="4 entities", linewidth=7,  markersize=15)
plt.plot(distance, time_8_approach_2, 'go-', label="8 entities", linewidth=7,  markersize=15)
plt.plot(distance, time_12_approach_2, 'bo-', label="12 entities", linewidth=7,  markersize=15)
plt.plot(distance, time_16_approach_2, 'o-', color="black", label="16 entities", linewidth=7,  markersize=15)
plt.legend()
plt.ylabel("computation time in seconds",  fontdict=dict(weight='bold'))
plt.xlabel("distance in meters", fontdict=dict(weight='bold'))
plt.yscale("log")
plt.xscale("symlog")
plt.ylim(0, 15000)
plt.show()
#
#
# plt.subplot(211)
# plt.plot(distance, event_time_4, 'ro-', label="4 entities", linewidth=7,  markersize=15)
# plt.plot(distance, event_time_8, 'go-', label="8 entities", linewidth=7,  markersize=15)
# plt.plot(distance, event_time_12, 'bo-', label="12 entities", linewidth=7,  markersize=15)
# plt.yticks((0,5,10,15,20,25,30,35))
# plt.plot(distance, event_time_16, 'o-', color="black", label="16 entities", linewidth=7,  markersize=15)
# plt.ylabel("computation time \n in seconds", fontdict=dict(weight='bold'))
# plt.xscale("symlog")
#
# plt.subplots_adjust(wspace=0.7)
# plt.subplot(212)
# plt.plot(distance, solution_time_4, 'ro-', label="4 entities", linewidth=7,  markersize=15)
# plt.plot(distance, solution_time_8, 'go-', label="8 entities", linewidth=7,  markersize=15)
# plt.plot(distance, solution_time_12, 'bo-', label="12 entities", linewidth=7,  markersize=15)
# plt.plot(distance, solution_time_16, 'o-', color="black", label="16 entities", linewidth=7,  markersize=15)
# plt.legend()
# plt.yticks((0,25,50,75,100,125,150,175,200))
# plt.ylabel("computation time \n in seconds", fontdict=dict(weight='bold'))
# plt.xlabel("distance in meters", fontdict=dict(weight='bold'))
# plt.xscale("symlog")
# plt.show()
# #

