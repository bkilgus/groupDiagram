import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'heavy'

x = [4, 8, 12, 16, 20, 24]
y = [3.375, 30.614, 155.68, 534.169, 1429.490, 2974.01]
y_2 = [k**5*3.375 for k in range(1, 7)]

# plt.plot(x, y, 'ro-', label="time in practice")
# plt.plot(x, y_2, 'bo-', label="runtime in theory")
# my_xticks = ['4', '8', '12','16', '20', '24']
# plt.xticks(x, my_xticks)
# plt.legend()
# plt.yscale("log")
# plt.ylabel("computation time in seconds")
# plt.xlabel("number of entities")
# plt.show()

# distance = [0, 3, 5, 10, 20, 40, 80, 160, 320, 640, 1280]
# time = [4.486000061035156, 6.302000045776367, 9.193000078201294, 7.2200000286102295, 6.45799994468689, 5.110000133514404, 5.4040000438690186, 5.19599986076355, 4.0970001220703125, 4.496999979019165, 4.874000072479248]
# plt.plot(distance, time, 'ro-', label="time in practice")
# my_xticks = ['0', '3', '5','10', '20', '40', '80', '160', '320', '640', '1280']
# plt.xticks(x, my_xticks)
# plt.legend()
# plt.ylabel("computation time in seconds")
# plt.xlabel("distance in m")
# plt.xscale("log")
# plt.show()

x_1 = [3.98200011253, 5.42000007629, 5.65599989891, 5.01300001144, 4.20600008965, 3.742000103, 3.61099982262,
       3.85000014305, 3.2650001049, 3.06100010872, 3.21499991417]
x_2_1 = [17.96, 146.8, 161.86, 89.29, 36.15, 24.81, 20.53, 17.05, 13.804, 11.304, 10.88]
x_3_1 = [32.56, 735.02, 822.43, 570.04, 203.39, 81.72, 62.36, 51.20, 36.49, 28.44, 33.24]
x_4_1 = [51.7960000038, 2161.04699993, 11326.0609999, 4283.773, 938.77699995, 419.799999952, 178.05599999, 119.37800002,  84.52399993, 58.17599988, 45.46700001]

x_2 = [x_2_1[j]/x_1[j] for j in range(0, len(x_1))]
x_3 = [x_3_1[j]/x_1[j] for j in range(0, len(x_1))]
x_4 = [x_4_1[j]/x_1[j] for j in range(0, len(x_1))]



distance = [0, 3, 5, 10, 20, 40, 80, 160, 320, 640, 1280]
#my_xticks = ['0', '3', '5','10', '20', '40', '80', '160', '320', '640', '1280']
#plt.xticks(x, my_xticks)
plt.plot(distance, x_1, 'ro-', label="4 entities", linewidth=7,  markersize=15)
plt.plot(distance, x_2, 'bo-', label="8 entities", linewidth=7,  markersize=15)
plt.plot(distance, x_3, 'go-', label="12 entities", linewidth=7,  markersize=15)
plt.plot(distance, x_4, 'o-', color="black", label="16 entities", linewidth=7,  markersize=15)

plt.legend()
plt.ylabel("computation time in seconds", fontdict=dict(weight='bold'))
plt.xlabel("distance in m", fontdict=dict(weight='bold'))
plt.xscale("symlog")
plt.yscale("log")

plt.show()

