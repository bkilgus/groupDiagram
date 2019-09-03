import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})
plt.rcParams['font.weight'] = 'heavy'
dic = {0: [0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
       320: [0.8305785123966942, 1.2175056348610067, 0.8745454545454545, 1.1290909090909091, 0.7996158770806658, 1.2797695262483995],
       3: [0.06963725929243171, 3.0868786386027764, 0.07740849810685739, 2.8834665544804374, 0.060794638583054096, 3.318334131163236],
       5: [0.1855585270520469, 2.6017280394980458, 0.21895299961788306, 2.322888803974016, 0.14661319073083778, 2.926916221033868],
       40: [0.5513296227581942, 1.7189239332096475, 0.7456011730205279, 1.31524926686217, 0.40962566844919784, 2.013368983957219],
       10: [0.3942500557165144, 2.1522175172721196, 0.5101952277657267, 1.7245119305856833, 0.27176901924839597, 2.604032997250229],
       160: [0.6936334857544847, 1.4006331340133662, 0.8073630136986302, 1.202054794520548, 0.6143283582089553, 1.5391044776119402],
       1280: [0.944991789819376, 1.055008210180624, 0.9765545361875637, 1.0234454638124364, 0.9237113402061856, 1.0762886597938144],
       80: [0.5922393949358764, 1.5840184149950673, 0.7795400475812847, 1.246629659000793, 0.45955056179775283, 1.8230337078651686],
       20: [0.4565041745219499, 1.9447885806625371, 0.6511357018054746, 1.4595224228305184, 0.28907815631262523, 2.3622244488977957],
       640: [0.9057131442269277, 1.110667199360767, 0.9603567888999008, 1.0406342913776017, 0.8688085676037484, 1.1579651941097724]}

land_rep=[]
land_together=[]
water_rep=[]
water_together=[]
dist=[3,5,10,20,40,80,160,320,640,1280]
for key in dist:
    land_rep.append(dic[key][3])
    land_together.append(dic[key][2])
    water_rep.append(dic[key][5])
    water_together.append(dic[key][4])

#plt.subplot(121)
plt.xscale("log")
plt.xlabel("distance in meter",  fontdict=dict(weight='bold'))
plt.ylabel("average number of \n representatives needed",  fontdict=dict(weight='bold'))
plt.plot(dist, land_rep, 'ro-', label='over land', linewidth=10,  markersize=20)
plt.plot(dist, water_rep, 'bo-', label='over water', linewidth=10,  markersize=20)
plt.legend()

# plt.subplot(122)
# plt.xscale("log")
# plt.xlabel("distance in meter",  fontdict=dict(weight='bold'))
# plt.ylabel("percentage of time travelled all together",  fontdict=dict(weight='bold'))
# plt.plot(dist, land_together, 'ro-', label='over land', linewidth=7,  markersize=15)
# plt.plot(dist, water_together, 'bo-', label='over water', linewidth=7,  markersize=15)
# plt.legend()


plt.show()