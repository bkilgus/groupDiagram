import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'heavy'
gain = [0.023599487295483678, 0.055821904900335964, 0.039906972696872656, 0.0064848692283749785, 0.004303850078652716,
        0.0010192760326424466, 0.0012521401845625557, 0.000934553694681977, 0.00018927840639874522,
        8.084270500830736e-05]

gain_shift =[0.09824323305436176, 0.056816552540362913, 0.020232841527386625, 0.004961569023537083,
             0.0021166656322016255, 0.00157246439851432, 0.001199485730169443, 0.0006303856858496943,
             0.0001567877836009899, 6.871706250012217e-05]


together = [0.0,
 0.07079846188645103,
 0.18244227168712296,
 0.38197713517148624,
 0.446825827455236,
 0.5329028290282903,
 0.5736738703339882,
 0.6738450850989927,
 0.823373676248109,
 0.8839427662957074,
 0.9356820975010242]

together_shift = [0.0,
 0.2947296991630853,
 0.4083628042438111,
 0.5095270118807442,
 0.559142702116115,
 0.6014760147601476,
 0.6643745907007204,
 0.7603334491142758,
 0.8611951588502269,
 0.9113672496025437,
 0.9553461696026219]

distance = [0, 3, 5, 10, 20, 40, 80, 160, 320, 640, 1280]

gain_1 = [(together[i+1]-together[i])*100/(distance[i+1]-distance[i]) for i in range(0, len(together)-1)]
gain_1_shift = [(together_shift[i+1]-together_shift[i])*100/(distance[i+1]-distance[i]) for i in range(0, len(together_shift)-1)]
print gain_1
print gain_1_shift
# diff_together = [0.102028815054, 0.12985121827, 0.18273289796, 0.073844746869, 0.07241188706400004, 0.04092684196899998, 0.10044643890799998, 0.13020937025400003, 0.052323158027999916, 0.05627636438600003]
x = range(0,len(gain)-2)
my_xticks = ['0-3', '3-5', '5-10','10-20', '20-40', '40-80', '80-160', '160-320']
plt.xticks(x, my_xticks)
plt.plot(gain_1[:-2], 'ro-', label="equal time", linewidth=7,  markersize=15)
plt.plot(gain_1_shift[:-2], 'bo-', label="similar time, 10s", linewidth=7,  markersize=15)
#plt.xscale("log")
plt.legend()
plt.ylabel("increase of time family is flying \n together in percentage per meter", fontdict=dict(weight='bold'))
plt.xlabel("distance in meters", fontdict=dict(weight='bold'))
plt.show()

# x = range(0,len(gain))
# my_xticks = ['3','5','10','20', '40', '80', '160', '320', '640', '1280']
# plt.xticks(x, my_xticks)
# plt.plot(together, 'ro-', label="equal time")
# plt.plot(together_shift, 'bo-', label="similar time, 10s")
# #plt.xscale("log")
# plt.legend()
# plt.ylabel("time (percentage)\n when family travels together")
# plt.xlabel("distance in meters")
# plt.show()