import matplotlib.pyplot as plt, numpy as np
import math

plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'heavy'


distance = [3, 5, 10, 20, 40, 80, 160, 320, 640, 1280]
#distance = [math.log(i) for i in [3, 5, 10, 20, 40, 80, 160, 320, 640, 1280]]
#average_number_of_representatives = [3.11044343141, 2.63692189599, 2.1650683534, 1.9378188001, 1.72176079734,1.59416909621, 1.41370716511, 1.23275862069, 1.13387495585, 1.08297258297]
#together = [0.0644426025694, 0.174246785646, 0.389920424403, 0.463201360214, 0.561184939092, 0.590962099125, 0.68722741433, 0.819297082228, 0.893323913811, 0.937229437229]
#average_number_of_representatives_shift = [1.750197571664631, 1.5303462321792261, 1.3991895631547737,
#                                           1.3224170250303231, 1.2384009691096305, 1.1491930192888073, 1.0831251734665557]


#average_number_of_representatives_shift = [2.31392457522, 1.99654576857, 1.78310548868, 1.65314549429, 1.53765227021, 1.40670553936,  1.27912772586, 1.16942970822, 1.10738255034, 1.063492063494]
#together_shift = [0.285951098218, 0.418537708693, 0.53050397878, 0.587320864707, 0.62292358804, 0.679591836735, 0.772274143302, 0.864389920424, 0.910985517485, 0.947691197691]



# print len(distance), len(average_number_of_representatives), len(average_number_of_representatives_shift), len(together), len(together_shift)

#average_number_of_representatives = [2.123797574236721, 1.9603535353535353, 1.6821345707656612, 1.5105067985166873,
#                                     1.3413272010512483, 1.1659513590844064, 1.0937972768532527]

# average_number_of_representatives = [3, 2.51308438695, 2.22799222438, 1.90229051899, 1.73101371696,
#                                      1.5902173913, 1.49738023952, 1.34802784223, 1.19072793819, 1.13851913478, 1.06064461408]
# together = [0, 0.102028815054, 0.231880033324, 0.414612931284, 0.488457678153, 0.560869565217, 0.601796407186, 0.702242846094, 0.832452216348, 0.884775374376, 0.941051738762]
#
# average_number_of_representatives_shift = [3, 2.01558365187, 1.8156067759, 1.66048129893, 1.55570424891,
#                                            1.45217391304, 1.35516467066, 1.25135344161, 1.13786091907,
#                                            1.09858569052, 1.03944020356]
# together_shift = [0, 0.336665686563, 0.450152735351, 0.53696723688, 0.586818333891, 0.635507246377,
#                   0.687874251497, 0.772235112142, 0.877999186661, 0.913893510815, 0.960559796438]
# growing percentage of time flying together relative to the steps of increasing the distance threshold

together = [
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

together = [i*100 for i in together]

together_shift = [
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

together_shift = [i*100 for i in together_shift]

average_number_of_representatives = [
 3.09590590364171,
 2.6197212398585394,
 2.1717103788388252,
 1.9454693434617472,
 1.7260147601476015,
 1.5893909626719056,
 1.4185481069815908,
 1.2295763993948563,
 1.1482511923688394,
 1.0643179024989757]


average_number_of_representatives_shift = [
 2.3080750961320966,
 2.0054087788641564,
 1.8081147724725397,
 1.675257731958763,
 1.545510455104551,
 1.4148657498362802,
 1.2851684612712748,
 1.1720877458396368,
 1.1029411764705883,
 1.044653830397378]

gain = []
gain2 = []
for i in range(1, len(distance)):
    gain.append((together[i]-together[i-1]) / (distance[i]-distance[i-1]))
    gain2.append((together_shift[i]-together_shift[i-1]) / (distance[i]-distance[i-1]))
print gain
print gain2

plt.subplot(121)
plt.plot(distance, average_number_of_representatives, 'ro-', label='equal time', linewidth=7,  markersize=15)
plt.plot(distance, average_number_of_representatives_shift, 'bo-', label='similar time, 10 s', linewidth=7,  markersize=15)

plt.legend()
plt.xscale("log")
plt.ylabel("average number of \n representatives needed", fontdict=dict(weight='bold'))
# plt.ylabel("difference of 'average of needed representatives' \n between equal- and similar-time distance")
plt.xlabel("distance in meters", fontdict=dict(weight='bold'))

plt.subplots_adjust(wspace=0.4)
#
plt.subplot(122)
plt.plot(distance, together, 'ro-', label='equal time', linewidth=7,  markersize=15)
plt.plot(distance, together_shift, 'bo-', label='similar time, 10 s', linewidth=7,  markersize=15)
plt.legend()
plt.xscale("log")
plt.ylabel("percentage of time all family \n members fly together", fontdict=dict(weight='bold'))
plt.xlabel("distance in meters", fontdict=dict(weight='bold'))


plt.show()