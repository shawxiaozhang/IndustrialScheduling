"""
__author__ = xxxzhang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 12})
# matplotlib.rcParams.update({'lines.linewidth': 5})
# matplotlib.rcParams.update({'lines.markersize': 9})


# plant1 MISO price
# makespan_obj = [41218.5, 63007.6, 74156.0, 87297.4]
# obj_acc_t15 = [39314.4, 57859.0, 69735.6, 86355.9]
# cpu_acc_t15 = [3005.0, 4766.0, 1005.0, 2975.0]
# obj_acc_heu_t15 = [39527.2, 58328.2, 70309.4, 87029.1]
# cpu_acc_heu_t15 = [394.0, 793.0, 103.0, 5.0]

# plant2 U shape price
makespan_obj = [210987.73, 247321.87, 271575.37, 319834.437500]
obj_acc_t15 = [104321.5, 171908.4, 222910.1, 299965.9]
cpu_acc_t15 = [9.0, 322.0, 3386.0, 1900.0]
obj_acc_heu_t15 = [104321.5, 172835.3, 224964.3, 309960.5]
cpu_acc_heu_t15 = [8.0, 142.0, 354.0, 7.0]

bene_heu = np.subtract(makespan_obj, obj_acc_heu_t15)
bene = np.subtract(makespan_obj, obj_acc_t15)
print bene
print bene_heu
ratio_bene = np.divide(bene_heu, bene)
ratio_cpu = np.divide(cpu_acc_heu_t15, cpu_acc_t15)
ratio_obj = np.divide(obj_acc_heu_t15, obj_acc_t15)
print ratio_obj
print np.mean(ratio_obj)
print ratio_cpu
print np.mean(ratio_cpu), 1-np.mean(ratio_cpu)
print ratio_bene
print np.mean(ratio_bene)

plt.figure()
plt.bar([3*i-1 for i in range(3, 7)], bene_heu, color='blue', width=1.0)
plt.bar([3*i for i in range(3, 7)], bene, color='red', width=1.0)
plt.legend(['with heuristics', 'w/o heuristics'])
plt.xticks([3*i for i in range(3, 7)], ['G1-G%d' % i for i in range(3, 7)])
plt.ylabel('Saving [$] compared with benchmark cost')
plt.savefig('bar_2.pdf')
plt.close()


makespan_obj = [41218.5, 63007.6, 74156.0, 87297.4]	    # t15
# makespan_obj = [40647.8, 62676.5, 73949.6, 87202.6]	# t10
# obj_heu_t10 = [39142.2, 57882.3, 69510.4, 85635.9]
# relax_obj_t10 = [39034.0, 57494.0, 69054.3, 85222.7]
obj_acc_heu_t10 = [39231.7, 57892.5, 69518.3, 85638.7]
relax_obj_acc_t10 = [39043.2, 57523.5, 69096.8, 85228.0]
obj_acc_t10_max_cpu = [39046.9, 57523.5, 69096.8, 85236.0]

bene_heu = np.subtract(obj_acc_heu_t10, makespan_obj)
bene = np.subtract(obj_acc_t10_max_cpu, makespan_obj)
print bene_heu
print bene
