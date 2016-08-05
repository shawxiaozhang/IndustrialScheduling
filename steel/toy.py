"""

__author__ = xxxzhang06
"""

import numpy as np


tasks = [78.0, 110.0, 130.0, 156.0, 222.0, 314.0, 370.0, 444.0]
resources = [91.0, 126.0, 147.0, 175.0, 166.0, 231.0, 270.0, 322.0]
num_var = [14379, 19833, 23235, 27758, 31617, 43882, 51503, 61715]
num_bin = [5357.0, 7416.0, 8781.0, 10588.0, 15320.0, 21280.0, 25118.0, 30286.0]
num_bin_10 = [8243, 11344, 13543, 16308, 23555, 32524, 38449, 46608]

print np.divide(num_bin, tasks)
print np.divide(np.subtract(num_var, num_bin), resources)
print np.divide(num_bin_10, num_bin)