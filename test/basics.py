# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 17:24:14 2018

@author: Jonas
"""

# loops

d = {'person': 2, 'cat': 4, 'spider': 8}

for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))

for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))



# Dictionary construction
nums = [0,1,2,3,4]
even_num_to_square = {x: x**2 for x in nums if x%2 == 0}
print(even_num_to_square)

squares = { x**2 for x in nums }
print(squares)


import numpy as np

j2 = np.arange(10)
j2[1:6:2]
# array([1, 3, 5])
j2[::2]  # slices are views on the original data, not copies
# array([0, 2, 4, 6, 8])
j3 = np.arange(10)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
j3[-2]
# 8
j3.shape = (2, 5)
# array([[0, 1, 2, 3, 4],
#       [5, 6, 7, 8, 9]])
j3[1, -1]
# 9

j3[:, 0:2]
#    Out[79]: 
#    array([[0, 1],
#           [5, 6]])
j4 = np.arange(35).reshape(5,7)
#    array([[ 0,  1,  2,  3,  4,  5,  6],
#           [ 7,  8,  9, 10, 11, 12, 13],
#           [14, 15, 16, 17, 18, 19, 20],
#           [21, 22, 23, 24, 25, 26, 27],
#           [28, 29, 30, 31, 32, 33, 34]])
j5 = np.arange(10,1,-2)
# array([10,  8,  6,  4,  2])
j5[np.array([3,3,-2,1])]
# array([4, 4, 4, 8])


