﻿import numpy as np

zero = np.asarray(
        [-1,+1,+1,+1,+1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,+1,+1,+1,+1,-1]).T

one = np.asarray(
        [-1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,+1,+1,-1,
        -1,-1,-1,+1,-1,+1,-1,
        -1,-1,+1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1]).T

two = np.asarray(
        [-1,+1,+1,+1,+1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,+1,-1,-1,
        -1,-1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,-1,
        -1,+1,+1,+1,+1,+1,-1]).T

three = np.asarray(
        [-1,+1,+1,+1,+1,+1,-1,
        -1,-1,-1,-1,+1,-1,-1,
        -1,-1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,-1,
        -1,+1,+1,+1,+1,+1,-1,
        -1,-1,-1,-1,+1,-1,-1,
        -1,-1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,-1,
        -1,+1,-1,-1,-1,-1,-1]).T

four = np.asarray(
        [-1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,+1,+1,+1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1]).T

five = np.asarray(
        [-1,+1,+1,+1,+1,+1,-1,
        -1,+1,-1,-1,-1,-1,-1,
        -1,+1,-1,-1,-1,-1,-1,
        -1,+1,-1,-1,-1,-1,-1,
        -1,+1,+1,+1,+1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,-1,+1,-1,
        -1,+1,+1,+1,+1,+1,-1]).T

six = np.asarray(
        [-1,-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,+1,-1,-1,
        -1,-1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,-1,
        -1,+1,+1,+1,+1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,+1,+1,+1,+1,-1]).T

seven = np.asarray(
        [-1,+1,+1,+1,+1,+1,-1,
        -1,-1,-1,-1,+1,-1,-1,
        -1,-1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,-1,
        -1,+1,-1,-1,-1,-1,-1,
        -1,+1,-1,-1,-1,-1,-1,
        -1,+1,-1,-1,-1,-1,-1,
        -1,+1,-1,-1,-1,-1,-1,
        -1,+1,-1,-1,-1,-1,-1]).T

eight = np.asarray(
        [-1,+1,+1,+1,+1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,+1,+1,+1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,+1,+1,+1,+1,-1]).T

nine =  np.asarray(
        [-1,+1,+1,+1,+1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,-1,-1,-1,+1,-1,
        -1,+1,+1,+1,+1,+1,-1,
        -1,-1,-1,-1,+1,-1,-1,
        -1,-1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,-1,
        -1,+1,-1,-1,-1,-1,-1]).T

digits = np.asarray([zero, one, two, three, four, five, six, seven, eight, nine])