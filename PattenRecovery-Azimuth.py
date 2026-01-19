# -*- coding: utf-8 -*-
"""
Created on January 15, 2026

@author: https://github.com/LyyyyRan
"""

import numpy as np
from np2mtlb import FFT_Azimuth, IFFT_Azimuth
import matplotlib.pyplot as plt

# load Patten over Azimuth:
Patten_Azimuth = np.load('./npy_files/Pattern_Azimuth.npy')
Patten_Azimuth = np.expand_dims(Patten_Azimuth, axis=1)

# Find peak:
peak_a = Patten_Azimuth.argmax()

# show Pattern over Azimuth:
show_length = 60
plt.figure('Patten-Azimuth[-{}:{}]'.format(int(show_length / 2), int(show_length / 2)))
plt.title('Patten-Azimuth[-{}:{}]'.format(int(show_length / 2), int(show_length / 2)))
plt.xlabel('Azimuth')
plt.ylabel('Modulus Rate')
plt.plot((Patten_Azimuth ** 1)[peak_a - int(show_length / 2):peak_a + int(show_length / 2) + 1], '*')
plt.plot((Patten_Azimuth ** 1)[peak_a - int(show_length / 2):peak_a + int(show_length / 2) + 1], '-')

# FFT:
Patten_Azimuth_F = FFT_Azimuth(Patten_Azimuth, shift=True)

Patten_Azimuth_F[0:399] = 0
Patten_Azimuth_F[-400:] = 0

plt.figure('Patten-Azimuth in Frequency Domain')
plt.title('Patten-Azimuth in Frequency Domain')
# plt.xlabel('Azimuth')
# plt.ylabel('Modulus Rate')
plt.plot(Patten_Azimuth_F)

# IFFT:
Patten_Azimuth = IFFT_Azimuth(Patten_Azimuth_F, shift=True)
Patten_Azimuth = Patten_Azimuth / Patten_Azimuth.max()

plt.figure('(Patten-Azimuth * sinc)[-{}:{}]'.format(int(show_length / 2), int(show_length / 2)))
plt.title('(Patten-Azimuth * sinc)[-{}:{}]'.format(int(show_length / 2), int(show_length / 2)))
# plt.xlabel('Azimuth')
# plt.ylabel('Modulus Rate')
plt.plot((Patten_Azimuth ** 1)[peak_a - int(show_length / 2):peak_a + int(show_length / 2) + 1], '*')
plt.plot((Patten_Azimuth ** 1)[peak_a - int(show_length / 2):peak_a + int(show_length / 2) + 1], '-')

# try to recover:
# alpha = 0.2 * np.pi
# t = np.linspace(-30, 30, 1600)
# plt.plot(t, np.sin(alpha * t) / (alpha * t))

# show all:
plt.show()
