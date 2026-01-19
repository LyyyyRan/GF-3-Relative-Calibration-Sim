# -*- coding: utf-8 -*-
"""
Created on January 15, 2026

@author: https://github.com/LyyyyRan
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from ly_utils import findpeak
from utils import img2View

# load Focused image:
Ori_Focused_Data = loadmat('./mat_files/Focused_Data_9_100.mat')['Focused_Data']
peak_y, peak_x = findpeak(Ori_Focused_Data)
Ori_ROI = Ori_Focused_Data[peak_y - 16:peak_y + 16 + 1, peak_x - 16:peak_x + 16 + 1]

# load Patten over Azimuth:
Patten_Azimuth = np.load('./npy_files/Pattern_Azimuth.npy')
Patten_Azimuth = np.expand_dims(Patten_Azimuth, axis=1)

# show Pattern over Azimuth:
plt.figure('Patten-Azimuth[-60:60]')
plt.title('Patten-Azimuth[-60:60]')
plt.xlabel('Azimuth')
plt.ylabel('Modulus Rate')
plt.plot((Patten_Azimuth ** 1)[800 - 60:800 + 60 + 1])

# Calibration over Azimuth:
Focused_Data = Ori_Focused_Data * Patten_Azimuth
ROI = Ori_ROI * (Patten_Azimuth[800 - 16:800 + 16 + 1] ** 1)

# show original data:
plt.figure('Original Data')
plt.title('Original Data')
plt.ylabel('Azimuth')
plt.xlabel('Range')
plt.imshow(img2View(Ori_Focused_Data, enhance=True))

# show Calibrated-Azimuth data:
plt.figure('Calibrated-Azimuth Data')
plt.title('Calibrated-Azimuth Data')
plt.ylabel('Azimuth')
plt.xlabel('Range')
plt.imshow(img2View(Focused_Data, enhance=True))

# show original ROI:
plt.figure('Original ROI')
plt.title('Original ROI')
plt.ylabel('Azimuth')
plt.xlabel('Range')
plt.imshow(img2View(Ori_ROI, enhance=False))

# show Calibrated-Azimuth ROI:
plt.figure('Calibrated-Azimuth ROI')
plt.title('Calibrated-Azimuth ROI')
plt.ylabel('Azimuth')
plt.xlabel('Range')
plt.imshow(img2View(ROI, enhance=False))

plt.show()
