# -*- coding: utf-8 -*-
"""
Created on January 14, 2026

@author: https://github.com/LyyyyRan
"""

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from utils import img2View, UpSampling
from ly_utils import findpeak, Rotate

# some flags:
__show = True  # whether to show images
__upsample = False  # whether to upsample

# Params:
H = 755e3  # 卫星轨道高度
phi = 20 * np.pi / 180  # 俯仰角+20°
incidence = 20.5 * np.pi / 180  # 入射角
R_eta_c = H / np.cos(incidence)  # 景中心斜距
R0 = H / np.cos(phi)
theta = np.acos(R0 / R_eta_c)  # 斜视角, 单位为 弧度, 斜视角为 4.6°

# load data:
mat_path = './mat_files/Focused_Data_9_100.mat'
mat_file = loadmat(mat_path)['Focused_Data']

# show original data:
if __show:
    plt.figure('Focused_Data')
    plt.title('Focused Data')
    plt.ylabel('Azimuth')
    plt.xlabel('Range')
    plt.imshow(img2View(mat_file, enhance=True))

# Find peak:
peak_Azimuth, peak_Range = findpeak(mat_file)

# Get ROI: shape == 33 * 33
ROI = mat_file[peak_Azimuth - 16:peak_Azimuth + 16 + 1, peak_Range - 16:peak_Range + 16 + 1]

# show Original ROI:
if __show:
    plt.figure('ROI')
    plt.title('ROI')
    plt.ylabel('Azimuth')
    plt.xlabel('Range')
    plt.imshow(img2View(ROI, enhance=False))

# Upsampling:
if __upsample:
    ROI = UpSampling(ROI)

    # show Upsampled ROI:
    if __show:
        plt.figure('Upsampled ROI')
        plt.title('Upsampled ROI')
        plt.ylabel('Azimuth')
        plt.xlabel('Range')
        plt.imshow(img2View(ROI, enhance=False))

# Rotate:
ROI = Rotate(ROI, theta=theta)

# show Rotated ROI:
if __show:
    plt.figure('Rotated ROI')
    plt.title('Rotated ROI')
    plt.ylabel('Azimuth')
    plt.xlabel('Range')
    plt.imshow(img2View(ROI, enhance=False))

# Get Pattern over Azimuth:
peak_Azimuth, peak_Range = findpeak(ROI)  # update peak location
Pattern_Azimuth = ROI[:, peak_Range]  # get original pattern
Pattern_Azimuth = Pattern_Azimuth / Pattern_Azimuth.max()  # Normalization

# show Pattern over Azimuth:
if __show:
    plt.figure('Pattern_Azimuth')
    plt.title('Pattern_Azimuth')
    plt.xlabel('Azimuth')
    plt.ylabel('Rate')
    plt.plot(Pattern_Azimuth)

# show all:
plt.show()
