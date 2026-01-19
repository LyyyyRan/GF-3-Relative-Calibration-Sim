# -*- coding: utf-8 -*-
"""
Created on January 15, 2026

@author: https://github.com/LyyyyRan
"""

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from ly_utils import findpeak

# load Echo and Focused Image:
RCS = 200
Echo_IMG = np.abs(loadmat('S_echo-NoNoise.mat')['S_echo'])
# mat_path = './mat_files/Focused_Data_9_{}'.format(RCS)
mat_path = './Focused_Data_9_{}-NoNoise'.format(RCS)
Focused_IMG = loadmat(mat_path + '.mat')['Focused_Data']

# find peak value and location:
peak_y, peak_x = findpeak(Focused_IMG)
DN_Max = np.max(Focused_IMG)

# 姿态参数
H = 755e3  # 卫星轨道高度
phi = 20 * np.pi / 180  # 俯仰角+20°
incidence = 20.5 * np.pi / 180  # 入射角
R_eta_c = H / np.cos(incidence)  # 景中心斜距
R0 = H / np.cos(phi)
theta_r_c = np.acos(R0 / R_eta_c)  # 斜视角, 单位为 弧度, 斜视角为 4.6°

# 卫星轨道速度Vr计算
EarthMass = 6e24  # 地球质量(kg)
EarthRadius = 6.37e6  # 地球半径6371km
Gravitational = 6.67e-11  # 万有引力常量

# 计算等效雷达速度(卫星做圆周运动的线速度)
Vr = np.sqrt(Gravitational * EarthMass / (EarthRadius + H))  # 第一宇宙速度

## 信号参数设置
# 电磁波参数
c = 3e+8  # 光速
Vs = Vr  # 卫星平台速度
Vg = Vr  # 波束扫描速度
La = 15  # 方位向天线长度->椭圆的长轴
Lr = 1.5  # 距离向天线尺寸—— > 椭圆的短轴
f0 = 5.4e+9  # 雷达工作频率
lamda = c / f0  # 电磁波波长

# 距离向信号参数
Tr = 40e-6  # 发射脉冲时宽
Br = 2.8 * 6e6  # 距离向信号带宽
Kr = Br / Tr  # 距离向调频率
alpha_os_r = 1.2  # 距离过采样率
Nrg = 2500  # 距离线采样点数
Fr = alpha_os_r * Br  # 距离向采样率

# 方位向信号参数
alpha_os_a = 1.23  # 方位过采样率(高过采样率避免鬼影目标)
Naz = 1600  # 距离线数
delta_f_dop = 2 * 0.886 * Vr * (np.cos(theta_r_c)) / La  # 多普勒带宽
Fa = alpha_os_a * delta_f_dop  # 方位向采样率

# Resolution over Azimuth and Range:
delta_Azimuth = Vs / Fa  # Azimuth
delta_Range = Vr / Fr  # Range (斜距向)

# Energy Extraction:
Energy = delta_Range * delta_Azimuth * np.abs(DN_Max) ** 2

plt.figure('Echo Image')
plt.title('Echo Image')
plt.imshow(Echo_IMG)

plt.figure('Focused_IMG')
plt.title('Focused_IMG')
plt.imshow(Focused_IMG)

plt.figure('Echo-Azimuth')
plt.title('Echo-Azimuth')
plt.ylabel('Modulus')
plt.xlabel('Azimuth')
plt.plot(Echo_IMG[:, 1249])

plt.figure('Focused_Data-Azimuth')
plt.title('Focused_Data-Azimuth')
plt.ylabel('Modulus')
plt.xlabel('Azimuth')
plt.plot(Focused_IMG[:, 1249])

plt.figure('Echo-Range')
plt.title('Echo-Range')
plt.ylabel('Modulus')
plt.xlabel('Range')
plt.plot(Echo_IMG[799, :])

plt.figure('Focused_Data-Range')
plt.title('Focused_Data-Range')
plt.ylabel('Modulus')
plt.xlabel('Range')
plt.plot(Focused_IMG[799, :])

plt.show()

