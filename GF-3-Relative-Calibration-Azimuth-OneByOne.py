import os.path

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from utils import img2View, rad2deg, mag2db
from np2mtlb import nextpow2, FFT_Range, FFT_Azimuth, FFTShift, apostrophe, pointwise_apostrophe, IFFT_Range, \
    IFFT_Azimuth
import cv2 as cv


def EnergyExtract(ROI, A_Y_min=145, A_Y_max=217 + 1, A_X_min=175, A_X_max=210 + 1, debug=False):
    CutResolution = 32

    # Upsampling:
    ROI_f = FFT_Azimuth(FFT_Range(ROI, shift=False), shift=False)  # Get Frequency Map
    Y_Insert = np.zeros([10 * CutResolution, 33])
    ROI_f = np.concatenate((ROI_f[0:17], Y_Insert, ROI_f[17:33]), axis=0)
    X_Insert = np.zeros([ROI_f.shape[0], 10 * CutResolution])
    ROI_f_Padded = np.concatenate((ROI_f[:, 0:22], X_Insert, ROI_f[:, 22:33]), axis=1)
    ROI_Upsampled = IFFT_Range(IFFT_Azimuth(ROI_f_Padded, shift=False),
                               shift=False)  # Back to Range-Azimuth Space Domain:

    # Find Max
    UP_Profile_Position_Ran, UP_Profile_Position_Azi = np.where(ROI_Upsampled == ROI_Upsampled.max())
    UP_Profile_Position_Ran, UP_Profile_Position_Azi = int(UP_Profile_Position_Ran[0]), int(UP_Profile_Position_Azi[0])

    # 姿态参数
    H = 755e3  # 卫星轨道高度
    phi = 20 * np.pi / 180  # 俯仰角+20°
    incidence = 20.5 * np.pi / 180  # 入射角
    R_eta_c = H / np.cos(incidence)  # 景中心斜距
    R0 = H / np.cos(phi)
    theta_r_c = np.acos(R0 / R_eta_c)  # 斜视角, 单位为 弧度, 斜视角为 4.6°

    # Rotate:
    Upsampled_ROI_Mod = np.abs(ROI_Upsampled)
    rotate_matrix = cv.getRotationMatrix2D(center=(UP_Profile_Position_Ran, UP_Profile_Position_Azi),
                                           angle=rad2deg(theta_r_c), scale=1)
    Rotated_ROI_Modulus = cv.warpAffine(Upsampled_ROI_Mod, rotate_matrix, Upsampled_ROI_Mod.shape[1::-1])

    # Find Max
    UP_Profile_Position_Ran, UP_Profile_Position_Azi = np.where(Rotated_ROI_Modulus == Rotated_ROI_Modulus.max())
    UP_Profile_Position_Ran, UP_Profile_Position_Azi = UP_Profile_Position_Ran[0], UP_Profile_Position_Azi[0]

    # Get shape of ROI:
    ROI_Modulus = Rotated_ROI_Modulus.copy()
    Na, Nr = int(ROI_Modulus.shape[0]), int(ROI_Modulus.shape[1])

    ########### integration B ###########
    SideLength_B = 100
    NumB = 4 * (SideLength_B ** 2)


    # [a, b) is used in numpy index:
    ROI_B1 = ROI_Modulus[: SideLength_B, : SideLength_B]
    ROI_B2 = ROI_Modulus[: SideLength_B, Nr - SideLength_B:]
    ROI_B3 = ROI_Modulus[Na - SideLength_B:, : SideLength_B]
    ROI_B4 = ROI_Modulus[Na - SideLength_B:, Nr - SideLength_B:]

    # DN_B ** 2:
    IntegrationB = (ROI_B1 ** 2).sum() + (ROI_B2 ** 2).sum() + (ROI_B3 ** 2).sum() + (ROI_B4 ** 2).sum()

    ########### integration A ###########
    A_Mask = np.zeros_like(ROI_Modulus, dtype=int)  # [a, b)
    A_Mask[:, A_X_min: A_X_max] = 1
    A_Mask[A_Y_min: A_Y_max, :] = 1
    NumA = A_Mask.sum()

    # Segmentation:
    ROI_A = ROI_Modulus * A_Mask

    # DN_A ** 2:
    IntegrationA = np.sum((ROI_A ** 2))

    # 卫星轨道速度Vr计算
    EarthMass = 6e24  # 地球质量(kg)
    EarthRadius = 6.37e6  # 地球半径6371km
    Gravitational = 6.67e-11  # 万有引力常量

    # 计算等效雷达速度(卫星做圆周运动的线速度)
    Vr = np.sqrt(Gravitational * EarthMass / (EarthRadius + H))  # 第一宇宙速度

    ##信号参数设置
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

    delta_Azimuth = Vs / Fa  # Azimuth
    delta_Range = Vr / Fr  # Range (斜距向)

    # Extract Energy:
    Energy = IntegrationA - (NumA * IntegrationB / NumB)

    if debug:
        print('delta_Azimuth = {}'.format(delta_Azimuth))
        print('delta_Range = {}'.format(delta_Range))
        print('Energy = {}'.format(Energy))

    #################### Visualize the Energy Extraction ####################
    if debug:
        ROI_BBox_show = Rotated_ROI_Modulus.copy()  # Copy
        ROI_BBox_show = ROI_BBox_show / ROI_BBox_show.max()  # squeeze val domain
        ROI_BBox_show = ROI_BBox_show * 255  # put val domain to be 0~255
        ROI_BBox_show = np.uint8(ROI_BBox_show)
        ROI_BBox_show = cv.cvtColor(ROI_BBox_show, cv.COLOR_GRAY2BGR)  # 1-Channel to 3-Channels

        # Drew B BBoxs: (Range, Azimuth) is used in OpenCV
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, 0), (SideLength_B - 1, SideLength_B - 1), color=(0, 0, 255),
                                     thickness=1)
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (Nr - SideLength_B, 0), (Nr - 1, SideLength_B - 1),
                                     color=(0, 0, 255),
                                     thickness=1)
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, Na - SideLength_B), (SideLength_B - 1, Na - 1),
                                     color=(0, 0, 255),
                                     thickness=1)
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (Nr - SideLength_B, Na - SideLength_B), (Nr, Na - 1),
                                     color=(0, 0, 255),
                                     thickness=1)

        # Drew A BBoxs: (Range, Azimuth) is used in OpenCV
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (A_X_min, 0), (A_X_max - 1, Na - 1), color=(255, 0, 0),
                                     thickness=1)  # Azimuth longer
        ROI_BBox_show = cv.rectangle(ROI_BBox_show, (0, A_Y_min), (Nr - 1, A_Y_max - 1), color=(255, 0, 0),
                                     thickness=1)  # Range longer

        # Show Mask_A
        plt.figure('Mask A')
        plt.imshow(A_Mask, cmap='gray')
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.title('"Mask A"')

        # Show Extraction:
        plt.figure('Energy Extraction')
        plt.imshow(ROI_BBox_show)
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.title('"Energy Extraction"')

    return Energy

positions_y = np.array([220, 292, 366, 438, 510, 583, 655, 729, 801, 873, 945, 1017, 1090, 1162, 1235, 1307, 1379]) - 1
positions_x = 1250 - 1

# load data:
Focused_Data = []

# get Focused Data:
for idx in range(len(positions_y)):
    mat_path = 'Focused_Data_Azimuth_{}.mat'.format(idx)
    if os.path.exists(mat_path):
        Focused_Data.append(loadmat(mat_path)['Focused_Data'])
    else:
        print('Focused data {} not found.'.format(idx))

# list to array:
Focused_Data = np.array(Focused_Data)

# init ROIs:
ROIs = []

for idx in range(len(Focused_Data)):
    y_min = int(positions_y[idx]) - 16
    y_max = int(positions_y[idx]) + 16 + 1
    x_min = positions_x - 16
    x_max = positions_x + 16 + 1

    ROI = Focused_Data[idx][y_min:y_max, x_min:x_max]
    ROIs.append(ROI)

# list 2 np:
ROIs = np.array(ROIs)

# show all ROIs:
# for idx in range(len(ROIs)):
#     ROI = ROIs[idx]
#     plt.figure()
#     plt.title('ROI[{}]'.format(idx))
#     plt.imshow(img2View(ROI, enhance=False))

# Get Energy:
Energy0 = EnergyExtract(ROIs[0], A_X_min=169, A_X_max=195, A_Y_min=135, A_Y_max=210, debug=False)
Energy1 = EnergyExtract(ROIs[1], A_X_min=169, A_X_max=194, A_Y_min=141, A_Y_max=215, debug=False)
Energy2 = EnergyExtract(ROIs[2], A_X_min=170, A_X_max=190, A_Y_min=129, A_Y_max=205, debug=False)
Energy3 = EnergyExtract(ROIs[3], A_X_min=170, A_X_max=192, A_Y_min=134, A_Y_max=207, debug=False)
Energy4 = EnergyExtract(ROIs[4], A_X_min=170, A_X_max=192, A_Y_min=135, A_Y_max=212, debug=False)
Energy5 = EnergyExtract(ROIs[5], A_X_min=170, A_X_max=190, A_Y_min=132, A_Y_max=204, debug=False)
Energy6 = EnergyExtract(ROIs[6], A_X_min=170, A_X_max=190, A_Y_min=134, A_Y_max=211, debug=False)
Energy7 = EnergyExtract(ROIs[7], A_X_min=170, A_X_max=190, A_Y_min=120, A_Y_max=190, debug=False)
Energy8 = EnergyExtract(ROIs[8], A_X_min=170, A_X_max=190, A_Y_min=120, A_Y_max=193, debug=False)
Energy9 = EnergyExtract(ROIs[9], A_X_min=170, A_X_max=190, A_Y_min=128, A_Y_max=199, debug=False)
Energy10 = EnergyExtract(ROIs[10], A_X_min=170, A_X_max=190, A_Y_min=130, A_Y_max=205, debug=False)
Energy11 = EnergyExtract(ROIs[11], A_X_min=170, A_X_max=190, A_Y_min=136, A_Y_max=208, debug=False)
Energy12 = EnergyExtract(ROIs[12], A_X_min=170, A_X_max=190, A_Y_min=128, A_Y_max=202, debug=False)
Energy13 = EnergyExtract(ROIs[13], A_X_min=170, A_X_max=190, A_Y_min=134, A_Y_max=207, debug=False)
Energy14 = EnergyExtract(ROIs[14], A_X_min=170, A_X_max=190, A_Y_min=128, A_Y_max=202, debug=False)
Energy15 = EnergyExtract(ROIs[15], A_X_min=170, A_X_max=195, A_Y_min=132, A_Y_max=206, debug=False)
Energy16 = EnergyExtract(ROIs[16], A_X_min=170, A_X_max=195, A_Y_min=142, A_Y_max=215, debug=False)

Energy = np.array([Energy0, Energy1, Energy2, Energy3, Energy4, Energy5, Energy6, Energy7, Energy8, Energy9, Energy10, Energy11, Energy12, Energy13, Energy14, Energy15, Energy16])

print('Energy0', Energy0)
print('Energy1', Energy1)
print('Energy2', Energy2)
print('Energy3', Energy3)
print('Energy4', Energy4)
print('Energy5', Energy5)
print('Energy6', Energy6)
print('Energy7', Energy7)
print('Energy8', Energy8)
print('Energy9', Energy9)
print('Energy10', Energy10)
print('Energy11', Energy11)
print('Energy12', Energy12)
print('Energy13', Energy13)
print('Energy14', Energy14)
print('Energy15', Energy15)
print('Energy16', Energy16)

plt.figure('Patten')
plt.title('Patten')
plt.plot(positions_y, Energy / Energy.max())

my_x_ticks = positions_y  # np.arange(1, 17, 1)
plt.xticks(my_x_ticks)

# show all targets:
all_targets = loadmat('Focused_Data_Azimuth.mat')['Focused_Data']
print(all_targets.shape)

plt.figure('All Targets')
plt.title('All Targets')
plt.imshow(img2View(all_targets, enhance=True))
plt.xlabel('Range')
plt.ylabel('Azimuth')

plt.figure('Central Target')
plt.title('Central Target')
plt.imshow(img2View(ROIs[8], enhance=False))
plt.xlabel('Range')
plt.ylabel('Azimuth')

########################################################################################
ROI = ROIs[8]
CutResolution = 32

# Upsampling:
ROI_f = FFT_Azimuth(FFT_Range(ROI, shift=False), shift=False)  # Get Frequency Map
Y_Insert = np.zeros([10 * CutResolution, 33])
ROI_f = np.concatenate((ROI_f[0:17], Y_Insert, ROI_f[17:33]), axis=0)
X_Insert = np.zeros([ROI_f.shape[0], 10 * CutResolution])
ROI_f_Padded = np.concatenate((ROI_f[:, 0:22], X_Insert, ROI_f[:, 22:33]), axis=1)
ROI_Upsampled = IFFT_Range(IFFT_Azimuth(ROI_f_Padded, shift=False),
                           shift=False)  # Back to Range-Azimuth Space Domain:

# Find Max
UP_Profile_Position_Ran, UP_Profile_Position_Azi = np.where(ROI_Upsampled == ROI_Upsampled.max())
UP_Profile_Position_Ran, UP_Profile_Position_Azi = int(UP_Profile_Position_Ran[0]), int(UP_Profile_Position_Azi[0])

# 姿态参数
H = 755e3  # 卫星轨道高度
phi = 20 * np.pi / 180  # 俯仰角+20°
incidence = 20.5 * np.pi / 180  # 入射角
R_eta_c = H / np.cos(incidence)  # 景中心斜距
R0 = H / np.cos(phi)
theta_r_c = np.acos(R0 / R_eta_c)  # 斜视角, 单位为 弧度, 斜视角为 4.6°

# Rotate:
Upsampled_ROI_Mod = np.abs(ROI_Upsampled)
rotate_matrix = cv.getRotationMatrix2D(center=(UP_Profile_Position_Ran, UP_Profile_Position_Azi),
                                       angle=rad2deg(theta_r_c), scale=1)
Rotated_ROI_Modulus = cv.warpAffine(Upsampled_ROI_Mod, rotate_matrix, Upsampled_ROI_Mod.shape[1::-1])

# Find Max
UP_Profile_Position_Ran, UP_Profile_Position_Azi = np.where(Rotated_ROI_Modulus == Rotated_ROI_Modulus.max())
UP_Profile_Position_Ran, UP_Profile_Position_Azi = UP_Profile_Position_Ran[0], UP_Profile_Position_Azi[0]

# Get shape of ROI:
ROI_Modulus = Rotated_ROI_Modulus.copy()

plt.figure('Central Target Upsampled')
plt.title('Central Target Upsampled')
plt.imshow(img2View(ROI_Modulus, enhance=False))
plt.xlabel('Range')
plt.ylabel('Azimuth')
########################################################################################

plt.figure('Azimuth Patten')
plt.title('Azimuth Patten')
Azimuth_line = ROI_Modulus[:, UP_Profile_Position_Azi]
plt.plot(Azimuth_line / Azimuth_line.max())

# show all:
plt.show()
