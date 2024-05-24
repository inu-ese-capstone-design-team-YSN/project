from PIL import Image
import numpy as np
import json
import cv2
from skimage import color
import math  # math 모듈 가져오기
from path_finder import PathFinder

path_finder = PathFinder()

'''
2024.05.14 kwc
RGB 기반 색온도 보정 알고리즘 개발
'''
# Outer Area의 기준 RGB
standard_r = 214
standard_g = 211
standard_b = 217
results = {}

original_image_dir = path_finder.swatch_original_directory_path
hue_corrected_image_dir = path_finder.swatch_HC_directory_path

# 몇 번 이미지부터 몇 번 이미지까지 처리할 것인지 정하는 반복 변수
start_num = 38
end_num = 60
extension = ".jpg"

# 지정된 영역에 대한 좌표
# areas = [
#     (0, 0, 100, 3040),      # 첫 번째 영역(Left)
#     (3956, 0, 4056, 3040),  # 두 번째 영역(Right)
#     (700, 260, 3200, 2760)  # 세 번재 영역(Center, Target Area)
# ]

areas = [
    (0, 0, 350,3040),      # 첫 번째 영역(Left)
    (3680, 0, 4056, 3040),  # 두 번째 영역(Right)
    (725, 220, 3325, 2830)  # 세 번재 영역(Center, Target Area)
]

# 2024.04.29, jdk
# 현재 촬영한 Post Card의 Label
labels = [
    (168, 62, 108),
    (121, 67, 132),
    (255, 178, 165),
    (64, 68, 102),
    (55, 65, 58),
    (67, 125, 109),
    (59, 114, 95),
    (243, 193, 44),
    (0, 129, 157),
    (219, 203, 190),
    (181, 182, 68),
    (221, 65, 50),
    (124, 41, 70),
    (43, 48, 66),
    (226, 88, 62),
    (207, 200, 189),
    (0, 99, 128),
    (185, 113, 79),
    (207, 223, 219),
    (141, 63, 45),
    (0, 126, 177),
    (148, 78, 135),
    (195, 124, 84)
]

def calculate_rgb_means(image, areas):
    """이미지와 영역 목록을 받아 각 영역의 RGB 평균값을 계산합니다."""
    # 지정된 영역의 픽셀 값을 numpy 배열로 추출
    extracted_areas = [np.array(image.crop(area)) for area in areas]
    left_area = extracted_areas[0] # Left Outer Area
    right_area = extracted_areas[1] # Right Outer Area

    # left_area의 R, G, B 채널별 평균 계산
    left_r_mean = np.mean(left_area[:, :, 0])
    left_g_mean = np.mean(left_area[:, :, 1])
    left_b_mean = np.mean(left_area[:, :, 2])

    # right_area의 R, G, B 채널별 평균 계산
    right_r_mean = np.mean(right_area[:, :, 0])
    right_g_mean = np.mean(right_area[:, :, 1])
    right_b_mean = np.mean(right_area[:, :, 2])

    # image의 outer area rgb 평균 계산
    total_r_mean  = round((left_r_mean + right_r_mean) / 2)
    total_g_mean  = round((left_g_mean + right_g_mean) / 2)
    total_b_mean  = round((left_b_mean + right_b_mean) / 2)
    
    return total_r_mean, total_g_mean, total_b_mean

def rgb_to_lab(rgb):
    """RGB를 LAB로 변환"""
    rgb = np.array(rgb).reshape(1, 1, 3) / 255.0
    lab = color.rgb2lab(rgb)
    return lab[0, 0]

def lab_to_rgb(lab):
    """LAB를 RGB로 변환"""
    lab = np.array(lab).reshape(1, 1, 3)
    rgb = color.lab2rgb(lab) * 255
    return rgb[0, 0]

def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    avg_L = (L1 + L2) / 2.0
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    avg_C = (C1 + C2) / 2.0
    
    G = 0.5 * (1 - math.sqrt(avg_C ** 7 / (avg_C ** 7 + 25 ** 7)))
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    
    C1_prime = math.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = math.sqrt(a2_prime ** 2 + b2 ** 2)
    avg_C_prime = (C1_prime + C2_prime) / 2.0
    
    h1_prime = math.atan2(b1, a1_prime) % (2 * math.pi)
    h2_prime = math.atan2(b2, a2_prime) % (2 * math.pi)
    
    avg_H_prime = (h1_prime + h2_prime) / 2.0
    if abs(h1_prime - h2_prime) > math.pi:
        avg_H_prime += math.pi
    
    T = 1 - 0.17 * math.cos(avg_H_prime - math.radians(30)) + \
        0.24 * math.cos(2 * avg_H_prime) + \
        0.32 * math.cos(3 * avg_H_prime + math.radians(6)) - \
        0.20 * math.cos(4 * avg_H_prime - math.radians(63))
    
    delta_H_prime = h2_prime - h1_prime
    if abs(delta_H_prime) > math.pi:
        if h2_prime <= h1_prime:
            delta_H_prime += 2 * math.pi
        else:
            delta_H_prime -= 2 * math.pi
    delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(delta_H_prime / 2.0)
    
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    SL = 1 + 0.015 * (avg_L - 50) ** 2 / math.sqrt(20 + (avg_L - 50) ** 2)
    SC = 1 + 0.045 * avg_C_prime
    SH = 1 + 0.015 * avg_C_prime * T
    
    delta_theta = math.radians(30) * math.exp(-((avg_H_prime - math.radians(275)) / math.radians(25)) ** 2)
    RC = 2 * math.sqrt(avg_C_prime ** 7 / (avg_C_prime ** 7 + 25 ** 7))
    RT = -math.sin(2 * delta_theta) * RC
    
    delta_E = math.sqrt(
        (delta_L_prime / SL) ** 2 +
        (delta_C_prime / SC) ** 2 +
        (delta_H_prime / SH) ** 2 +
        RT * (delta_C_prime / SC) * (delta_H_prime / SH)
    )
    
    return delta_E

for i in range(start_num, end_num + 1):
    file_name = f"{i}"
    
    # 이미지를 불러오기
    original_image = Image.open(f'{original_image_dir}/{file_name}{extension}')
    total_r_mean, total_g_mean, total_b_mean = calculate_rgb_means(original_image, areas)
    
    extracted_areas = [np.array(original_image.crop(area)) for area in areas]
    target_area = extracted_areas[2] # Target Area

    # 보정 전 Standard와 Outer Area RGB의 차이 계산(diff)
    diff_r = total_r_mean - standard_r
    diff_g = total_g_mean - standard_g
    diff_b = total_b_mean - standard_b

    # RGB to LAB 변환
    total_lab_mean = rgb_to_lab((total_r_mean, total_g_mean, total_b_mean))
    standard_lab = rgb_to_lab((standard_r, standard_g, standard_b))
    
    # LAB 차이 계산 (CIEDE2000)
    delta_e = ciede2000(total_lab_mean, standard_lab)

    print(f"image {i})")
    print(f"보정 전 Outer Area Value: {total_r_mean} {total_g_mean} {total_b_mean}") # Outer Area RGB
    print(f"Delta E (CIEDE2000) with Standard LAB: {delta_e}") # Standard와의 CIEDE2000 차이
    print(f"Diff with Standard {diff_r} {diff_g} {diff_b}") # Standard와의 차이
    
    
    # Target Area의 R, G, B 채널별 평균 계산
    target_area_r_mean = round(np.mean(target_area[:, :, 0]))
    target_area_g_mean = round(np.mean(target_area[:, :, 1]))
    target_area_b_mean = round(np.mean(target_area[:, :, 2]))
    
    cur_label_r = labels[i-start_num][0]
    cur_label_g = labels[i-start_num][1]
    cur_label_b = labels[i-start_num][2]
    
    print(f"Target Label: {cur_label_r} {cur_label_g} {cur_label_b}") # Target Swatch의 Label
    print(f"Target Area Value: {target_area_r_mean} {target_area_g_mean} {target_area_b_mean}") # Target Area의 Mean
    print(f"Diff: {cur_label_r - target_area_r_mean} {cur_label_g - target_area_g_mean} {cur_label_b - target_area_b_mean}")
    
    # image를 np array로 변환
    np_image = np.array(original_image, dtype=np.float32)

    # 보정값 생성
    R_correction_factor = (standard_r / total_r_mean)
    G_correction_factor = (standard_g / total_g_mean)
    B_correction_factor = (standard_b / total_b_mean)
    # 빼려는 RGB 보정값 배열 생성, broadcasting
    correction_values = np.array([R_correction_factor, G_correction_factor, B_correction_factor])
    np_image = np_image * correction_values
    np_image = np.clip(np_image, 0, 255)
    np_image = np_image.astype(np.uint8)  # uint8 타입으로 최종 변환
    
    # np array를 다시 image로 변환
    WB_image = Image.fromarray(np_image)
    WB_image.save(f"{hue_corrected_image_dir}/{i}_hue_corrected_{extension}")

    hue_corrected_image = Image.open(f'{hue_corrected_image_dir}/{file_name}_hue_corrected_{extension}')
    corrected_total_r_mean, corrected_total_g_mean, corrected_total_b_mean = calculate_rgb_means(hue_corrected_image, areas)

    diff_correction_r = corrected_total_r_mean - standard_r
    diff_correction_g = corrected_total_g_mean - standard_g
    diff_correction_b = corrected_total_b_mean - standard_b
    
    # RGB to LAB 변환
    corrected_total_lab_mean = rgb_to_lab((corrected_total_r_mean, corrected_total_g_mean, corrected_total_b_mean))
    
    # LAB 차이 계산 (CIEDE2000)
    delta_e2 = ciede2000(corrected_total_lab_mean, standard_lab)
    
    print("\n")
    print(f"보정 후 Outer Area Value: {corrected_total_r_mean} {corrected_total_g_mean} {corrected_total_b_mean}") # Outer Area RGB
    print(f"Delta E (CIEDE2000) with Standard LAB: {delta_e2}") # Standard와의 CIEDE2000 차이
    print(f"Diff with Standard {diff_correction_r} {diff_correction_g} {diff_correction_b}") # Standard와의 차이
    
    extracted_areas = [np.array(hue_corrected_image.crop(area)) for area in areas]
    target_area = extracted_areas[2] # Target Area

    # Target Area의 R, G, B 채널별 평균 계산
    corrected_target_area_r_mean = round(np.mean(target_area[:, :, 0]))
    corrected_target_area_g_mean = round(np.mean(target_area[:, :, 1]))
    corrected_target_area_b_mean = round(np.mean(target_area[:, :, 2]))
    
    cur_label_r = labels[i-start_num][0]
    cur_label_g = labels[i-start_num][1]
    cur_label_b = labels[i-start_num][2]
    
    print(f"Target Label: {cur_label_r} {cur_label_g} {cur_label_b}") # Target Swatch의 Label
    print(f"Corrected Target Area Value: {corrected_target_area_r_mean} {corrected_target_area_g_mean} {corrected_target_area_b_mean}") # Target Area의 Mean
    print(f"Diff: {cur_label_r - corrected_target_area_r_mean} {cur_label_g - corrected_target_area_g_mean} {cur_label_b - corrected_target_area_b_mean}")
    print("\n")

    results[i] = {
        'Target Label': [cur_label_r, cur_label_g, cur_label_b],
        'Target Area Value': [target_area_r_mean, target_area_g_mean, target_area_b_mean],
        'Corrected Target Area Value': [corrected_target_area_r_mean, corrected_target_area_g_mean, corrected_target_area_b_mean]
    }
    
with open('image_processing_results.json', 'w') as f:
    json.dump(results, f)
