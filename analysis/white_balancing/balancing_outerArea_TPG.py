from PIL import Image
import numpy as np
import os
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
standard_r = 228
standard_g = 224
standard_b = 230
results = {}

original_image_dir = path_finder.tcx_original_directory_path
hue_corrected_image_dir = path_finder.tcx_HC_directory_path

# 몇 번 이미지부터 몇 번 이미지까지 처리할 것인지 정하는 반복 변수
extension = ".png"

areas = [
    (0, 0, 100, 3040),       # 첫 번째 영역(Left)
    (3956, 0, 4056, 3040),  # 두 번째 영역(Right)
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
    total_r_mean = round((left_r_mean + right_r_mean) / 2)
    total_g_mean = round((left_g_mean + right_g_mean) / 2)
    total_b_mean = round((left_b_mean + right_b_mean) / 2)
    
    return total_r_mean, total_g_mean, total_b_mean


def main():
    # 디렉토리 내의 모든 PNG 파일을 가져오기
    for file_name in os.listdir(original_image_dir):
        if file_name.endswith(extension):
            file_path = os.path.join(original_image_dir, file_name)
            
            # 이미지를 불러오기
            original_image = Image.open(file_path)
            total_r_mean, total_g_mean, total_b_mean = calculate_rgb_means(original_image, areas)

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
            corrected_file_name = f"{os.path.splitext(file_name)[0]}{extension}"
            WB_image.save(os.path.join(hue_corrected_image_dir, corrected_file_name))

            hue_corrected_image = Image.open(os.path.join(hue_corrected_image_dir, corrected_file_name))
            corrected_total_r_mean, corrected_total_g_mean, corrected_total_b_mean = calculate_rgb_means(hue_corrected_image, areas)

            print(f"{file_name} HB 보정 완료")

if __name__ == "__main__":
    main()
