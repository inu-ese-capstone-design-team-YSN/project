from PIL import Image
import numpy as np

from path_finder import PathFinder

path_finder = PathFinder()

# Outer Area의 기준 RGB
standard_r = 214
standard_g = 211
standard_b = 217

# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #

"""
2024.04.29, jdk
start_num부터 end_num까지의 image에 대해서 standard RGB와의 차이를 구하고,
해당 차이 만큼 전체 이미지에 보정한 다음 새로운 이미지를 생성하는 코드

"""

original_image_dir = path_finder.tpg_original_directory_path
hue_corrected_image_dir = path_finder.tpg_HC_directory_path

# 몇 번 이미지부터 몇 번 이미지까지 처리할 것인지 정하는 반복 변수
start_num = 38
end_num = 60

left_area_weight = 0.333
right_area_weight = 0.666

# 지정된 영역에 대한 좌표
areas = [
    (0, 0, 210, 3040),      # 첫 번째 영역(Left)
    (3620, 0, 4056, 3040),  # 두 번째 영역(Right)
    (700, 260, 3200, 2760)  # 세 번재 영역(Center, Target Area)
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

extension = ".jpg"

for i in range(start_num, end_num+1):
    file_name = f"{i}"

    # 이미지를 불러오기
    image = Image.open(f'{original_image_dir}/{file_name}{extension}')

    # 지정된 영역의 픽셀 값을 numpy 배열로 추출
    extracted_areas = [np.array(image.crop(area)) for area in areas]

    left_area = extracted_areas[0] # Left Outer Area
    right_area = extracted_areas[1] # Right Outer Area
    target_area = extracted_areas[2] # Target Area

    # left_area의 R, G, B 채널별 평균 계산
    left_r_mean = np.mean(left_area[:, :, 0])
    left_g_mean = np.mean(left_area[:, :, 1])
    left_b_mean = np.mean(left_area[:, :, 2])

    # right_area의 R, G, B 채널별 평균 계산
    right_r_mean = np.mean(right_area[:, :, 0])
    right_g_mean = np.mean(right_area[:, :, 1])
    right_b_mean = np.mean(right_area[:, :, 2])

    # image의 outer area rgb 평균 계산
    total_r_mean  = round((left_r_mean*left_area_weight+right_r_mean*right_area_weight))
    total_g_mean  = round((left_g_mean*left_area_weight+right_g_mean*right_area_weight))
    total_b_mean  = round((left_b_mean*left_area_weight+right_b_mean*right_area_weight))

    # Standard와 Outer Area RGB의 차이 계산(diff)
    diff_r = total_r_mean - standard_r
    diff_g = total_g_mean - standard_g
    diff_b = total_b_mean - standard_b

    print(f"{i})")
    print(f"Total Value: {total_r_mean} {total_g_mean} {total_b_mean}") # Outer Area RGB
    print(f"Diff with Standard")
    print(f"R: {diff_r} G: {diff_g} B: {diff_b}") # Standard와의 차이

    # Target Area의 R, G, B 채널별 평균 계산
    target_area_r_mean = round(np.mean(target_area[:, :, 0]))
    target_area_g_mean = round(np.mean(target_area[:, :, 1]))
    target_area_b_mean = round(np.mean(target_area[:, :, 2]))

    cur_label_r = labels[i-start_num][0]
    cur_label_g = labels[i-start_num][1]
    cur_label_b = labels[i-start_num][2]

    print(f"Target Label: {cur_label_r} {cur_label_g} {cur_label_b}") # Target Swatch의 Label
    print(f"Target Area Value: {target_area_r_mean} {target_area_g_mean} {target_area_b_mean}") # Target Area의 Mean

    # Mean에서 diff를 뺀 corrected mean
    corrected_r = target_area_r_mean - diff_r
    corrected_g = target_area_g_mean - diff_g
    corrected_b = target_area_b_mean - diff_b

    print(f"Corrected Target Area Value: {corrected_r} {corrected_g} {corrected_b}") # corrected mean value

    # target area mean rgb와 label의 차이
    diff_with_label_r = target_area_r_mean - cur_label_r
    diff_with_label_g = target_area_g_mean - cur_label_g
    diff_with_label_b = target_area_b_mean - cur_label_b

    # 보정된 rgb와 label의 차이
    corrected_diff_with_label_r = corrected_r - cur_label_r
    corrected_diff_with_label_g = corrected_g - cur_label_g
    corrected_diff_with_label_b = corrected_b - cur_label_b

    print(f"Diff with Label: {diff_with_label_r} {diff_with_label_g} {diff_with_label_b}")
    print(f"Diff with Label after Correction: {corrected_diff_with_label_r} {corrected_diff_with_label_g} {corrected_diff_with_label_b}\n")

    # image를 np array로 변환
    np_image = np.array(image)

    # 빼려는 RGB 보정값 배열 생성, broadcasting
    correction_values = np.array([diff_r, diff_g, diff_b])
    np_image = np_image - correction_values
    np_image = np_image.astype(np.int32)
    np_image = np.clip(np_image, 0, 255)
    np_image = np_image.astype(np.uint8) # uint8 타입으로 최종 변환

    # 2024.04.29, jdk
    # 보정 후 numpy 배열의 원소들이 int64로 변경되어 Image로 생성 불가했음.
    # 이에 따라 data type을 uint8로 변경하는 코드를 추가하여 오류를 해결하였음.

    # np array를 다시 image로 변환
    correcte_image = Image.fromarray(np_image)
    correcte_image.save(f"{hue_corrected_image_dir}/{i}_hue_corrected_{extension}")

# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #