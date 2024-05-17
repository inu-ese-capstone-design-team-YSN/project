from PIL import Image
import numpy as np

from path_finder import PathFinder

path_finder = PathFinder()

# Outer Area의 기준 RGB
standard_r = 235
standard_g = 233
standard_b = 238

# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #

"""
2024.04.29, jdk
여러 개의 이미지에 대해서 Outer Area 값의 평균에 대한 평균을 구하는 코드
"""

# 최종 rgb를 계산하기 위한 변수
total_left_r = 0
total_left_g = 0
total_left_b = 0

total_right_r = 0
total_right_g = 0
total_right_b = 0

original_image_dir = path_finder.tpg_original_dir_path

# 지정된 영역에 대한 좌표
areas = [
    (0, 0, 210, 3040),      # 첫 번째 영역(Left)
    (3620, 0, 4056, 3040)   # 두 번째 영역(Right)
]

# 몇 번 이미지부터 몇 번 이미지까지 처리할 것인지 정하는 반복 변수
start_num = 61
end_num = 70
image_cnt = end_num-start_num+1

extension = ".jpg"

for i in range(start_num, end_num+1):
    file_name = f"{i}"

    # 이미지를 불러오기
    image = Image.open(f'{original_image_dir}/{file_name}{extension}')

    # 지정된 영역의 픽셀 값을 numpy 배열로 추출
    extracted_areas = [np.array(image.crop(area)) for area in areas]

    left_area = extracted_areas[0]
    right_area = extracted_areas[1]

    # left_area의 R, G, B 채널별 평균 계산
    left_r_mean = np.mean(left_area[:, :, 0])
    left_g_mean = np.mean(left_area[:, :, 1])
    left_b_mean = np.mean(left_area[:, :, 2])

    total_left_r += left_r_mean
    total_left_g += left_g_mean
    total_left_b += left_b_mean

    # right_area의 R, G, B 채널별 평균 계산
    right_r_mean = np.mean(right_area[:, :, 0])
    right_g_mean = np.mean(right_area[:, :, 1])
    right_b_mean = np.mean(right_area[:, :, 2])

    total_right_r += right_r_mean
    total_right_g += right_g_mean
    total_right_b += right_b_mean

# 최종적으로 Standard rgb를 저장하는 변수
total_r = (total_left_r + total_right_r)/2
total_g = (total_left_g + total_right_g)/2
total_b = (total_left_b + total_right_b)/2

print(f"Total Value: {round(total_r/image_cnt)} {round(total_g/image_cnt)} {round(total_b/image_cnt)}")

# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #