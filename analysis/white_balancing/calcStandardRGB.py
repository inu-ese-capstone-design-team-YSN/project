from PIL import Image
import numpy as np

from path_finder import PathFinder

path_finder = PathFinder()

# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #

"""
2024.04.29, jdk
한 개의 이미지에 대해서 Outer Area 값의 평균을 구하고, Standard와의 차이를 구하는 코드
"""

original_image_dir = path_finder.tpg_original_directory_path

file_name = "11"
extension = ".jpg"

# 이미지를 불러오기
image = Image.open(f'{original_image_dir}/{file_name}{extension}')

# 지정된 영역에 대한 좌표
areas = [
    (0, 0, 210, 3040),      # 첫 번째 영역(Left)
    (3620, 0, 4056, 3040)   # 두 번째 영역(Right)
]

# 지정된 영역의 픽셀 값을 numpy 배열로 추출
extracted_areas = [np.array(image.crop(area)) for area in areas]

left_area = extracted_areas[0]
right_area = extracted_areas[1]

# left_area의 R, G, B 채널별 평균 계산
left_r_mean = np.mean(left_area[:, :, 0])
left_g_mean = np.mean(left_area[:, :, 1])
left_b_mean = np.mean(left_area[:, :, 2])

# right_area의 R, G, B 채널별 평균 계산
right_r_mean = np.mean(right_area[:, :, 0])
right_g_mean = np.mean(right_area[:, :, 1])
right_b_mean = np.mean(right_area[:, :, 2])

# 
standard_r = round((left_r_mean + right_r_mean)/2)
standard_g = round((left_g_mean + right_g_mean)/2)
standard_b = round((left_b_mean + right_b_mean)/2)

print(f"Standard Value: {standard_r} {standard_g} {standard_b}")

# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #