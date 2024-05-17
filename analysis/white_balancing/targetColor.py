from PIL import Image
import numpy as np
from path_finder import PathFinder

path_finder = PathFinder()

tpg_combined_dir = path_finder.tpg_combined_dir_path

file_name = "18-6018"
extension = ".jpg"
# 이미지 불러오기
combined_image = Image.open(f'{tpg_combined_dir}/{file_name}_combined{extension}')
image_array = np.array(combined_image)

# 영역 정의 (height, width)
regions = [
    (0, 610, 0, 1250),        # 0~610, 0~1250
    (610, 1250, 0, 1250),     # 610~1250, 0~1250
    (1250, 1890, 0, 1250),    # 1250~1890, 0~1250
    (1890, 2500, 0, 1250),    # 1890~2500, 0~1250
    (0, 610, 1250, 2500),     # 0~610, 1250~2500
    (640, 1250, 1250, 2500),  # 640~1250, 1250~2500
    (1250, 1890, 1250, 2500), # 1250~1890, 1250~2500
    (1890, 2500, 1250, 2500)  # 1890~2500, 1250~2500
]

def calculate_rgb_mean(image_array, region):
    """
    주어진 영역의 RGB 평균을 계산하는 함수
    """
    region_array = image_array[region[0]:region[1], region[2]:region[3]]
    r_mean = np.mean(region_array[:, :, 0])
    g_mean = np.mean(region_array[:, :, 1])
    b_mean = np.mean(region_array[:, :, 2])
    return r_mean, g_mean, b_mean

# 각 영역의 RGB 평균 계산
rgb_means = []
for region in regions:
    r_mean, g_mean, b_mean = calculate_rgb_mean(image_array, region)
    # 소수점 첫째자리에서 반올림하여 정수로 변환
    r_mean = round(r_mean)
    g_mean = round(g_mean)
    b_mean = round(b_mean)
    rgb_means.append((r_mean, g_mean, b_mean))

# 결과 출력
for i, (r_mean, g_mean, b_mean) in enumerate(rgb_means):
    print(f"Region {i + 1}: R mean = {r_mean}, G mean = {g_mean}, B mean = {b_mean}")
