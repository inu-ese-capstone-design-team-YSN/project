from PIL import Image
import numpy as np
import cv2
import numpy

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

original_image_dir = path_finder.tpg_original_dir_path
hue_corrected_image_dir = path_finder.tpg_hue_corrected_dir_path

# 몇 번 이미지부터 몇 번 이미지까지 처리할 것인지 정하는 반복 변수
start_num = 38
end_num = 60
extension = ".jpg"

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

def white_balance_using_white_patch(img, white_patch):
    # 하얀 영역의 평균 BGR 값을 계산합니다.
    average_white = np.mean(white_patch, axis=(0, 1))

    # 이미지의 각 픽셀을 흰색 영역의 평균으로 나누어 정규화합니다.
    scaling_factors = 255 / average_white
    img_balanced = img * scaling_factors[np.newaxis, np.newaxis, :]

    # 값이 255를 넘지 않도록 조정합니다.
    img_balanced = np.clip(img_balanced, 0, 255).astype(np.uint8)
    return img_balanced

for i in range(start_num, end_num+1):
    file_name = f"{i}"

    # 이미지를 불러오기
    image = cv2.imread(f'{original_image_dir}/{file_name}{extension}')

    # 이미지에서 흰색 영역을 선택합니다. 예를 들어, 이미지의 (x, y) 위치에 10x10 픽셀 크기의 흰색 패치가 있다고 가정합니다.
    # 실제 사용시에는 흰색 영역의 좌표를 정확히 알아야 합니다.
    x, y = 50, 50  # 흰색 패치의 시작 위치
    white_patch = image[y:y+10, x:x+10]

    # 화이트 밸런싱 적용
    image_balanced = white_balance_using_white_patch(image, white_patch)

    # 지정된 영역의 픽셀 값을 numpy 배열로 추출
    left_area = image_balanced[0:3040, 0:210]          # Left
    right_area = image_balanced[0:3040, 3620:4056]     # Right
    target_area = image_balanced[260:2760, 700:3200]   # Center, Target Area

    # Target Area의 R, G, B 채널별 평균 계산
    target_area_r_mean = round(np.mean(target_area[:, :, 0]))
    target_area_g_mean = round(np.mean(target_area[:, :, 1]))
    target_area_b_mean = round(np.mean(target_area[:, :, 2]))

    # 2024.04.29, jdk
    # 보정 후 numpy 배열의 원소들이 int64로 변경되어 Image로 생성 불가했음.
    # 이에 따라 data type을 uint8로 변경하는 코드를 추가하여 오류를 해결하였음.

    diff_r = target_area_r_mean - labels[i-start_num][0]
    diff_g = target_area_g_mean - labels[i-start_num][1]
    diff_b = target_area_b_mean - labels[i-start_num][2]

    print(f"{i}")
    print(f"R mean: {target_area_r_mean}, G mean: {target_area_g_mean}, B mean: {target_area_b_mean}")
    print(f"R: {diff_r}, G: {diff_g}, B: {diff_b}")

    # np array를 다시 image로 변환
    correcte_image = Image.fromarray(image_balanced)
    correcte_image.save(f"{hue_corrected_image_dir}/{i}_white_balanced_{extension}")

# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #