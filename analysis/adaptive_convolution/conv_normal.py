import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
from path_finder import PathFinder

pf = PathFinder()

# 이미지 파일 경로 설정 (PathFinder 대신 직접 지정)
original_image = Image.open(f"{pf.tpg_combined_dir_path}/122_123.jpg")

# 이미지를 NumPy 배열로 변환
image_array = np.array(original_image)

# 컨볼루션 커널 사이즈 계산 (원본 크기 / 목표 크기)
kernel_size = original_image.size[0] // 200
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

# 각 채널에 대해 컨볼루션 적용
resized_image_channels = [
    convolve2d(image_array[:, :, i], kernel, mode='valid')[::kernel_size, ::kernel_size]
    for i in range(3)  # RGB 채널 수는 3
]

# 컨볼루션 결과를 다시 합쳐서 최종 이미지 생성
resized_image = np.stack(resized_image_channels, axis=-1).astype(np.uint8)
resized_image = Image.fromarray(resized_image)
resized_image.save(f"{pf.tpg_test_dir_path}/test_normal.jpg")
# 원본 이미지와 축소된 이미지 시각화
