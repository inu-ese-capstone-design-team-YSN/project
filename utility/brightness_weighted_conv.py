import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
from path_finder import PathFinder

pf = PathFinder()

def weighted_average_convolution(image_array, kernel_size):
    # 이미지의 높이, 너비, 채널
    height, width, channels = image_array.shape
    
    # 결과 이미지 배열 초기화 (축소된 크기)
    new_height = height // kernel_size
    new_width = width // kernel_size
    result = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    # 커널 단위로 이미지를 순회
    for i in range(new_height):
        for j in range(new_width):
            # 현재 커널의 RGB 값 추출
            current_patch = image_array[i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size]
            
            # 커널 내의 각 픽셀 밝기 계산
            brightness = np.mean(current_patch, axis=2)
            
            # 밝기의 총합
            total_brightness = np.sum(brightness)
            
            if total_brightness > 0:
                # 각 픽셀의 밝기 가중치 계산 (각 픽셀 밝기 / 밝기의 총합)
                weights = brightness / total_brightness
            else:
                # 모든 밝기가 0인 경우 동일 가중치 부여
                weights = np.ones_like(brightness) / (kernel_size * kernel_size)
            
            # 가중치를 적용하여 각 채널의 평균 계산
            for c in range(channels):
                result[i, j, c] = np.sum(current_patch[:, :, c] * weights)
    
    return result

original_image = Image.open(f"{pf.tpg_combined_directory_path}/122_123.jpg")
image_array = np.array(original_image)

# 컨볼루션 커널 크기
kernel_size = 10  # 이미지 축소 비율

# 가중치가 적용된 컨볼루션 수행
resized_image = weighted_average_convolution(image_array, kernel_size)
resized_image = Image.fromarray(resized_image)

resized_image.save(f"{pf.tpg_test_dir_path}/test.jpg")