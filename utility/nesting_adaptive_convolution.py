import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from path_finder import PathFinder

'''

2024.05.14 kwc
mcu ac 알고리즘 개발

'''
# PathFinder 인스턴스를 생성하여 디렉토리 경로를 가져옴
path_finder = PathFinder()
extension = ".png"  # 파일 확장자 설정
combined_image_dir = path_finder.tpg_combined_dir_path  # TPG 결합 이미지 디렉토리 경로

# 이미지 파일 이름과 경로 설정
image_filename = f"18-6018_combined{extension}"
image_path = f"{combined_image_dir}/{image_filename}"

# 이미지를 열기
image = Image.open(image_path)

# 이미지를 NumPy 배열로 변환
image_array = np.array(image)

def reduce_image(image_array, MCU_SIZE, stride):
    # 축소된 이미지의 크기를 계산
    new_height = (image_array.shape[0] - MCU_SIZE) // stride + 1
    new_width = (image_array.shape[1] - MCU_SIZE) // stride + 1

    if new_height <= 0 or new_width <= 0:
        return np.zeros((0, 0)), np.zeros((0, 0, 3), dtype=np.uint8)

    # 축소된 이미지 배열 생성 (흑백)
    reduced_image_gray = np.zeros((new_height, new_width))

    # 축소된 이미지 배열 생성 (컬러)
    reduced_image_color = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # LUMA 공식을 이용한 밝기 가중치 적용된 Convolution 함수 정의
    def brightness_weighted_convolution(block):
        # 블록의 밝기(픽셀 값) 가중치를 적용한 평균을 계산합니다.
        R, G, B = block[:,:,0], block[:,:,1], block[:,:,2]
        brightness = 0.299 * R + 0.587 * G + 0.114 * B
        weighted_sum = np.sum(block * brightness[:, :, np.newaxis], axis=(0, 1))
        return weighted_sum / np.sum(brightness)

    # 컬러 평균 계산 함수 정의
    def color_average_convolution(block):
        # 블록의 RGB 채널 각각에 대해 평균을 계산합니다.
        return np.mean(block, axis=(0, 1))

    # Convolution을 수행하여 이미지를 축소합니다.
    for i in range(0, image_array.shape[0] - MCU_SIZE + 1, stride):
        for j in range(0, image_array.shape[1] - MCU_SIZE + 1, stride):
            # 현재 블록을 추출
            block = image_array[i:i + MCU_SIZE, j:j + MCU_SIZE]
            
            # 흑백 이미지용 밝기 가중치 적용
            reduced_pixel_gray = brightness_weighted_convolution(block)
            
            # 컬러 이미지용 색상 평균 적용
            reduced_pixel_color = color_average_convolution(block)
            
            # 축소된 이미지 배열에 결과 저장
            reduced_image_gray[i // stride, j // stride] = np.mean(reduced_pixel_gray)
            reduced_image_color[i // stride, j // stride] = reduced_pixel_color

    return reduced_image_gray, reduced_image_color

# 첫 번째 단계: n을 선택 (예: 64으로 가정)
n = 8

# 각 단계의 MCU_SIZE와 stride 설정
mcu_sizes = [n, n // 2]
strides = [n // 2, n // 4]


# 이미지를 반복적으로 축소
for mcu_size, stride in zip(mcu_sizes, strides):
    reduced_image_gray, reduced_image_color = reduce_image(image_array, mcu_size, stride)
    if reduced_image_color.size == 0:
        break
    image_array = reduced_image_color  # 다음 단계 축소를 위해 컬러 이미지를 사용

# 최종 축소된 이미지 크기 출력
print("최종 축소된 이미지 크기 (흑백):", reduced_image_gray.shape)
print("최종 축소된 이미지 크기 (컬러):", reduced_image_color.shape)

# 결과 이미지를 확인합니다.
if reduced_image_gray.size > 0:
    # 흑백 이미지 저장
    plt.imshow(reduced_image_gray, cmap='gray')
    plt.title("최종 축소된 이미지 (흑백)")
    plt.savefig("final_reduced_image_gray.png")
    print("이미지가 'final_reduced_image_gray.png' 파일로 저장되었습니다.")

if reduced_image_color.size > 0:
    # 컬러 이미지 저장
    plt.imshow(reduced_image_color)
    plt.title("최종 축소된 이미지 (컬러)")
    plt.savefig("final_reduced_image_color.png")
    print("이미지가 'final_reduced_image_color.png' 파일로 저장되었습니다.")
