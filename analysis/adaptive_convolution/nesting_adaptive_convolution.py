import numpy as np
from PIL import Image
from path_finder import PathFinder

'''
2024.05.14 kwc
mcu ac 알고리즘 개발
'''

# 밝기 가중치를 적용한 Convolution 함수 정의
def brightness_weighted_convolution(block):
    # 각 채널 분리
    R, G, B = block[:, :, 0], block[:, :, 1], block[:, :, 2]
    # 밝기 계산 (LUMA 공식 사용)
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    # 밝기 가중치를 적용한 합 계산
    weighted_sum = np.sum(block * brightness[:, :, np.newaxis], axis=(0, 1))
    # 가중치를 적용한 평균 반환
    return np.sum(weighted_sum) / np.sum(brightness)

# RGB 평균 Convolution 함수 정의
def rgb_average_convolution(block):
    # 각 채널의 평균 계산
    R = np.mean(block[:, :, 0])
    G = np.mean(block[:, :, 1])
    B = np.mean(block[:, :, 2])
    # RGB 값 배열로 반환
    return np.array([R, G, B], dtype=np.uint8)

# 이미지를 처리하는 함수 정의
def process_image(image_array, n, mode='L'):
    # 새로운 이미지의 높이와 너비 계산
    new_height = image_array.shape[0] // n
    new_width = image_array.shape[1] // n
    
    if mode == 'L':
        # 그레이스케일 모드일 경우
        new_image_array = np.zeros((new_height, new_width), dtype=np.uint8)  # 단일 채널 배열
        
        
        if image_array.ndim == 3:
            # RGB 이미지 처리
            for i in range(0, new_height):
                for j in range(0, new_width):
                    # 각 MCU 블록을 가져옴
                    block = image_array[i*n:(i+1)*n, j*n:(j+1)*n, :]
                    # 블록의 밝기 가중치 평균 계산
                    new_pixel = brightness_weighted_convolution(block)
                    # 새로운 이미지 배열에 할당
                    new_image_array[i, j] = new_pixel
        else:
            # 그레이스케일 이미지 처리
            for i in range(0, new_height):
                for j in range(0, new_width):
                    block = image_array[i*n:(i+1)*n, j*n:(j+1)*n]
                    new_pixel = np.mean(block)  # 블록의 평균 밝기 계산
                    new_image_array[i, j] = new_pixel
                    
        # 최대값과 최소값 계산 및 출력
        max_pixel_value = np.max(new_image_array)
        min_pixel_value = np.min(new_image_array)
        print(f"최대 밝기 값: {max_pixel_value}")
        print(f"최소 밝기 값: {min_pixel_value}")
       
                    
    elif mode == 'RGB':
        # RGB 모드일 경우
        new_image_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # RGB 배열
        if image_array.ndim == 3:
            # RGB 이미지 처리
            for i in range(0, new_height):
                for j in range(0, new_width):
                    block = image_array[i*n:(i+1)*n, j*n:(j+1)*n, :]
                    new_pixel = rgb_average_convolution(block)  # 블록의 RGB 평균 계산
                    new_image_array[i, j, :] = new_pixel
        else:
            raise ValueError("Input image must be RGB for RGB processing.")
    
    return new_image_array

# 메인 함수 정의
def main():
    # PathFinder 인스턴스를 생성하여 디렉토리 경로를 가져옴
    path_finder = PathFinder()
    extension = ".png"  # 파일 확장자 설정
    combined_image_dir = path_finder.tpg_combined_dir_path  # TPG 결합 이미지 디렉토리 경로

    # 이미지 파일 이름과 경로 설정 (예: 2500 x 2500 크기)
    # image_filename = f"19-1606_combined{extension}"
    image_path = f"./analysis/fringing_correction/temp_images/fringing_corrected.png"
    # image_path = f"{combined_image_dir}/{image_filename}"

    # 이미지 열기
    image = Image.open(image_path)
    image_array = np.array(image)

    # MCU 크기 설정
    n = 5

    # 첫 번째 MCU 과정 (그레이스케일)
    first_pass_image_array_gray = process_image(image_array, n, mode='L')
    # 두 번째 MCU 과정 (그레이스케일)
    second_pass_image_array_gray = process_image(first_pass_image_array_gray, n, mode='L')
    
    # 새로운 그레이스케일 이미지 생성 및 저장
    # new_image_gray = Image.fromarray(first_pass_image_array_gray, mode='L')  # 'L' 모드로 그레이스케일 이미지 생성
    new_image_gray = Image.fromarray(second_pass_image_array_gray, mode='L')  # 'L' 모드로 그레이스케일 이미지 생성
    new_image_gray.save(f"16-1450_combined_mcu_gray{extension}")

    # 첫 번째 MCU 과정 (RGB)
    first_pass_image_array_rgb = process_image(image_array, n, mode='RGB')
    # 두 번째 MCU 과정 (RGB)
    second_pass_image_array_rgb = process_image(first_pass_image_array_rgb, n, mode='RGB')
    # 새로운 RGB 이미지 생성 및 저장
    new_image_rgb = Image.fromarray(second_pass_image_array_rgb, mode='RGB')  # 'RGB' 모드로 컬러 이미지 생성
    new_image_rgb.save(f"16-1450_combined_mcu_rgb{extension}")

if __name__ == "__main__":
    main()
