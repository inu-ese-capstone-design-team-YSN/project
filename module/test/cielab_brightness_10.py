import numpy as np
import os
from skimage import io, color

# 주어진 영역의 Luminance(명도) 평균을 계산하는 함수
def calculate_region_mean_luminance(L, start_row, end_row, start_col, end_col):
    region = L[start_row:end_row, start_col:end_col]
    region_mean = np.mean(region)
    return region_mean

# 주어진 영역의 RGB 평균을 계산하는 함수
def calculate_corner_rgb_mean(image, start_row, end_row, start_col, end_col):
    region = image[start_row:end_row, start_col:end_col]
    mean_rgb = np.mean(region, axis=(0, 1))
    return mean_rgb

# Luminance(명도)를 조정하는 함수
def adjust_luminance(L, a, b):
    # 각 모서리의 명도 평균을 계산
    mean_1 = calculate_region_mean_luminance(L, 0, 10, 0, 10)
    mean_2 = calculate_region_mean_luminance(L, 90, 100, 0, 10)
    mean_3 = calculate_region_mean_luminance(L, 0, 10, 90, 100)
    mean_4 = calculate_region_mean_luminance(L, 90, 100, 90, 100)

    # 전체 평균 명도를 계산
    total_mean_luminance = np.mean([mean_1, mean_2, mean_3, mean_4])

    # 평균 명도로 새로운 L 채널을 생성
    L_adjusted = np.full_like(L, total_mean_luminance)
    lab_adjusted = np.stack((L_adjusted, a, b), axis=-1)
    
    return lab_adjusted

# RGB 비율을 조정하는 함수
def adjust_rgb_ratios(img):
    # 각 모서리의 RGB 평균을 계산
    mean_rgb_1 = calculate_corner_rgb_mean(img, 0, 10, 0, 10)
    mean_rgb_2 = calculate_corner_rgb_mean(img, 90, 100, 0, 10)
    mean_rgb_3 = calculate_corner_rgb_mean(img, 0, 10, 90, 100)
    mean_rgb_4 = calculate_corner_rgb_mean(img, 90, 100, 90, 100)
    
    # 전체 평균 RGB를 계산
    total_mean_rgb = np.mean([mean_rgb_1, mean_rgb_2, mean_rgb_3, mean_rgb_4], axis=0)
    
    # RGB 비율을 계산
    rgb_sum = np.sum(total_mean_rgb)
    r_ratio, g_ratio, b_ratio = total_mean_rgb / rgb_sum
    
    # 이미지의 각 픽셀에 RGB 비율을 적용
    adjusted_img = np.zeros_like(img, dtype=np.float32)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            pixel_sum = np.sum(img[row, col])
            adjusted_img[row, col] = [pixel_sum * r_ratio, pixel_sum * g_ratio, pixel_sum * b_ratio]
    
    adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)
    
    return adjusted_img

# 밝기와 RGB 비율을 조정하는 함수
def adjust_brightness_and_rgb(image_path):
    # 이미지 읽기
    img = io.imread(image_path)
    
    # 이미지가 RGBA 형식인 경우 RGB로 변환
    if img.shape[-1] == 4:
        img = color.rgba2rgb(img)
    # RGB를 LAB 색공간으로 변환
    lab = color.rgb2lab(img)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    # 명도 조정
    lab_adjusted = adjust_luminance(L, a, b)
    
    # LAB를 RGB로 변환
    img_adjusted = color.lab2rgb(lab_adjusted)
    img_adjusted = (img_adjusted * 255).astype(np.uint8)
    
    # RGB 비율 조정
    adjusted_img = adjust_rgb_ratios(img_adjusted)
    
    # RGB를 다시 LAB로 변환
    lab = color.rgb2lab(adjusted_img)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    # 명도 다시 조정
    final_lab_adjusted = adjust_luminance(L, a, b)
    
    # 최종 이미지를 RGB로 변환
    final_img = color.lab2rgb(final_lab_adjusted)
    final_img = (final_img * 255).astype(np.uint8)
    
    # 결과 이미지 저장
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_adjusted_combined{ext}"
    io.imsave(output_path, final_img, check_contrast=False)

if __name__ == "__main__":
    for filename in os.listdir('.'):
        if filename.lower().endswith('_ac.png'):
            print(f"Processing {filename}")
            adjust_brightness_and_rgb(filename)
            # adjust_rgb_then_brightness(filename)
