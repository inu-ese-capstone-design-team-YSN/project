import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from image_utility import ImageUtility
import json
import os

image_utility = ImageUtility()

# JSON 파일 위치 확인 및 로드
json_file_path = 'image_processing_results.json'

if not os.path.exists(json_file_path):
    raise FileNotFoundError(f"No such file or directory: '{json_file_path}'")
else:
    with open(json_file_path, 'r') as f:
        results = json.load(f)

# 각 이미지에 대해 유사도 및 거리 계산
for image_id, data in results.items():
    target_rgb = tuple(data['Target Area Value'])
    corrected_rgb = tuple(data['Corrected Target Area Value'])
    label_rgb = tuple(data['Target Label'])

    # Target Area Value와 Label RGB 비교
    similarity_target = image_utility.calculateLABSimilarity(target_rgb, label_rgb)
    similarity_target_percentage = 100 - similarity_target
    distance_target = image_utility.calculateRGBDistance(target_rgb, label_rgb)
    channel_distance_target = image_utility.calculateChannelIndependentRGBDistance(target_rgb, label_rgb)

    # Corrected Target Area Value와 Label RGB 비교
    similarity_corrected = image_utility.calculateLABSimilarity(corrected_rgb, label_rgb)
    similarity_corrected_percentage = 100 - similarity_corrected
    distance_corrected = image_utility.calculateRGBDistance(corrected_rgb, label_rgb)
    channel_distance_corrected = image_utility.calculateChannelIndependentRGBDistance(corrected_rgb, label_rgb)

    print(f"Image {image_id}:")
    print(f"  Target Area Value vs Label RGB - LAB 색상 유사도 (Delta E): {similarity_target}")
    print(f"  Target Area Value vs Label RGB - 유사도 퍼센테이지: {similarity_target_percentage}%")
    print(f"  Target Area Value vs Label RGB - RGB 거리: {distance_target}")
    print(f"  Target Area Value vs Label RGB - 채널별 RGB 거리: {channel_distance_target}")
    print(f"  Corrected Target Area Value vs Label RGB - LAB 색상 유사도 (Delta E): {similarity_corrected}")
    print(f"  Corrected Target Area Value vs Label RGB - 유사도 퍼센테이지: {similarity_corrected_percentage}%")
    print(f"  Corrected Target Area Value vs Label RGB - RGB 거리: {distance_corrected}")
    print(f"  Corrected Target Area Value vs Label RGB - 채널별 RGB 거리: {channel_distance_corrected}")
