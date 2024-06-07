import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def convert_to_grayscale_heatmap(image_path, output_path):
    # 이미지 로드 및 그레이스케일 변환
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    
    # 히트맵 생성
    plt.imshow(image_array, cmap='hot', interpolation='nearest')
    plt.colorbar()
    
    # 히트맵 저장
    plt.savefig(output_path)
    plt.close()

    print(f"Heatmap saved at: {output_path}")

# 이미지 경로
image_path = "/home/pi/project/images/swatch/BC/99-9997_BC.png"
# image_path = "./image/15-1262_combined.png"
output_path = "/home/pi/project/images/swatch/HC/99-9997_BC.png"

convert_to_grayscale_heatmap(image_path, output_path)

# # 이미지 경로
# image_path = "./reduced_image/reduced_rgb.png"
# # image_path = "./image/15-1262_combined.png"
# output_path = "./reduced_image/reduced_rgb_heatmap.png"

# # 그레이스케일 히트맵으로 변환 및 저장
# convert_to_grayscale_heatmap(image_path, output_path)

# # 이미지 경로
# image_path = "./reduced_image/reduced_rgb_BC.png"
# # image_path = "./image/15-1262_combined.png"
# output_path = "./reduced_image/reduced_rgb_BC_heatmap.png"

# # 그레이스케일 히트맵으로 변환 및 저장
# convert_to_grayscale_heatmap(image_path, output_path)