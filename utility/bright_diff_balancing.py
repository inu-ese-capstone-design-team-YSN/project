import numpy as np
from PIL import Image

# 표준 RGB 값 설정
standard_r = 88
standard_g = 111
standard_b = 62

# 이미지 파일 경로 설정
image_path = "reduced_image_color.png"
gray_image_path = "reduced_image_gray.png"

# 컬러 이미지를 열기
color_image = Image.open(image_path)
reduced_image_color = np.array(color_image)

# 흑백 이미지를 열기
gray_image = Image.open(gray_image_path)
reduced_image_gray = np.array(gray_image)

# 첫 번째 픽셀의 RGB 값 추출
first_pixel_color = reduced_image_color[380, 280]

# 첫 번째 픽셀의 흑백 값 추출
first_pixel_gray = reduced_image_gray[380, 280]

# 표준 RGB 값과 흑백 값 이용해 미지수 x 계산
x_r = standard_r / (first_pixel_color[0] * first_pixel_gray)
x_g = standard_g / (first_pixel_color[1] * first_pixel_gray)
x_b = standard_b / (first_pixel_color[2] * first_pixel_gray)


# 첫 번째 픽셀 값 출력
print("첫 번째 픽셀의 R 값:", first_pixel_color[0])
print("첫 번째 픽셀의 G 값:", first_pixel_color[1])
print("첫 번째 픽셀의 B 값:", first_pixel_color[2])
print("첫 번째 픽셀의 흑백 값:", first_pixel_gray)
print("미지수 x_r:", x_r)
print("미지수 x_g:", x_g)
print("미지수 x_b:", x_b)


