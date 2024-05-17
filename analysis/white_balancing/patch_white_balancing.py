import cv2
import numpy as np

def white_balance_using_white_patch(img, white_patch):
    # 하얀 영역의 평균 BGR 값을 계산합니다.
    average_white = np.mean(white_patch, axis=(0, 1))

    # 이미지의 각 픽셀을 흰색 영역의 평균으로 나누어 정규화합니다.
    scaling_factors = 255 / average_white
    img_balanced = img * scaling_factors[np.newaxis, np.newaxis, :]

    # 값이 255를 넘지 않도록 조정합니다.
    img_balanced = np.clip(img_balanced, 0, 255).astype(np.uint8)
    return img_balanced

# 이미지 로드
img = cv2.imread('39.jpg')

# 이미지에서 흰색 영역을 선택합니다. 예를 들어, 이미지의 (x, y) 위치에 10x10 픽셀 크기의 흰색 패치가 있다고 가정합니다.
# 실제 사용시에는 흰색 영역의 좌표를 정확히 알아야 합니다.
x, y = 50, 50  # 흰색 패치의 시작 위치
white_patch = img[y:y+10, x:x+10]

# 화이트 밸런싱 적용
image_balanced = white_balance_using_white_patch(img, white_patch)

# 결과 이미지 저장
cv2.imwrite("white_balanced_39.jpg", image_balanced)
