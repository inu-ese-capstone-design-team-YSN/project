import numpy as np
from PIL import Image

# 예제 이미지 배열 생성 (8x8 크기의 RGB 이미지)

def shuffleFringingRowsPixels(corrected_fringing_rows):
    """
        2024.05.20, jdk
        전달받은 array의 모든 pixel을 random하게 섞는 함수
    """

    height, width, channels = corrected_fringing_rows.shape

    indicies = np.arange(height * width)
    np.random.shuffle(indicies)

    flattend = corrected_fringing_rows.reshape(-1, channels)
    shuffled_array = np.empty_like(flattend)

    for i in range(len(indicies)):
        pixel_index = indicies[i] # index를 얻어냄
        rgb_value = flattend[pixel_index]
        shuffled_array[i] = rgb_value
    
    shuffled_array = shuffled_array.reshape(height, width, channels)

    return shuffled_array

a = np.array(
    [[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], 
     [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]], 
     [[9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12]], 
     [[13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16]]]
)
print(a)
a = shuffleFringingRowsPixels(a)
print("\n\n")
print(a)




# flattened = np.arange(10)
# np.random.shuffle(flattened)
# print(flattened)

# flattend = flattened.reshape(5, 2)
# print(flattend)