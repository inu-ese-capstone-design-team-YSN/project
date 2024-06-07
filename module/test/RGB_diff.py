import os
from PIL import Image
import numpy as np

class RGB_diff:
    def __init__(self):
        self.codes = [
            "11-0601",
            "11-4302",
            "13-4305",
            "14-1910",
            "15-1214",
            "17-0145",
            "17-0340",
            "17-4041",
            "18-1246",
            "18-1764",
            "19-1554",
            "19-2311",
            "19-3911",
            "19-3952",
        ]

        self.labels = [
            [244, 249, 255], # 11-0601
            [241, 240, 226], # 11-4302
            [209, 213, 208], # 13-4305
            [247, 185, 194], # 14-1910
            [197, 174, 145], # 15-1214
            [68, 136, 60], # 17-0145
            [114, 136, 57], # 17-0340
            [80, 133, 195], # 17-4041
            [148, 74, 32], # 18-1246
            [204, 28, 59], # 18-1764
            [149, 21, 40], # 19-1554
            [98, 63, 76], # 19-2311
            [39, 39, 42], # 19-3911
            [32, 60, 127], # 19-3952
        ]
    
    ''' 첫 픽셀과 중간 픽셀 하나 골라서 비교하는 코드 '''
    
    def process_image(self, image_path, label_row):
        image = Image.open(image_path)
        image_array = np.array(image, dtype=np.int16)  # np.int16으로 변환

        # 이미지 크기 확인
        height, width, _ = image_array.shape
        if height <= 50 or width <= 50:
            print(f"Image: {image_path} is too small for (50, 50) pixel comparison.")
            return

        # 첫 번째 픽셀의 RGB 값 추출
        first_pixel_rgb = image_array[0:5, 0:5, :3]
        rgb_mean = np.mean(first_pixel_rgb, axis=(0, 1)).astype(np.int16)

        # (50, 50) 위치의 픽셀 RGB 값 추출
        pixel_50_50_rgb = image_array[50, 50, :3]

        # labels 배열과 비교
        diff_label = rgb_mean - label_row
        diff_50_50 = pixel_50_50_rgb - rgb_mean
        
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Label: {label_row}")
        print(f"First pixel RGB mean: {rgb_mean}")
        print(f"(50, 50) pixel RGB: {pixel_50_50_rgb}")
        print(f"Difference with Label: {diff_label}")
        print(f"Difference with (50, 50) pixel: {diff_50_50}\n")

    def calc_diff(self):
        # 현재 디렉토리의 모든 PNG 파일 찾기
        for filename in os.listdir('.'):
            if filename.lower().endswith('.png'):
                parts = filename.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                code = parts[0]
                suffix = parts[1]
                if code in self.codes and suffix in ["BC.png", "ACadjusted.png"]:
                    label_row = np.array(self.labels[self.codes.index(code)], dtype=np.int16)
                    image_path = os.path.join('.', filename)
                    self.process_image(image_path, label_row)

if __name__ == "__main__":
    rgb_diff = RGB_diff()
    rgb_diff.calc_diff()