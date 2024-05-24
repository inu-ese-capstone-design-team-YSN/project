from PIL import Image
import numpy as np
from path_finder import PathFinder

class RGB_diff:
    def __init__(self):
        self.path_finder = PathFinder()
        
        self.TCX_BC_dir = self.path_finder.tcx_BC_directory_path
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
    def calc_diff(self):
        for i, code in enumerate(self.codes):
            image_path = f'{self.TCX_BC_dir}/{code}_BC.png'
            image = Image.open(image_path)
            image_array = np.array(image)

            # 첫 번째 픽셀의 RGB 값 추출
            first_pixel_rgb = image_array[0:5, 0:5, :3]  # 알파 채널 무시
            rgb_mean = np.mean(first_pixel_rgb, axis=(0, 1))
            rgb_mean = rgb_mean.astype(np.uint8)

            # labels 배열과 비교
            label_row = self.labels[i]
            diff = np.abs(rgb_mean - label_row)
            
            print(f"Image: {code}.png")
            # print(f"First pixel RGB: {first_pixel_rgb}")
            print(f"Label: {label_row}")
            print(f"Difference: {diff}\n")

if __name__ == "__main__":
    rgb_diff = RGB_diff()
    rgb_diff.calc_diff()