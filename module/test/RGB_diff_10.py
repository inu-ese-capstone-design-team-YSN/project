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
    
    def calculate_region_mean(self, image_array, start_row, end_row, start_col, end_col):
        region = image_array[start_row:end_row, start_col:end_col, :3]  # 알파 채널 무시
        region_mean = np.mean(region, axis=(0, 1))
        return region_mean

    def process_image(self, image_path, label_row):
        image = Image.open(image_path)
        image_array = np.array(image, dtype=np.int16)  # np.int16으로 변환

        # 이미지 크기 확인
        height, width, _ = image_array.shape
        if height != 100 or width != 100:
            print(f"Image: {image_path} is not 100x100 pixels.")
            return

        # 4개의 특정 영역의 RGB 평균 계산
        mean_1 = self.calculate_region_mean(image_array, 0, 10, 0, 10)
        mean_2 = self.calculate_region_mean(image_array, 90, 100, 0, 10)
        mean_3 = self.calculate_region_mean(image_array, 0, 10, 90, 100)
        mean_4 = self.calculate_region_mean(image_array, 90, 100, 90, 100)

        # 4개의 평균 RGB들의 총 평균 계산
        total_mean = np.mean([mean_1, mean_2, mean_3, mean_4], axis=0)
        total_mean = total_mean.astype(np.int16)

        # (46, 46)부터 (55, 55) 영역의 RGB 평균 계산
        center_mean = self.calculate_region_mean(image_array, 46, 55, 46, 55)
        center_mean = center_mean.astype(np.int16)
        
        
        
        # labels 배열과 비교
        diff_label = total_mean - label_row
        
        # total_mean과 center_mean의 차이 계산
        diff_center = total_mean - center_mean
        
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Label: {label_row}")
        print(f"Total RGB mean: {total_mean}")
        print(f"Difference with Label: {diff_label}")
        #print(f"Center region mean: {center_mean}")
        print(f"Difference with Center region: {diff_center}\n")

    def calc_diff(self):
        # 현재 디렉토리의 모든 PNG 파일 찾기
        bc_files = []
        ac_files = []
        for filename in os.listdir('.'):
            if filename.lower().endswith('.png'):
                parts = filename.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                code = parts[0]
                suffix = parts[1]
                if code in self.codes and suffix == "BC.png":
                    bc_files.append(filename)
                elif code in self.codes and suffix == "ACconverted.png":
                    ac_files.append(filename)
        
        bc_files.sort()
        ac_files.sort()

        # 번갈아가면서 출력
        max_length = max(len(bc_files), len(ac_files))
        for i in range(max_length):
            if i < len(bc_files):
                code = bc_files[i].rsplit('_', 1)[0]
                label_row = np.array(self.labels[self.codes.index(code)], dtype=np.int16)
                self.process_image(bc_files[i], label_row)
            if i < len(ac_files):
                code = ac_files[i].rsplit('_', 1)[0]
                label_row = np.array(self.labels[self.codes.index(code)], dtype=np.int16)
                self.process_image(ac_files[i], label_row)

if __name__ == "__main__":
    rgb_diff = RGB_diff()
    rgb_diff.calc_diff()
