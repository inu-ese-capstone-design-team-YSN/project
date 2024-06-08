from skimage.color import deltaE_ciede2000
from skimage import color
import numpy as np
import pandas as pd
from path_finder import PathFinder

class ColorSimilarityCalculator():
    def __init__(self, cluster_images_name, inferenced_Lab):
        self.cluster_images_name = cluster_images_name
        self.inferenced_Lab = inferenced_Lab
        
        self.image_1_inferenced_Lab = []
        self.image_2_inferenced_Lab = []

        self.inferenced_Lab_pair = []
        self.inferenced_RGB_pair = []

        self.deltaE = []

        self.isSwapped = False
        self.path_finder = PathFinder()

    def lab2rgb(self, lab_array):
        return np.round(color.lab2rgb(lab_array) * 255)

    def cie_distance(self, cie_array1, cie_array2):
        return deltaE_ciede2000(cie_array1, cie_array2)
    
    def swap_lists(self, list1, list2):
        return list2, list1
    
    def swap_Lab_pair(self):
        for i in range(len(self.inferenced_Lab_pair)):
            new_Lap_pair = (self.inferenced_Lab_pair[i][1], self.inferenced_Lab_pair[i][0])
            self.inferenced_Lab_pair[i] = new_Lap_pair

    def getRGBFromLab(self):
        for i in range(len(self.inferenced_Lab_pair)):
            new_RGB_pair = self.lab2rgb(self.inferenced_Lab_pair[i][0]), self.lab2rgb(self.inferenced_Lab_pair[i][1])
            self.inferenced_RGB_pair.append(new_RGB_pair)

    def calculate_deltaE(self):
        """
            image_1의 cluster image와
            image_2의 cluster image를 분리하고,
            각각의 이미지에서 가장 비슷한 색상끼리
            매칭하는 함수
        """

        # 2) 이름에 따라서 inferenced Lab 분리
        for image_index in range(len(self.cluster_images_name)):

            if "image_1" in self.cluster_images_name[image_index]:
                self.image_1_inferenced_Lab.append(self.inferenced_Lab[image_index])
            elif "image_2" in self.cluster_images_name[image_index]:
                self.image_2_inferenced_Lab.append(self.inferenced_Lab[image_index])
        
        # 만약 image_2의 색상 클러스터 개수가 더 적을 경우

        if len(self.image_2_inferenced_Lab) < len(self.image_1_inferenced_Lab):
            self.image_1_inferenced_Lab, self.image_2_inferenced_Lab = self.swap_lists(self.image_1_inferenced_Lab, self.image_2_inferenced_Lab)
            self.isSwapped = True

        # 2) image별로 deltaE 계산
        for i in range(len(self.image_1_inferenced_Lab)):
            current_image_1_Lab = self.image_1_inferenced_Lab[i]

            min_deltaE = np.inf
            min_deltaE_j = np.inf

            for j in range(len(self.image_2_inferenced_Lab)):
                current_image_2_Lab = self.image_2_inferenced_Lab[j]

                deltaE = self.cie_distance(current_image_1_Lab, current_image_2_Lab)
                if min_deltaE > deltaE:
                    min_deltaE_j = j
                    min_deltaE = deltaE

            self.inferenced_Lab_pair.append((self.image_1_inferenced_Lab[i], self.image_2_inferenced_Lab[min_deltaE_j]))
            self.deltaE.append(min_deltaE)

        if self.isSwapped:
            # 결과 바꿔주기
            self.swap_Lab_pair()
        
        self.getRGBFromLab()

        return (self.inferenced_Lab_pair, self.inferenced_RGB_pair, self.deltaE)
    
    def saveResult(self):
        RGB1 = []
        RGB2 = []
        Lab1 = []
        Lab2 = []

        for i in range(len(self.deltaE)):
            RGB1.append(self.inferenced_RGB_pair[i][0])
            RGB2.append(self.inferenced_RGB_pair[i][1])
            Lab1.append(self.inferenced_Lab_pair[i][0])
            Lab2.append(self.inferenced_Lab_pair[i][1])

        data = {
            'RGB1': RGB1,
            'Lab1': Lab1,
            'RGB2': RGB2,
            'Lab2': Lab2,
            'DeltaE': self.deltaE 
        }

        # 데이터프레임 생성
        df = pd.DataFrame(data)

        # CSV 파일로 저장
        df.to_csv(f"{self.path_finder.result_data_dir_path}/analysis.csv", index=False)