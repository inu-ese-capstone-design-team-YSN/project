from PIL import Image
import numpy as np
import os
from path_finder import PathFinder
import json

class HueCorrection:
    def __init__(self, standard_r=228, standard_g=224, standard_b=230, extension=".png"):
        self.standard_r = standard_r 
        self.standard_g = standard_g
        self.standard_b = standard_b 
        self.extension = extension 
        self.path_finder = PathFinder()
        
        self.TCX_original_dir = self.path_finder.tcx_original_directory_path
        self.TCX_hue_balanced_dir = self.path_finder.tcx_HC_directory_path
        self.TPG_original_dir = self.path_finder.tpg_original_directory_path
        self.TPG_hue_balanced_dir = self.path_finder.tpg_HC_directory_path
        self.swatch_original_dir = self.path_finder.swatch_original_directory_path
        self.swatch_hue_balanced_dir = self.path_finder.swatch_HC_directory_path

        self.hue_corrected_images = []
        
        # HC 영역
        self.areas = [
            (0, 0, 100, 3040),       # 첫 번째 영역(Left)
            (3956, 0, 4056, 3040),  # 두 번째 영역(Right)
        ]

        with open(self.path_finder.image_settings_file_path, 'r') as file:
            image_settings = json.load(file)
            self.crop_size = image_settings["crop_size"]

    def calculateRGBMeans(self, image):
        
        """이미지와 영역 목록을 받아 각 영역의 RGB 평균값을 계산합니다."""
        
        # 지정된 영역의 픽셀 값을 numpy 배열로 추출
        extracted_areas = [np.array(image.crop(area)) for area in self.areas]
        left_area = extracted_areas[0]  # Left Outer Area
        right_area = extracted_areas[1]  # Right Outer Area

        # left_area의 R, G, B 채널별 평균 계산
        left_r_mean = np.mean(left_area[:, :, 0])
        left_g_mean = np.mean(left_area[:, :, 1])
        left_b_mean = np.mean(left_area[:, :, 2])

        # right_area의 R, G, B 채널별 평균 계산
        right_r_mean = np.mean(right_area[:, :, 0])
        right_g_mean = np.mean(right_area[:, :, 1])
        right_b_mean = np.mean(right_area[:, :, 2])

        # image의 outer area rgb 평균 계산
        total_r_mean = round((left_r_mean + right_r_mean) / 2)
        total_g_mean = round((left_g_mean + right_g_mean) / 2)
        total_b_mean = round((left_b_mean + right_b_mean) / 2)

        return total_r_mean, total_g_mean, total_b_mean

    def correctHueOfCropArea(self, np_image, correction_value, cur_area_index=-1):
        for crop_area_index in range(0, 8):
            if self.mode == 'TPG' and crop_area_index != cur_area_index:
                continue

            row_start = self.crop_size[crop_area_index][1]
            row_end = self.crop_size[crop_area_index][3]
            col_start = self.crop_size[crop_area_index][0]
            col_end = self.crop_size[crop_area_index][2]

            # [col_start, row_start, col_end, row_end]

            # print(np_image[col_start:col_start+2, row_start:row_start+2, 0])
            np_image[row_start:row_end, col_start:col_end, :] = np_image[row_start:row_end, col_start:col_end, :] * correction_value
            # print(np_image[col_start:col_start+2, row_start:row_start+2, 0])
        
        return np_image

    def correctImageTCX(self, file_name):
        # 이미지를 불러오기
        original_image = Image.open(f'{self.TCX_original_dir}/{file_name}{self.extension}')
        total_r_mean, total_g_mean, total_b_mean = self.calculateRGBMeans(original_image)
        # image를 np array로 변환
        np_image = np.array(original_image, dtype=np.float32)

        # 보정값 생성
        R_correction_factor = (self.standard_r / total_r_mean)
        G_correction_factor = (self.standard_g / total_g_mean)
        B_correction_factor = (self.standard_b / total_b_mean)
        
        # 빼려는 RGB 보정값 배열 생성, broadcasting
        correction_value = np.array([R_correction_factor, G_correction_factor, B_correction_factor])
        # np_image = np_image * correction_value
        np_image = self.correctHueOfCropArea(np_image, correction_value)

        np_image = np.clip(np_image, 0, 255)
        np_image = np_image.astype(np.uint8)  # uint8 타입으로 최종 변환

        # np array를 다시 image로 변환
        HB_image = Image.fromarray(np_image)  # 수정된 부분
        self.hue_corrected_images.append(HB_image)
        # HB_image.save(f"{self.TCX_hue_balanced_dir}/{file_name}_HC{self.extension}")
        
    def correctImageTPG(self, file_name):   

        for i in range(1, 9):
            print(f"image ({i})")
            TPG_file_name = f"{file_name}_{i}"
            
            file_path = os.path.join(self.TPG_original_dir, TPG_file_name)
            file_path = file_path + ".png"
            
            # 이미지를 불러오기
            original_image = Image.open(file_path)
            total_r_mean, total_g_mean, total_b_mean = self.calculateRGBMeans(original_image)

            # image를 np array로 변환
            np_image = np.array(original_image, dtype=np.float32)

            # 보정값 생성
            R_correction_factor = (self.standard_r / total_r_mean)
            G_correction_factor = (self.standard_g / total_g_mean)
            B_correction_factor = (self.standard_b / total_b_mean)
            # 빼려는 RGB 보정값 배열 생성, broadcasting
            correction_value = np.array([R_correction_factor, G_correction_factor, B_correction_factor])
            # np_image = np_image * correction_value
            np_image = self.correctHueOfCropArea(np_image, correction_value, i-1)

            np_image = np.clip(np_image, 0, 255)
            np_image = np_image.astype(np.uint8)  # uint8 타입으로 최종 변환
            
            # np array를 다시 image로 변환
            HB_image = Image.fromarray(np_image)  # 수정된 부분
            self.hue_corrected_images.append(HB_image)
            # HB_image.save(f"{self.TPG_hue_balanced_dir}/{TPG_file_name}_HC{self.extension}")

    def correctImageSwatch(self, file_name):

        print(f"cur filename : {file_name}")

        base_path = None

        if file_name == "image_1" or file_name == "image_2":
            base_path = self.path_finder.capture_SM_dir_path
        elif file_name == "image":
            base_path = self.path_finder.capture_CI_dir_path

        # 이미지를 불러오기
        original_image = Image.open(f'{base_path}/{file_name}{self.extension}')
        total_r_mean, total_g_mean, total_b_mean = self.calculateRGBMeans(original_image)
        print(total_r_mean, total_g_mean, total_b_mean)
        # image를 np array로 변환
        np_image = np.array(original_image, dtype=np.float32)

        # 보정값 생성
        R_correction_factor = (self.standard_r / total_r_mean)
        G_correction_factor = (self.standard_g / total_g_mean)
        B_correction_factor = (self.standard_b / total_b_mean)
        
        # 빼려는 RGB 보정값 배열 생성, broadcasting
        correction_value = np.array([R_correction_factor, G_correction_factor, B_correction_factor])
        # np_image = np_image * correction_value
        np_image = self.correctHueOfCropArea(np_image, correction_value)

        np_image = np.clip(np_image, 0, 255)
        np_image = np_image.astype(np.uint8)  # uint8 타입으로 최종 변환

        # np array를 다시 image로 변환
        HB_image = Image.fromarray(np_image)  # 수정된 부분
        self.hue_corrected_images.append(HB_image)
        
    def correctHue(self, mode, image_file_name):
        
        self.hue_corrected_images.clear()
        self.mode = mode

        if mode == 'TPG':
            self.correctImageTPG(image_file_name)
        elif mode == 'TCX':
            self.correctImageTCX(image_file_name)
        elif mode == 'Swatch':
            self.correctImageSwatch(image_file_name)

        return self.hue_corrected_images

# 사용 예시
if __name__ == "__main__":
    corrector = HueCorrection()
    image_code = input("이미지 코드 번호를 입력하세요: ")
    corrector.correct_image_TCX(image_code)
