import numpy as np
from PIL import Image
from path_finder import PathFinder
import cv2

class AdaptiveConvolution:
    
    def __init__(self, extension=".png"):
        self.extension = extension
        self.path_finder = PathFinder()
        self.tcx_combined_image_dir = self.path_finder.tcx_combined_directory_path
        self.tcx_adaptive_convolution_image_dir = self.path_finder.tcx_AC_directory_path
        self.tpg_fringing_correction_image_dir = self.path_finder.tpg_FC_directory_path
        self.tpg_adaptive_convolution_image_dir = self.path_finder.tpg_AC_directory_path
        self.swatch_combined_image_dir = self.path_finder.swatch_combined_directory_path
        self.swatch_adaptive_convolution_image_dir = self.path_finder.swatch_AC_directory_path
        

    # 밝기 가중치를 적용한 Convolution 함수 정의
    # def BrightnessWeightedConvolution(self, block):
    #     # 각 채널 분리
    #     R, G, B = block[:, :, 0], block[:, :, 1], block[:, :, 2]
    #     # 밝기 계산 (LUMA 공식 사용)
    #     brightness = 0.299 * R + 0.587 * G + 0.114 * B
    #     # 밝기 가중치를 적용한 합 계산
    #     weighted_sum = np.sum(block * brightness[:, :, np.newaxis], axis=(0, 1))
    #     # 가중치를 적용한 평균 반환
    #     return np.sum(weighted_sum) / np.sum(brightness)

    # def getBrightnessKernel(self, kernel):
    #     brightness_kernel = []
    #     for pixel in kernel:
    #         R = pixel[0]
    #         G = pixel[1]
    #         B = pixel[2]
    #         brightness = 0.299*R + 0.587*G + 0.114*B
    #         brightness_kernel.append(brightness)
    #     return np.array(brightness_kernel)

    def rgb_to_lab(self, rgb):
        lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
        return lab

    def conv(self, kernel):
        # width, height, _ = kernel.shape
        # kernel_flattened = kernel.reshape(-1, 3)
        # # brightness_weight = 0.299 * kernel_flattened[:, 0] + 0.587 * kernel_flattened[:, 1] + 0.114 * kernel_flattened[:, 2]
        # brightness_weight = 0.333 * kernel_flattened[:, 0] + 0.333 * kernel_flattened[:, 1] + 0.333 * kernel_flattened[:, 2]
        # brightness_weighted_mean = np.average(kernel_flattened, weights=brightness_weight, axis=0)
        # return brightness_weighted_mean

        width, height, _ = kernel.shape
        kernel_flattened = kernel.reshape(-1, 3)
        lab_values = self.rgb_to_lab(kernel_flattened.reshape((width, height, 3)))
        luminance = lab_values[:, :, 0].flatten()
        weighted_mean = np.average(kernel_flattened, weights=luminance, axis=0)
        return weighted_mean.astype(np.uint8)

    # 이미지를 처리하는 함수 정의
    def ProcessImage(self, image_array, n):
        """
        2024.05.28, jdk
        함수 인자에서 mode는 사용하지 않으므로, mode='L' 제거
        """
        new_height = image_array.shape[0] // n
        new_width = image_array.shape[1] // n
        new_image_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # RGB 배열
        if image_array.ndim == 3:
            for i in range(0, new_height):
                for j in range(0, new_width):
                    kernel = image_array[i*n:(i+1)*n, j*n:(j+1)*n, :]
                    new_pixel = self.conv(kernel)  # 블록의 RGB 평균 계산
                    new_image_array[i, j, :] = new_pixel
        return new_image_array

    # 메인 함수 정의
    def doAdaptiveConvolution(self, mode, combined_image, image_file_name):
        # if mode == 'TPG':
        #     image_path = f"{self.tpg_fringing_correction_image_dir}/{image_filename}_FC{self.extension}"
        # elif mode == 'TCX':
        #     image_path = f"{self.tcx_combined_image_dir}/{image_filename}_comb{self.extension}"
        # elif mode == 'Swatch':
        #     image_path = f"{self.swatch_combined_image_dir}/{image_filename}_comb{self.extension}"

        # 이미지 파일 이름과 경로 설정 (예: 2500 x 2500 크기)

        # 이미지 열기
        # image = Image.open(image_path)
        image = combined_image
        image = image.convert("RGB")
        image_array = np.array(image)

        # MCU 크기 설정
        n = 5

        if image_file_name == "image_1" or image_file_name == "image_2":
            base_path = self.path_finder.capture_SM_dir_path
        elif image_file_name == "image":
            base_path = self.path_finder.capture_CI_dir_path

        # 첫 번째 MCU 과정 (RGB)
        first_pass_image_array_rgb = self.ProcessImage(image_array, n)
        # 두 번째 MCU 과정 (RGB)
        second_pass_image_array_rgb = self.ProcessImage(first_pass_image_array_rgb, n)
        # 새로운 RGB 이미지 생성 및 저장
        new_image_rgb = Image.fromarray(second_pass_image_array_rgb, mode='RGB')

        if mode == 'TPG':
            new_image_rgb.save(f"{self.tpg_adaptive_convolution_image_dir}/{image_file_name}{self.extension}")
            print(f"{image_file_name}_FC{self.extension} AC 보정 완료")
        elif mode == 'TCX':
            new_image_rgb.save(f"{self.tcx_adaptive_convolution_image_dir}/{image_file_name}{self.extension}")
            print(f"{image_file_name}_comb{self.extension} AC 보정 완료")
        elif mode == 'Swatch':
            new_image_rgb.save(f"{base_path}/{image_file_name}{self.extension}")
            # print(f"{image_file_name}_comb{self.extension} AC 보정 완료")

            
            
        

        

if __name__ == "__main__":
    ac = AdaptiveConvolution()
    ac.main("19-3952_combined")
