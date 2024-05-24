import numpy as np
from PIL import Image
from path_finder import PathFinder

class BrightnessCorrection:
    def __init__(self):
        self.pf = PathFinder()

    # 밝기 계산 함수
    def calculate_brightness(self, R, G, B):
        # return 0.299 * R + 0.587 * G + 0.114 * B
        return 0.333 * R + 0.333 * G + 0.333 * B

    # 모든 픽셀의 밝기를 첫 번째 픽셀의 밝기와 동일하게 맞추는 함수
    def match_brightness(self, image_array, target_brightness):
        height, width = image_array.shape[:2]

        for i in range(height):
            for j in range(width):
                R, G, B = image_array[i, j, 0], image_array[i, j, 1], image_array[i, j, 2]
                current_brightness = self.calculate_brightness(R, G, B)
                delta_brightness = target_brightness / current_brightness

                # RGB 값이 유효한 범위 [0, 255] 내에 있도록 조정
                image_array[i, j, 0] = np.clip(R*delta_brightness, 0, 255)
                image_array[i, j, 1] = np.clip(G*delta_brightness, 0, 255)
                image_array[i, j, 2] = np.clip(B*delta_brightness, 0, 255)
        
        return image_array

    def correctBrightness(self, mode, image_file_name, reduced_image):

        # 이미지 파일 경로 설정
        if mode == 'Swatch':
            from_dir_path = self.pf.swatch_AC_directory_path
            to_dir_path = self.pf.swatch_BC_directory_path
        elif mode == 'TPG':
            from_dir_path = self.pf.tpg_AC_directory_path
            to_dir_path = self.pf.tpg_BC_directory_path
        elif mode == 'TCX':
            from_dir_path = self.pf.tcx_AC_directory_path
            to_dir_path = self.pf.tcx_BC_directory_path

        # image_file_path = f"{from_dir_path}/{image_file_name}_AC.png"
        output_image_file_path = f"{to_dir_path}/{image_file_name}_BC.png"

        # 이미지 열기
        # image = Image.open(image_file_path)
        image = reduced_image
        image_array = np.array(image)

        # 첫 번째 픽셀의 밝기 계산
        R1, G1, B1 = image_array[0, 0, 0], image_array[0, 0, 1], image_array[0, 0, 2]
        target_brightness = self.calculate_brightness(R1, G1, B1)
        print(target_brightness)

        # 모든 픽셀의 밝기를 첫 번째 픽셀의 밝기와 동일하게 맞춤
        adjusted_image_array = self.match_brightness(image_array, target_brightness)

        # 새로운 이미지 생성 및 저장
        new_image = Image.fromarray(adjusted_image_array.astype(np.uint8), mode='RGB')
        new_image.save(output_image_file_path)
        print(f"조정된 이미지를 저장했습니다: {output_image_file_path}")
        
        # image2 = Image.open(output_image_file_path)
        # image_array2 = np.array(image2)
        
        # 첫 번째 픽셀의 밝기 계산
        # R2, G2, B2 = image_array[50, 50, 0], image_array[50, 50, 1], image_array[50, 50, 2]
        # target_brightness = calculate_brightness(R2, G2, B2)