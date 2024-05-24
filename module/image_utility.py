import math
import numpy as np
from path_finder import PathFinder
from PIL import Image
import json

class ImageUtility:
    """
        image display, visualization 및 editing과 관련한
        모든 메서드를 포함하는 이미지 유틸리티 클래스
    """

    def __init__(self):
        self.pf = PathFinder()
        self.cropped_image_list_step_1 = []
        self.cropped_image_list_step_2 = []
        self.cropped_image_list_step_3 = []

        # image crop을 진행하기 위한 crop_size 변수
        with open(self.pf.image_settings_file_path, 'r') as file:
            image_settings = json.load(file)
            self.crop_size = image_settings["crop_size"]

    def clearCroppedImageLists(self):
        self.cropped_image_list_step_1.clear()
        self.cropped_image_list_step_2.clear()
        self.cropped_image_list_step_3.clear()

    def cropImageBySize(self, image, left, upper, right, lower):
        crop_area = (left, upper, right, lower)
        return image.crop(crop_area)

    def cropImage(self, mode, hue_corrected_images):
        """
            2024.05.28, jdk
            시스템 통합 과정에서 cropImage의 동작을 수정하도록 한다.
            본 함수는 mode와 image_file_name을 전달받고, 사전에
            load한 json file을 통해 crop_size에 맞게 image를 잘라
            지정된 directory에 저장하는 역할을 한다.
        """

        self.clearCroppedImageLists()

        if mode == 'Swatch':
            from_dir_path = self.pf.swatch_HC_directory_path
        elif mode == 'TPG':
            from_dir_path = self.pf.tpg_HC_directory_path
        elif mode == 'TCX':
            from_dir_path = self.pf.tcx_HC_directory_path
        
        if mode in ['Swatch', 'TCX']:
            # image_path = f"{from_dir_path}/{image_file_name}_HC.png"
            # image = Image.open(image_path)
            image = hue_corrected_images[0]

            for image_index in range(1, 9):
                left = self.crop_size[image_index-1][0]
                upper = self.crop_size[image_index-1][1]
                right = self.crop_size[image_index-1][2]
                lower = self.crop_size[image_index-1][3]

                cropped_image = self.cropImageBySize(image, left, upper, right, lower)
                self.cropped_image_list_step_1.append(cropped_image)

        elif mode == 'TPG':
            for image_index in range(1, 9):
                # image_path = f"{from_dir_path}/{image_file_name}_{image_index}_HC.png"
                # image = Image.open(image_path)
                image = hue_corrected_images[image_index-1]

                left = self.crop_size[image_index-1][0]
                upper = self.crop_size[image_index-1][1]
                right = self.crop_size[image_index-1][2]
                lower = self.crop_size[image_index-1][3]

                cropped_image = self.cropImageBySize(image, left, upper, right, lower)
                self.cropped_image_list_step_1.append(cropped_image)

    def combineImage(self, mode):
        """
            2024.05.28, jdk
            crop한 image를 combine하는 함수
        """

        # if mode == 'Swatch':
        #     to_dir_path = self.pf.swatch_combined_directory_path
        # elif mode == 'TPG':
        #     to_dir_path = self.pf.tpg_combined_directory_path
        # elif mode == 'TCX':
        #     to_dir_path = self.pf.tcx_combined_directory_path
        
        # TPG Image Combine
        # Combine Vertically - 1
        for image_index in range(0, 8, 2):

            """
                2024.05.13, jdk
                subprocess 내부에서 에러가 발생하면 result로 나타나는데,
                이것을 체크하지 않으므로 에러가 체크되지 않는다.

                2024.05.28, jdk
                subprocess 사용을 중단하였으므로, try/exception을 사용하지 않는다.
            """

            self.combineImagesVertically(step=1, upper_image_index=image_index, lower_image_index=image_index+1)

        # Combine Vertically - 2
        for tpg_index in range(0, 4, 2):
            self.combineImagesVertically(step=2, upper_image_index=tpg_index, lower_image_index=tpg_index+1)

        # Combine Horizontally - 3
        # 0: left, 1: right

        # combined_image_file_name = f"{image_file_name}_comb.png"
        # combined_image_file_path = f"{to_dir_path}/{combined_image_file_name}"

        combined_image = self.combineImagesHorizontally(upper_image_index=0, lower_image_index=1)
        # combined_image.save(combined_image_file_path)
        return combined_image

    def combineImagesVertically(self, step, upper_image_index, lower_image_index):
        """
            가로 사이즈가 동일한 두 개의 이미지를 전달받고,
            절반을 잘라 수직적으로 이어붙이는 함수이다.
        """

        # combine 1단계: 8개의 이미지를 결합하는 과정
        if step == 1:
            # 이미지 오픈
            upper_image = self.cropped_image_list_step_1[upper_image_index]
            lower_image = self.cropped_image_list_step_1[lower_image_index]
        # combine 2단계: 4개의 이미지를 결합하는 과정
        elif step == 2:
            upper_image = self.cropped_image_list_step_2[upper_image_index]
            lower_image = self.cropped_image_list_step_2[lower_image_index]

        # 가로 길이가 동일한지 체크
        # upper_image_size = upper_image.size[0]
        # lower_image_size = lower_image.size[0]

        """
            이미지의 width가 다른지 체크하고,
            width가 다르다면 Error를 일으킨다.

            2024.05.28, jdk
            원활한 동작을 위해 일시적으로 try/exception 제거
        """
        # if (not upper_image_size == lower_image_size):
        #     raise ValueError("Two images have different widths. The widths of the two images have to be the same.")

        # 2024.05.13, jdk
        # 코드 수정: width는 동일하므로 하나의 변수로 사용하고,
        # height가 달라졌으므로 height를 서로 다른 변수로 구분한다.
        width, upper_image_height = upper_image.size[0],  upper_image.size[1]
        _, lower_image_height = lower_image.size[0],  lower_image.size[1]

        # 새 이미지 생성
        combined_image = Image.new('RGB', (width, upper_image_height + lower_image_height), (255, 255, 255))
        combined_image.paste(upper_image, (0, 0))
        combined_image.paste(lower_image, (0, upper_image_height))

        # 이미지 저장
        if step == 1:
            self.cropped_image_list_step_2.append(combined_image)
        elif step == 2:
            self.cropped_image_list_step_3.append(combined_image)

    def combineImagesHorizontally(self, upper_image_index, lower_image_index):
        """
        가로 사이즈가 동일한 두 개의 이미지를 전달받고,
        두 이미지를 수평적으로 이어붙이는 함수이다.
        """

        left_image = self.cropped_image_list_step_3[upper_image_index]
        right_image = self.cropped_image_list_step_3[lower_image_index]

        # 이미지의 높이가 동일한지 확인
        left_image_height = left_image.size[1]
        right_image_height = right_image.size[1]

        # if left_image_height != right_image_height:
        #     raise ValueError("Images have different heights. The heights of both images must be the same.")

        # 이미지의 가로 길이를 계산하여 새 이미지의 크기 결정
        total_width = left_image.size[0] + right_image.size[0]
        height = left_image_height  # 이미지의 높이는 동일

        # 새 이미지 생성
        combined_image = Image.new('RGB', (total_width, height))

        # 이미지 이어붙이기
        combined_image.paste(left_image, (0, 0))  # 왼쪽 이미지 위치
        combined_image.paste(right_image, (left_image.size[0], 0))  # 오른쪽 이미지 위치

        return combined_image

##############################################################################################################################

    # def getImageSize(self):
    #     """
    #         지정된 image의 Size(Width, Height)를 저장하고, 반환하는 함수이다.
    #     """

    #     image_width, image_height = self.image.size

    #     return (image_width, image_height)

    # def cropImage(self, left, upper, right, lower):
    #     """
    #         지정된 image를 crop하는 함수이다.
    #         전달받은 left, upper, right, lower를 crop area로 설정하며,
    #         open된 img 객체를 crop하고 변형한다.

    #         2024.05.14, jdk
    #         이때, right, lower는 포함되지 않고 이전까지만 잘리게 된다.
    #     """

    #     crop_area = (left, upper, right, lower)
    #     self.image = self.image.crop(crop_area)

    # def calcImageBrightness(self):
    #     """
    #         현재 지정된 image의 밝기를 Pixel별로 알아내는 함수이다.
    #     """

    #     # image.load()를 통해 pixel 값에 접근할 수 있도록 변경
    #     # Image Library의 mode에 따라서 L(Gray Scale)RGB, RGBA, CMYK, HSV에 접근할 수 있음.
    #     # 1100x950
    #     pixels = self.image.load()

    #     # 2024.04.13, jdk
    #     # width, height 기준으로 뽑아오므로, 주의 필요함.
    #     # print(pixels[1099, 949])

    #     (width, height) = self.getImageSize()

    #     for w in range(width):
    #         image_brightness_row = []

    #         for h in range(height):
    #             pixel_value = pixels[w, h]
    #             brightness = self.calcBrightnessOfPixel(pixel_value)
    #             image_brightness_row.append(brightness)

    #         self.image_brightness.append(image_brightness_row)

    # def getBrightnessOfImage(self):
    #     """
    #         지정된 image의 밝기를 반환한다.
    #     """

    #     return self.image_brightness

    # # def setHistogramBins(self, rows_len, columns_len):
    #     # """
    #     #     Histogram을 그리기 위한 Bin을 설정하는 함수이
    #     #     본 Class에서는 특정 이차원 배열의 index에 설정된
    #     #     값을 빈도로 상정하여 Visualization하는 방식을 채택한다.

    #     #     TODO 2024.04.13, jdk
    #     #     현재는 Brightness 분석에 초점을 맞추므로, self.brightness
    #     #     배열에 대해서만 동작하도록 프로그래밍 했음. 추후에 수정 필요.

    #     #     + OpenGL을 사용함에도 3D Histogram Visualization에 시간적 한계가 존재함.
    #     #     이에 따라 2D Heatmap을 사용하는 방식으로 시각화를 변경하여, 이전 함수는 제거함.
    #     # """

    # def getHeatmapSeaborn(self, heatmap_image_path):
    #     imsize = self.image.size
    #     width = imsize[0]
    #     height = imsize[1]

    #     print(f"Original image size: width {width}, height {height}")

    #     brightness_values = np.array(self.image_brightness)
    #     min = brightness_values.min()
    #     max = brightness_values.max()

    #     normalized_values = ((brightness_values - min) / (max - min)).T

    #     # 히트맵 설정
    #     plt.figure(figsize=(width/100, height/100))
    #     ax = sns.heatmap(normalized_values, xticklabels=False, yticklabels=False, cbar=False)
    #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    #     # 축 및 라벨 숨기기
    #     plt.axis('off')

    #     # 파일로 저장: 원본 이미지와 동일한 해상도와 크기
    #     plt.savefig(heatmap_image_path, dpi=100, pad_inches=0)
    #     plt.close()

    #     heatmap_image = Image.open(heatmap_image_path)
    #     heatmap_size = heatmap_image.size
    #     print(f"Heatmap image size: width {heatmap_size[0]}, height {heatmap_size[1]}")


    # # def displayHistogram3DVispy(self, heatmap_image_path):
    # #     """
    # #         3D Histogram을 display하는 함수

    # #         TODO 2024.04.13, jdk
    # #         현재는 Brightness 분석에 초점을 맞추므로, self.brightness
    # #         배열에 대해서만 동작하도록 프로그래밍 했음. 추후에 수정 필요.

    # #     """

    # #     # image_brightness 배열의 길이 변수 선언
    # #     # 이때, image_brightness이 현재 지정된 image의
    # #     # 밝기를 담고 있지 않을 수도 있으므로 주의가 필요함.
    # #     rows_len = len(self.image_brightness)
    # #     columns_len = len(self.image_brightness[0])

    # #     # # image_brightness 배열을 통해 Histogram의 Bin을 설정하고 값을 받아온다.
    # #     # (xpos, ypos, zpos) = self.setHistogramBins(rows_len, columns_len)

    # #     # 캔버스 및 뷰 설정
    # #     """
    # #         2024.04.13, jdk
    # #         https://github.com/vispy/vispy/issues/904

    # #         지속적으로 이미지가 회전되어 디스플레이 되는 현상이 발생함.
    # #         검색 결과, PanZoomCamera 옵션을 사용할 경우 flip 현상이 발생하는 것을 확인함.
    # #         아는 PanZoomCamera를 사용할 경우, +y axis가 위로 가지만, 일반적으로 이미지
    # #         데이터는 +y axis가 아래로 향하기 때문에 발생한 것으로 생각됨.
    # #         이에 따라 view.camera.flip에서 y axis만 flip해 주었음.
    # #     """

    # #     canvas = scene.SceneCanvas(keys='interactive', size=(rows_len, columns_len), show=True)
    # #     view = canvas.central_widget.add_view()
    # #     view.camera = scene.PanZoomCamera(aspect=1)
    # #     view.camera.flip = (False, True, False)
    # #     view.camera.set_range((0, columns_len), (0, rows_len))

    # #     def saveCanvas(event):
    # #         img_data = canvas.render(alpha=False)
    # #         io.write_png(heatmap_image_path, img_data)

    # #     canvas.events.draw.connect(saveCanvas)

    # #     # 이미지로 데이터 시각화
    # #     """
    # #         2024.04.13, jdk
    # #         0을 최솟값, 255를 최댓값으로 설정하니 조도 차이가 눈에 띄게 나타나지 않는 것을 확인함.
    # #         이에 따라 이미지 자체에서의 최솟값과 최댓값을 찾아서 조도 차이가 잘 드러나도록 코드를 변경하였음.
    # #     """

    # #     brightness_values = np.array(self.image_brightness)
    # #     min = brightness_values.min()
    # #     max = brightness_values.max()

    # #     normalized_values = (brightness_values - min) / (max - min)

    # #     cmap = get_cmap('inferno')
    # #     colors = cmap(np.linspace(0, 1, 256))
    # #     vispy_cmap = color.Colormap(colors[:, :3])

    # #     image = scene.visuals.Image(normalized_values, parent=view.scene, cmap=vispy_cmap, clim=(0, 1))

    # #     app.run()

    # def RGBtoLAB(self, RGB):
    #     # RGB 값을 [0, 1] 범위로 정규화합니다.
    #     rgb_normalized = np.array(RGB) / 255.0
    #     # 3차원 배열로 재구성합니다.
    #     rgb_array = rgb_normalized.reshape(1, 1, 3)
    #     # RGB에서 LAB으로 변환합니다.
    #     lab = rgb2lab(rgb_array)

    #     return lab[0, 0, :]  # 첫 번째 픽셀의 LAB 값 반환

    # def calculateLABSimilarity(self, rgb1, rgb2):
    #     """
    #         2024.05.02, jdk
    #         RGB color 두 개를 전달받고, 두 색을 CIELAB으로 변환한 다음
    #         유사도를 비교하여 반환하는 메서드이다.
    #     """

    #     lab1 = self.RGBtoLAB(rgb1)
    #     lab2 = self.RGBtoLAB(rgb2)

    #     # Delta E 2000을 사용하여 두 색상의 차이를 계산합니다.
    #     delta_e = deltaE_ciede2000(lab1[None, None, :], lab2[None, None, :])
        
    #     return delta_e.item()  # numpy array에서 스칼라 값 추출
    
    # def calculateRGBDistance(self, rgb1, rgb2):
    #     # 전체 RGB에 대한 유클리드 거리 계산
    #     distance = math.sqrt((rgb1[0] - rgb2[0])**2 + (rgb1[1] - rgb2[1])**2 + (rgb1[2] - rgb2[2])**2)
    #     return distance

    # def calculateChannelIndependentRGBDistance(self, rgb1, rgb2):
    #     # R, G, B 각각에 대한 유클리드 거리 계산
    #     r_distance = abs(rgb1[0] - rgb2[0])
    #     g_distance = abs(rgb1[1] - rgb2[1])
    #     b_distance = abs(rgb1[2] - rgb2[2])

    #     return (r_distance+g_distance+b_distance)/3