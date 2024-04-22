from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
# from vispy import app, scene, color, io

class ImageUtility:
    """
        image display, visualization 및 editing과 관련한
        모든 메서드를 포함하는 이미지 유틸리티 클래스
    """

    def __init__(self):
        self.image_path = None
        self.image = None
        self.image_brightness = []

    def setImagePathAndOpen(self, image_path):
        """
            현재는 한 개의 image에 대해서만 다루는 것으로 설정하여
            instance 생성 시 image path를 받아오도록 한다.

            setImagePathAndOpen 함수를 실행할 시,
            기존에 저장된 다른 이미지에 대한 데이터는 모두 None으로 초기화한다.
        """

        self.image_path = image_path
        self.image = Image.open(self.image_path)

        self.image_brightness = []

    def saveImage(self, path):
        """
            open된 image를 지정된 path로 저장하는 함수이다.
        """

        self.image.save(path)

    def displayImage(self):
        """
            open된 image를 display하는 함수이다.
        """

        plt.imshow(self.image)
        plt.axis('off')
        plt.show()
        plt.close()

    def getImageSize(self):
        """
            지정된 image의 Size(Width, Height)를 저장하고, 반환하는 함수이다.
        """

        image_width, image_height = self.image.size

        return (image_width, image_height)

    def cropImage(self, left, upper, right, lower):
        """
            지정된 image를 crop하는 함수이다.
            전달받은 left, upper, right, lower를 crop area로 설정하며,
            open된 img 객체를 crop하고 변형한다.
        """

        crop_area = (left, upper, right, lower)
        self.image = self.image.crop(crop_area)

    def combineImagesVertically(self, upper_image_path, lower_image_path, combined_image_path):
        """
            가로 사이즈가 동일한 두 개의 이미지를 전달받고,
            절반을 잘라 수직적으로 이어붙이는 함수이다.
        """

        # 이미지 오픈
        upper_image = Image.open(upper_image_path)
        lower_image = Image.open(lower_image_path)

        # 가로 길이가 동일한지 체크
        upper_image_size = upper_image.size
        lower_image_size = lower_image.size

        # 이미지의 크기가 동일한지 체크하고
        # 크기가 다르다면 Error를 일으킨다.
        if (not upper_image_size == lower_image_size):
            raise ValueError("Two images have different widths. The widths of the two images have to be the same.")

        width, height = upper_image_size # 두 이미지의 크기가 동일하므로 크기 변수 통일
        # crop_height = height // 2

        """
            TODO 2024.04.16, jdk
            이전에는 스와치를 사용하였기 때문에 crop하는 자동화 코드가 있었지만,
            이제는 TPG에 대해 동작하도록 해야 하므로 crop을 주석처리 함.
            이후에는 추가적인 모듈화를 통해 두 동작을 분리해 주어야 함.
        """

        # upper_half_image = upper_image.crop((0, 0, width, crop_height))
        # lower_half_image = lower_image.crop((0, crop_height, width, height))

        # 새 이미지 생성
        # combined_image = Image.new('RGB', (width, height), (255, 255, 255))
        # combined_image.paste(upper_half_image, (0, 0))
        # combined_image.paste(lower_half_image, (0, crop_height))

        # upper_half_image = upper_image.crop((0, 0, width, crop_height))
        # lower_half_image = lower_image.crop((0, crop_height, width, height))

        # 새 이미지 생성
        combined_image = Image.new('RGB', (width, height*2), (255, 255, 255))
        combined_image.paste(upper_image, (0, 0))
        combined_image.paste(lower_image, (0, height))

        # 이미지 저장.
        # 기존에 저장되어 있던 image 변수를 temp_image에 옮겨놨다가
        # cropped_image의 저장이 완료되면 다시 복사한다.
        temp_image = self.image
        self.image = combined_image

        self.saveImage(combined_image_path)
        self.image = temp_image

    def combineImagesHorizontally(self, left_image_path, right_image_path, combined_image_path):
        """
        가로 사이즈가 동일한 두 개의 이미지를 전달받고,
        두 이미지를 수평적으로 이어붙이는 함수이다.
        """

        # 이미지 열기
        left_image = Image.open(left_image_path)
        right_image = Image.open(right_image_path)

        # 이미지의 높이가 동일한지 확인
        left_image_height = left_image.size[1]
        right_image_height = right_image.size[1]

        if left_image_height != right_image_height:
            raise ValueError("Images have different heights. The heights of both images must be the same.")

        # 이미지의 가로 길이를 계산하여 새 이미지의 크기 결정
        total_width = left_image.size[0] + right_image.size[0]
        height = left_image_height  # 이미지의 높이는 동일

        # 새 이미지 생성
        combined_image = Image.new('RGB', (total_width, height))

        # 이미지 이어붙이기
        combined_image.paste(left_image, (0, 0))  # 왼쪽 이미지 위치
        combined_image.paste(right_image, (left_image.size[0], 0))  # 오른쪽 이미지 위치

        temp_image = self.image
        self.image = combined_image

        self.saveImage(combined_image_path)
        self.image = temp_image

    def getImageMode(self):
        """
            현재 지정된 image의 mode를 확인한다.
        """

        mode = self.image.mode

        return mode

    def setImageMode(self, mode):
        """
            Image의 Pixel 값을 어떻게 표현할 것인지 mode를 설정하는 함수이다.
            이때, mode는 L(Gray Scale)RGB, RGBA, CMYK, HSV로 설정 가능하며,
            Open한 image에 대해서 convert를 적용해 변환한다.
        """

        if (not (mode == "L" or mode == "RGB" or mode == "RGBA" or mode == "CMYK" or mode == "HSV")):
            print(f"Wrong mode. The mode have to be one of (L, RGB, RGBA, CMYK, HSV)")
            print(f"Mode you set: {mode}")
            return

        self.image = self.image.convert(mode)

    def calcBrightnessOfPixel(self, RGB):
        """
            RGB값을 기반으로 특정 픽셀의 밝기를 계산하는 함수이다.
            본 방식은 YIQ 색 공간에서 Y 성분을 계산할 때 사용되는 방식이다.
            최솟값 0, 최댓값 255이다.

            TODO 2024.04.18, jdk
            RGB로 밝기를 수치화하는 것이 아니라, CIELAB으로 판단하는 것이
            더 나을 수도 있겠다는 생각이 들었음. 추가적인 논의가 필요해 보임.
        """

        (R, G, B) = RGB

        return 0.299*R + 0.587*G + 0.114*B

    def calcImageBrightness(self):
        """
            현재 지정된 image의 밝기를 Pixel별로 알아내는 함수이다.
        """

        # image.load()를 통해 pixel 값에 접근할 수 있도록 변경
        # Image Library의 mode에 따라서 L(Gray Scale)RGB, RGBA, CMYK, HSV에 접근할 수 있음.
        # 1100x950
        pixels = self.image.load()

        # 2024.04.13, jdk
        # width, height 기준으로 뽑아오므로, 주의 필요함.
        # print(pixels[1099, 949])

        (width, height) = self.getImageSize()

        for w in range(width):
            image_brightness_row = []

            for h in range(height):
                pixel_value = pixels[w, h]
                brightness = self.calcBrightnessOfPixel(pixel_value)
                image_brightness_row.append(brightness)

            self.image_brightness.append(image_brightness_row)

    def getBrightnessOfImage(self):
        """
            지정된 image의 밝기를 반환한다.
        """

        return self.image_brightness

    # def setHistogramBins(self, rows_len, columns_len):
        # """
        #     Histogram을 그리기 위한 Bin을 설정하는 함수이
        #     본 Class에서는 특정 이차원 배열의 index에 설정된
        #     값을 빈도로 상정하여 Visualization하는 방식을 채택한다.

        #     TODO 2024.04.13, jdk
        #     현재는 Brightness 분석에 초점을 맞추므로, self.brightness
        #     배열에 대해서만 동작하도록 프로그래밍 했음. 추후에 수정 필요.

        #     + OpenGL을 사용함에도 3D Histogram Visualization에 시간적 한계가 존재함.
        #     이에 따라 2D Heatmap을 사용하는 방식으로 시각화를 변경하여, 이전 함수는 제거함.
        # """

    def getHeatmapSeaborn(self, heatmap_image_path):
        imsize = self.image.size
        width = imsize[0]
        height = imsize[1]

        print(f"Original image size: width {width}, height {height}")

        brightness_values = np.array(self.image_brightness)
        min = brightness_values.min()
        max = brightness_values.max()

        normalized_values = ((brightness_values - min) / (max - min)).T

        # 히트맵 설정
        plt.figure(figsize=(width/100, height/100))
        ax = sns.heatmap(normalized_values, xticklabels=False, yticklabels=False, cbar=False)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

        # 축 및 라벨 숨기기
        plt.axis('off')

        # 파일로 저장: 원본 이미지와 동일한 해상도와 크기
        plt.savefig(heatmap_image_path, dpi=100, pad_inches=0)
        plt.close()

        heatmap_image = Image.open(heatmap_image_path)
        heatmap_size = heatmap_image.size
        print(f"Heatmap image size: width {heatmap_size[0]}, height {heatmap_size[1]}")


    # def displayHistogram3DVispy(self, heatmap_image_path):
    #     """
    #         3D Histogram을 display하는 함수

    #         TODO 2024.04.13, jdk
    #         현재는 Brightness 분석에 초점을 맞추므로, self.brightness
    #         배열에 대해서만 동작하도록 프로그래밍 했음. 추후에 수정 필요.

    #     """

    #     # image_brightness 배열의 길이 변수 선언
    #     # 이때, image_brightness이 현재 지정된 image의
    #     # 밝기를 담고 있지 않을 수도 있으므로 주의가 필요함.
    #     rows_len = len(self.image_brightness)
    #     columns_len = len(self.image_brightness[0])

    #     # # image_brightness 배열을 통해 Histogram의 Bin을 설정하고 값을 받아온다.
    #     # (xpos, ypos, zpos) = self.setHistogramBins(rows_len, columns_len)

    #     # 캔버스 및 뷰 설정
    #     """
    #         2024.04.13, jdk
    #         https://github.com/vispy/vispy/issues/904

    #         지속적으로 이미지가 회전되어 디스플레이 되는 현상이 발생함.
    #         검색 결과, PanZoomCamera 옵션을 사용할 경우 flip 현상이 발생하는 것을 확인함.
    #         아는 PanZoomCamera를 사용할 경우, +y axis가 위로 가지만, 일반적으로 이미지
    #         데이터는 +y axis가 아래로 향하기 때문에 발생한 것으로 생각됨.
    #         이에 따라 view.camera.flip에서 y axis만 flip해 주었음.
    #     """

    #     canvas = scene.SceneCanvas(keys='interactive', size=(rows_len, columns_len), show=True)
    #     view = canvas.central_widget.add_view()
    #     view.camera = scene.PanZoomCamera(aspect=1)
    #     view.camera.flip = (False, True, False)
    #     view.camera.set_range((0, columns_len), (0, rows_len))

    #     def saveCanvas(event):
    #         img_data = canvas.render(alpha=False)
    #         io.write_png(heatmap_image_path, img_data)

    #     canvas.events.draw.connect(saveCanvas)

    #     # 이미지로 데이터 시각화
    #     """
    #         2024.04.13, jdk
    #         0을 최솟값, 255를 최댓값으로 설정하니 조도 차이가 눈에 띄게 나타나지 않는 것을 확인함.
    #         이에 따라 이미지 자체에서의 최솟값과 최댓값을 찾아서 조도 차이가 잘 드러나도록 코드를 변경하였음.
    #     """

    #     brightness_values = np.array(self.image_brightness)
    #     min = brightness_values.min()
    #     max = brightness_values.max()

    #     normalized_values = (brightness_values - min) / (max - min)

    #     cmap = get_cmap('inferno')
    #     colors = cmap(np.linspace(0, 1, 256))
    #     vispy_cmap = color.Colormap(colors[:, :3])

    #     image = scene.visuals.Image(normalized_values, parent=view.scene, cmap=vispy_cmap, clim=(0, 1))

    #     app.run()
