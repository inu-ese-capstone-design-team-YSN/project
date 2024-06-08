from image_utility import ImageUtility
from hue_correction import HueCorrection
from fringing_correction import FringingCorrection
from adaptive_convolution import AdaptiveConvolution
from brightness_correction import BrightnessCorrection

class ImageCorrection:
###################################################################################################################
    """
        Initializer
    """

    def __init__(self):
        """
            2024.05.28, jdk
            현재 프로그램의 보정 모드이다.
            mode == TPG일 경우, TPG에 대한 보정으로 코드가 동작하며,
            mode == TCx일 경우, TCX에 대한 보정으로 코드가 동작한다.
            기본 모드는 추론을 위한 Swatch로 설정한다.
        """
        self.mode_list = ['Swatch', 'TPG', 'TCX']
        self.mode = 'Swatch'

        # 현재 보정할 이미지의 객체를 저장하기 위한 변수
        # 보정을 진행하는 도중이나 완료한 후에는 corrected_image에 저장한다.
        self.image_file_name = None # image file의 이름
        
        """
            image processing을 위한 전체 Correction 객체 선언
        """
        self.hue_correction = HueCorrection() # 색조 보정
        self.image_utility = ImageUtility() # Crop & Combine
        self.fringing_correction = FringingCorrection() # 색수차 보정
        self.adaptive_convolution = AdaptiveConvolution() # 밝기 가중 적응형 컨볼루션
        self.brightness_correction = BrightnessCorrection() # 조도 보정

        self.tcx_codes = [
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

###################################################################################################################
    """
        Getters
    """

    # @property
    # def __mode(self):
    #     return self.__mode
    
###################################################################################################################
    """
        Setters
    """

    # @mode.setter
    # def __mode(self, mode):
    #     if mode not in self.__mode_list:
    #         # mode가 TCX, TPG, Swatch 중 하나가 아닐 경우 에러 발생
    #         raise Exception(f"mode는 반드시 TCX 혹은 TPG로 설정되어야 합니다. (현재 값: {mode})")
        
    #     self.__mode = mode

    def setMode(self, mode):
        self.mode = mode

###################################################################################################################

    def setImageFileName(self, image_file_name):
        """
            2024.05.28, jdk
            이미지 파일의 이름을 전달 받고 저장하는 함수

            ex)
            image_file_name: 19-3952.png
        """

        self.image_file_name = image_file_name
    
    def correctImage(self):
        """
            2024.05.28, jdk
            현재 ImageCorrection 클래스 내부에서
            객체로 가지고 있는 __image에 대한 보정을 수행하는 함수

            1) Swatch: HC -> Crop -> Combine -> AC -> BC

            2) TPG: HC -> Crop -> Combine -> FC -> AC -> BC

            3) TCX: HC -> Crop -> Combine -> AC -> BC
        """

        # 1) HC
        # mode와 image_file_name을 전달하고 보정을 진행한다. 보정 결과는 정해진 디렉터리에 저장된다.
        # print(f"{self.image_file_name} HC Started...")
        print(f"image file name: {self.image_file_name}")
        hue_corrected_images = self.hue_correction.correctHue(self.mode, self.image_file_name) 
        # print(f"{self.image_file_name} HC Completed...")

        # 2) Crop
        # print(f"{self.image_file_name} Crop Started...")
        self.image_utility.cropImage(self.mode, hue_corrected_images)
        # print(f"{self.image_file_name} Crop Completed...")

        # 3) Comb
        # print(f"{self.image_file_name} Comb Started...")
        combined_image = self.image_utility.combineImage(self.mode)
        # print(f"{self.image_file_name} Comb Completed...")

        # 4) FC
        if self.mode == 'TPG':
            print(f"{self.image_file_name} FC Started...")
            fringing_corrected_image = self.fringing_correction.correctFringing(combined_image)
            combined_image = fringing_corrected_image
            print(f"{self.image_file_name} FC Completed...")

        # 5) AC
        # print(f"{self.image_file_name} AC Started...")
        self.adaptive_convolution.doAdaptiveConvolution(self.mode, combined_image, self.image_file_name)
        # print(f"{self.image_file_name} AC Completed...")

        # 6) BC
        # print(f"{self.image_file_name} BC Started...")
        # self.brightness_correction.correctBrightness(self.mode, self.image_file_name, reduced_image)
        # print(f"{self.image_file_name} BC Completed...\n\n\n")
##########################################################################################################

if __name__ == "__main__":
    image_correction = ImageCorrection()

    code="99-9997"

    # for code in image_correction.tcx_codes:
    image_correction.setMode(mode='Swatch')
    image_correction.setImageFileName(image_file_name=code)
    image_correction.correctImage()