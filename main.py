import subprocess # 외부 프로세스를 실행하고 그 결과를 다루기 위해 사용
import os # 운영체제와 상호작용을 위한 모듈, 파일 및 디렉토리 관리에 사용
import sys # 시스템 관련 파라미터와 함수를 다루기 위해 사용
from path_finder import PathFinder  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스
from function.swatch_capture import CameraCapture
from analysis.white_balancing.balancing_outerArea_swatch import outerAreaBalancing

camera = CameraCapture()
HB = outerAreaBalancing()
'''

이 파일은 총체적인 main program이다.

프로그램의 Flow는 다음과 같다.
1. GUI 모드 on
2. 촬영 시
    2-1. Image Correction
        2-1-1. WB
        2-1-2. Target Area만큼 crop
        2-1-3. Addaptive Convolution
        2-1-4. Brightness 보정
    2-2. Clustering
    2-3. Color Similarity
        2-3-1. Color Inference
        2-3-2. Calc Similiarity (단일 촬영 모드 시 x)
        
'''

'''
2024.05.22 kwc
실전 스와치 capture 모드 파일 ~/project/function에 개발중

'''



def main():
    image_code = camera.capture_image('test_image_code')  # 'test_image.png' 파일이 'captures' 디렉토리에 저장됨
    HB.correct_image(image_code) # HB 작동 -> ~/project/capture_images/outer_balanced에 이미지 저장됨
    
    

if __name__ == "__main__":
    main()