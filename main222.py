import subprocess # 외부 프로세스를 실행하고 그 결과를 다루기 위해 사용
import os # 운영체제와 상호작용을 위한 모듈, 파일 및 디렉토리 관리에 사용
import sys # 시스템 관련 파라미터와 함수를 다루기 위해 사용
from path_finder import PathFinder  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스
from function.swatch_capture import CameraCapture


'''

이 파일은 총체적인 main program이다.

프로그램의 Flow는 다음과 같다.
1. 단일 원단 색상 추론
2. 유사도 비교 모드
    2-1. 촬영 및 사진 확인
        2-1-1. 결정 - 폴더: original_1
        2-1-2. 재촬영 - 동일명으로 재촬영
    2-2. 촬영 및 사진 확인
        2-2-1. 결정 - 폴더: original_2
        2-2-2. 재촬영 - 동일명으로 재촬영
    2-3. Image Correction
    2-4. Clustering
    2-5. Color Similarity
        2-5-1. Color Inference
        2-5-2. Calc Similiarity
        
'''

'''
2024.05.22 kwc
실전 스와치 capture 모드 파일 ~/project/function/swatch_capture.py에 개발중

'''

class MainProgram:
    def __init__(self):
        self.camera = CameraCapture()
        self.path_finder = PathFinder()

    def main():
        camera.capture_image() 
        
    

if __name__ == "__main__":
    main()