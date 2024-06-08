from camera_capture import CameraCapture
import argparse
import sys

# Argument parser 생성
parser = argparse.ArgumentParser(description='capture image index')

# 인자를 정의
parser.add_argument('--n', type=int, choices=[0, 1, 2])

# 인자 파싱
args = parser.parse_args()
index = None

# 입력 값에 따른 작업 수행
index = args.n

camera_capture = CameraCapture()
camera_capture.capture_image(index)