import subprocess  # 외부 프로세스를 실행하고 그 결과를 다루기 위해 사용
import os  # 운영체제와 상호작용을 위한 모듈, 파일 및 디렉토리 관리에 사용
import sys  # 시스템 관련 파라미터와 함수를 다루기 위해 사용
from google.cloud import storage  # Google Cloud Storage 서비스를 사용하기 위한 클라이언트 라이브러리

# 프로젝트의 data_class 디렉토리를 모듈 검색 경로에 추가하여 해당 디렉토리의 모듈을 사용 가능하게 함
from path_finder import PathFinder  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스
from cloud_controller import CloudController  # 클라우드 관련 작업을 관리하는 클래스

# ----------------------------------------------------------------------------- #

'''
2024.04.17, kwc

SWATCH, TPG, TCX별로 촬영 모드를 선택하여
촬영 시 이미지 이름 설정, 이미지 로그 기록,
GCS 자동 업로드 기능 구현을 동작하는 총체적인 프로그램 구현
'''

'''
변수 설명:

- preview_command: 카메라 미리보기를 시작하는 외부 명령어
- capture_command: 이미지를 캡처하는 외부 명령어로, 실행 시 파일명이 동적으로 설정

- preview_stop_key: 미리보기를 중단하는데 사용되는 키
- user_input_exit: 프로그램을 종료하는데 사용되는 사용자 입력 키
- user_input_capture: 이미지 캡처를 실행하는데 사용되는 사용자 입력 키
- user_input_preview: 미리보기 모드를 시작하는데 사용되는 사용자 입력 키

- image_file_extension: 캡처된 이미지 파일의 확장자를 정의하는 문자열, 기본적으로 '.png'로 설정

- image_type: 사용자가 선택한 이미지 유형('1'은 Swatch, '2'는 TPG, '3'은 TCX)을 저장하는 변수
- cmd_input: 메인 루프에서 사용자로부터 받은 명령을 저장하는 변수
- directory: 선택된 이미지 유형에 따라 이미지가 저장될 경로를 저장하는 변수
- image_name: 캡처된 이미지의 이름을 저장하는 변수
- upload_path: 클라우드에 업로드할 때 사용될 경로를 저장하는 변수
- file_name: 캡처 명령을 실행할 때 사용되는 전체 파일 경로를 저장하는 변수
'''

# 클래스 인스턴스 생성 및 사용
path_finder = PathFinder()  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스
path_finder.ensureDirectoriesExist()  # 필요한 디렉토리가 존재하는지 확인하고 없으면 생성
cloud_controller = CloudController(path_finder)  # 클라우드 관련 작업을 관리하는 클래스

# 카메라 커맨드 지정
# 카메라 미리보기 커맨드 설정, '-t 0'은 타이머 없음을 의미
preview_command = ['libcamera-hello', '-t', '0']
# 이미지 캡처 커맨드, '-o'는 출력 파일 경로
capture_command = ['libcamera-still', '-o', '']
capture_command[2] = file_name
subprocess.run(capture_command)