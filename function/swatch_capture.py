'''
2024.05.22 kwc
실전 main프로그램에서 동작하는 capture 파일이다.

'''
import subprocess
import os
from path_finder import PathFinder  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스

class CameraCapture:
    def __init__(self, image_file_extension='.png'):
        self.path_finder = PathFinder()
        self.output_dir = self.path_finder.capture_original_dir_path
        self.image_file_extension = image_file_extension
        self.preview_command = ['rpicam-hello', '-t', '0']
        self.capture_command = ['rpicam-still', '-o', '', '-t', '100', '-n']

    def capture_image(self, file_name):
        # 출력 디렉토리 존재 확인 및 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 이미지 파일 경로 생성
        image_path = os.path.join(self.output_dir, file_name + self.image_file_extension)
        
        # 명령어에 이미지 경로 설정
        self.capture_command[2] = image_path
        
        # 캡처 명령어 실행
        try:
            subprocess.run(self.capture_command, check=True)
            print(f"이미지 {image_path}가 저장되었습니다.")
        except subprocess.CalledProcessError as e:
            print(f"이미지 촬영 오류가 발생하였습니다: {e}")
        except Exception as e:
            print(f"예상치 못한 오류가 발생하였습니다: {e}")
        
        return file_name

# 사용 예시
if __name__ == "__main__":
    camera = CameraCapture()
    camera.capture_image('test_image_code')  # 'test_image.png' 파일이 'captures' 디렉토리에 저장됨
