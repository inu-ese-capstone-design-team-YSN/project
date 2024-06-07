import subprocess
import os
from path_finder import PathFinder  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스

class CameraCapture:
    def __init__(self, image_file_extension='.png'):
        self.path_finder = PathFinder()
        self.output_dir_1 = self.path_finder.capture_original_1_dir_path
        self.output_dir_2 = self.path_finder.capture_original_2_dir_path
        self.image_1 = "image_1"
        self.image_2 = "image_2"
        self.image_file_extension = image_file_extension
        self.capture_command = ['rpicam-still', '-o', '', '-t', '100', '-n']
    
    def capture_image_mode(self, file_name, output_dir):
        # 출력 디렉토리 존재 확인 및 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 이미지 파일 경로 생성
        image_path = os.path.join(output_dir, file_name + self.image_file_extension)
        
        # 명령어에 이미지 경로 설정
        self.capture_command[2] = image_path
        
        # 캡처 명령어 실행
        try:
            subprocess.run(self.capture_command, check=True)
            print(f"이미지 {image_path}가 저장되었습니다.")
            return image_path
        except subprocess.CalledProcessError as e:
            print(f"이미지 촬영 오류가 발생하였습니다: {e}")
        except Exception as e:
            print(f"예상치 못한 오류가 발생하였습니다: {e}")
        
        return None

    def capture_image(self):
        # image_1 촬영
        image_path_1 = self.capture_image_mode(self.image_1, self.output_dir_1)
        
        # image_2 촬영
        image_path_2 = self.capture_image_mode(self.image_2, self.output_dir_2)