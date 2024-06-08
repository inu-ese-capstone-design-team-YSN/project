import subprocess
import os
from path_finder import PathFinder  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스

class CameraCapture:
    def __init__(self, image_file_extension='.png'):
        self.path_finder = PathFinder()
        self.CI_dir_path = self.path_finder.capture_CI_dir_path
        self.SM_dir_path = self.path_finder.capture_SM_dir_path
        self.image_filename = "image"
        self.image_1_filename = "image_1"
        self.image_2_filename = "image_2"
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

    def capture_image(self, index):

        if index == 0:
            self.capture_image_mode(self.image_filename, self.CI_dir_path)
        elif index == 1:
            # image_1 촬영
            self.capture_image_mode(self.image_1_filename, self.SM_dir_path)
        elif index == 2:
            self.capture_image_mode(self.image_2_filename, self.SM_dir_path)


# 사용 예시
if __name__ == "__main__":
    camera = CameraCapture()
    camera.capture_image()
