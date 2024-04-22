import os

'''
    2024.04.16, kwc
    경로 모듈화
'''

class PathFinder:
    def __init__(self, base_dir_path="/home/pi/project"):
        # 기본 디렉토리 경로 설정, 기본값은 "/home/pi/project"
        self.base_dir_path = base_dir_path

        # 이미지 저장을 위한 루트 디렉토리
        self.image_dir_path = os.path.join(base_dir_path, "image")

        # swatch 이미지 관련 디렉토리
        self.swatch_image_dir_path = os.path.join(self.image_dir_path, "swatch_image")
        self.swatch_original_dir_path = os.path.join(self.swatch_image_dir_path, "original")  # 원본 이미지 디렉토리
        self.swatch_cropped_dir_path = os.path.join(self.swatch_image_dir_path, "cropped")  # 크롭된 이미지 디렉토리
        self.swatch_heatmap_dir_path = os.path.join(self.swatch_image_dir_path, "heatmap")  # 히트맵 이미지 디렉토리
        self.swatch_combined_dir_path = os.path.join(self.swatch_image_dir_path, "combined")  # 결합된 이미지 디렉토리
        self.swatch_test_dir_path = os.path.join(self.swatch_image_dir_path, "test")  # 테스트 이미지 디렉토리
        self.tpg_image_dir_path = os.path.join(self.image_dir_path, "tpg_image") # TPG 이미지 관련 디렉토리
        self.tcx_image_dir_path = os.path.join(self.image_dir_path, "tcx_image") # TCX 이미지 관련 디렉토리
        
        self.config_dir_path = os.path.join(base_dir_path, "config") # 설정 파일이 저장되는 디렉토리
        self.key_dir_path = self.config_dir_path  # API 키 또는 인증 관련 파일을 저장하는 디렉토리
        self.service_account_file_path = os.path.join(self.config_dir_path, "key.json")  # Google 서비스 계정 키 파일 경로
        
        self.image_number_file_path = os.path.join(self.config_dir_path, "image_number.txt")  # 이미지 순번을 기록하는 파일 경로

    def ensureDirectoriesExist(self):
        # 필요한 모든 디렉토리가 있는지 확인하고 없으면 생성
        os.makedirs(self.image_dir_path, exist_ok=True)
        os.makedirs(self.swatch_image_dir_path, exist_ok=True)
        os.makedirs(self.swatch_original_dir_path, exist_ok=True)
        os.makedirs(self.swatch_cropped_dir_path, exist_ok=True)
        os.makedirs(self.swatch_heatmap_dir_path, exist_ok=True)
        os.makedirs(self.tpg_image_dir_path, exist_ok=True)
        os.makedirs(self.tcx_image_dir_path, exist_ok=True)
        os.makedirs(self.config_dir_path, exist_ok=True)
        os.makedirs(self.key_dir_path, exist_ok=True)
        