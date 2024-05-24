import os


class PathFinder:
    def __init__(self, base_dir_path="/home/pi/project"):
        # 기본 디렉토리 경로 설정, 기본값은 "/home/pi/project"
        self.base_dir_path = base_dir_path

        # 이미지 저장을 위한 루트 디렉토리
        self.image_dir_path = os.path.join(base_dir_path, "image")
        self.images_directory_path = os.path.join(base_dir_path, "images")
        
        # self.capture_dir_path = os.path.join(base_dir_path, "capture_images")

        # self.capture_original_dir_path = os.path.join(self.capture_dir_path, "original")
        # self.capture_HB_dir_path = os.path.join(self.capture_dir_path, "outer_balanced")
        # self.capture_cropped_dir_path = os.path.join(self.capture_dir_path, "cropped")
        # self.capture_AC_dir_path = os.path.join(self.capture_dir_path, "addaptive_convolution")
        # self.capture_BB_dir_path = os.path.join(self.capture_dir_path, "brigntness_balanced")
        
        # image의 swatch 이미지 관련 디렉토리
        self.swatch_images_dir_path = os.path.join(self.image_dir_path, "swatch_image")
        self.swatch_original_directory_path = os.path.join(self.swatch_images_dir_path, "original")  # 원본 이미지 디렉토리
        self.swatch_cropped_dir_path = os.path.join(self.swatch_images_dir_path, "cropped")  # 크롭된 이미지 디렉토리
        self.swatch_heatmap_dir_path = os.path.join(self.swatch_images_dir_path, "heatmap")  # 히트맵 이미지 디렉토리
        self.swatch_combined_dir_path = os.path.join(self.swatch_images_dir_path, "combined")  # 결합된 이미지 디렉토리
        self.swatch_HC_dir_path = os.path.join(self.swatch_images_dir_path, "hue_corrected") # 색조 보정 이미지 디렉토리
        self.swatch_test_dir_path = os.path.join(self.swatch_images_dir_path, "test")  # 테스트 이미지 디렉토리
        
        # image의 TPG 이미지 관련 디렉토리
        self.tpg_image_dir_path = os.path.join(self.image_dir_path, "tpg_image")
        self.tpg_original_directory_path = os.path.join(self.tpg_image_dir_path, "original")  # 원본 이미지 디렉토리
        self.tpg_cropped_dir_path = os.path.join(self.tpg_image_dir_path, "cropped")  # 크롭된 이미지 디렉토리
        self.tpg_heatmap_dir_path = os.path.join(self.tpg_image_dir_path, "heatmap")  # 히트맵 이미지 디렉토리
        self.tpg_combined_dir_path = os.path.join(self.tpg_image_dir_path, "combined")  # 결합된 이미지 디렉토리
        self.tpg_HC_dir_path = os.path.join(self.tpg_image_dir_path, "hue_corrected") # 색조 보정 이미지 디렉토리
        self.tpg_test_dir_path = os.path.join(self.tpg_image_dir_path, "test")  # 테스트 이미지 디렉토리
        self.tpg_compressed_dir_path = os.path.join(self.tpg_image_dir_path, "compressed")

        # image의 TCX 이미지 관련 디렉토리
        self.tcx_image_dir_path = os.path.join(self.image_dir_path, "tcx_image")
        self.tcx_original_directory_path = os.path.join(self.tcx_image_dir_path, "original")  # 원본 이미지 디렉토리
        self.tcx_heatmap_dir_path = os.path.join(self.tcx_image_dir_path, "heatmap")  # 히트맵 이미지 디렉토리
        self.tcx_combined_dir_path = os.path.join(self.tcx_image_dir_path, "combined")  # 결합된 이미지 디렉토리
        self.tcx_cropped_dir_path = os.path.join(self.tcx_image_dir_path, "cropped")  # 크롭된 이미지 디렉토리
        self.tcx_HC_dir_path = os.path.join(self.tcx_image_dir_path, "hue_corrected") # 색조 보정 이미지 디렉토리
        
        #---------------------------------------------------------------------------------------------------------------------
        
        # images의 swatch 이미지 관련 디렉토리
        self.swatch_images_directory_path = os.path.join(self.images_directory_path, "swatch")
        self.swatch_original_directory_path = os.path.join(self.swatch_images_directory_path, "original")  # 원본 이미지 디렉토리
        self.swatch_combined_directory_path = os.path.join(self.swatch_images_directory_path, "comb")  # 결합된 이미지 디렉토리
        self.swatch_HC_directory_path = os.path.join(self.swatch_images_directory_path, "HC") # 색조 보정 이미지 디렉토리
        self.swatch_AC_directory_path = os.path.join(self.swatch_images_directory_path, "AC") # 밝기 가중 축소 이미지 디렉토리
        self.swatch_BC_directory_path = os.path.join(self.swatch_images_directory_path, "BC") # 조도 보정 이미지 디렉토리
        self.swatch_corrected_directory_path = os.path.join(self.swatch_images_directory_path, "corrected") # 모든 보정 완료 이미지 디렉토리
        
        # images의 tpg 이미지 관련 디렉토리
        self.tpg_images_directory_path = os.path.join(self.images_directory_path, "tpg")
        self.tpg_original_directory_path = os.path.join(self.tpg_images_directory_path, "original")  # 원본 이미지 디렉토리
        self.tpg_combined_directory_path = os.path.join(self.tpg_images_directory_path, "comb")  # 결합된 이미지 디렉토리
        self.tpg_HC_directory_path = os.path.join(self.tpg_images_directory_path, "HC") # 색조 보정 이미지 디렉토리
        self.tpg_AC_directory_path = os.path.join(self.tpg_images_directory_path, "AC") # 밝기 가중 축소 이미지 디렉토리
        self.tpg_BC_directory_path = os.path.join(self.tpg_images_directory_path, "BC") # 조도 보정 이미지 디렉토리
        self.tpg_FC_directory_path = os.path.join(self.tpg_images_directory_path, "FC") # 색수차 보정 이미지 디렉토리
        self.tpg_corrected_directory_path = os.path.join(self.tpg_images_directory_path, "corrected") # 모든 보정 완료 이미지 디렉토리
        

        # images의 tcx 이미지 관련 디렉토리
        self.tcx_images_directory_path = os.path.join(self.images_directory_path, "tcx")
        self.tcx_original_directory_path = os.path.join(self.tcx_images_directory_path, "original")  # 원본 이미지 디렉토리
        self.tcx_combined_directory_path = os.path.join(self.tcx_images_directory_path, "comb")  # 결합된 이미지 디렉토리
        self.tcx_HC_directory_path = os.path.join(self.tcx_images_directory_path, "HC") # 색조 보정 이미지 디렉토리
        self.tcx_AC_directory_path = os.path.join(self.tcx_images_directory_path, "AC") # 밝기 가중 축소 이미지 디렉토리
        self.tcx_BC_directory_path = os.path.join(self.tcx_images_directory_path, "BC") # 조도 보정 이미지 디렉토리
        self.tcx_corrected_directory_path = os.path.join(self.tcx_images_directory_path, "corrected") # 모든 보정 완료 이미지 디렉토리
        
        
        #---------------------------------------------------------------------------------------------------------------------
        
        
        self.config_dir_path = os.path.join(base_dir_path, "config") # 설정 파일이 저장되는 디렉토리
        self.key_dir_path = self.config_dir_path  # API 키 또는 인증 관련 파일을 저장하는 디렉토리
        self.service_account_file_path = os.path.join(self.config_dir_path, "key.json")  # Google 서비스 계정 키 파일 경로
        self.image_settings_file_path = os.path.join(self.config_dir_path, "image_settings.json")
        
        self.image_number_file_path = os.path.join(self.config_dir_path, "image_number.txt")  # 이미지 순번을 기록하는 파일 경로
        self.image_settings_file_path = os.path.join(self.config_dir_path, "image_settings.json")  # 이미지 설정 파일 경로
    
    
    def ensureDirectoriesExist(self):
        
        # 필요한 모든 디렉토리가 있는지 확인하고 없으면 생성
        os.makedirs(self.image_dir_path, exist_ok=True)
        os.makedirs(self.swatch_images_dir_path, exist_ok=True)
        os.makedirs(self.swatch_original_directory_path, exist_ok=True)
        os.makedirs(self.swatch_cropped_dir_path, exist_ok=True)
        os.makedirs(self.swatch_heatmap_dir_path, exist_ok=True)
        os.makedirs(self.tcx_image_dir_path, exist_ok=True)
        os.makedirs(self.config_dir_path, exist_ok=True)
        os.makedirs(self.key_dir_path, exist_ok=True)
        os.makedirs(self.tpg_image_dir_path, exist_ok=True)
        os.makedirs(self.tpg_original_directory_path, exist_ok=True)
        os.makedirs(self.tpg_cropped_dir_path, exist_ok=True)
        os.makedirs(self.tpg_heatmap_dir_path, exist_ok=True)
        os.makedirs(self.tpg_image_dir_path, exist_ok=True)
        # os.makedirs(self.capture_dir_path, exist_ok=True)
        # os.makedirs(self.capture_swatch_dir_path, exist_ok=True)
        # os.makedirs(self.capture_OB_dir_path, exist_ok=True)
        # os.makedirs(self.capture_cropped_dir_path, exist_ok=True)
        # os.makedirs(self.capture_AC_dir_path, exist_ok=True)
        # os.makedirs(self.capture_BB_dir_pat, exist_ok=True)
        
    def get_directory(self, image_type):
        # 이미지 유형에 따라 적절한 디렉토리 경로 반환
        if image_type == '1':
            return self.swatch_original_directory_path
        elif image_type == '2':
            return self.tpg_original_directory_path  # TPG 이미지의 원본 디렉토리를 반환하도록 수정
        elif image_type == '3':
            return self.tcx_image_dir_path
        else:
            raise ValueError("Unknown image type")