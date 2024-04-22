import os
from google.cloud import storage  # Google Cloud Storage 클라이언트 라이브러리를 사용하기 위한 임포트
from google.cloud import logging  # Google Cloud Logging 서비스를 사용하기 위한 임포트
from path_finder import PathFinder  # 경로 찾기를 담당하는 클래스를 임포트
from google.oauth2 import service_account  # Google API 서비스 계정을 사용하기 위한 임포트

'''
2024.04.16, kwc
촬영 시 google cloud로 즉각 업로드, 로그 기록 클래스 생성
'''

# # GCS(Google Cloud Service)를 제어하는 함수
class CloudController:
    def __init__(self, path_finder):
        # 서비스 계정 및 버킷 설정
        self.service_account_file_path = path_finder.service_account_file_path  # 서비스 계정 파일의 경로
        self.bucket_name = 'ysn-bucket'  # 사용할 Google Cloud Storage 버킷의 이름
        self.storage_client = storage.Client.from_service_account_json(self.service_account_file_path)  # GCS 클라이언트 인스턴스 생성
        self.bucket = self.storage_client.bucket(self.bucket_name)  # GCS에서 사용할 버킷 설정

        # 자격증명을 직접 로드하여 로깅 클라이언트 생성
        self.credentials = service_account.Credentials.from_service_account_file(self.service_account_file_path)  # 서비스 계정으로부터 자격증명 로드
        self.logging_client = logging.Client(credentials=self.credentials)  # 로깅 클라이언트 인스턴스 생성
        
        # 로깅 클라이언트 설정
        self.logger = self.logging_client.logger("gcs-uploads")  # 로그 기록을 위한 로거 설정

    def upload_file(self, source_file_path, destination_blob_name, image_type=None):
        """
            파일을 Google Cloud Storage에 업로드하고 로그를 기록하는 함수

            destination_blob_name은 파일 이름 ex) image123.jpg
            source_file_path은 파일 전체 경로 ex) /home/pi/project/image/swatch_image/original/image123.jpg
            image_type은 업로드할 이미지의 타입: 1) swatch, 2) tpg, 3) tcx
        """
        # 폴더 이름을 이미지 유형에 따라 설정
        # folder_path는 GCS에서 저장될 파일의 상위 디렉터리 ex) swatch
        folder_path = {
            '1': 'swatch',  # Swatch 이미지 폴더
            '2': 'tpg',     # TPG 이미지 폴더
            '3': 'tcx'      # TCX 이미지 폴더
        }.get(image_type, 'other')  # 이미지 유형이 지정되지 않은 경우 'other' 폴더 사용

        # 전체 파일 경로 생성
        # full_destination_path는 GCS에서 저장 경로 ex) swatch/image123.jpg
        full_destination_path = f"{folder_path}/{destination_blob_name}"
        
        try:
            # 저장할 위치의 Blob 참조 생성
            blob = self.bucket.blob(full_destination_path) 
            blob.upload_from_filename(source_file_path)  # 파일 업로드 실행
            # 성공 로그 기록
            self.logger.log_text(f"File {source_file_path} uploaded to {full_destination_path} successfully.")
            print(f"File {source_file_path} uploaded to {full_destination_path} successfully.")  # 성공 메시지 출력
        except Exception as e:
            # 실패 로그 기록
            error_message = f"Failed to upload {source_file_path} to {full_destination_path}: {str(e)}"
            self.logger.log_text(error_message)  # 로그에 실패 메시지 기록
            print(error_message)  # 콘솔에 실패 메시지 출력