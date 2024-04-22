"""
    2024.04.18, jdk
    서버와 통신하기 위한 네트워크 컨트롤러 클래스
"""
import requests
from network_url import NetworkURL

class NetworkController:
    def __init__(self):
        self.network_url = NetworkURL()

        # TCP 연결을 유지하여 재사용함으로써
        # TCP 연결에 소요되는 시간과 자원을 줄이고
        # 헤더와 쿠키를 유지할 수 있다.
        self.session = requests.Session()

    def getHeatmapImage(self, original_image_path):
        url = self.network_url.base_url + self.network_url.im2hmap_url

        path_splited_by_slash = original_image_path.split('/')
        path_len = len(path_splited_by_slash)
        file_name = path_splited_by_slash[path_len-1]
        print(file_name)

        with open(original_image_path, 'rb') as file:
            files = {'image': file}
            data = {'image_name': file_name}
            response = self.session.post(url, files=files, data=data)
        
        # with문을 벗어나도 reponse는 이미 정상적으로 동작하였으므로
        # with 밖에서 참조해도 올바른 값을 얻을 수 있다.
        
        if response.status_code == 200:
            # 요청이 올바르게 처리됨
            return response
        else:
            return f"Error: {response.status_code} {response.text}"

    def DBSCAN(self):
        url = self.network_url.dbscan_url