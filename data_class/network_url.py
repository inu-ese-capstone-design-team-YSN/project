"""
    웹 서버와 통신할 때 사용할 network url을 저장하는 데이터 클래스
"""

class NetworkURL:
    def __init__(self):
        self.base_url = "http://211.226.237.118:6500"
        self.im2hmap_url = "/im2hmap"
        self.dbscan_url = "/dbscan"