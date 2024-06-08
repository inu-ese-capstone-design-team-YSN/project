import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from collections import Counter
from PIL import Image
from path_finder import PathFinder
import cv2

class Clustering:
    def __init__(self, eps=3, min_samples=300):
        """
        파일 경로, eps, 및 min_samples로 Clustering 초기화
        :param file_path: 이미지 파일 경로
        :param eps: DBSCAN의 두 샘플 간 최대 거리
        :param min_samples: 코어 포인트로 간주되기 위한 이웃 샘플의 수
        """

        self.eps = eps
        self.min_samples = min_samples
        self.img = None
        self.color_tbl = None # RGB pixel을 갖는 list
        self.labels = None # 각 pixel 별 label
        self.unique_labels = None # 고유 Label을 담는 변수
        self.cluster_centers = None # 각 cluster의 대표 색상(평균)
        self.inpated_images = [] # inpaint를 수행한 image 객체를 담는 list
        self.image_file_name = None
        self.path_finder = PathFinder()

    def setImageFileName(self, image_file_name):
        self.image_file_name = image_file_name

    def load_image(self):
        """
        이미지를 로드 및 RGB 변환 시각화
        """

        base_path = None

        if self.image_file_name == "image_1" or self.image_file_name == "image_2":
            base_path = self.path_finder.capture_SM_dir_path
        elif self.image_file_name == "image":
            base_path = self.path_finder.capture_CI_dir_path

        file_path = f"{base_path}/{self.image_file_name}.png"

        self.img = Image.open(file_path)  # 이미지 로드
        self.img = self.img.convert("RGB")  # PNG 이미지를 RGBA에서 RGB로 변환
        self.img = np.array(self.img)  # 이미지를 numpy 배열로 변환

        # 로드된 이미지를 시각화
        # plt.imshow(self.img)
        # plt.show()

    def extract_colors(self):
        """
        이미지 배열을 각 RGB 픽셀 2D 배열로 재구성
        """
        self.color_tbl = self.img.reshape(-1, 3)  # 이미지 배열을 2D RGB 값 배열로 재구성

    def cluster_colors(self):
        """
        DBSCAN을 사용하여 이미지의 색상을 클러스터링하고 클러스터된 색상을 시각화
        """
        # DBSCAN 모델 생성 및 학습
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        dbscan.fit(self.color_tbl)

        # 클러스터 레이블 가져오기
        self.labels = dbscan.labels_

        # 고유 클러스터 레이블 가져오기
        self.unique_labels = set(self.labels)

        # # 클러스터를 위한 색상 팔레트 생성
        # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(self.unique_labels))]

        # # 각 클러스터 색상 플로팅
        # plt.figure(figsize=(6, 6))
        # plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, hspace=0.3, wspace=0.3)
        # plt.suptitle("Clustered Colors", fontsize=10)
        # num_cols = 2  # 한 행에 표시할 색상 수

        # for i, k in enumerate(self.unique_labels):
        #     if k == -1:
        #         continue  # 노이즈 건너뛰기 (레이블 -1)

        #     # 각 클러스터의 픽셀 RGB 값 가져오기
        #     cluster_rgb = self.color_tbl[self.labels == k]

        #     # 대표 색상 계산
        #     representative_color = np.mean(cluster_rgb, axis=0)

        #     # 대표 색상 표시
        #     plt.subplot(len(self.unique_labels) // num_cols + 1, num_cols, i + 1)
        #     cluster_color = np.full((10, 10, 3), representative_color, dtype=int)
        #     plt.imshow(cluster_color)
        #     plt.axis('off')
        #     plt.title(f"Cluster {k} RGB: {representative_color.astype(int)}", fontsize=6)

        # plt.tight_layout()
        # plt.show()

        # # 3D 그래프 생성
        # self.plot_cluster_colors_3d(self.color_tbl, self.labels)

    def plot_cluster_colors_3d(self, colors, labels):
        plt.ion()  # 인터랙티브 모드 켜기
        fig = plt.figure(figsize=(7, 7))  # 그래프 크기 조정
        ax = fig.add_subplot(111, projection='3d')

        # 색상을 더 부드럽게 표현
        normalized_colors = colors / 255

        # 각 클러스터별 점의 개수를 계산하여 텍스트로 표시
        cluster_counts = Counter(labels)
        cluster_info = []
        for k, v in cluster_counts.items():
            if k == -1:
                continue
            cluster_rgb = colors[labels == k]
            mean_rgb = cluster_rgb.mean(axis=0)
            cluster_info.append((k, v, mean_rgb))

        # 점의 크기를 더 키움
        ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2], c=normalized_colors, marker='o', s=10)

        # 노이즈 점 추가
        noise_mask = (labels == -1)
        ax.scatter(colors[noise_mask, 0], colors[noise_mask, 1], colors[noise_mask, 2], c='grey', marker='o', s=3, alpha=0.5, label='Noise')

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue', rotation=90)  # Z 축 레이블을 세로로 설정

        # 그래프의 위치를 조정하여 여백을 확보
        pos = ax.get_position()
        pos.x0 += 0.1  # 좌측 여백 증가
        pos.x1 += 2.0  # 우측 여백 증가
        pos.y0 += 0.1  # 하단 여백 증가
        pos.y1 -= 0.1  # 상단 여백 감소
        ax.set_position(pos)

        # 그래프 여백 조정
        plt.subplots_adjust(left=0.1, right=1.0, top=0.9, bottom=0.1)

        plt.title('RGB Distribution in 3D', fontsize=12)

        # 클러스터 정보 텍스트와 색상 박스 추가
        for i, (k, v, mean_rgb) in enumerate(cluster_info):
            text_x = 0.95
            text_y = 0.95 - (i * 0.05)
            fig.text(text_x, text_y, f'Cluster {k}: {v} points', fontsize=8, ha='left', va='center', transform=fig.transFigure)
            fig.patches.extend([plt.Rectangle((text_x - 0.05, text_y - 0.02), 0.03, 0.03, color=mean_rgb / 255, transform=fig.transFigure, figure=fig)])

        plt.show()

    def calculate_cluster_sizes(self):
        """
        클러스터 별 사이즈를 계산하고, 대표 색상을 추출 및 각 클러스터 시각화
        """
        # 각 클러스터의 픽셀 수를 계산
        cluster_sizes = Counter(self.labels)
        # 노이즈를 제외한 클러스터 크기 계산
        cluster_sizes_without_noise = {label: size for label, size in cluster_sizes.items() if label != -1}

        # 각 클러스터의 대표 색상 추출
        self.cluster_centers = np.array([np.mean(self.color_tbl[self.labels == label], axis=0) for label in set(self.labels)])
        # 레이블을 원래 이미지 형태로 재구성
        cluster_image = self.cluster_centers[self.labels].reshape(self.img.shape).astype(int)

        # 클러스터 크기를 내림차순으로 정렬
        sorted_cluster_labels = sorted(cluster_sizes_without_noise, key=cluster_sizes_without_noise.get, reverse=True)

        # 크기별로 각 클러스터를 시각화
        for label in sorted_cluster_labels:
            color = self.cluster_centers[label].astype(int)
            cluster_size = cluster_sizes_without_noise[label]

            # 클러스터 마스크 생성
            color_mask = np.all(cluster_image == color, axis=-1)
            segmented_image = np.zeros_like(self.img)
            segmented_image[color_mask] = self.img[color_mask]

            # 클러스터 이미지 표시
            # plt.figure(figsize=(5, 5))
            # plt.imshow(segmented_image)
            # plt.title(f"Cluster {label} RGB: {color.astype(int)} - Pixel_Size: {cluster_size}")
            # plt.axis('off')
            # plt.show()

        return cluster_image, sorted_cluster_labels, cluster_sizes_without_noise

    def pixel_grid_images(self, cluster_image, sorted_cluster_labels, cluster_sizes_without_noise):
        """
        클러스터 별로 픽셀을 그리드 이미지로 저장하고 출력
        :param cluster_image: 클러스터링된 이미지 배열
        :param sorted_cluster_labels: 정렬된 클러스터 레이블
        :param cluster_sizes_without_noise: 클러스터 크기 (노이즈 제외)
        """
        grid_size = 100  # 그리드에 표시할 이미지 크기
        output_dir = self.path_finder.capture_SM_clusters_dir_path  # 절대 경로 사용
        
        # cluster_list 디렉토리를 비움
        # if os.path.exists(output_dir):
        #     for file in os.listdir(output_dir):
        #         file_path = os.path.join(output_dir, file)
        #         if os.path.isfile(file_path):
        #             os.unlink(file_path)
        
        #  os.makedirs(output_dir, exist_ok=True)  # 저장 디렉토리 생성

        for label in sorted_cluster_labels:
            cluster_size = cluster_sizes_without_noise[label]  # 클러스터 크기

            # 클러스터 크기가 500개 이하인 경우 건너뜀
            if cluster_size <= 500:
                continue

            color = self.cluster_centers[label].astype(int)  # 클러스터 중심 색상

            # 클러스터 이미지 생성
            color_mask = np.all(cluster_image == color, axis=-1)  # 클러스터 중심 색상과 같은 모든 픽셀 찾기
            cluster_pixels = np.argwhere(color_mask)  # 해당 색상의 모든 픽셀 좌표

            # 벡터화된 방식으로 픽셀 값 추출
            cluster_pixels_values = self.img[cluster_pixels[:, 0], cluster_pixels[:, 1]]

            # 이미지를 채울 픽셀 수 계산
            num_fill_pixels = grid_size * grid_size

            if cluster_size >= num_fill_pixels:
                cluster_pixels_values = cluster_pixels_values[:num_fill_pixels]
            else:
                num_repeats = num_fill_pixels // cluster_size + 1
                repeated_pixels = np.tile(cluster_pixels_values, (num_repeats, 1))
                cluster_pixels_values = repeated_pixels[:num_fill_pixels]

            # 그리드 이미지 생성
            num_cols = grid_size
            num_rows = grid_size
            grid_image = np.zeros((num_rows, num_cols, 3), dtype=int)

            # 클러스터 픽셀 값의 순서를 섞음
            shuffled_indices = np.random.permutation(len(cluster_pixels_values))
            shuffled_pixels_values = cluster_pixels_values[shuffled_indices]

            # 각 픽셀 값을 이미지 그리드에 배치
            for i, pixel_value in enumerate(shuffled_pixels_values):
                row = i // num_cols
                col = i % num_cols
                grid_image[row, col] = pixel_value

            # 그리드 이미지 저장
            file_path = os.path.join(output_dir, f'{self.image_file_name}_{label}.png')
            plt.imsave(file_path, grid_image.astype(np.uint8))

            # 디버깅 출력 추가
            # print(f"Saving cluster grid image {label} to {file_path}")

            # 저장된 이미지 크기 확인
            # saved_image = Image.open(file_path)
            # print(f"Saved image size for cluster {label}: {saved_image.size}")

            # 그리드 이미지 출력 (필요시 활성화)
            # plt.imshow(grid_image)
            # plt.show()