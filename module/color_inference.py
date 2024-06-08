import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import os
from path_finder import PathFinder

# 모델 클래스 정의 (Lab_separate 클래스가 이미 정의되어 있다고 가정)
class Lab_separate(nn.Module):
    def __init__(self):
        super(Lab_separate, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
    
class ColorInference():
    def __init__(self):
        self.path_finder = PathFinder()

        self.model_L = Lab_separate()
        self.model_a = Lab_separate()
        self.model_b = Lab_separate()

        model_L_params_name = 'L.pth'
        model_a_params_name = 'a.pth'
        model_b_params_name = 'b.pth'

        self.model_L.load_state_dict(torch.load(f"{self.path_finder.model_dir_path}/{model_L_params_name}"))
        self.model_a.load_state_dict(torch.load(f'{self.path_finder.model_dir_path}/{model_a_params_name}'))
        self.model_b.load_state_dict(torch.load(f'{self.path_finder.model_dir_path}/{model_b_params_name}'))

        self.model_L.eval()
        self.model_a.eval()
        self.model_b.eval()

        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()
        ])

        self.cluster_images = []
        self.cluster_images_name = []
        self.inferenced_Lab = []

    def load_cluster_images(self):
        """
            SM_clusters 디렉터리에서 현재 저장되어 있는
            모든 이미지를 로드하여 list에 저장하는 함수
        """

        self.cluster_images.clear()
        self.cluster_images_name.clear()

        for filename in os.listdir(self.path_finder.capture_SM_clusters_dir_path):
            # PNG 파일만 처리
            if filename.lower().endswith('.png'):
                file_path = os.path.join(self.path_finder.capture_SM_clusters_dir_path, filename)

                # 이미지 로드 및 리스트에 추가
                image = Image.open(file_path)
                self.cluster_images.append(image)
                self.cluster_images_name.append(filename)

    def inference_colors(self):

        # 이전 추론 결과 비우기
        self.inferenced_Lab.clear()
        
        # cluster image 개수 만큼 추론 실행
        num_images = len(self.cluster_images)
        for image_index in range(num_images):
            image = self.cluster_images[image_index]
            image = image.convert('RGB')

            image_tensor = self.transform(image).unsqueeze(0)

            # 모델 추론
            with torch.no_grad():
                L_output = self.model_L(image_tensor)
                a_output = self.model_a(image_tensor)
                b_output = self.model_b(image_tensor)

            # 결과 출력
            L_output = L_output.item() * 100
            a_output = a_output.item() * 255 - 128
            b_output = b_output.item() * 255 - 128

            self.inferenced_Lab.append([L_output, a_output, b_output])
        
        return (self.cluster_images_name, self.inferenced_Lab)



            