from image_correction import ImageCorrection
from clustering import Clustering
from color_inference import ColorInference
from calculate_deltaE import ColorSimilarityCalculator

#########################################################################################################

# 1) 이미지 보정
# 2) 클러스터링
# 3) 색상 추론
# 4) 유사도 계산

#Lab_1,RGB_1,Lab_2,RGB_2,Lab_distance
#[30.00 10.01 -10.00],[255 0 0],#FFFFFF,[100 0 0 1],[0.00 10.00 -10.00],[0 255 0],#AAAAAA,[100 0 0 2],11.11
#[30.00 10.00 -10.00],[0 255 0],#FFFFFF,[100 0 0 3],[10.00 10.00 -10.00],[0 0 255],#AAAAAA,[100 0 0 4],22.11
#[30.00 10.00 -10.00],[0 0 255],#FFFFFF,[100 0 0 5],[20.00 10.00 -10.00],[255 0 0],#AAAAAA,[100 0 0 6],33.11
# 가게 요청사항: 이런식으로 csv파일에 저장되게 해주세요 (row의 개수는 다색상원단에서 색상의 개수)

#########################################################################################################
# 1) 이미지 보정

image_correction = ImageCorrection()

print("Started Image Correction...")
for i in range(1, 3):
    # 이미지 1번과 2번에 대해서 동작

    # i번 이미지에 대한 보정 수행
    image_correction.setImageFileName(f"image_{i}")
    image_correction.correctImage()
print("Image Correction is Done...\n\n")

# 보정된 이미지는 capture_original_dir_1, capture_original_dir_2에 저장되어 있음.


#########################################################################################################
# 2) 클러스터링(DBSCAN)

clustering = Clustering()

print("Started Image Clustering...")
for i in range(1, 3):
    image_file_name = f"image_{i}"

    image_cluster = Clustering()

    # 이미지 로드
    image_cluster.setImageFileName(image_file_name)

    image_cluster.load_image()

    # 이미지에서 색상 추출
    image_cluster.extract_colors()

    # 색상을 클러스터링하고 클러스터 표시
    image_cluster.cluster_colors()  

    # 클러스터 크기 계산 및 각 클러스터 표시
    cluster_image, sorted_cluster_labels, cluster_sizes_without_noise = image_cluster.calculate_cluster_sizes()  

    # 픽셀 처리 그리드 이미지 변환
    image_cluster.pixel_grid_images(cluster_image, sorted_cluster_labels, cluster_sizes_without_noise)
print("Clustering is Done...\n\n")

#########################################################################################################
# 3) 색상 추론(CI)

print("Started Color Inference...")
color_inference = ColorInference()
color_inference.load_cluster_images()
cluster_images_name, inferenced_Lab = color_inference.inference_colors()
print("Color Inference is done...\n\n")

#########################################################################################################
# 4) 유사도 계산

print("Started Similarity Calculation...")
color_similarity_calculator = ColorSimilarityCalculator(cluster_images_name, inferenced_Lab)
color_similarity_calculator.calculate_deltaE()
color_similarity_calculator.saveResult()
print("Similarity Calculation is Done...")

#########################################################################################################





