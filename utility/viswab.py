"""
    촬영한 원단을 잘라 이어붙여 하나의 원단 이미지로 만들고,
    Heatmap Visualization을 진행하는 프로그램
"""

# import python modules
import os
import subprocess
import argparse

# import custom modules
from image_utility import ImageUtility
from path_finder import PathFinder
from cloud_controller import CloudController

# 20240.04.19, jdk
# swatch의 이름을 숫자로 넘버링하므로,
# 변환을 시작할 숫자와 마지막 숫자를 입력한다.
# viswab 프로그램을 bash command로 실행했을 때
# argument를 주지 않는다면 코드에서 hard coding 된
# default value를 바탕으로 변환이 시작된다.

file_num_start = 208
file_num_end = 217
combine_extension = ".jpg"

# create argument parser
# argument로 crop 인자를 전달받음.
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, required=False)
parser.add_argument("--end", type=int, required=False)

# argument parsing
args = parser.parse_args()

if args.start:
    file_num_start = args.start

if args.end:
    file_num_end = args.end

util = ImageUtility()
path_finder = PathFinder()
cloud_controller = CloudController(path_finder=path_finder)

original_image_dir = path_finder.tpg_original_directory_path
cropped_image_dir = path_finder.tpg_cropped_dir_path
combined_image_dir = path_finder.tpg_combined_directory_path
heatmap_image_dir = path_finder.tpg_heatmap_dir_path

# image crop 영역 설정
# 정사각형을 기준으로 자르고자 하므로,
# 자르는 영역 또한 정사각형이 되도록 설정해야 한다.

# Orig. 4056x3040
# Cropped. 2500x2500

# Left, Upper, Right, Lower
crop_size = (700, 260, 3200, 2760)

# 촬영된 모든 이미지에 대해서 imcrop 실행
for file_num in range(file_num_start, file_num_end+1):
    file_name = f"{file_num}.jpg"
    
    print(file_name)
    result = subprocess.run(['imcrop', '-s', f'{crop_size[0]}', f'{crop_size[1]}', f'{crop_size[2]}', f'{crop_size[3]}', '-n', file_name], capture_output=True, text=True)
    print(file_name + " cropping done")

print("\n")

# # crop 된 모든 이미지에 대해서 imcomb 실행
# # 이때, 짝수는 upper image이고, 홀수는 lower image이다.
for file_num in range(file_num_start, file_num_end+1, 2):
    upper_file_num = file_num
    lower_file_num = file_num+1

    upper_file_name = f"{upper_file_num}_cropped{combine_extension}"
    lower_file_name = f"{lower_file_num}_cropped{combine_extension}"
 
    combined_image_file_name = f"{upper_file_num}_{lower_file_num}{combine_extension}"

    print(f"{upper_file_num}, {lower_file_num}")
    result = subprocess.run(['imcomb', '-u', upper_file_name, '-l', lower_file_name, '-n', combined_image_file_name], capture_output=True, text=True)
    print(f"{upper_file_num}, {lower_file_num} combining done")

# print("\n")

# # combine 된 이미지에 대해서 heatmap 변환 실행
for file_num in range(file_num_start, file_num_end+1, 2):
    upper_file_num = file_num
    lower_file_num = file_num+1

    combined_image_file_name = f"{upper_file_num}_{lower_file_num}{combine_extension}"

    print(f"{combined_image_file_name}")
    result = subprocess.run(['im2hmap', '-o', combined_image_file_name])
    print(f"{combined_image_file_name} heatmap converting done")

# upload to google cloud service
# combine 된 이미지에 대해서 heatmap 변환 실행
for file_num in range(file_num_start, file_num_end+1, 2):
    upper_file_num = file_num
    lower_file_num = file_num+1

    heatmap_image_file_name = f"{upper_file_num}_{lower_file_num}-hmap{combine_extension}"

    print(f"{heatmap_image_file_name}")
    
    cloud_controller.upload_file(heatmap_image_dir + f"/{heatmap_image_file_name}", heatmap_image_file_name, 1)

    print(f"{heatmap_image_file_name} upload done")


