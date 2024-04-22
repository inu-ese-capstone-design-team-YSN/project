import subprocess
import argparse

from image_utility import ImageUtility
from path_finder import PathFinder

# create utility instances
util = ImageUtility()
pf = PathFinder()

# 이미지 파일 시작 이름
base_num=-1

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--base", type=int, required=False)

args = parser.parse_args()

extension=".jpg"

comb_type_vetically="v"
comb_type_horizontally="h"

# Left and Right
lr_1 = 700
lr_2 = 1950
lr_3 = 3200

# Upper and Lower
ul_1 = 260
ul_2 = 885
ul_3 = 1510
ul_4 = 2135
ul_5 = 2760

# TPG를 크롭할 사이즈 지정
# TPG의 순서는 다음과 같이 되어 있다고 상정한다.
# 1 5
# 2 6
# 3 7
# 4 8
crop_size = [
    (lr_1, ul_1, lr_2, ul_2),
    (lr_1, ul_2, lr_2, ul_3),
    (lr_1, ul_3, lr_2, ul_4),
    (lr_1, ul_4+100, lr_2, ul_5+100),

    (lr_2, ul_1, lr_3, ul_2),
    (lr_2, ul_2+30, lr_3, ul_3+30),
    (lr_2, ul_3+70, lr_3, ul_4+70),
    (lr_2, ul_4+80, lr_3, ul_5+80)
]

# 시작 번호 지정
if args.base:
    base_num=args.base

# TPG Image Crop
for tpg_index in range(0, 8):
    file_name = f"{base_num+tpg_index}{extension}"
    
    left = crop_size[tpg_index][0]
    upper = crop_size[tpg_index][1]
    right = crop_size[tpg_index][2]
    lower = crop_size[tpg_index][3]

    print(f"crop {file_name}")
    result = subprocess.run(['imcrop', '-s', f'{left}', f'{upper}', f'{right}', f'{lower}', '-n', file_name], capture_output=True, text=True)
    print(file_name + " cropping done")

print("\n\n")

# TPG Image Combine
# Combine Vertically - 1
for tpg_index in range(0, 8, 2):
    upper_image_num = f"{base_num+tpg_index}"
    lower_image_num = f"{base_num+tpg_index+1}"

    upper_image_name = f"{base_num+tpg_index}_cropped{extension}"
    lower_image_name = f"{base_num+tpg_index+1}_cropped{extension}"

    combined_image_file_name = f"{base_num}_combined_{tpg_index}{extension}"

    print(f"combine {upper_image_name} and {lower_image_name} vertically")
    result = subprocess.run(['imcomb', '-u', upper_image_name, '-l', lower_image_name, '-n', combined_image_file_name, '-t', comb_type_vetically], capture_output=True, text=True)
    print(f"{upper_image_name} and {lower_image_name} combining done")
    
    # Image가 combined에 저장되므로, mv가 필요하다.
    # combined_dir에 있는 이미지를 cropped_dir로 이동한다.
    result = subprocess.run(['mv', f'{pf.swatch_combined_dir_path}/{combined_image_file_name}', f'{pf.swatch_cropped_dir_path}/'], capture_output=True, text=True)

print("\n\n")

# Combine Vertically - 2
for tpg_index in range(0, 8, 4):
    upper_image_name = f"{base_num}_combined_{tpg_index}{extension}"
    lower_image_name = f"{base_num}_combined_{tpg_index+2}{extension}"

    # 위치에 따라 combined_image_file_name을 지정
    if tpg_index == 0:
        combined_image_file_name = f"{base_num}_left{extension}"
    elif tpg_index == 4:
        combined_image_file_name = f"{base_num}_right{extension}"

    print(f"combine {upper_image_name} and {lower_image_name} vertically")
    result = subprocess.run(['imcomb', '-u', upper_image_name, '-l', lower_image_name, '-n', combined_image_file_name, '-t', comb_type_vetically], capture_output=True, text=True)
    print(f"{upper_image_name} and {lower_image_name} combining done")
    
    # Image가 combined에 저장되므로, mv가 필요하다.
    # combined_dir에 있는 이미지를 cropped_dir로 이동한다.
    result = subprocess.run(['mv', f'{pf.swatch_combined_dir_path}/{combined_image_file_name}', f'{pf.swatch_cropped_dir_path}/'], capture_output=True, text=True)

print("\n\n")

# Combine Horizontally - 3
left_image_name = f"{base_num}_left{extension}"
right_image_name = f"{base_num}_right{extension}"

combined_image_file_name = f"{base_num}_combined{extension}"

print(f"combine {left_image_name} and {right_image_name} horizontally")
result = subprocess.run(['imcomb', '-u', left_image_name, '-l', right_image_name, '-n', combined_image_file_name, '-t', comb_type_horizontally], capture_output=True, text=True)
print(f"{left_image_name} and {right_image_name} combining done")
# 최종적으로 만들어진 이미지가 combined 디렉터리에 저장되었음.

print("\n\n")

# convert combined image to heatmap
combined_image_file_name = f"{base_num}_combined{extension}"

print(f"{combined_image_file_name}")
result = subprocess.run(['im2hmap', '-o', combined_image_file_name])
print(f"{combined_image_file_name} heatmap converting done")

print("\n\n")