import subprocess
import argparse

from image_utility import ImageUtility
from path_finder import PathFinder

'''

2024.05.03 kwc
tpg2sw로 입력 인자 두 개를 입력 받아 combined 및 히트맵 이미지까지 생성되도록 변경
두 번째 입력값까지만 동작

'''

# 이미지 처리와 경로 탐색을 위한 유틸리티 클래스 인스턴스 생성
util = ImageUtility()
pf = PathFinder()

# 명령줄 인수 파싱을 설정
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, required=True, help="Starting number")
parser.add_argument("--end", type=int, required=True, help="Ending number")

args = parser.parse_args()

extension=".jpg" # 파일 확장자 지정

comb_type_vetically="v" # 세로 결합 유형 지정
comb_type_horizontally="h" # 가로 결합 유형 지정

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
    (lr_1, ul_4, lr_2, ul_5),

    (lr_2, ul_1, lr_3, ul_2),
    (lr_2, ul_2, lr_3, ul_3),
    (lr_2, ul_3, lr_3, ul_4),
    (lr_2, ul_4, lr_3, ul_5)
]

# 시작 번호 지정
if args.start:
    base_num = args.start

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
        result = subprocess.run(['mv', f'{pf.swatch_combined_directory_path}/{combined_image_file_name}', f'{pf.swatch_cropped_dir_path}/'], capture_output=True, text=True)

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
        result = subprocess.run(['mv', f'{pf.swatch_combined_directory_path}/{combined_image_file_name}', f'{pf.swatch_cropped_dir_path}/'], capture_output=True, text=True)

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

    # 시작 번호 업데이트
    base_num += 8
