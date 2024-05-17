import subprocess
import argparse
import os
import json
from image_utility import ImageUtility
from path_finder import PathFinder
path_finder = PathFinder()

'''

2024.05.03 kwc
tpg2sw로 입력 인자 두 개를 입력 받아 combined 및 히트맵 이미지까지 생성되도록 변경
두 번째 입력값까지만 동작

'''

# 이미지 처리와 경로 탐색을 위한 유틸리티 클래스 인스턴스 생성
util = ImageUtility()
pf = PathFinder()

# 명령줄 인수 파싱을 설정
"""
    2024.05.16, jdk
    --code 인자를 전달하지 않아도 동작하게 하려면
    default option을 None으로 지정해 주어야 한다.
"""
parser = argparse.ArgumentParser()
parser.add_argument("--code", type=str, required=None, help="Target Code", default=None, nargs='?')
args = parser.parse_args()

target_code = None
if args.code:
    target_code = args.code

extension=".png" # 파일 확장자 지정

comb_type_vetically="v" # 세로 결합 유형 지정
comb_type_horizontally="h" # 가로 결합 유형 지정

with open(pf.image_settings_file_path, 'r') as file:
    image_settings = json.load(file)
    
crop_size = image_settings["crop_size"]

def crop_tpg(tpg_code):
    # TPG Image Crop
    for tpg_index in range(0, 8):
        file_name = f"{tpg_code}_{tpg_index+1}{extension}"
        
        left = crop_size[tpg_index][0]
        upper = crop_size[tpg_index][1]
        right = crop_size[tpg_index][2]
        lower = crop_size[tpg_index][3]

        try:
            print(f"crop {file_name}") 
            result = subprocess.run(['imcrop', '-s', f'{left}', f'{upper}', f'{right}', f'{lower}', '-n', file_name], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(result.stderr)

            print(file_name + " cropping done")
        except Exception as e:
            print(e)
            continue

    print("\n\n")

def combine_tpg(tpg_code):
    # TPG Image Combine
    # Combine Vertically - 1
    for tpg_index in range(0, 8, 2):
        upper_image_name = f"{tpg_code}_{tpg_index+1}_cropped{extension}"
        lower_image_name = f"{tpg_code}_{tpg_index+2}_cropped{extension}"

        combined_image_file_name = f"{tpg_code}_{tpg_index+1}_combined_{tpg_code}_{tpg_index+2}{extension}"

        """
            2024.05.13, jdk
            subprocess 내부에서 에러가 발생하면 result로 나타나는데,
            이것을 체크하지 않으므로 에러가 체크되지 않는다.
        """
        try:
            print(f"combine {upper_image_name} and {lower_image_name} vertically")
            result = subprocess.run(['imcomb', '-u', upper_image_name, '-l', lower_image_name, '-n', combined_image_file_name, '-t', comb_type_vetically], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(result.stderr)

            # Image가 combined에 저장되므로, mv가 필요하다.
            # combined_dir에 있는 이미지를 cropped_dir로 이동한다.
            result = subprocess.run(['mv', f'{pf.tpg_combined_dir_path}/{combined_image_file_name}', f'{pf.tpg_cropped_dir_path}/'], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(result.stderr)

        except Exception as e:
            print(e)
            return

    print("\n\n")

    # Combine Vertically - 2
    for tpg_index in range(0, 8, 4):
        upper_image_name = f"{tpg_code}_{tpg_index+1}_combined_{tpg_code}_{tpg_index+2}{extension}"
        lower_image_name = f"{tpg_code}_{tpg_index+3}_combined_{tpg_code}_{tpg_index+4}{extension}"

        # 위치에 따라 combined_image_file_name을 지정
        if tpg_index == 0:
            combined_image_file_name = f"{tpg_code}_{tpg_index+1}_left{extension}"
        elif tpg_index == 4:
            combined_image_file_name = f"{tpg_code}_{tpg_index+1}_right{extension}"

        try:
            print(f"combine {upper_image_name} and {lower_image_name} vertically")
            result = subprocess.run(['imcomb', '-u', upper_image_name, '-l', lower_image_name, '-n', combined_image_file_name, '-t', comb_type_vetically], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(result.stderr)

        except Exception as e:
            print(e)
            return
            
        try:
            # Image가 combined에 저장되므로, mv가 필요하다.
            # combined_dir에 있는 이미지를 cropped_dir로 이동한다.
            result = subprocess.run(['mv', f'{pf.tpg_combined_dir_path}/{combined_image_file_name}', f'{pf.tpg_cropped_dir_path}/'], capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(result.stderr)

            print(f"{combined_image_file_name} combining done")

        except Exception as e:
            print(e)
            return

    print("\n\n")

    # Combine Horizontally - 3
    left_image_name = f"{tpg_code}_1_left{extension}"
    right_image_name = f"{tpg_code}_5_right{extension}"

    combined_image_file_name = f"{tpg_code}_combined{extension}"

    try:
        print(f"combine {left_image_name} and {right_image_name} horizontally")
        result = subprocess.run(['imcomb', '-u', left_image_name, '-l', right_image_name, '-n', combined_image_file_name, '-t', comb_type_horizontally], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(result.stderr)

        print(f"{left_image_name} and {right_image_name} combining done")
        # 최종적으로 만들어진 이미지가 combined 디렉터리에 저장되었음.
    except Exception as e:
        print(e)
        return

    print("\n\n")

def get_captured_tpg_codes():
    """
        2024.05.13, jdk
        tpg directory 내에 있는 unique한 
        code set을 얻어내고, 해당 set을 반환한다.
    """

    unique_codes = set()

    file_names = os.listdir(pf.tpg_original_dir_path)

    for file_name in file_names:
        code = file_name.split('_')[0]
        unique_codes.add(code)
    
    return unique_codes

def convert_tpg_to_swatch_image(tpg_code):
    crop_tpg(tpg_code)
    combine_tpg(tpg_code)

# ------------------------------------------------------------------ #
# Main

# 1) target code를 지정하지 않은 경우
# 이 경우에는 모든 코드에 대해서 tpg2sw를 동작한다.
if target_code == None:
    # 현재 촬영된 tpg_codes를 모두 얻어냄
    tpg_codes = get_captured_tpg_codes()


    print("TPG Code Set")
    print(tpg_codes)
    print("\n\n")

    for tpg_code in tpg_codes:
        convert_tpg_to_swatch_image(tpg_code)
# 2) target code를 지정한 경우
# 이 경우에는 특정한 코드에 대해서 tpg2sw를 동작한다.
else: 
    print("Target Code")
    print(target_code)
    print("\n\n")

    convert_tpg_to_swatch_image(target_code)

# ------------------------------------------------------------------ #

# # 히트맵 변환 파트
# # convert combined image to heatmap
# combined_image_file_name = f"{base_num}_combined{extension}"

# print(f"{combined_image_file_name}")
# result = subprocess.run(['im2hmap', '-o', combined_image_file_name])
# print(f"{combined_image_file_name} heatmap converting done")

# print("\n\n")