import os
import tarfile
from path_finder import PathFinder

pf = PathFinder()
tpg_original_image_dir_path = pf.tpg_original_directory_path
tpg_compressed_dir_path = pf.tpg_compressed_dir_path

tpg_lowest_hue = 0
tpg_highest_hue = 64

def compress_tpg_images():
    files = os.listdir(tpg_original_image_dir_path)
    
    # 00부터 64까지의 숫자 문자열 리스트 생성
    # {i:02}와같이 사용하면 한 글자 숫자를 prefix 0을 붙여 가능
    hue_list = [f"{i:02}" for i in range(51, 65)]
    
    for hue in hue_list:
        matching_files = [f for f in files if f[3:5] == hue]
        
        print(hue)
        print(len(matching_files))
        print("\n\n")

        if len(matching_files) == 0:
            continue

        tar_filename = f'tpg_{hue}.tar.gz'
        tar_path = f'{tpg_compressed_dir_path}/{tar_filename}'
        
        with tarfile.open(tar_path, 'w:gz') as tar:
            for file in matching_files:
                file_path = f"{tpg_original_image_dir_path}/{file}"
                tar.add(file_path, arcname=os.path.basename(file_path))
        
# 스크립트를 직접 실행할 때 수행되는 부분
# 모듈로서 임포트 될 때는 수행되지 않는다.
if __name__ == "__main__":
    compress_tpg_images()