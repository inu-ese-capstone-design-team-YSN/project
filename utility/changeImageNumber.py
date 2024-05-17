import os
import subprocess
from path_finder import PathFinder

path_finder = PathFinder()

start_num = 94
end_num = 265

# 이름이 잘못되어 이름을 수정할 경우,
# 이름에 더해줄 숫자를 지정
add_num = -2

original_image_dir_path = path_finder.tpg_original_dir_path

for i in range(start_num, end_num+1):
    original_number = i
    changed_number = i + add_num

    original_file_name = f"{original_number}.jpg"
    original_file_path = f"{original_image_dir_path}/{original_file_name}"
    print(original_file_path)

    changed_file_name = f"{changed_number}.jpg"
    changed_file_path = f"{original_image_dir_path}/{changed_file_name}"
    print(changed_file_path)

    result = subprocess.run(['mv', f'{original_file_path}', f'{changed_file_path}'], capture_output=True, text=True)