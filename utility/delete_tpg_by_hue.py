import os
import re
from path_finder import PathFinder

pf = PathFinder()

def delete_files_by_yy(directory, hue):
    # 정규식 패턴 정의
    # 디렉토리의 파일 목록 가져오기
    files = os.listdir(directory)
    
    count = 0

    # 파일 삭제
    for file in files:
        if file[3:5] == hue:
            file_path = os.path.join(directory, file)
            print(file_path)
            count += 1
            
            # os.remove(file_path)
            print(f"{file}")
    
    print(f"len: {len(files)}")
    print(f"count: {count}")

if __name__ == "__main__":
    # 예시: '/path/to/directory'에서 '05' 값을 가진 파일 삭제
    delete_files_by_yy(pf.tpg_original_dir_path, '15')