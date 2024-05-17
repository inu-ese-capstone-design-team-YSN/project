from path_finder import PathFinder
from PIL import Image
from pathlib import Path
#전체 이미지 크기 4056x3040
# PathFinder 인스턴스 생성
path_finder = PathFinder()
# 경로를 Path 객체로 변환
original_image_dir = Path(path_finder.tcx_image_dir_path)

# 원본 이미지 파일 경로
input_image_path = original_image_dir / 'test.jpg'

# 크롭된 이미지를 저장할 파일 경로
# output_image_path = original_image_dir / 'cropped_target_test.jpg'
# output_image_path = original_image_dir / 'cropped_left_test.jpg'
# output_image_path = original_image_dir / 'cropped_right_test.jpg'

# 크롭할 영역의 좌표 (left, upper, right, lower)
# crop_area = (725, 220, 3325, 2830) #target
# crop_area = (0, 0, 350,3040) #left
# crop_area = (3680, 0, 4056, 3040) #right


# 이미지 열기
with Image.open(input_image_path) as img:
    # 이미지 크롭
    cropped_img = img.crop(crop_area)
    
    # 크롭된 이미지 저장
    cropped_img.save(output_image_path)

print(f"Cropped image saved to {output_image_path}")
