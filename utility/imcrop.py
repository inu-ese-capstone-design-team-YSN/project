import argparse

# import custom modules
from image_utility import ImageUtility
from path_finder import PathFinder

# create utility instances
util = ImageUtility()
pf = PathFinder()

# 파일 동작에 필요한 경로 설정 변수들 선언
image_file_name="example"
extension=".jpg"

crop_size = (0, 0, 0, 0)

# create argument parser
# argument로 crop 인자를 전달받음.
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, nargs=4, required=False)
parser.add_argument("--name", type=str, required=False)

# argument parsing
args = parser.parse_args()

if args.size:
    size = args.size
    crop_size = (size[0], size[1], size[2], size[3])

if args.name:
    file_name = args.name.split('.')
    image_file_name = file_name[0]
    extension=f".{file_name[1]}"

# 2024.04.17, jdk
# 이미지의 촬영은 반드시 'capture.py'를 거치므로,
# 촬영된 이미지는 /project/image/swatch_image/original에 저장된다.
# 따라서, 별도의 path는 지정하지 않고 file의 이름만 argument로 전달받는다.
# 그리고 crop된 이미지는 역시 /project/image/swatch_image/cropped에 저장된다.
image_file_path=f"{pf.tpg_original_dir_path}/{image_file_name}{extension}"
cropped_image_path=f"{pf.tpg_cropped_dir_path}/{image_file_name}_cropped{extension}"

util.setImagePathAndOpen(image_file_path)

im_size_before_crop = util.getImageSize()
util.cropImage(crop_size[0], crop_size[1], crop_size[2], crop_size[3])
im_size_after_crop = util.getImageSize()

print(f"Image size: {im_size_before_crop[0]}, {im_size_before_crop[1]} ->  \
{im_size_after_crop[0]}, {im_size_after_crop[1]}")

try:
    util.saveImage(cropped_image_path)
    print(f"Successfully saved {image_file_path}")
    print(f"Cropped image name: {cropped_image_path}")
except:
    print(f"Failed to save {cropped_image_path}")