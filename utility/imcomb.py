from PIL import Image
import argparse

from image_utility import ImageUtility
from path_finder import PathFinder

# create utility instances
util = ImageUtility()
pf = PathFinder()

# 이미지 파일 이름
upper_image_file_name = "example"
lower_image_file_name = "example"
combined_image_file_name = "example"

upper_image_extension = '.jpg'
lower_image_extension = '.jpg'
combined_image_extension = '.jpg'

# combine type
comb_type_vertically='v'
comb_type_horizontally='h'

# combine default value
comb_type=comb_type_vertically

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--u", type=str, required=False)
parser.add_argument("--l", type=str, required=False)
parser.add_argument("--n", type=str, required=False)
parser.add_argument("--t", type=str, required=False)

# argument parsing
args = parser.parse_args()

if args.u:
    u = args.u.split('.')
    upper_image_file_name = u[0]
    upper_image_extension=f".{u[1]}"

if args.l:
    l = args.l.split('.')
    lower_image_file_name = l[0]
    lower_image_extension=f".{l[1]}"

if args.n:
    n = args.n.split('.')
    combined_image_file_name = n[0]
    combined_image_extension=f".{n[1]}"

if args.t:
    t = args.t
    comb_type = t
    
# 2024.04.17, jdk
# combine할 이미지는 반드시 cropped 디렉터리에 존재하므로, 이름만 전달받는다.
# 그리고 comb된 이미지는 역시 /project/image/swatch_image/combined에 저장된다.

first_image_file_path=f"{pf.tpg_cropped_dir_path}/{upper_image_file_name}{upper_image_extension}"
second_image_file_path=f"{pf.tpg_cropped_dir_path}/{lower_image_file_name}{lower_image_extension}"
combined_image_file_path=f"{pf.tpg_combined_directory_path}/{combined_image_file_name}{combined_image_extension}"

if comb_type == comb_type_vertically:
    util.combineImagesVertically(first_image_file_path, second_image_file_path, combined_image_file_path)
elif comb_type == comb_type_horizontally:
    util.combineImagesHorizontally(first_image_file_path, second_image_file_path, combined_image_file_path)
