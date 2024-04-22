from PIL import Image
import numpy as np
import argparse

from path_finder import PathFinder
from image_utility import ImageUtility

# 유틸리티 인스턴스 생성
pf = PathFinder()
util = ImageUtility()

# 이미지 이름 및 경로 설정 변수
image_file_name = "example"
extension = ".jpg"

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--o", type=str, required=False)

# parse the argument
args = parser.parse_args()

if args.o:
    o = args.o.split('.')
    image_file_name = o[0]
    extension = f'.{o[1]}'

# TODO 2024.04.18, jdk
# 현재는 image_path가 /combined로 지정되어 있으므로,
# 추후에 다른 디렉터리의 이미지에 적용하려면 변경해야 함.
image_path = f"{pf.swatch_combined_dir_path}/{image_file_name}{extension}"
hmap_path = f"{pf.swatch_heatmap_dir_path}/{image_file_name}-hmap{extension}"

util.setImagePathAndOpen(image_path)
util.calcImageBrightness()
util.getHeatmapSeaborn(hmap_path)