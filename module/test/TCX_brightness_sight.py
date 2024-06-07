import cv2
from PIL import Image
import numpy as np

def extract_pixel_values(image_path, coordinates):
    # Read the image
    img = cv2.imread(image_path)
    image = Image.open(image_path)
    
    image_array = np.array(image, dtype=np.int16)
    height, width, _ = image_array.shape
    
    if height != 100 or width != 100:
        print(f"Image: {image_path} is not 100x100 pixels.")
        return
    
     # RGB 평균을 구하고 LAB으로 변환
    for coord in coordinates:
        x, y = coord
        region = image_array[max(0, y-9):y+1, max(0, x-9):x+1]
        avg_rgb = np.mean(region, axis=(0, 1)).astype(int)
        print(f"Pixel ({x}, {y}) RGB: ({avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]})")


    print("\n")
     # RGB 평균을 구하고 LAB으로 변환
    for coord in coordinates:
        x, y = coord
        region = image_array[max(0, y-9):y+1, max(0, x-9):x+1]
        avg_rgb = np.mean(region, axis=(0, 1)).astype(int)
        avg_lab = cv2.cvtColor(np.uint8([[avg_rgb]]), cv2.COLOR_RGB2Lab)[0][0]
        print(f"Pixel ({x}, {y}) LAB: ({avg_lab[0]}, {avg_lab[1]}, {avg_lab[2]})")

def rgb_to_lab(rgb):
    # 입력된 RGB 값을 단일 픽셀 이미지로 변환
    rgb_image = np.uint8([[rgb]])
    # RGB에서 LAB 색공간으로 변환
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)
    # LAB 값 추출
    l, a, b = lab_image[0, 0]
    print(f"L: {l}, a: {a}, b: {b}")


if __name__ == "__main__":
    filename = "17-4041_AC_adjusted_combined.png"
    pixel_coordinates = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (70, 70), (80, 80), (90, 90), (100, 100)]
    # pixel_coordinates = [ (30, 30), (40, 40), (50, 50), (60, 60), (70, 70), (80, 80)]
    # extract_pixel_values(filename, pixel_coordinates)
    rgb_to_lab((91, 137, 203))
    

