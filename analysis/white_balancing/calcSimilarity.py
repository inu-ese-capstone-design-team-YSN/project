import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from image_utility import ImageUtility

image_utility = ImageUtility()

rgb1 = (120, 100, 81)
rgb2 = (120, 100, 80)

similarity = image_utility.calculateLABSimilarity(rgb1, rgb2)
print(f"LAB 색상 유사도 (Delta E): {similarity}")
similarity = int(similarity)
similarity = 100 - similarity
print(similarity)

distance = image_utility.calculateRGBDistance(rgb1, rgb2)
print(distance)

distance= image_utility.calculateChannelIndependentRGBDistance(rgb1, rgb2)
print(distance)