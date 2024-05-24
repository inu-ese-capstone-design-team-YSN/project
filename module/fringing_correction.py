import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import json
import copy
from dataclasses import dataclass
import cv2
from scipy.stats import kurtosis

from path_finder import PathFinder

@dataclass(frozen=False)
class ImageCorrectionSetting:
    # fringing_row_margin과 fringing_col_margin은
    # fringing_row가 1개 이상으로 잡혔을 때, 잡히지 않았을 수도 있는
    # 약간의 fringing_row를 보정하기 위하여 fringing_rows의 범위를
    # 넓히기 위해 적용하는 margin 값이다.
    fringing_row_margin: int
    fringing_col_margin: int

    # non_fringing_row_margin과 non_fringing_col_margin은
    # fringing_rows가 잡히지 않았더라도 혹시 존재할 수 있기 때문에
    # 기본적으로 어느 정도의 pixel은 보정해 주기 위한 margin 값이다.
    non_fringing_row_margin: int
    non_fringing_col_margin: int

    # upper_pruning_index_value와 lower_pruning_index_value는
    # 중앙을 기준으로 나타나지 않은 fringing_rows가 아니면
    # fringing_rows라고 판단하기 어려우므로, 해당 부분을 제거하기 위해
    # 이미지 height의 중앙에 위치한 두 픽셀의 index를 기록하는 변수이다.
    upper_pruning_index_value: int
    lower_pruning_index_value: int
    col_upper_pruning_index_value: int
    col_lower_pruning_index_value: int

    brightness_margin: int

    shuffle_margin: int
    random_shuffle_kernel_size: int
    random_shuffle_stride_size: int

    col_shuffle_margin: int
    col_random_shuffle_kernel_size: int
    col_random_shuffle_stride_size: int

    gaussian_smoothing_kernel_size: int
    gaussian_smoothing_stride_size: int

# --------------------------------------------------------------------------- #

class FringingCorrection():
    def __init__(self):
        self.pf = PathFinder()

    def getImageArraySize(self, image_setting):
        crop_size = image_setting['crop_size']

        height_1 = crop_size[0][3] - crop_size[0][1]
        height_2 = crop_size[1][3] - crop_size[1][1]
        height_3 = crop_size[2][3] - crop_size[2][1]
        height_4 = crop_size[3][3] - crop_size[3][1]

        width = crop_size[4][2] - crop_size[0][0]
        height = height_1 + height_2 + height_3 + height_4

        row_margin = image_setting['row_margin']
        col_margin = image_setting['col_margin']

        """
            2024.05.16, jdk
            보정할 기준 pixel 설정
            1) row_joint_surface_1: area_1
            2) row_joint_surface_2: area_2
            3) row_joint_surface_3: area_3
            4) col_joint_surface: area_4

            해당 joint_surface들의 pixel을 area_size로 지정한다.
        """
        row_joint_surface_1 = height_1
        area_1_start = row_joint_surface_1 - row_margin
        area_1_end = row_joint_surface_1 + row_margin
        # print(f"row_joint_surface_1: {row_joint_surface_1}")

        row_joint_surface_2 = height_1 + height_2
        area_2_start = row_joint_surface_2 - row_margin
        area_2_end = row_joint_surface_2 + row_margin
        # print(f"row_joint_surface_2: {row_joint_surface_2}")

        row_joint_surface_3 = height_1 + height_2 + height_3
        area_3_start = row_joint_surface_3 - row_margin
        area_3_end = row_joint_surface_3 + row_margin
        # print(f"row_joint_surface_3: {row_joint_surface_3}")

        col_joint_surface = int(width / 2)
        area_4_start = col_joint_surface - col_margin
        area_4_end = col_joint_surface + col_margin

        area_size = [
            [area_1_start, area_1_end, 0, width],
            [area_2_start, area_2_end, 0, width],
            [area_3_start, area_3_end, 0, width],
            [0, height, area_4_start, area_4_end]
        ]

        return area_size

    def getColumnFringingAreaArray(self, image, area_size, image_height):
        image_array = np.array(image)

        column_fringing_area_array = image_array[
            0:image_height, 
            area_size[3][2]:area_size[3][3],
            :
        ]

        return column_fringing_area_array

    def changeCorrectedFringingAreaOfOriginalImage(self, image, area_size, image_height, column_fringing_area_array):
        image_array = np.array(image)

        image_array[
            0:image_height, 
            area_size[3][2]:area_size[3][3],
            :
        ] = column_fringing_area_array

        return Image.fromarray(image_array)

    def getAreaArrayList(self, image, area_size):
        image_array = np.array(image)

        area_1_array = image_array[
            area_size[0][0]:area_size[0][1], 
            area_size[0][2]:area_size[0][3],
            :
        ]

        area_2_array = image_array[
            area_size[1][0]:area_size[1][1], 
            area_size[1][2]:area_size[1][3],
            :
        ]

        area_3_array = image_array[
            area_size[2][0]:area_size[2][1], 
            area_size[2][2]:area_size[2][3],
            :
        ]

        area_array_list = [
            area_1_array,
            area_2_array,
            area_3_array,
        ]

        return area_array_list

    def makeNewImageFromCorrectedArrays(self, image, area_array_list, area_size):
        image_array = np.array(image)
        
        image_array[
            area_size[0][0]:area_size[0][1], 
            area_size[0][2]:area_size[0][3],
            :
        ] = area_array_list[0]

        image_array[
            area_size[1][0]:area_size[1][1], 
            area_size[1][2]:area_size[1][3],
            :
        ] = area_array_list[1]

        image_array[
            area_size[2][0]:area_size[2][1], 
            area_size[2][2]:area_size[2][3],
            :
        ] = area_array_list[2]

        return Image.fromarray(image_array)

    def calcRowsBlueMean(self, area_array):
        """
            2024.05.16, jdk
            전달받은 array의 모든 row에 대해서
            B값 평균을 계산하고 이를 반환한다.
            반드시 3차원 array가 전달되어야 한다.
        """

        area_array_rows_B_mean = area_array[:, :, 2].mean(axis=1, keepdims=True)
        # print(f"mean shape: {area_array_rows_B_mean.shape}")
        return area_array_rows_B_mean

    def getFringingRowsAnomalyPoint(self, rows_mean):
        """
            2024.05.14, jdk
            전달받은 data의 BoxPlot Anomaly Point를 계산하는 함수
            B값이 강한 부분만 찾게 되므로, whishi만 반환한다.
        """

        # boxplot 통계량 계산
        # whis가 너무 작으면 오류가 발생함. 1.5 가량으로 낮춰준다.
        stats = boxplot_stats(rows_mean, whis=1.5)
        plt.close()  # plot을 그리지 않도록 닫기
        
        # 이상치 추출
        anomaly_high = stats[0]['whishi']
        return anomaly_high

    def getOuterFringingPixelsAnomalyPoint(self, outer_calc_boundary_pixels):
        """
            2024.05.16, jdk
            upper/lower boundary pixel을 바탕으로
            B값에 대한 anomaly point를 구하여 반환하는 함수
        """

        blue_pixels = outer_calc_boundary_pixels[:, :, 2]
        col_len = blue_pixels.shape[1]
        anomaly_points = [] # 각 column에 대한 anomaly point를 추가하는 함수

        for i in range(col_len):
            # boxplot 통계량 계산
            # whis=3.0으로 설정하여, 이미지 상에서 매우 이상한 값만 잡아낸다.
            
            stats = boxplot_stats(blue_pixels[:, i], whis=3.0)
            plt.close()  # plot을 그리지 않도록 닫기

            # 이상치 추출
            anomaly_high = stats[0]['whishi']
            # print(f"anomaly_high: {anomaly_high}")
            anomaly_points.append(anomaly_high)

        return anomaly_points

    def getFringingRowsIdentifier(self, array_blue_mean, anomaly_high):
        """
            2024.05.14, jdk
            Anomaly Row의 index를 얻기 위하여 
            True False flatten array를 반환하는 함수
        """

        fringing_rows = array_blue_mean > anomaly_high
        return fringing_rows

    def getFringingRowsIndicies(self, fringing_rows_identifier, image_correction_setting):
        # anomaly_rows의 index를 반환한다.
        # 이때, anomaly_rows_indicies[0]에 
        # 1차원 array가 들어가게 되므로 주의해야 함.
        fringing_rows_indicies = np.where(fringing_rows_identifier)[0]
        # print(f"indicies: {fringing_rows_indicies}")
        # print(f"\n(getFringingRowsIndicies)")
        # print(f"fringing_rows_indicies: {fringing_rows_indicies}")

        """
            2024.05.19, jdk
            반드시 height가 640이므로, 319 또는 320을 기준으로 
            1 간격보다 크게 떨어져서 나타나는 것이 있다면 fringing rows에서 배제해야 한다.
        """
        
        # area_1 ~ area_3에 대한 보정의 경우
        # upper: 319, lower: 320
        
        # area_4에 대한 보정의 경우
        # upper: 1249, lower: 1250

        upper_pruning_index_value = image_correction_setting.upper_pruning_index_value
        lower_pruning_index_value = image_correction_setting.lower_pruning_index_value
        # print(f"upper_pruning_index_value: {upper_pruning_index_value}")
        # print(f"lower_pruning_index_value: {lower_pruning_index_value}")
        # print(f"fringing_rows_indicies: {fringing_rows_indicies}")

        # 1) 길이가 2 이상인 경우
        if len(fringing_rows_indicies) >= 2:
            # 길이가 2 이상임에도 319와 320이 모두 들어있지 않은 경우 (잘못된 경우)
            # 현재 촬영 구조상 절대로 둘 중 하나도 포함되지 않을 수가 없음.
            if upper_pruning_index_value not in fringing_rows_indicies and lower_pruning_index_value not in fringing_rows_indicies:
                # print(fringing_rows_indicies)
                # print("Something went wrong on choosing checking fringing_rows_indicies...")
                
                # fringing_rows_indicies를 빈 list로 반환하여
                # 의미있는 보정이 수행되지 못하게 한다.
                fringing_rows_indicies = []

            # 319와 320이 둘 다 들어있는 경우 (정상)
            elif upper_pruning_index_value in fringing_rows_indicies and lower_pruning_index_value in fringing_rows_indicies:
                # print(f"fringing_rows_indicies-1: {fringing_rows_indicies}")

                lower_pruning_index = np.where(fringing_rows_indicies == upper_pruning_index_value)[0][0]
                # print(f"lower_pruning_index: {lower_pruning_index}")
                upper_pruning_index = np.where(fringing_rows_indicies == lower_pruning_index_value)[0][0]

                # 둘 다 들어있다고 하더라도 끊기는 부분이 존재할 수 있으므로,
                # 이를 판단해서 끊어주어야 한다.
                
                for i in range(lower_pruning_index-1, -1, -1):
                    lower_pruning_index = i+1

                    if fringing_rows_indicies[i+1] - fringing_rows_indicies[i] > 1:
                        break
                
                for j in range(upper_pruning_index+1, len(fringing_rows_indicies)):
                    upper_pruning_index = j-1

                    if fringing_rows_indicies[j] - fringing_rows_indicies[j-1] > 1:
                        break
                
                # 정해진 pruning 포인트에 따라서 index를 끊어준다.
                fringing_rows_indicies = fringing_rows_indicies[lower_pruning_index:upper_pruning_index+1]

            # 2) 319와 320이 둘 다 들어있거나 둘 다 들어있지 않은 경우는 위에서 필터링 됨.
            # 따라서 여기부터는 319와 320 둘 중 하나만 들어있는 경우 (정상이지만 간격 판단은 필요)
            elif upper_pruning_index_value in fringing_rows_indicies:
                # 319만 들어있으므로, 320보다 큰 것이 fringing _rows_indicies 들어 있어서는 안됨.
                # 만약 들어 있다면 잘못 나온 것일 가능성이 매우 크다. 따라서 제거함.
                fringing_rows_indicies = fringing_rows_indicies[fringing_rows_indicies < lower_pruning_index_value]
                # print(f"fringing_rows_indicies-2: {fringing_rows_indicies}")

                # 319부터 시작하여 1차이를 넘어서는 것은 잘라내야 함.
                upper_pruning_index = np.where(fringing_rows_indicies == upper_pruning_index_value)[0][0]
                
                # 간격이 1보다 큰 지점을 찾아낸다.
                for i in range(upper_pruning_index-1, -1, -1):
                    upper_pruning_index = i+1

                    if fringing_rows_indicies[i+1] - fringing_rows_indicies[i] > 1:
                        break  

                # 정해진 pruning 포인트에 따라서 index를 끊어준다.
                fringing_rows_indicies = fringing_rows_indicies[fringing_rows_indicies >= fringing_rows_indicies[upper_pruning_index]]

            elif lower_pruning_index_value in fringing_rows_indicies:
                # 320만 들어있으므로, 319보다 작은 것은 들어있어서는 안됨.
                # 만약 들어 있다면 잘못 나온 것일 가능성이 매우 크다. 따라서 제거함.
                fringing_rows_indicies = fringing_rows_indicies[fringing_rows_indicies > upper_pruning_index_value]
                # print(f"fringing_rows_indicies-3: {fringing_rows_indicies}")

                lower_pruning_index = np.where(fringing_rows_indicies == lower_pruning_index_value)[0][0]
                # print(lower_pruning_index)
                
                # 간격이 1보다 큰 지점을 찾아낸다.
                for j in range(lower_pruning_index+1, len(fringing_rows_indicies)):
                    lower_pruning_index = j-1

                    if fringing_rows_indicies[j] - fringing_rows_indicies[j-1] > 1:
                        break
                
                # 정해진 pruning 포인트에 따라서 index를 끊어준다.
                fringing_rows_indicies = fringing_rows_indicies[fringing_rows_indicies <= fringing_rows_indicies[lower_pruning_index]]

        # 길이가 1 이하인 경우는 보정할 것이 없다고 판단.
        else:
            fringing_rows_indicies = []

        return fringing_rows_indicies

    def getOuterBoundaryArray(self, area_array, bluing_rows_indicies):
        """ 
            2024.05.16, jdk
            area_array로부터 outer_array를 반환받는 함수
            upper/lower outer boundary array를 구분하여 반환한다.
        """
        
        upper_row_index = bluing_rows_indicies[0]
        lower_row_index = bluing_rows_indicies[-1]

        upper_outer_array = area_array[:upper_row_index]
        lower_outer_array = area_array[lower_row_index+1:]
        
        return upper_outer_array, lower_outer_array

    def isFringingPixel(self, anomaly_point_indicies, row_idx, col_idx):
        """
            2024.05.16, jdk
            전달받은 idx가 bluing pixel인지 아닌지 판단하는 함수
        """

        if row_idx in anomaly_point_indicies[0] and col_idx in anomaly_point_indicies[1]:
            return True
        else:
            return False

    def getClosestNonFringingPixelByColumn(self, row_len, outer_anomaly_point, pixel_index):
        """
            2024.05.16, jdk
            전달받은 (i, j) 픽셀에서 row를 기준으로 
            bluing pixel이 아니고 가장 가까운 pixel의 값을 반환한다.
        """

        row_idx = pixel_index[0]
        col_idx = pixel_index[1]

        # bluing pixel이 존재하지 않는 경우는 거의 100% 존재하지 않으므로,
        # 별도의 예외 처리는 하지 않고, 반드시 값을 찾을 수 있는 것으로 가정한다.

        # 2024.05.22, jdk
        # Not Exists가 나오는 경우가 존재한다.
        for cur_row_idx in range(row_idx+1, row_len):
            if self.isFringingPixel(outer_anomaly_point, cur_row_idx, col_idx):
                # 현재 찾는 row가 bluing pixel인 경우
                # 다음 row를 탐색하러 진행한다.
                continue
            else:  
                # 현재 찾는 row가 bluing pixel이 아닌 경우
                # 현재 pixel의 index를 반환한다.
                # print("found!")
                return (cur_row_idx, col_idx)
        
        """
            2024.05.16, jdk
            문제가 발생할 경우, 연산이 가능하도록
            자기 자신의 index를 그대로 반환한다.
            단, 문제가 발생한 것을 알 수 있도록
            pirint를 지정해 둔다.
        """
        print("Not Exists! (getClosestNonBluingPixelByColumn)")

        return (-1, -1)
        

    def getClosestNonFringingPixelByRow(self, col_len, anomaly_point_indicies, pixel_index):
        """
            2024.05.16, jdk
            전달받은 (i, j) 픽셀에서 col을 기준으로 
            bluing pixel이 아니고 가장 가까운 pixel의 값을 반환한다.
        """

        # 현재 bluing pixel의 index
        row_idx = pixel_index[0]
        col_idx = pixel_index[1]

        # non-bluing pixel이 존재하지 않는 경우는 거의 100% 존재하지 않으므로,
        # 별도의 예외 처리는 하지 않고, 반드시 값을 찾을 수 있는 것으로 가정한다.
        for cur_col_idx in range(col_idx+1, col_len):
            # 현재 column index의 다음부터, column의 전체 길이로 진행
            if self.isFringingPixel(anomaly_point_indicies, row_idx, cur_col_idx):
                # 현재 찾는 row가 bluing pixel인 경우
                # 다음 row를 탐색하러 진행한다.
                continue
            else:  
                # 현재 찾는 row가 bluing pixel이 아닌 경우
                # 현재 pixel의 index를 반환한다.
                return (row_idx, cur_col_idx)
        
        # 2024.05.16, jdk
        # 문제가 발생할 경우, 연산이 가능하도록
        # 자기 자신의 index를 그대로 반환한다.
        # 단, 문제가 발생한 것을 알 수 있도록
        # pirint를 지정해 둔다.
        
        return (-1, -1)

    def calcRGBMeanOfPixels(self, pixel_list):
        """
            2024.05.17, jdk
            전달받은 pixel list에서 rgb 평균을 계산하고 반환
        """
        list_num = len(pixel_list)

        if list_num == 0:
            print("Could not find the closest pixels. RGB mean calculation failed.")
            return

        pixel_value = []

        r = 0
        g = 0
        b = 0

        for i in range(list_num):
            r = r + pixel_list[i][0]
            g = g + pixel_list[i][1]
            b = b + pixel_list[i][2]

        r = r / list_num
        g = g / list_num
        b = b / list_num

        # int로 표현하기 위하여 반올림
        pixel_value.append(round(r))
        pixel_value.append(round(g))
        pixel_value.append(round(b))

        return pixel_value

    def correctOuterFringingPixels(self, outer_array, anomaly_point_indicies):
        """
            2024.05.16, jdk
            Upper Outer Bluing Pixel에 대한 Anomaly Point를 기반으로 
            Bluing Pixel Calibration을 수행하는 함수
        """

        # 2024.05.16, jdk
        # anomaly_point의 모든 index를 얻어내었음.
        # 전달받은 anomaly point는 좌상단 순서대로 정리되어 있다.
        # 이제 알고리즘 대로 보정을 시작한다.

        anomaly_point_num = len(anomaly_point_indicies[0])
        row_len = outer_array.shape[0]
        col_len = outer_array.shape[1]

        # anomaly point의 개수 만큼 동작한다.
        for idx in range(anomaly_point_num):
            i = anomaly_point_indicies[0][idx]
            j = anomaly_point_indicies[1][idx]

            # 현재 pixel의 index
            pixel_index = (i, j)

            # 1) bluing pixel이 첫 row에 존재할 경우
            if i == 0:

                # 1-1) 첫 번째 픽셀인 경우
                if j == 0:
                    # bluing pixel이 아닌 가장 가까운 row/col 값의 평균으로 보간한다.

                    pixels = []

                    # non-bluing pixel인 가장 가까운 row pixel의 값을 가져온다.
                    (row_idx_of_closest_row, col_idx_of_closest_row) = self.getClosestNonFringingPixelByRow(col_len, anomaly_point_indicies, pixel_index)

                    if not (row_idx_of_closest_row == -1 and col_idx_of_closest_row == -1):
                        value_of_closest_row = outer_array[row_idx_of_closest_row, col_idx_of_closest_row, :]
                        pixels.append(value_of_closest_row)

                    # non-bluing pixel인 가장 가까운 column pixel의 값을 가져온다.
                    (row_idx_of_closest_col, col_idx_of_closest_col) = self.getClosestNonFringingPixelByColumn(row_len, anomaly_point_indicies, pixel_index)
                    
                    if not (row_idx_of_closest_col == -1 and col_idx_of_closest_col == -1):
                        value_of_closest_col = outer_array[row_idx_of_closest_col, col_idx_of_closest_col, :]
                        pixels.append(value_of_closest_col)

                    interpolation_value = self.calcRGBMeanOfPixels(pixels)

                    # 보간 값을 찾을 수 있을 때만 값 변경한다.
                    if interpolation_value != None:
                        # 현재 pixel을 interpolation value로 교체한다
                        outer_array[i, j, :] = interpolation_value 

                # 1-2) 첫 번째 row이고, 첫 번째 col이 아닌 경우
                # 왼쪽 pixel의 rgb와 non-bluing이며 같은 column에서 
                # 가장 가까운 pixel의 rgb의 평균으로 변경한다.
                elif j == col_len-1:

                    pixels = []

                    # 왼쪽 pixel의 index를 가져온다.
                    left_pixel_row_idx = pixel_index[0] - 1
                    left_pixel_col_idx = pixel_index[1]

                    # 왼쪽 pixel의 값을 가져온다.
                    left_pixel_value = outer_array[left_pixel_row_idx, left_pixel_col_idx, :]
                    pixels.append(left_pixel_value)

                    # non-bluing pixel인 가장 가까운 column pixel의 값을 가져온다.
                    (row_idx_of_closest_col, col_idx_of_closest_col) = self.getClosestNonFringingPixelByColumn(row_len, anomaly_point_indicies, pixel_index)
                    
                    if not (row_idx_of_closest_col == -1 and col_idx_of_closest_col == -1):
                        value_of_closest_col = outer_array[row_idx_of_closest_col, col_idx_of_closest_col, :]
                        pixels.append(value_of_closest_col)

                    # interpolation value로 평균을 계산한다.
                    interpolation_value = self.calcRGBMeanOfPixels(pixels)

                    # 현재 pixel을 interpolation value로 교체한다;
                    outer_array[i, j, :] = interpolation_value

            # bluing pixel이 첫 row가 아닌 경우
            else:
                # 첫 row가 아니며, 첫 column인 경우
                if j == 0:
                    pixels = []

                    upper_pixel_row_idx = pixel_index[0] - 1
                    upper_pixel_col_idx = pixel_index[1]
                    upper_pixel_value = outer_array[upper_pixel_row_idx, upper_pixel_col_idx, :]
                    pixels.append(upper_pixel_value)

                    right_upper_pixel_row_idx = pixel_index[0] - 1
                    right_upper_pixel_col_idx = pixel_index[1] + 1
                    right_upper_pixel_value = outer_array[right_upper_pixel_row_idx, right_upper_pixel_col_idx, :]
                    pixels.append(right_upper_pixel_value)

                    # non-bluing pixel인 가장 가까운 row pixel의 값을 가져온다.
                    (row_idx_of_closest_row, col_idx_of_closest_row) = self.getClosestNonFringingPixelByRow(col_len, anomaly_point_indicies, pixel_index)

                    if not (row_idx_of_closest_row == -1 and col_idx_of_closest_row == -1):
                        value_of_closest_row = outer_array[row_idx_of_closest_row, col_idx_of_closest_row, :]
                        pixels.append(value_of_closest_row)
                    
                    # RGB 평균 반환
                    interpolation_value = self.calcRGBMeanOfPixels(pixels)
                    
                    # 현재 pixel을 interpolation value로 교체한다;
                    outer_array[i, j, :] = interpolation_value

                # 첫 row가 아니며, 마지막 column인 경우
                elif j == col_len - 1:
                    pixels = []

                    left_upper_pixel_row_idx = pixel_index[0] - 1
                    left_upper_pixel_col_idx = pixel_index[1] - 1
                    left_upper_pixel_value = outer_array[left_upper_pixel_row_idx, left_upper_pixel_col_idx, :]
                    pixels.append(left_upper_pixel_value)

                    upper_pixel_row_idx = pixel_index[0] - 1
                    upper_pixel_col_idx = pixel_index[0] - 1
                    upper_pixel_value = outer_array[upper_pixel_row_idx, upper_pixel_col_idx, :]
                    pixels.append(upper_pixel_value)
                    
                    left_pixel_row_idx = pixel_index[0]
                    left_pixel_col_idx = pixel_index[1] - 1
                    left_pixel_value = outer_array[left_pixel_row_idx, left_pixel_col_idx, :]
                    pixels.append(left_pixel_value)

                    interpolation_value = self.calcRGBMeanOfPixels(pixels)

                    outer_array[i, j, :] = interpolation_value

                # 첫 row가 아니며, 중간 column인 경우
                else:
                    pixels = []

                    left_upper_pixel_row_idx = pixel_index[0] - 1
                    left_upper_pixel_col_idx = pixel_index[1] - 1
                    left_upper_pixel_value = outer_array[left_upper_pixel_row_idx, left_upper_pixel_col_idx, :]
                    pixels.append(left_upper_pixel_value)

                    upper_pixel_row_idx = pixel_index[0] - 1
                    upper_pixel_col_idx = pixel_index[0] - 1
                    upper_pixel_value = outer_array[upper_pixel_row_idx, upper_pixel_col_idx, :]
                    pixels.append(upper_pixel_value)
                    
                    right_upper_pixel_row_idx = pixel_index[0] - 1
                    right_upper_pixel_col_idx = pixel_index[1] + 1
                    right_upper_pixel_value = outer_array[right_upper_pixel_row_idx, right_upper_pixel_col_idx, :]
                    pixels.append(right_upper_pixel_value)

                    left_pixel_row_idx = pixel_index[0]
                    left_pixel_col_idx = pixel_index[1] - 1
                    left_pixel_value = outer_array[left_pixel_row_idx, left_pixel_col_idx, :]
                    pixels.append(left_pixel_value)

                    # non-bluing pixel인 가장 가까운 row pixel의 값을 가져온다.
                    (row_idx_of_closest_row, col_idx_of_closest_row) = self.getClosestNonFringingPixelByRow(col_len, anomaly_point_indicies, pixel_index)
                
                    if row_idx_of_closest_row == -1 and col_idx_of_closest_row == -1:
                        value_of_closest_row = [0, 0, 0]
                    else:
                        value_of_closest_row = outer_array[row_idx_of_closest_row, col_idx_of_closest_row, :]
                        pixels.append(value_of_closest_row)

                        interpolation_value = self.calcRGBMeanOfPixels(pixels)
                        # interpolation_value = left_upper_pixel_value 

                        outer_array[i, j, :] = interpolation_value

        return outer_array

    def interpolateFringingRowPixels(self, loop_start, loop_end, area_array_col_len, area_array):
        """
            2024.05.20, jdk
            FringingRow에 대해 반복문을 돌며 각 Pixel에 대한
            interpolation을 수행하는 함수이다.
        """

        # 보정 시작
        for i in range(loop_start, loop_end):
            cur_row_idx = i

            for j in range(area_array_col_len):
                cur_col_idx = j

                # 첫 column일 경우
                if j == 0:
                    upper_pixel_row_idx = cur_row_idx - 1
                    upper_pixel_col_idx = cur_col_idx
                    upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

                    right_upper_pixel_row_idx = cur_row_idx - 1
                    right_upper_pixel_col_idx = cur_col_idx + 1
                    right_upper_pixel_value = area_array[right_upper_pixel_row_idx, right_upper_pixel_col_idx, :]

                    pixels = []
                    pixels.append(upper_pixel_value)
                    pixels.append(right_upper_pixel_value)

                    interpolation_value = self.calcRGBMeanOfPixels(pixels)
                    area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

                # 마지막 column일 경우
                elif j == area_array_col_len-1:
                    left_upper_pixel_row_idx = cur_row_idx - 1
                    left_upper_pixel_col_idx = cur_col_idx - 1
                    left_upper_pixel_value = area_array[left_upper_pixel_row_idx, left_upper_pixel_col_idx, :]

                    upper_pixel_row_idx = cur_row_idx - 1
                    upper_pixel_col_idx = cur_col_idx
                    upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

                    # left_pixel_row_idx = cur_row_idx
                    # left_pixel_col_idx = cur_col_idx - 1
                    # left_pixel_value = area_array[left_pixel_row_idx, left_pixel_col_idx, :]

                    pixels = []
                    pixels.append(left_upper_pixel_value)
                    pixels.append(upper_pixel_value)
                    # pixels.append(left_pixel_value)

                    interpolation_value = self.calcRGBMeanOfPixels(pixels)
                    area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

                # 중간 column일 경우
                else:
                    left_upper_pixel_row_idx = cur_row_idx - 1
                    left_upper_pixel_col_idx = cur_col_idx - 1
                    left_upper_pixel_value = area_array[left_upper_pixel_row_idx, left_upper_pixel_col_idx, :]

                    upper_pixel_row_idx = cur_row_idx - 1
                    upper_pixel_col_idx = cur_col_idx
                    upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

                    right_upper_pixel_row_idx = cur_row_idx - 1
                    right_upper_pixel_col_idx = cur_col_idx + 1
                    right_upper_pixel_value = area_array[right_upper_pixel_row_idx, right_upper_pixel_col_idx, :]

                    # left_pixel_row_idx = cur_row_idx
                    # left_pixel_col_idx = cur_col_idx
                    # left_pixel_value = area_array[left_pixel_row_idx, left_pixel_col_idx, :]

                    pixels = []
                    pixels.append(left_upper_pixel_value)
                    pixels.append(upper_pixel_value)
                    pixels.append(right_upper_pixel_value)
                    # pixels.append(left_pixel_value)

                    """ 
                        2024.05.17, jdk
                        에러가 났던 이유는 numpy array가 아니기 때문에
                        boradcasting이 불가능해서 column wise 덧셈이
                        안되었기 때문이다...
                    """

                    interpolation_value = self.calcRGBMeanOfPixels(pixels)
                    area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

        return area_array

    def correctFringingRows(self, area_array, fringing_rows_indicies):
        """
            2024.05.17, jdk
            fringing rows의 outer area는 fringing pixel이 모두 처리된 상황이다.
            좀 더 자연스러운 fringing rows 보간을 수행하기 위하여
            기존의 방식을 개선하여 
        """

        fringing_rows_len = len(fringing_rows_indicies) # anomaly row의 개수

        # 1) row가 없는 경우 
        if fringing_rows_len == 0:
            # 전달받은 array를 그대로 반환한다.
            return area_array
        # 이 아래부터는 anomaly row가 존재한다는 것이다.
        # 따라서 우선 anomaly row의 index를 얻어낸다.

        """
            2024.05.14, jdk
            현재 작업의 특성 상, anomaly row는 반드시 중앙에만 나타난다고 고려한다.
            그러므로 list에서 발생할 수 있는 잘못된 참조의 문제는 고려하지 않았다.
        """
        # 2) row가 한 개인 경우
        if fringing_rows_len == 1:

            row_index = fringing_rows_indicies[0]

            # row가 한 개만 있으므로, 위 아래의 평균을 계산하여 보정한다.
            upper_rgb = area_array[row_index-1]
            lower_rgb = area_array[row_index+1]
            
            # 평균값은 반올림을 통해 소수점 제거
            mean_rgb = (upper_rgb + lower_rgb)/2
            mean_rgb = np.round(mean_rgb)

            area_array[row_index] = mean_rgb        

        # 3) row가 두 개 이상인 경우.
        # 연속적이지 않은 row에 대해서는 정상적으로 동작할 수 없으므로 주의가 필요하다.
        
        # 2024.05.20, jdk
        # 1 간격으로 연속적으로 나타나지 않는 index는 (upper_pruning_index_value: 319) (upper_pruning_index_value: 320)
        # 319와 320을 기준으로 잘라내므로, 현재 fringing index는 모두 연속적으로 나타난다고 가정한다.
        
        if fringing_rows_len >= 2:
            """
                boundary는 upper_low와 lower_low의 값을
                지정하기 위하여 bluing row의 위/아래로 참조하게 될
                row의 boundary를 지정하는 값이다.
                크기는 hyper parameter이며, 이는
                upper_row와 lower_row의 값을 지정하기 위하여
                boundary 만큼 바깥쪽의 row를 참조한다는 의미이다.

                2024.05.17, jdk
                보정 방식이 바뀌게 되어 boundary는 삭제하도록 함.
            """
            
            area_array_row_len = area_array.shape[0]
            area_array_col_len = area_array.shape[1]

            # fringing rows의 index를 넣고,
            # interpolateBluingRowPixels 함수 내에서
            # 값에 접근해 calibration을 하게 된다.
        
            row_idx_start = fringing_rows_indicies[0]
            row_idx_end = fringing_rows_indicies[int(fringing_rows_len/2)]

            area_array = self.interpolateFringingRowPixels(row_idx_start, row_idx_end+1, area_array_col_len, area_array)

            # 2024.05.19, jdk
            # 주변값 반영하도록 추가로 보정

            # 전체 이미지를 뒤집고 보정 시작
            area_array = self.flipArray(area_array)
            # 이미지를 뒤집었으므로 index를 변경해 주어야 함.
            # row만 신경쓰면 되기 때문에 bluing_rows_indicies list에서
            # bluing_rows_len/2+1부터 bluing_rows_len까지의 값들을 2500에서 빼준다. 
            # 그렇게 하고, 역순으로 돌려주면 된다. column은 그대로 0번부터 돌리면 된다.
            fringing_rows_indicies[int(fringing_rows_len/2)+1:fringing_rows_len] = (area_array_row_len - 1) - fringing_rows_indicies[int(fringing_rows_len/2)+1:fringing_rows_len]

            row_idx_start = fringing_rows_indicies[fringing_rows_len-1]
            row_idx_end = fringing_rows_indicies[int(fringing_rows_len/2)+1]

            area_array = self.interpolateFringingRowPixels(row_idx_start, row_idx_end+1, area_array_col_len, area_array)

            # 작업이 끝났으므로 flip한 index를 다시 돌려준다.
            fringing_rows_indicies[int(fringing_rows_len/2)+1:fringing_rows_len] = (area_array_row_len - 1) - fringing_rows_indicies[int(fringing_rows_len/2)+1:fringing_rows_len]

            # 보정이 끝났으므로 다시 뒤집는다.
            area_array = self.flipArray(area_array)

            corrected_fringing_rows = area_array[fringing_rows_indicies[0]:fringing_rows_indicies[-1] + 1]

            return corrected_fringing_rows

    def flipArray(self, array):
        """
            2024.05.17, jdk
            전달받은 numpy array를
            상하좌우로 뒤집는 함수
        """

        fliped_array = array[::-1, ::-1, :]

        return fliped_array

    def flipAnomalyIndicies(self, anomaly_indicies, row_len, col_len):
        """
            2024.05.17, jdk
            anomaly_indicies를 상하좌우 뒤집는 함수
        """

        anomaly_indicies = np.array(anomaly_indicies)

        anomaly_indicies[0] = row_len - anomaly_indicies[0] - 1
        anomaly_indicies[1] = col_len - anomaly_indicies[1] - 1
        
        # indicies를 완벽히 뒤집기 위해서는 sort를 하는게 아니라
        # reverse를 수행해야 함.
        anomaly_indicies[0] = anomaly_indicies[0][::-1]
        anomaly_indicies[1] = anomaly_indicies[1][::-1]
        
        return anomaly_indicies   

    def calcBrightness(self, rgb_value):
        R = rgb_value[0]
        G = rgb_value[1]
        B = rgb_value[2]

        return 0.299*R+0.587*G+0.114*B


    def shuffle_kernel_pixels(self, image, kernel_size=3, stride=2):
        """
        이미지 내의 각 커널 영역에 대해 픽셀의 위치를 랜덤하게 섞습니다.
        
        Args:
        image (numpy.ndarray): 입력 이미지 배열, 크기는 (n, m, 3).
        kernel_size (int): 커널의 크기, 정사각형 형태.
        stride (int): 커널 이동 시의 스트라이드 값.
        
        Returns:
        numpy.ndarray: 픽셀이 섞인 이미지.
        """
        height, width, _ = image.shape
        shuffled_image = np.copy(image)

        # 커널을 이동시키며 각 영역에 대해 처리
        # for i in range(0, height - kernel_size + 1, stride):

        #     if i+kernel_size
            
        #     for j in range(0, width - kernel_size + 1, stride):
        #         # 커널 영역 추출
        #         kernel = image[i:i+kernel_size, j:j+kernel_size, :]
        #         # 커널 영역을 1D로 변환
        #         flattened = kernel.reshape(-1, 3)
        #         # 픽셀 인덱스 섞기
        #         np.random.shuffle(flattened)
        #         # 섞인 픽셀을 원래 차원으로 변환
        #         shuffled_kernel = flattened.reshape(kernel_size, kernel_size, 3)
        #         # 섞인 커널을 이미지에 할당
        #         shuffled_image[i:i+kernel_size, j:j+kernel_size, :] = shuffled_kernel       

        i = 0
        j = 0

        # Sliding Kernel Shuffle 시작
        while (True):
            height_kernel_size = kernel_size

            # 만약 kenel이 이미지의 height를 초과하여 넘어간다면,
            # 이미지 내부만 처리되도록 height_kernel_size를 조절한다.
            if i + height_kernel_size > height:
                height_kernel_size = height - i - 1

            # print(f"i: {i}")
            # print(f"height_kernel_size: {height_kernel_size}")

            while (True):  
                width_kernel_size = kernel_size
                
                # 만약 kernel이 이미지의 width를 초과하여 넘어간다면,
                # 이미지 내부만 처리되도록 width_kernel_size를 조절한다.
                if j + width_kernel_size > width:
                    # print(f"j: {j}")
                    # print(f"j+width_kernel_size: {j+width_kernel_size}, width: {width}")
                    width_kernel_size = width - j - 1

                # print(f"width_kernel_size: {width_kernel_size}")

                kernel = image[i:i+height_kernel_size, j:j+width_kernel_size, :]
                flattend = kernel.reshape(-1, 3)
                np.random.shuffle(flattend)
                shuffled_kernel = flattend.reshape(height_kernel_size, width_kernel_size, 3)
                shuffled_image[i:i+height_kernel_size, j:j+width_kernel_size, :] = shuffled_kernel
                # print(f"shuffled_image.shape: {shuffled_image.shape}")

                # 현재 위치에 대해서 shuffle이 끝났으므로,
                # 같은 row에서 col의 위치인 j를 stride 만큼 옮겨준다.
                j += stride

                # j를 stride 만큼 옮겼을 때,
                # 만약 j가 width보다 크거나 같다면
                # 한 row에 대한 shuffle이 끝난 것이므로
                # inner while loop을 탈출한다.
                if j >= width:
                    break
            

            # 한 row에 대해서 shuffle이 끝났으므로,
            # 다음 row로 넘어가기 위해 i를 stride 만큼 옮겨준다.
            i += stride

            j = 0

            # i를 stride 만큼 옮겼을 때,
            # 만약 i가 hieght보다 크거나 같다면
            # 모든 영역에 대한 shuffle이 끝난 것이므로
            # outer while loop을 탈출한다.
            if i >= height:
                break

        return shuffled_image

    def shuffleFringingRowsPixels(self, corrected_fringing_rows):
        """
            2024.05.20, jdk
            전달받은 array의 모든 pixel을 random하게 섞는 함수
        """

        height, width, channels = corrected_fringing_rows.shape

        indicies = np.arange(height * width)
        np.random.shuffle(indicies)

        flattend = corrected_fringing_rows.reshape(-1, channels)
        # original array를 pixel만 유지한 채로 flatten한다.

        flattend = flattend[indicies]
        shuffled_array = flattend.reshape(height, width, channels)

        return shuffled_array

    def calcBrightnessRatioByColumn(self, upper_brightness, lower_brightness, height):

        # 현재 column에서 pixel별로 가져야 
        # 하는 밝기를 저장하는 list
        pixels_brightness = []
        diff = (upper_brightness - lower_brightness)/(height+1)

        # 각 픽셀이 가져야 하는 밝기를 저장
        for h in range(1, height+1):
            pixels_brightness.append(upper_brightness - h*diff)

        return pixels_brightness

    def correctPixelsByLinearBrightness(self, corrected_fringing_rows, brightness_margin):
        """
            2024.05.20, jdk
            전달받은 array에 대해서 밝기 보정을 실시한다.
            위/아래로 세 픽셀은 밝기 판단용이다.
        """

        height, width, _ = corrected_fringing_rows.shape

        for w in range(width):
            
            upper_pixels = corrected_fringing_rows[0:brightness_margin, w, :]

            upper_r_mean = np.mean(upper_pixels[:, 0])
            upper_g_mean = np.mean(upper_pixels[:, 1])
            upper_b_mean = np.mean(upper_pixels[:, 2])

            rgb_mean = (upper_r_mean, upper_g_mean, upper_b_mean)
            rgb_mean = np.round(rgb_mean)

            upper_brightness = self.calcBrightness(rgb_mean)

            lower_pixels = corrected_fringing_rows[height-brightness_margin:height, w, :]

            lower_r_mean = np.mean(lower_pixels[:, 0])
            lower_g_mean = np.mean(lower_pixels[:, 1])
            lower_b_mean = np.mean(lower_pixels[:, 2])

            rgb_mean = (lower_r_mean, lower_g_mean, lower_b_mean)
            rgb_mean = np.round(rgb_mean)

            lower_brightness = self.calcBrightness(rgb_mean)

            area_height = (height-brightness_margin-1)*brightness_margin

            # pixel 별로 밝기의 비율을 저장하는 리스트
            pixels_brightness = self.calcBrightnessRatioByColumn(upper_brightness, lower_brightness, area_height)
            # print("brightness")
            # print(upper_brightness)
            # print(lower_brightness)
            # print(pixels_brightness)

            for h in range(brightness_margin, height-brightness_margin):
                # 이제 column에서 pixel 별 밝기의 값이 정해졌으므로,
                # 상대적인 비를 구하고 r, g, b값을 보정해 준다.
                cur_pixel_rgb = corrected_fringing_rows[h, w, :]
                cur_pixel_brightness = self.calcBrightness(cur_pixel_rgb)

                ratio = pixels_brightness[h-brightness_margin] / cur_pixel_brightness
                corrected_pixel_rgb = ratio*cur_pixel_rgb

                corrected_fringing_rows[h, w, :] = corrected_pixel_rgb
    
        return corrected_fringing_rows

    def applyGaussianSmoothing(self, area_array, height_start, height_end, width_start, width_end, kernel_size=5, stride=1):
        """
        이미지의 특정 구간에 가우시안 블러를 적용하여 스무딩하는 함수
        
        Args:
        area_array (numpy.ndarray): 입력 이미지 배열.
        x_start (int): 스무딩을 적용할 구간의 시작 x좌표.
        y_start (int): 스무딩을 적용할 구간의 시작 y좌표.
        x_end (int): 스무딩을 적용할 구간의 끝 x좌표.
        y_end (int): 스무딩을 적용할 구간의 끝 y좌표.
        kernel_size (int): 가우시안 커널 크기.
        stride (int): 커널 이동 시의 스트라이드 값.
        
        Returns:
        numpy.ndarray: 스무딩이 적용된 이미지.
        """

        region_of_interest = area_array[height_start:height_end, width_start:width_end]
        height, width, _ = region_of_interest.shape

        i = 0
        j = 0

        # gaussian smoothing 시작
        while (True):
            height_kernel_size = kernel_size

            # 만약 kenel이 이미지의 height를 초과하여 넘어간다면,
            # 이미지 내부만 처리되도록 height_kernel_size를 조절한다.
            if i + height_kernel_size > height:
                height_kernel_size = height - i - 1

                # height_kernel_size가 0인 경우는 존재할 수 없으므로 고려하지 않음.
                # 왜냐면, stride가 kernel_size보다 작기 때문이다.
                if height_kernel_size % 2 == 0:
                    height_kernel_size -= 1


            while (True):  
                width_kernel_size = kernel_size
                
                # 만약 kernel이 이미지의 width를 초과하여 넘어간다면,
                # 이미지 내부만 처리되도록 width_kernel_size를 조절한다.
                if j + width_kernel_size > width:
                    width_kernel_size = width - j - 1

                    if width_kernel_size % 2 == 0:
                        width_kernel_size -= 1
                        
                kernel = area_array[i:i+height_kernel_size, j:j+width_kernel_size, :]
                blurred_kernel = cv2.GaussianBlur(kernel, (height_kernel_size, width_kernel_size), 0)
                region_of_interest[i:i+height_kernel_size, j:j+width_kernel_size, :] = blurred_kernel

                # 현재 위치에 대해서 shuffle이 끝났으므로,
                # 같은 row에서 col의 위치인 j를 stride 만큼 옮겨준다.
                j += stride

                # j를 stride 만큼 옮겼을 때,
                # 만약 j가 width보다 크거나 같다면
                # 한 row에 대한 shuffle이 끝난 것이므로
                # inner while loop을 탈출한다.
                if j >= width:
                    break
            

            # 한 row에 대해서 shuffle이 끝났으므로,
            # 다음 row로 넘어가기 위해 i를 stride 만큼 옮겨준다.
            i += stride

            j = 0

            # i를 stride 만큼 옮겼을 때,
            # 만약 i가 hieght보다 크거나 같다면
            # 모든 영역에 대한 shuffle이 끝난 것이므로
            # outer while loop을 탈출한다.
            if i >= height:
                break
        
        return area_array

    def correctFringingRowsOfArea(self, area_array, image_correction_setting):

        """
            2024.05.14, jdk
            fringing이 나타난 pixel을 보정하는 함수이다.
            fringing correction 전체 과정을 포함한다.
            
            parameter
            1) area_array: 전체 이미지에 대해서 특정 부분만 잘라낸 array
        """
        # print("\n\n(correctFringingRowsOfArea)")

        height, width, _ = area_array.shape

        # 1) 
        # 전달 받은 하나의 area_array에 대해 각 row별 Blue Pixel의 평균을 계산한다.
        area_array_rows_B_mean = self.calcRowsBlueMean(area_array)
        # print(f"area_array_rows_B_mean: {area_array_rows_B_mean}")

        # 2) 
        # BoxPlot을 통해 Bluing row의 평균에 대한 High Anomaly Point 획득
        area_array_rows_B_anomaly_high = self.getFringingRowsAnomalyPoint(area_array_rows_B_mean.flatten()) 
        # print(f"area_array_rows_B_anomaly_high: {area_array_rows_B_anomaly_high}")

        # 3) 
        # 얻어낸 Anomaly Point를 바탕으로 Anomaly Rows 정보를 Bool List로 반환
        # 즉, area_array의 row 길이 만큼에 대해서, anomaly가 발생한 부분은 True로, 
        # 아닌 부분은 False로 표현하는 Boolean List를 전달 받는다.
        fringing_rows_identifier = self.getFringingRowsIdentifier(area_array_rows_B_mean, area_array_rows_B_anomaly_high) 
        # print(f"fringing_rows_identifier True count: {len(fringing_rows_identifier[fringing_rows_identifier == True])}")

        # 4) 
        # 얻어낸 bluing rows identifier(boolean list)를 바탕으로 
        # bluing rows의 indicies를 얻어낸다.
        # 해당 indicies는 area_array를 기준으로 하는 indicies이다.
        fringing_rows_indicies = self.getFringingRowsIndicies(fringing_rows_identifier, image_correction_setting)
        # 실제로 area_array에서 어느 위치가 anomaly인지 index를 제공하는 함수이다.

        # 만약 fringing_rows_indicies가 empty array라면
        # 보정할 것이 없다고 판단한다.
        if len(fringing_rows_indicies) == 0:
            # anoamly로 잡힌 것이 없으니, 위 아래로 20픽셀 가량만 보정을 실시한다.
            upper_pruning_index_value = image_correction_setting.upper_pruning_index_value
            lower_pruning_index_value = image_correction_setting.lower_pruning_index_value
            non_fringing_row_margin = image_correction_setting.non_fringing_row_margin

            fringing_rows_indicies = np.arange(upper_pruning_index_value - non_fringing_row_margin + 1, lower_pruning_index_value + non_fringing_row_margin)
            # print(f"modified_fringing_rows_indicies: {fringing_rows_indicies}")
            # print("fringing_rows_indexes: [] (empty). Set default correction boundary")
        else:
            # 5) add margin to fringing rows
            # 위의 과정으로는 실제로 나타나는 fringing row를 모두 잡아낼 수가 없다.
            # 따라서 약간의 margin을 설정하여 영역을 추가로 보정하고, 이를 통해
            # 추가적으로 나타날 수 있는 fringing pixel을 잡아낸다.

            # fringing rows는 반드시 선형적으로 나타난다고 가정하고,
            # 만약 선형적이지 않더라도 영향을 피해갈 수는 없기 때문에
            # margin을 추가하여 fringing rows의 index를 새롭게 만들어준다.
            min_index = fringing_rows_indicies.min()
            max_index = fringing_rows_indicies.max()

            fringing_row_margin = image_correction_setting.fringing_row_margin
            
            # arange는 1을 더해주어야 함.
            # print("\ncurrent fringing rows indicies")
            # print(f"min: {min_index}, max: {max_index}")
            # print(fringing_rows_indicies)
            fringing_rows_indicies = np.arange(min_index - fringing_row_margin, max_index + fringing_row_margin + 1)
        
        """
            2024.05.19, jdk
            fringing row가 존재하지 않을 수 있으므로, 해당 경우에 대한 대처가 필요하다.
        """

        # 6) 
        # upper_outer_rows와 lower_outer_rows로 구성된 outer boundary array를 얻어낸다.
        # rint(f"fringing_rows_indicies max and min: {fringing_rows_indicies.min()}, {fringing_rows_indicies.max()}")
        upper_outer_array, lower_outer_array = self.getOuterBoundaryArray(area_array, fringing_rows_indicies)
        # print(f"{upper_outer_array.shape}")
        # print(f"{lower_outer_array.shape}")

        """
            2024.05.22, jdk
            anomaly point 계산에 대한 문제 발생.
            만약, blue값이 거의 동일하게 찍힌다면 제대로 된 anomaly를 잡아내지 못하라 가능성이 매우 높다.
            이에 따라 첨도(kurtosis)를 계산해서 첨도가 일정 값 이상이라면 아예 보정하지 않도록 하는 방식을
            도입하여 추가적인 filtering을 진행하도록 해야 할 것 같다.
        """

        upper_outer_array_kurtosis = kurtosis(upper_outer_array[:, :, 2].flatten(), fisher=False)
        lower_outer_array_kurtosis = kurtosis(lower_outer_array[:, :, 2].flatten(), fisher=False)

        """"""

        # print(f"upper_outer_array_kurtosis: {upper_outer_array_kurtosis}")
        # print(f"lower_outer_array_kurtosis: {lower_outer_array_kurtosis}")

        # 7-1) 
        # 얻어낸 upper/outer array에 대해서 Anomaly Point를 얻어낸다.
        # outer anomaly point는 outer array의 column 별로 구해낸다.
        # column 별로 B값에 대한 BoxPlot을 수행하여 Anomaly Point를 반환한다.
        upper_outer_anomaly_point = self.getOuterFringingPixelsAnomalyPoint(upper_outer_array)
        lower_outer_anomaly_point = self.getOuterFringingPixelsAnomalyPoint(lower_outer_array)

        # 위에서 얻어낸 Anomaly Point에 해당하는 pixel들의 index를 얻어낸다.
        upper_anomaly_indicies = np.where(upper_outer_array[:, :, 2] > upper_outer_anomaly_point)
        lower_anomaly_indicies = np.where(lower_outer_array[:, :, 2] > lower_outer_anomaly_point)
        # print(f"upper_anomaly_indicies: {upper_anomaly_indicies}")
        # print(f"lower_anomaly_indicies: {lower_anomaly_indicies}")

        # 7-2) 
        # upper area의 보정 코드를 활용하여
        # lower area를 보정을 수행하기 위해 lower_anomaly_indcies를 flip한다.
        lower_outer_area_row_len = lower_outer_array.shape[0]
        lower_outer_area_col_len = lower_outer_array.shape[1]

        # lower area를 flip했을 때 기준으로 anomaly index를 변환한다.
        lower_anomaly_indicies = self.flipAnomalyIndicies(lower_anomaly_indicies, lower_outer_area_row_len, lower_outer_area_col_len)

        # 8-1) 
        # 얻어낸 Anomaly Point를 바탕으로 upper outer Array의 Fringing Pixel을 보정한다.
        # Fringing Rows에 대한 보정보다 Outer Fringing Pixels에 대한 보정을 먼저 수행하는 이유는,
        # 상대적으로 Fringing Rows의 영역이 훨씬 크기 때문에 적절한 값을 확보하기 어렵기 때문이다.
        # 다시 말하면, Fringing rows는 interpolation 값을 얻기 위해 upper/lower outer array의 값을
        # 참조할 수밖에 없게 되는데, 이때 outer array fringing pixel의 값이 적절한 값이라고 신뢰할 수 없다면
        # fringing rows의 값도 신뢰할 수 없게 된다. 따라서 Outer Array의 값에 대한 interpolation을 우선적으로 
        # 수행하고, 그 이후에 fringing rows에 대한 interpolation을 수행해야 한다.
        upper_outer_array = self.correctOuterFringingPixels(upper_outer_array, upper_anomaly_indicies)
        
        # 8-2)
        # lower_outer_array를 flip하고 보정한다.
        lower_outer_array = self.flipArray(lower_outer_array)
        lower_outer_array = self.correctOuterFringingPixels(lower_outer_array, lower_anomaly_indicies)
        
        # 보정이 끝났으면 다시 flip 해서 원본으로 되돌린다.
        lower_outer_array = self.flipArray(lower_outer_array)
        
        # 9-1) 
        # Upper/Lower Outer Fringing Pixel에 대한 보정이 끝났으므로,
        # 이전에 구했던 Fringing Rows에 대한 보정을 수행한다.
        
        # 원본 array를 보정된 outer array로 변경한다.
        area_array[:fringing_rows_indicies[0], :, :] = upper_outer_array
        area_array[fringing_rows_indicies[-1]+1:, :, :] = lower_outer_array

        # 변경된 outer array에 대해서 fringing rows 보정을 수행한다.
        corrected_fringing_rows = self.correctFringingRows(area_array, fringing_rows_indicies)
        # print(f"no margin fringing rows shape: {corrected_fringing_rows.shape}")

        # # 9-2)
        # # Fringing rows의 모든 pixel을 random하게 섞는다.
        # corrected_fringing_rows = shuffleFringingRowsPixels(corrected_fringing_rows)

        # # 10-1)
        # # 섞은 pixel에 대해 correctPixelsByLinearBrightness를 실행
        # area_array[fringing_rows_indicies[0]:fringing_rows_indicies[-1]+1, :, :] = corrected_fringing_rows
        # brightness_margin = image_correction_setting.brightness_margin

        # # 10-2)
        # # 밝기 판단을 위하여 위/아래로 3칸 정도를 추가로 가져와 저장한다.
        # # 이후 보정이 수행되면 보정된 부분만 잘라낸다.
        # corrected_fringing_rows = area_array[fringing_rows_indicies[0]-brightness_margin:fringing_rows_indicies[-1]+1+brightness_margin, :, :]
        # corrected_fringing_rows = correctPixelsByLinearBrightness(corrected_fringing_rows, brightness_margin)
        # corrected_fringing_rows = corrected_fringing_rows[brightness_margin:-brightness_margin, :, :]

        # # 10-3)
        # # 보정된 array를 원본 array에 삽입한다.
        # area_array[fringing_rows_indicies[0]:fringing_rows_indicies[-1]+1, :, :] = corrected_fringing_rows
        # # print(f"{fringing_rows_indicies[0]}, {fringing_rows_indicies[-1]}: {fringing_rows_indicies[-1] - fringing_rows_indicies[0] + 1}")

        # 11) 보정된 밝기에 따라 kernel slide shuffle을 진행한다.
        shuffle_margin = image_correction_setting.shuffle_margin
        random_shuffle_kernel_size = image_correction_setting.random_shuffle_kernel_size
        random_shuffle_stride_size = image_correction_setting.random_shuffle_stride_size

        corrected_fringing_rows = area_array[fringing_rows_indicies[0]-shuffle_margin:fringing_rows_indicies[-1]+shuffle_margin, :, :]
        # print(f"corrected_fringing_rows.shape: {corrected_fringing_rows.shape}")

        corrected_fringing_rows = self.shuffle_kernel_pixels(corrected_fringing_rows, kernel_size=random_shuffle_kernel_size, stride=random_shuffle_stride_size)
        area_array[fringing_rows_indicies[0]-shuffle_margin:fringing_rows_indicies[-1]+shuffle_margin, :, :] = corrected_fringing_rows

        # 12) Brightness Smoothing을 진행하기 위한 index 반환
        # smoothing_index_for_gaussian_smoothing = (fringing_rows_indicies[0]-shuffle_margin, fringing_rows_indicies[-1]+shuffle_margin, width)

        # gaussian_smoothing_kernel_size = image_correction_setting.gaussian_smoothing_kernel_size
        # gaussian_smoothing_stride_size = image_correction_setting.gaussian_smoothing_stride_size

        # area_array = applyGaussianSmoothing(area_array, smoothing_index_for_gaussian_smoothing[0], smoothing_index_for_gaussian_smoothing[1], 
        #                        0, width, 
        #                        gaussian_smoothing_kernel_size, gaussian_smoothing_stride_size)

        # 13) 결과 반환
        return area_array

    def correctFringingColumnsOfArea(self, area_array, image_correction_setting):
        """
            2024.05.20, jdk
            fringing columns를 보정하는 함수이다. (area_4)
        """
        area_array = np.transpose(area_array, (1, 0, 2))
        # print(f"transposed shape: {area_array.shape}")
        
        # 2024.05.20, jdk
        # 현재 column을 전치하여 보정하고 있으므로,
        # column에 맞는 값으로 세팅값을 바꿔주어야 한다.

        cur_image_correction_setting = copy.deepcopy(image_correction_setting)

        cur_image_correction_setting.fringing_row_margin = image_correction_setting.fringing_col_margin
        cur_image_correction_setting.upper_pruning_index_value = image_correction_setting.col_upper_pruning_index_value
        cur_image_correction_setting.lower_pruning_index_value = image_correction_setting.col_lower_pruning_index_value
        cur_image_correction_setting.non_fringing_col_margin = image_correction_setting.non_fringing_row_margin
        cur_image_correction_setting.shuffle_margin = image_correction_setting.col_shuffle_margin
        cur_image_correction_setting.random_shuffle_kernel_size = image_correction_setting.col_random_shuffle_kernel_size
        cur_image_correction_setting.random_shuffle_stride_size = image_correction_setting.col_random_shuffle_stride_size

        area_array = self.correctFringingRowsOfArea(area_array, cur_image_correction_setting)
        # print(f"shape: {area_array.shape}")

        area_array = np.transpose(area_array, (1, 0, 2))

        return area_array

    # --------------------------------------------------------------------------- #

    def correctFringing(self, combined_image):
        """
        2024.05.15, jdk
        이미지 세팅 파일 분리를 통해 모듈화로 세팅값 가져오기
        """

        image_settings_file_path = "./config/image_settings.json"
        with open(image_settings_file_path, 'r', encoding='utf-8') as file:
            image_setting = json.load(file)

        area_size = self.getImageArraySize(image_setting)

        # FC를 적용하기 위해 불러올 image의 경로 지정
        # image_file_name = f"{tpg_code}_comb.png"
        # image_file_path = f"{self.pf.tpg_combined_directory_path}/{image_file_name}"

        # image = Image.open(image_file_path)
        image = combined_image

        image_height, image_width = image.size

        # Fringing Row Correction을 수행하기 위한 세팅값
        fringing_row_margin = image_setting['fringing_row_margin']
        fringing_col_margin = image_setting['fringing_col_margin']
        non_fringing_row_margin = image_setting['non_fringing_row_margin']
        non_fringing_col_margin = image_setting['non_fringing_col_margin']
        upper_pruning_index_value = image_setting['upper_pruning_index_value']
        lower_pruning_index_value = image_setting['lower_pruning_index_value']

        # Fringing Column Correction을 수행하기 위한 세팅값
        # fringing_col_width = area_size[3][3] - area_size[3][2]
        col_upper_pruning_index_value = image_setting['col_upper_pruning_index_value']
        col_lower_pruning_index_value = image_setting['col_lower_pruning_index_value']

        # Brightness Correction을 수행하기 위한 세팅값
        brightness_margin = image_setting['brightness_margin']

        # Random Shuffle Kernel Sliding을 수행하기 위한 세팅값
        shuffle_margin = image_setting['shuffle_margin']
        random_shuffle_kernel_size = image_setting['random_shuffle_kernel_size']
        random_shuffle_stride_size = image_setting['random_shuffle_stride_size']

        col_shuffle_margin = image_setting['col_shuffle_margin']
        col_random_shuffle_kernel_size = image_setting['col_random_shuffle_kernel_size']
        col_random_shuffle_stride_size = image_setting['col_random_shuffle_stride_size']

        gaussian_smoothing_kernel_size = image_setting['gaussian_smoothing_kernel_size']
        gaussian_smoothing_stride_size = image_setting['gaussian_smoothing_stride_size']

        # 4개의 영역을 기반으로 Image를 분할하여 얻은 numpy array를 갖고 있는 list
        column_fringing_area_array = self.getColumnFringingAreaArray(image, area_size, image_height)

        image_correction_setting = ImageCorrectionSetting(
            brightness_margin= brightness_margin,
            fringing_row_margin=fringing_row_margin,
            fringing_col_margin=fringing_col_margin,
            non_fringing_row_margin=non_fringing_row_margin,
            non_fringing_col_margin=non_fringing_col_margin,
            upper_pruning_index_value=upper_pruning_index_value,
            lower_pruning_index_value=lower_pruning_index_value,
            col_upper_pruning_index_value=col_upper_pruning_index_value,
            col_lower_pruning_index_value=col_lower_pruning_index_value,
            shuffle_margin=shuffle_margin,
            random_shuffle_kernel_size=random_shuffle_kernel_size,
            random_shuffle_stride_size=random_shuffle_stride_size,
            col_shuffle_margin=col_shuffle_margin,
            col_random_shuffle_kernel_size=col_random_shuffle_kernel_size,
            col_random_shuffle_stride_size=col_random_shuffle_stride_size,
            gaussian_smoothing_kernel_size=gaussian_smoothing_kernel_size,
            gaussian_smoothing_stride_size=gaussian_smoothing_stride_size
        )

        # --------------------------------------------------------------------------- #

        column_fringing_area_array = self.correctFringingColumnsOfArea(column_fringing_area_array, image_correction_setting)
        image = self.changeCorrectedFringingAreaOfOriginalImage(image, area_size, image_height, column_fringing_area_array)

        # print("Correcting column fringing area")
        area_array_list = self.getAreaArrayList(image, area_size)

        # correction 실시
        # print("\n\nCorrecting row fringing area_1")
        area_array_list[0] = self.correctFringingRowsOfArea(area_array_list[0], image_correction_setting)

        # print("\n\nCorrecting row fringing area_2")
        area_array_list[1] = self.correctFringingRowsOfArea(area_array_list[1], image_correction_setting)

        # print("\n\nCorrecting row fringing area_3")
        area_array_list[2] = self.correctFringingRowsOfArea(area_array_list[2], image_correction_setting)

        # corrected_image_file_path = f"{self.pf.tpg_FC_directory_path}/{tpg_code}_FC.png"

        # corrected_area_image를 하나로 합성
        image = self.makeNewImageFromCorrectedArrays(image, area_array_list, area_size)
        return image
        # image.save(corrected_image_file_path)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

