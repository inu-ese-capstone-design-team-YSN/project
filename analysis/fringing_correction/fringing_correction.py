import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import json

from path_finder import PathFinder
pf = PathFinder()

np.set_printoptions(threshold=np.inf)

def getImageArraySize(image_setting):
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
    print(f"row_joint_surface_1: {row_joint_surface_1}")

    row_joint_surface_2 = height_1 + height_2
    area_2_start = row_joint_surface_2 - row_margin
    area_2_end = row_joint_surface_2 + row_margin
    print(f"row_joint_surface_2: {row_joint_surface_2}")

    row_joint_surface_3 = height_1 + height_2 + height_3
    area_3_start = row_joint_surface_3 - row_margin
    area_3_end = row_joint_surface_3 + row_margin
    print(f"row_joint_surface_3: {row_joint_surface_3}")

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

def getAreaArrayList(image, area_size):
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

    area_4_array = image_array[
        area_size[3][0]:area_size[3][1], 
        area_size[3][2]:area_size[3][3],
        :
    ]

    area_array_list = [
        area_1_array,
        area_2_array,
        area_3_array,
        area_4_array
    ]

    return area_array_list

def makeNewImageFromCorrectedArrays(image, area_array_list, area_size):
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

    # image_array[
    #     area_size[3][0]:area_size[3][1], 
    #     area_size[3][2]:area_size[3][3],
    #     :
    # ]

    return Image.fromarray(image_array)


def saveArrayImages(area_1_array, area_2_array, area_3_array, area_4_array):
    """
        2024.05.14, jdk
        네 개의 영역으로 잘린 이미지를 저장하는 함수
    """

    area_1_image = Image.fromarray(area_1_array)
    area_2_image = Image.fromarray(area_2_array)
    area_3_image = Image.fromarray(area_3_array)
    area_4_image = Image.fromarray(area_4_array)

    area_1_image.save(f"{temp_image_dir_path}/area_1{extension}")
    area_2_image.save(f"{temp_image_dir_path}/area_2{extension}")
    area_3_image.save(f"{temp_image_dir_path}/area_3{extension}")
    area_4_image.save(f"{temp_image_dir_path}/area_4{extension}")

def calcRowsBlueMean(area_array):
    """
        2024.05.16, jdk
        전달받은 array의 모든 row에 대해서
        B값 평균을 계산하고 이를 반환한다.
        반드시 3차원 array가 전달되어야 한다.
    """

    area_array_rows_B_mean = area_array[:, :, 2].mean(axis=1, keepdims=True)
    return area_array_rows_B_mean

def getFringingAnomalyPoint(rows_mean):
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

def getFringingRowsIdentifier(array_blue_mean, anomaly_high):
    """
        2024.05.14, jdk
        Anomaly Row의 index를 얻기 위하여 
        True False flatten array를 반환하는 함수
    """

    fringing_rows = array_blue_mean > anomaly_high
    return fringing_rows

def getFringingRowsIndicies(fringing_rows_identifier):
    # anomaly_rows의 index를 반환한다.
    # 이때, anomaly_rows_indicies[0]에 
    # 1차원 array가 들어가게 되므로 주의해야 함.
    fringing_rows_indicies = np.where(fringing_rows_identifier)[0]

    """
        2024.05.19, jdk
        반드시 height가 640이므로, 320을 기준으로 떨어져서 나타나는 것이 있다면
        fringing rows에서 배제해야 한다.
    """
    
    # 만약 길이가 2 이상인 경우
    if len(fringing_rows_indicies) >= 2:
        # 길이가 2 이상임에도 319와 320이 모두 들어있지 않은 경우 (잘못된 경우)
        # 현재 촬영 구조상 절대로 둘 중 하나도 포함되지 않을 수가 없음.
        if 319 not in fringing_rows_indicies and 320 not in fringing_rows_indicies:
            print("Something went wrong on choosing checking fringing_rows_indicies...")
            
            # fringing_rows_indicies를 None으로 설정하여
            # 별도의 보정을 수행하지 못하게 만든다.
            fringing_rows_indicies = None

        # 319와 320이 둘 다 들어있는 경우 (정상)
        elif 319 in fringing_rows_indicies and 320 in fringing_rows_indicies:
            print(f"fringing_rows_indicies-1: {fringing_rows_indicies}")

            lower_pruning_index = fringing_rows_indicies.index(319)
            upper_pruning_index = fringing_rows_indicies.index(320)

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

        # 둘 다 들어있거나 둘 다 들어있지 않은 경우는 위에서 필터링 됨.
        # 319와 320 둘 중 하나만 들어있는 경우 (정상이지만 판단 필요)
        elif 319 in fringing_rows_indicies:
            # 319만 들어있으므로, 319보다 큰 것은 들어있어서는 안됨.
            fringing_rows_indicies = fringing_rows_indicies[fringing_rows_indicies < 320]
            print(f"fringing_rows_indicies-2: {fringing_rows_indicies}")

            # 319부터 시작하여 1차이를 넘어서는 것은 잘라내야 함.
            lower_pruning_index = np.where(fringing_rows_indicies == 319)[0][0]
            
            for i in range(lower_pruning_index-1, -1, -1):
                lower_pruning_index = i+1

                if fringing_rows_indicies[i+1] - fringing_rows_indicies[i] > 1:
                    break  
            # 정해진 pruning 포인트에 따라서 index를 끊어준다.
            fringing_rows_indicies = fringing_rows_indicies[fringing_rows_indicies >= fringing_rows_indicies[lower_pruning_index]]

        elif 320 in fringing_rows_indicies:
            # 320만 들어있으므로, 319보다 작은 것은 들어있어서는 안됨.
            fringing_rows_indicies = fringing_rows_indicies[fringing_rows_indicies > 319]
            print(f"fringing_rows_indicies-3: {fringing_rows_indicies}")

            upper_pruning_index = np.where(fringing_rows_indicies == 320)[0][0]
            print(upper_pruning_index)
            
            for j in range(upper_pruning_index+1, len(fringing_rows_indicies)):
                upper_pruning_index = j-1

                if fringing_rows_indicies[j] - fringing_rows_indicies[j-1] > 1:
                    break
            
            # 정해진 pruning 포인트에 따라서 index를 끊어준다.
            fringing_rows_indicies = fringing_rows_indicies[fringing_rows_indicies <= fringing_rows_indicies[upper_pruning_index]]

    # 길이가 1 이하인 경우는 보정할 것이 없다고 판단.
    else:
        fringing_rows_indicies = []

    return fringing_rows_indicies

def getOuterBoundaryArray(area_array, bluing_rows_indicies):
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

def getOuterFringingPixelsAnomalyPoint(outer_calc_boundary_pixels):
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
        anomaly_points.append(anomaly_high)
    
    return anomaly_points

def isFringingPixel(anomaly_point_indicies, row_idx, col_idx):
    """
        2024.05.16, jdk
        전달받은 idx가 bluing pixel인지 아닌지 판단하는 함수
    """

    if row_idx in anomaly_point_indicies[0] and col_idx in anomaly_point_indicies[1]:
        return True
    else:
        return False

def getClosestNonFringingPixelByColumn(row_len, outer_anomaly_point, pixel_index):
    """
        2024.05.16, jdk
        전달받은 (i, j) 픽셀에서 row를 기준으로 
        bluing pixel이 아니고 가장 가까운 pixel의 값을 반환한다.
    """

    row_idx = pixel_index[0]
    col_idx = pixel_index[1]

    # bluing pixel이 존재하지 않는 경우는 거의 100% 존재하지 않으므로,
    # 별도의 예외 처리는 하지 않고, 반드시 값을 찾을 수 있는 것으로 가정한다.
    for cur_row_idx in range(row_idx+1, row_len):
        if isFringingPixel(outer_anomaly_point, cur_row_idx, col_idx):
            # 현재 찾는 row가 bluing pixel인 경우
            # 다음 row를 탐색하러 진행한다.
            continue
        else:  
            # 현재 찾는 row가 bluing pixel이 아닌 경우
            # 현재 pixel의 index를 반환한다.
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
    

def getClosestNonFringingPixelByRow(col_len, anomaly_point_indicies, pixel_index):
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
        if isFringingPixel(anomaly_point_indicies, row_idx, cur_col_idx):
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

def calcRGBMeanOfPixels(pixel_list):
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

def correctOuterFringingPixels(outer_array, anomaly_point_indicies):
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
                (row_idx_of_closest_row, col_idx_of_closest_row) = getClosestNonFringingPixelByRow(col_len, anomaly_point_indicies, pixel_index)

                if not (row_idx_of_closest_row == -1 and col_idx_of_closest_row == -1):
                    value_of_closest_row = outer_array[row_idx_of_closest_row, col_idx_of_closest_row, :]
                    pixels.append(value_of_closest_row)

                # non-bluing pixel인 가장 가까운 column pixel의 값을 가져온다.
                (row_idx_of_closest_col, col_idx_of_closest_col) = getClosestNonFringingPixelByColumn(row_len, anomaly_point_indicies, pixel_index)
                
                if not (row_idx_of_closest_col == -1 and col_idx_of_closest_col == -1):
                    value_of_closest_col = outer_array[row_idx_of_closest_col, col_idx_of_closest_col, :]
                    pixels.append(value_of_closest_col)

                interpolation_value = calcRGBMeanOfPixels(pixels)

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
                (row_idx_of_closest_col, col_idx_of_closest_col) = getClosestNonFringingPixelByColumn(row_len, anomaly_point_indicies, pixel_index)
                
                if not (row_idx_of_closest_col == -1 and col_idx_of_closest_col == -1):
                    value_of_closest_col = outer_array[row_idx_of_closest_row, col_idx_of_closest_col, :]
                    pixels.append(value_of_closest_col)

                # interpolation value로 평균을 계산한다.
                interpolation_value = calcRGBMeanOfPixels(pixels)

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
                (row_idx_of_closest_row, col_idx_of_closest_row) = getClosestNonFringingPixelByRow(col_len, anomaly_point_indicies, pixel_index)

                if not (row_idx_of_closest_row == -1 and col_idx_of_closest_row == -1):
                    value_of_closest_row = outer_array[row_idx_of_closest_row, col_idx_of_closest_row, :]
                    pixels.append(value_of_closest_row)
                
                # RGB 평균 반환
                interpolation_value = calcRGBMeanOfPixels(pixels)
                
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

                interpolation_value = calcRGBMeanOfPixels(pixels)

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
                (row_idx_of_closest_row, col_idx_of_closest_row) = getClosestNonFringingPixelByRow(col_len, anomaly_point_indicies, pixel_index)
            
                if row_idx_of_closest_row == -1 and col_idx_of_closest_row == -1:
                   value_of_closest_row = [0, 0, 0]
                else:
                   value_of_closest_row = outer_array[row_idx_of_closest_row, col_idx_of_closest_row, :]
                   pixels.append(value_of_closest_row)

                interpolation_value = calcRGBMeanOfPixels(pixels)
                # interpolation_value = left_upper_pixel_value 

                outer_array[i, j, :] = interpolation_value

    return outer_array

def interpolateFringingRowPixels(loop_start, loop_end, area_array_col_len, area_array):
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

                interpolation_value = calcRGBMeanOfPixels(pixels)
                area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

            # 마지막 column일 경우
            elif j == area_array_col_len-1:
                left_upper_pixel_row_idx = cur_row_idx - 1
                left_upper_pixel_col_idx = cur_col_idx - 1
                left_upper_pixel_value = area_array[left_upper_pixel_row_idx, left_upper_pixel_col_idx, :]

                upper_pixel_row_idx = cur_row_idx - 1
                upper_pixel_col_idx = cur_col_idx
                upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

                left_pixel_row_idx = cur_row_idx
                left_pixel_col_idx = cur_col_idx - 1
                left_pixel_value = area_array[left_pixel_row_idx, left_pixel_col_idx, :]

                pixels = []
                pixels.append(left_upper_pixel_value)
                pixels.append(upper_pixel_value)
                # pixels.append(left_pixel_value)

                interpolation_value = calcRGBMeanOfPixels(pixels)
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

                left_pixel_row_idx = cur_row_idx
                left_pixel_col_idx = cur_col_idx
                left_pixel_value = area_array[left_pixel_row_idx, left_pixel_col_idx, :]

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

                interpolation_value = calcRGBMeanOfPixels(pixels)
                area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

    return area_array

# def interpolateFringingRowPixelsFliped(loop_start, loop_end, area_array_col_len, area_array):
#     # 보정 시작
#     for i in range(loop_start, loop_end):
#         cur_row_idx = i

#         for j in range(area_array_col_len):
#             cur_col_idx = j

#             # 첫 column일 경우
#             if j == 0:
#                 upper_pixel_row_idx = cur_row_idx - 1
#                 upper_pixel_col_idx = cur_col_idx
#                 upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

#                 right_upper_pixel_row_idx = cur_row_idx - 1
#                 right_upper_pixel_col_idx = cur_col_idx + 1
#                 right_upper_pixel_value = area_array[right_upper_pixel_row_idx, right_upper_pixel_col_idx, :]

#                 pixels = []
#                 pixels.append(upper_pixel_value)
#                 pixels.append(right_upper_pixel_value)

#                 interpolation_value = calcRGBMeanOfPixels(pixels)
#                 area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

#             # 마지막 column일 경우
#             elif j == area_array_col_len-1:
#                 left_upper_pixel_row_idx = cur_row_idx - 1
#                 left_upper_pixel_col_idx = cur_col_idx - 1
#                 left_upper_pixel_value = area_array[left_upper_pixel_row_idx, left_upper_pixel_col_idx, :]

#                 upper_pixel_row_idx = cur_row_idx - 1
#                 upper_pixel_col_idx = cur_col_idx
#                 upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

#                 left_pixel_row_idx = cur_row_idx
#                 left_pixel_col_idx = cur_col_idx - 1
#                 left_pixel_value = area_array[left_pixel_row_idx, left_pixel_col_idx, :]

#                 pixels = []
#                 pixels.append(left_upper_pixel_value)
#                 pixels.append(upper_pixel_value)
#                 pixels.append(left_pixel_value)

#                 interpolation_value = calcRGBMeanOfPixels(pixels)
#                 area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

#             # 중간 column일 경우
#             else:
#                 left_upper_pixel_row_idx = cur_row_idx - 1
#                 left_upper_pixel_col_idx = cur_col_idx - 1
#                 left_upper_pixel_value = area_array[left_upper_pixel_row_idx, left_upper_pixel_col_idx, :]

#                 upper_pixel_row_idx = cur_row_idx - 1
#                 upper_pixel_col_idx = cur_col_idx
#                 upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

#                 right_upper_pixel_row_idx = cur_row_idx - 1
#                 right_upper_pixel_col_idx = cur_col_idx + 1
#                 right_upper_pixel_value = area_array[right_upper_pixel_row_idx, right_upper_pixel_col_idx, :]

#                 left_pixel_row_idx = cur_row_idx
#                 left_pixel_col_idx = cur_col_idx
#                 left_pixel_value = area_array[left_pixel_row_idx, left_pixel_col_idx, :]

#                 pixels = []
#                 pixels.append(left_upper_pixel_value)
#                 pixels.append(upper_pixel_value)
#                 pixels.append(right_upper_pixel_value)
#                 pixels.append(left_pixel_value)
                
#                 interpolation_value = calcRGBMeanOfPixels(pixels)
#                 area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

#     return area_array

def correctFringingRows(area_array, fringing_rows_indicies):
    """
        2024.05.14, jdk
        각 영역별로 찾아낸 anomaly_rows를 사용하여
        fringing row를 필터링하는 함수이다.
        anomaly_rows는 boolean list이다.
        
        2024.05.17, jdk
        fringing rows의 바깥으로는 Anomaly가 처리된 상황이다.
        좀 더 자연스러운 fringing rows 보간을 수행하기 위하여
        기존의 방식을 개선하도록 한다.

        - fringing_rows_indicies: fringing rows의 row indicies
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

        # bluing rows의 index를 넣고,
        # interpolateBluingRowPixels 함수 내에서
        # 값에 접근해 calibration을 하게 된다.
    
        row_idx_start = fringing_rows_indicies[0]
        row_idx_end = fringing_rows_indicies[int(fringing_rows_len/2)]

        area_array = interpolateFringingRowPixels(row_idx_start, row_idx_end+1, area_array_col_len, area_array)

        # 2024.05.19, jdk
        # 주변값 반영하도록 추가로 보정

        # 전체 이미지를 뒤집고 보정 시작
        area_array = flipArray(area_array)
        # 이미지를 뒤집었으므로 index를 변경해 주어야 함.
        # row만 신경쓰면 되기 때문에 bluing_rows_indicies list에서
        # bluing_rows_len/2+1부터 bluing_rows_len까지의 값들을 2500에서 빼준다. 
        # 그렇게 하고, 역순으로 돌려주면 된다. column은 그대로 0번부터 돌리면 된다.
        fringing_rows_indicies[int(fringing_rows_len/2)+1:fringing_rows_len] = (area_array_row_len - 1) - fringing_rows_indicies[int(fringing_rows_len/2)+1:fringing_rows_len]

        row_idx_start = fringing_rows_indicies[fringing_rows_len-1]
        row_idx_end = fringing_rows_indicies[int(fringing_rows_len/2)+1]

        # area_array = interpolateFringingRowPixelsFliped(row_idx_start, row_idx_end+1, area_array_col_len, area_array)
        area_array = interpolateFringingRowPixels(row_idx_start, row_idx_end+1, area_array_col_len, area_array)

        # 작업이 끝났으므로 flip한 index를 다시 돌려준다.
        fringing_rows_indicies[int(fringing_rows_len/2)+1:fringing_rows_len] = (area_array_row_len - 1) - fringing_rows_indicies[int(fringing_rows_len/2)+1:fringing_rows_len]

        # 보정이 끝났으므로 다시 뒤집는다.
        area_array = flipArray(area_array)

        corrected_fringing_rows = area_array[fringing_rows_indicies[0]:fringing_rows_indicies[-1] + 1]

        return corrected_fringing_rows

def flipArray(array):
    """
        2024.05.17, jdk
        전달받은 numpy array를
        상하좌우로 뒤집는 함수
    """

    fliped_array = array[::-1, ::-1, :]

    return fliped_array

def flipAnomalyIndicies(anomaly_indicies, row_len, col_len):
    """
        2024.05.17, jdk
        anomaly_indicies를 상하좌우 뒤집는 함수
    """

    anomaly_indicies = np.array(anomaly_indicies)

    anomaly_indicies[0] = row_len - anomaly_indicies[0] - 1
    anomaly_indicies[1] = col_len - anomaly_indicies[1] - 1
    
    # 완벽히 뒤집기 위해서는 sort를 하는게 아니라
    # reverse를 수행해야 함.
    anomaly_indicies[0] = anomaly_indicies[0][::-1]
    anomaly_indicies[1] = anomaly_indicies[1][::-1]
    
    return anomaly_indicies   

def correctFringingPixels(area_array):
    """
        2024.05.14, jdk
        fringing이 나타난 pixel을 보정하는 함수이다.
        fringing correction 전체 과정을 포함한다.
        
        parameter
        1) area_array: 전체 이미지에 대해서 특정 부분만 잘라낸 array
    """
    print(f"\n\narea_array.shape: {area_array.shape}")

    # 1) 
    # 전달 받은 하나의 area_array에 대해 각 row별 Blue Pixel의 평균을 계산한다.
    area_array_rows_B_mean = calcRowsBlueMean(area_array)

    # 2) 
    # BoxPlot을 통해 Bluing row의 평균에 대한 High Anomaly Point 획득
    area_array_rows_B_anomaly_high = getFringingAnomalyPoint(area_array_rows_B_mean.flatten()) 

    # 3) 
    # 얻어낸 Anomaly Point를 바탕으로 Anomaly Rows 정보를 Bool List로 반환
    # 즉, area_array의 row 길이 만큼에 대해서, anomaly가 발생한 부분은 True로, 
    # 아닌 부분은 False로 표현하는 Boolean List를 전달 받는다.
    fringing_rows_identifier = getFringingRowsIdentifier(area_array_rows_B_mean, area_array_rows_B_anomaly_high) 

    # 4) 
    # 얻어낸 bluing rows identifier(boolean list)를 바탕으로 
    # bluing rows의 indicies를 얻어낸다.
    # 해당 indicies는 area_array를 기준으로 하는 indicies이다.
    fringing_rows_indicies = getFringingRowsIndicies(fringing_rows_identifier)
    # => 실제로 area_array에서 어느 위치가 anomaly인지 index를 제공하는 함수이다.

    # 만약 fringing_rows_indicies가 empty array라면
    # 보정할 것이 없다고 판단한다.
    if len(fringing_rows_indicies) == 0:
        # anoamly로 잡힌 것이 없으니, 위 아래로 10픽셀 가량만 보정을 실시한다.
        fringing_rows_indicies = np.arange(319 - 20 + 1, 320 + 20)
        print("fringing_rows_indexes: [] (empty!). set default correction boundary")
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

        global fringing_row_margin
        
        # arange는 1을 더해주어야 함.
        print("current fringing rows indicies")
        print(f"min: {min_index}, max: {max_index}")
        print(fringing_rows_indicies)
        fringing_rows_indicies = np.arange(min_index - fringing_row_margin, max_index + fringing_row_margin + 1)
    
    """
        2024.05.19, jdk
        fringing row가 존재하지 않을 수 있으므로, 해당 경우에 대한 대처가 필요하다.
    """

    # 6) 
    # upper_outer_rows와 lower_outer_rows로 구성된 outer boundary array를 얻어낸다.
    upper_outer_array, lower_outer_array = getOuterBoundaryArray(area_array, fringing_rows_indicies)

    # 7-1) 
    # 얻어낸 upper/outer array에 대해서 Anomaly Point를 얻어낸다.
    # outer anomaly point는 outer array의 column 별로 구해낸다.
    # column 별로 B값에 대한 BoxPlot을 수행하여 Anomaly Point를 반환한다.
    upper_outer_anomaly_point = getOuterFringingPixelsAnomalyPoint(upper_outer_array)
    lower_outer_anomaly_point = getOuterFringingPixelsAnomalyPoint(lower_outer_array)

    # 위에서 얻어낸 Anomaly Point에 해당하는 pixel들의 index를 얻어낸다.
    upper_anomaly_indicies = np.where(upper_outer_array[:, :, 2] > upper_outer_anomaly_point)
    lower_anomaly_indicies = np.where(lower_outer_array[:, :, 2] > lower_outer_anomaly_point)

    # 7-2) 
    # upper area의 보정 코드를 활용하여
    # lower area를 보정을 수행하기 위해 lower_anomaly_indcies를 flip한다.
    lower_outer_area_row_len = lower_outer_array.shape[0]
    lower_outer_area_col_len = lower_outer_array.shape[1]

    # lower area를 flip했을 때 기준으로 anomaly index를 변환한다.
    lower_anomaly_indicies = flipAnomalyIndicies(lower_anomaly_indicies, lower_outer_area_row_len, lower_outer_area_col_len)

    # 8-1) 
    # 얻어낸 Anomaly Point를 바탕으로 upper outer Array의 Fringing Pixel을 보정한다.
    # Fringing Rows에 대한 보정보다 Outer Fringing Pixels에 대한 보정을 먼저 수행하는 이유는,
    # 상대적으로 Fringing Rows의 영역이 훨씬 크기 때문에 적절한 값을 확보하기 어렵기 때문이다.
    # 다시 말하면, Fringing rows는 interpolation 값을 얻기 위해 upper/lower outer array의 값을
    # 참조할 수밖에 없게 되는데, 이때 outer array fringing pixel의 값이 적절한 값이라고 신뢰할 수 없다면
    # fringing rows의 값도 신뢰할 수 없게 된다. 따라서 Outer Array의 값에 대한 interpolation을 우선적으로 
    # 수행하고, 그 이후에 fringing rows에 대한 interpolation을 수행해야 한다.
    upper_outer_array = correctOuterFringingPixels(upper_outer_array, upper_anomaly_indicies)
    
    # 8-2)
    # lower_outer_array를 flip하고 보정한다.
    lower_outer_array = flipArray(lower_outer_array)
    lower_outer_array = correctOuterFringingPixels(lower_outer_array, lower_anomaly_indicies)
    
    # 보정이 끝났으면 다시 flip 해서 원본으로 되돌린다.
    lower_outer_array = flipArray(lower_outer_array)
    
    # 9) 
    # Upper/Lower Outer Fringing Pixel에 대한 보정이 끝났으므로,
    # 이전에 구했던 Fringing Rows에 대한 보정을 수행한다.
    
    # 원본 array를 보정된 outer array로 변경한다.
    area_array[:fringing_rows_indicies[0], :, :] = upper_outer_array
    area_array[fringing_rows_indicies[-1]+1:, :, :] = lower_outer_array

    # 변경된 outer array에 대해서 fringing rows 보정을 수행한다.
    corrected_fringing_rows = correctFringingRows(area_array, fringing_rows_indicies)
    area_array[fringing_rows_indicies[0]:fringing_rows_indicies[-1]+1, :, :] = corrected_fringing_rows

    # 10)
    # Fringing Rows에 대한 보정을 수행하였으나, 현재까지는 다소 부자연스러운 이미지가 나타난다.
    # 이에 따라 자연스러운 이미지로 수정하기 위해 추가적인 보정을 수행한다.

    return area_array

# --------------------------------------------------------------------------- #

temp_image_dir_path = "./analysis/fringing_correction/temp_images"
tpg_combined_dir_path = "./image/tpg_image/combined"

tpg_code = "18-1025"
postfix = "combined"
extension = ".png"
corrected_image_file_name = f"fringing_corrected{extension}"
corrected_image_file_path = f"{temp_image_dir_path}/{corrected_image_file_name}"

image_file_name = f"{tpg_code}_{postfix}{extension}"
image_file_path = f"{tpg_combined_dir_path}/{image_file_name}"

image = Image.open(image_file_path)

# --------------------------------------------------------------------------- #

"""
    2024.05.15, jdk
    이미지 세팅 파일 분리를 통해 모듈화로 세팅값 가져오기
"""

image_settings_file_path = pf.image_settings_file_path
with open(image_settings_file_path, 'r', encoding='utf-8') as file:
    image_setting = json.load(file)

area_size = getImageArraySize(image_setting)
fringing_row_margin = image_setting['fringing_row_margin']

# 4개의 영역을 기반으로 Image를 분할하여 얻은 numpy array를 갖고 있는 list
area_array_list = getAreaArrayList(image, area_size)
area_array_num = len(area_array_list)

# --------------------------------------------------------------------------- #

# correction 실시
area_array_list[0] = correctFringingPixels(area_array_list[0])
area_array_list[1] = correctFringingPixels(area_array_list[1])
area_array_list[2] = correctFringingPixels(area_array_list[2])

# --------------------------------------------------------------------------- #

# image 합성
image = makeNewImageFromCorrectedArrays(image, area_array_list, area_size)
image.save(corrected_image_file_path)