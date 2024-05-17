import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import json

from path_finder import PathFinder
pf = PathFinder()

np.set_printoptions(threshold=np.inf)

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

def getBluingRowsAnomalyPoint(rows_mean):
    """
        2024.05.14, jdk
        전달받은 data의 BoxPlot Anomaly Point를 계산하는 함수
        B값이 강한 부분만 찾게 되므로, whishi만 반환한다.
    """

    # boxplot 통계량 계산
    stats = boxplot_stats(rows_mean, whis=0.8)
    plt.close()  # plot을 그리지 않도록 닫기
    
    # 이상치 추출
    anomaly_high = stats[0]['whishi']
    return anomaly_high

def getBluingRowsIdentifier(array_blue_mean, anomaly_high):
    """
        2024.05.14, jdk
        Anomaly Row의 index를 얻기 위하여 
        True False flatten array를 반환하는 함수
    """

    bluing_rows = array_blue_mean > anomaly_high
    return bluing_rows

def getBluingRowsIndicies(bluing_rows_identifier):
    # anomaly_rows의 index를 반환한다.
    # 이때, anomaly_rows_indicies[0]에 
    # 1차원 array가 들어가게 되므로 주의해야 함.
    bluing_rows_indicies = np.where(bluing_rows_identifier)[0]
    # print(f"bluing_rows_indicies: {bluing_rows_indicies}\n\n")

    return bluing_rows_indicies

def getOuterBoundaryArray(area_array, bluing_rows_indicies):
    """ 
        2024.05.16, jdk
        area_array로부터 outer_array를 반환받는 함수
        upper/lower outer boundary array를 구분하여 반환한다.
    """
    
    upper_row_index = bluing_rows_indicies[0]
    lower_row_index = bluing_rows_indicies[-1]
    # print(f"upper_row_index: {upper_row_index}")
    # print(f"lower_row_index: {lower_row_index}\n\n")

    upper_outer_array = area_array[:upper_row_index]
    lower_outer_array = area_array[lower_row_index+1:]
    
    return upper_outer_array, lower_outer_array

def getOuterBluingPixelsAnomalyPoint(outer_calc_boundary_pixels):
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
        stats = boxplot_stats(blue_pixels[:, i], whis=1.5)
        plt.close()  # plot을 그리지 않도록 닫기
    
        # 이상치 추출
        anomaly_high = stats[0]['whishi']
        anomaly_points.append(anomaly_high)
    
    return anomaly_points

def isBluingPixel(anomaly_point_indicies, row_idx, col_idx):
    """
        2024.05.16, jdk
        전달받은 idx가 bluing pixel인지 아닌지 판단하는 함수
    """

    if row_idx in anomaly_point_indicies[0] and col_idx in anomaly_point_indicies[1]:
        return True
    else:
        return False

def getClosestNonBluingPixelByColumn(row_len, outer_anomaly_point, pixel_index):
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
        if isBluingPixel(outer_anomaly_point, cur_row_idx, col_idx):
            # 현재 찾는 row가 bluing pixel인 경우
            # 다음 row를 탐색하러 진행한다.
            continue
        else:  
            # 현재 찾는 row가 bluing pixel이 아닌 경우
            # 현재 pixel의 index를 반환한다.
            return (cur_row_idx, col_idx)
    
    # 2024.05.16, jdk
    # 문제가 발생할 경우, 연산이 가능하도록
    # 자기 자신의 index를 그대로 반환한다.
    # 단, 문제가 발생한 것을 알 수 있도록
    # pirint를 지정해 둔다.
    # print("Not Exists! (getClosestNonBluingPixelByColumn)")
    return (-1, -1)
    

def getClosestNonBluingPixelByRow(col_len, anomaly_point_indicies, pixel_index):
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
        if isBluingPixel(anomaly_point_indicies, row_idx, cur_col_idx):
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
    
    # print(f"{row_idx} {col_idx} Not Exists! (getClosestNonBluingPixelByRow)")
    return (-1, -1)

def calcRGBMeanOfPixels(pixel_list):
    """
        2024.05.17, jdk
        전달받은 pixel list에서
        rgb 평균을 계산하고 반환
    """
    list_num = len(pixel_list)

    if list_num == 0:
        print("mean of rgb lists is 0!")

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

def calibrateOuterBluingPixels(outer_array, anomaly_point_indicies):
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
                (row_idx_of_closest_row, col_idx_of_closest_row) = getClosestNonBluingPixelByRow(col_len, anomaly_point_indicies, pixel_index)

                if row_idx_of_closest_row == -1 and col_idx_of_closest_row == -1:
                    value_of_closest_row = [0, 0, 0]
                else:
                    value_of_closest_row = outer_array[row_idx_of_closest_row, col_idx_of_closest_row, :]
                    pixels.append(value_of_closest_row)

                # non-bluing pixel인 가장 가까운 column pixel의 값을 가져온다.
                (row_idx_of_closest_col, col_idx_of_closest_col) = getClosestNonBluingPixelByColumn(row_len, anomaly_point_indicies, pixel_index)
                
                if row_idx_of_closest_col == -1 and col_idx_of_closest_col == -1:
                     value_of_closest_col = [0, 0, 0]
                else:
                    value_of_closest_col = outer_array[row_idx_of_closest_col, col_idx_of_closest_col, :]
                    pixels.append(value_of_closest_col)

                # 2024.05.17, jdk
                # interpolation_value로 평균을 계산한다.
                # 이 경우, [0, 0, 0] 이 반환될 가능성이 있지만
                # 존재할 확률이 거의 없으므로, print만 찍고 넘어간다.
                interpolation_value = calcRGBMeanOfPixels(pixels)
                
                # 현재 pixel을 interpolation value로 교체한다;
                outer_array[i, j, :] = interpolation_value

            # 1-2) 첫 번째 row이고, 첫 번째 col이 아닌 경우
            # 왼쪽 pixel의 rgb와 non-bluing이며 같은 column에서 가장 가까운 pixel의
            # rgb의 평균으로 변경한다.
            elif j == col_len-1:

                pixels = []

                # 왼쪽 pixel의 index를 가져온다.
                left_pixel_row_idx = pixel_index[0] - 1
                left_pixel_col_idx = pixel_index[1]

                # 왼쪽 pixel의 값을 가져온다.
                left_pixel_value = outer_array[left_pixel_row_idx, left_pixel_col_idx, :]
                pixels.append(left_pixel_value)

                # non-bluing pixel인 가장 가까운 column pixel의 값을 가져온다.
                (row_idx_of_closest_col, col_idx_of_closest_col) = getClosestNonBluingPixelByColumn(row_len, anomaly_point_indicies, pixel_index)
                
                if row_idx_of_closest_col == -1 and row_idx_of_closest_col == -1:
                    value_of_closest_col = [0, 0, 0]
                else:
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
                (row_idx_of_closest_row, col_idx_of_closest_row) = getClosestNonBluingPixelByRow(col_len, anomaly_point_indicies, pixel_index)

                if row_idx_of_closest_row == -1 and col_idx_of_closest_row == -1:
                    value_of_closest_row = [0, 0, 0]
                else:
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
                (row_idx_of_closest_row, col_idx_of_closest_row) = getClosestNonBluingPixelByRow(col_len, anomaly_point_indicies, pixel_index)
            
                if row_idx_of_closest_row == -1 and col_idx_of_closest_row == -1:
                   value_of_closest_row = [0, 0, 0]
                else:
                   value_of_closest_row = outer_array[row_idx_of_closest_row, col_idx_of_closest_row, :]
                   pixels.append(value_of_closest_row)

                interpolation_value = calcRGBMeanOfPixels(pixels)
                # interpolation_value = left_upper_pixel_value 

                outer_array[i, j, :] = interpolation_value

    return outer_array

def interpolateBluingRowPixels(loop_start, loop_end, area_array_col_len, area_array):

    # loop start와 loop end는 ok
    print(f"loop start: {loop_start}")
    print(f"loop end: {loop_end}")

    # 보정 시작
    for i in range(loop_start, loop_end):
        for j in range(area_array_col_len):
            cur_row_idx = i
            cur_col_idx = j

            # 첫 column일 경우
            if j == 0:
                upper_pixel_row_idx = cur_row_idx - 1
                upper_pixel_col_idx = cur_col_idx
                upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

                right_upper_pixel_row_idx = cur_row_idx - 1
                right_upper_pixel_col_idx = cur_col_idx + 1
                right_upper_pixel_value = area_array[right_upper_pixel_row_idx, right_upper_pixel_col_idx, :]

                interpolation_value = np.round((upper_pixel_value + right_upper_pixel_value)/2)
                area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

            # 마지막 column일 경우
            elif j == area_array_col_len-1:
                left_upper_pixel_row_idx = cur_row_idx - 1
                left_upper_pixel_col_idx = cur_col_idx - 1
                left_upper_pixel_value = area_array[left_upper_pixel_row_idx, left_upper_pixel_col_idx, :]

                upper_pixel_row_idx = cur_row_idx - 1
                upper_pixel_col_idx = cur_col_idx
                upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

                interpolation_value = np.round((left_upper_pixel_value + upper_pixel_value)/2)
                area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

            # 중간 column일 경우
            else:
                left_upper_pixel_row_idx = cur_row_idx - 1
                left_upper_pixel_col_idx = cur_col_idx - 1
                left_upper_pixel_value = area_array[left_upper_pixel_row_idx, left_upper_pixel_col_idx, :]
                # print(f"left_upper_pixel_row_idx: {left_upper_pixel_row_idx}, left_upper_pixel_col_idx: {left_upper_pixel_col_idx}, left_upper_pixel_value: {left_upper_pixel_value}")

                upper_pixel_row_idx = cur_row_idx - 1
                upper_pixel_col_idx = cur_col_idx
                upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]
                # print(f"upper_pixel_row_idx: {upper_pixel_row_idx}, upper_pixel_col_idx: {upper_pixel_col_idx}, upper_pixel_value: {upper_pixel_value}")


                right_upper_pixel_row_idx = cur_row_idx - 1
                right_upper_pixel_col_idx = cur_col_idx + 1
                right_upper_pixel_value = area_array[right_upper_pixel_row_idx, right_upper_pixel_col_idx, :]
                # print(f"right_upper_pixel_row_idx: {right_upper_pixel_row_idx}, right_upper_pixel_col_idx: {right_upper_pixel_col_idx}, right_upper_pixel_value: {right_upper_pixel_value}")

                pixels = []
                pixels.append(left_upper_pixel_value)
                pixels.append(upper_pixel_value)
                pixels.append(right_upper_pixel_value)

                """ 
                    2024.05.17, jdk
                    에러가 났던 이유는 numpy array가 아니기 때문에
                    boradcasting이 불가능해서 column wise 덧셈이
                    안되었기 때문이다...
                """

                interpolation_value = calcRGBMeanOfPixels(pixels)
                area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

    return area_array

def interpolateBluingRowPixelsFliped(loop_start, loop_end, area_array_col_len, area_array):

    # loop start와 loop end는 ok
    # print(f"fliped_loop_start: {loop_start}")
    # print(f"fliped_loop_end: {loop_end}")

    # 보정 시작
    for i in range(loop_start, loop_end):
        for j in range(area_array_col_len):
            cur_row_idx = i
            cur_col_idx = j

            # 첫 column일 경우
            if j == 0:
                upper_pixel_row_idx = cur_row_idx - 1
                upper_pixel_col_idx = cur_col_idx
                upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

                right_upper_pixel_row_idx = cur_row_idx - 1
                right_upper_pixel_col_idx = cur_col_idx + 1
                right_upper_pixel_value = area_array[right_upper_pixel_row_idx, right_upper_pixel_col_idx, :]

                interpolation_value = np.round((upper_pixel_value + right_upper_pixel_value)/2)
                # print(f"upper_pixel_value: {upper_pixel_value}, right_upper_pixel_value: {right_upper_pixel_value}, interpolation_value: {interpolation_value}")
                area_array[cur_row_idx, cur_col_idx, :] = interpolation_value

            # 마지막 column일 경우
            elif j == area_array_col_len-1:
                left_upper_pixel_row_idx = cur_row_idx - 1
                left_upper_pixel_col_idx = cur_col_idx - 1
                left_upper_pixel_value = area_array[left_upper_pixel_row_idx, left_upper_pixel_col_idx, :]

                upper_pixel_row_idx = cur_row_idx - 1
                upper_pixel_col_idx = cur_col_idx
                upper_pixel_value = area_array[upper_pixel_row_idx, upper_pixel_col_idx, :]

                interpolation_value = np.round((left_upper_pixel_value + upper_pixel_value)/2)
                # print(f"upper_pixel_value: {upper_pixel_value}, right_upper_pixel_value: {left_upper_pixel_value}, interpolation_value: {interpolation_value}")

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

                pixels = []
                pixels.append(left_upper_pixel_value)
                pixels.append(upper_pixel_value)
                pixels.append(right_upper_pixel_value)
                
                interpolation_value = calcRGBMeanOfPixels(pixels)
                area_array[cur_row_idx, cur_col_idx, :] = interpolation_value
                # print(f"upper_pixel_value: {upper_pixel_value}, upper_pixel_value: {upper_pixel_value}, right_upper_pixel_value: {right_upper_pixel_value}, interpolation_value: {interpolation_value}")

    return area_array

def calibrateBluingRows(area_array, bluing_rows_indicies):
    """
        2024.05.14, jdk
        각 영역별로 찾아낸 anomaly_rows를 사용하여
        bluing row를 필터링하는 함수이다.
        anomaly_rows는 boolean list이다.
        
        2024.05.17, jdk
        bluing rows 위로는 Anomaly가 처리된 상황이다.
        좀 더 자연스러운 bluing rows 보간을 수행하기 위하여
        기존의 방식을 개선하도록 한다.

        - bluing_rows_indicies: bluing rows의 row indicies
    """

    # print(f"bluing_rows_indicies: {bluing_rows_indicies}")
    bluing_rows_len = len(bluing_rows_indicies) # anomaly row의 개수

    # 1) row가 없는 경우 
    if bluing_rows_len == 0:
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
    if bluing_rows_len == 1:

        row_index = bluing_rows_indicies[0]

        # row가 한 개만 있으므로, 위 아래의 평균을 계산하여 보정한다.
        upper_rgb = area_array[row_index-1]
        lower_rgb = area_array[row_index+1]
        
        # 평균값은 반올림을 통해 소수점 제거
        mean_rgb = (upper_rgb + lower_rgb)/2
        mean_rgb = np.round(mean_rgb)

        area_array[row_index] = mean_rgb        

    # 3) row가 두 개 이상인 경우.
    # 연속적이지 않은 row에 대해서는 정상적으로 동작할 수 없으므로 주의가 필요하다.
    
    if bluing_rows_len >= 2:
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
        # print(f"bluing_rows_len: {bluing_rows_len}")
    
        row_idx_start = bluing_rows_indicies[0]
        row_idx_end = bluing_rows_indicies[int(bluing_rows_len/2)]

        area_array = interpolateBluingRowPixels(row_idx_start, row_idx_end+1, area_array_col_len, area_array)

        # 전체 이미지를 뒤집고 보정 시작
        area_array = flipArray(area_array)

        # 이미지를 뒤집었으므로 index를 변경해 주어야 함.
        # row만 신경쓰면 되기 때문에 bluing_rows_indicies list에서
        # bluing_rows_len/2+1부터 bluing_rows_len까지의 값들을 2500에서 빼준다. 
        # 그렇게 하고, 역순으로 돌려주면 된다. column은 그대로 0번부터 돌리면 된다.
        # print("here")
        # print(bluing_rows_indicies[int(bluing_rows_len/2)+1:bluing_rows_len])
        bluing_rows_indicies[int(bluing_rows_len/2)+1:bluing_rows_len] = (area_array_row_len - 1) - bluing_rows_indicies[int(bluing_rows_len/2)+1:bluing_rows_len]
        # print(bluing_rows_indicies[int(bluing_rows_len/2)+1:bluing_rows_len])

        row_idx_start = bluing_rows_indicies[bluing_rows_len-1]
        row_idx_end = bluing_rows_indicies[int(bluing_rows_len/2)+1]

        area_array = interpolateBluingRowPixelsFliped(row_idx_start, row_idx_end+1, area_array_col_len, area_array)

        # 작업이 끝났으므로 flip한 index를 다시 돌려준다.
        bluing_rows_indicies[int(bluing_rows_len/2)+1:bluing_rows_len] = (area_array_row_len - 1) - bluing_rows_indicies[int(bluing_rows_len/2)+1:bluing_rows_len]
        
        # 보정이 끝났으므로 다시 뒤집는다.
        area_array = flipArray(area_array)

        # print(f"bluing_rows_indicies[0]: {bluing_rows_indicies[0]}")
        # print(f"bluing_rows_indicies[-1]: {bluing_rows_indicies[-1]}")
        calibrated_bluing_rows = area_array[bluing_rows_indicies[0]:bluing_rows_indicies[-1] + 1]

        return calibrated_bluing_rows

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

    # print(row_len)
    # print(col_len)

    anomaly_indicies = np.array(anomaly_indicies)
    print(len(anomaly_indicies[0]))

    anomaly_indicies[0] = row_len - anomaly_indicies[0] - 1
    anomaly_indicies[1] = col_len - anomaly_indicies[1] - 1
    
    # sort를 하면 안된다. sort를 하면 순서가 엉망이 되어버림.
    # row 순서
    anomaly_indicies[0] = anomaly_indicies[0][::-1]
    anomaly_indicies[1] = anomaly_indicies[1][::-1]
    
    return anomaly_indicies   

def bluingPixelCalibration(area_array):
    """
        2024.05.14, jdk
        Bluing Pixel을 보정하는 함수
        Bluing Pixel Calibration 전체 과정을 포함한다.
    """

    # 1) 전달 받은 하나의 area_array에 대해 각 row별 Blue Pixel의 평균을 계산한다.
    area_array_rows_B_mean = calcRowsBlueMean(area_array)

    # 2) BoxPlot을 통해 Bluing row의 평균에 대한 High Anomaly Point 획득
    area_array_rows_B_anomaly_high = getBluingRowsAnomalyPoint(area_array_rows_B_mean.flatten()) 

    # 3) 얻어낸 Anomaly Point를 바탕으로 Anomaly Rows 정보를 Bool List로 반환
    bluing_rows_identifier = getBluingRowsIdentifier(area_array_rows_B_mean, area_array_rows_B_anomaly_high) 

    # 4) 얻어낸 bluing rows identifier(boolean list)를 바탕으로 
    # bluing rows의 indicies를 얻어낸다.
    # 해당 indicies는 area_array를 기준으로 하는 indicies이다.
    bluing_rows_indicies = getBluingRowsIndicies(bluing_rows_identifier)
    # => 실제로 area_array에서 어느 위치가 anomaly인지 index를 제공하는 함수이다.

    # 5) outer boundary array를 얻어낸다.
    # upper_outer_rows와
    # lower_outer_rows를 반환한다.
    upper_outer_array, lower_outer_array = getOuterBoundaryArray(area_array, bluing_rows_indicies)

    # 6) 얻어낸 upper/outer array에 대해서 Anomaly Point를 얻어낸다.
    upper_outer_anomaly_point = getOuterBluingPixelsAnomalyPoint(upper_outer_array)
    lower_outer_anomaly_point = getOuterBluingPixelsAnomalyPoint(lower_outer_array)

    upper_anomaly_indicies = np.where(upper_outer_array[:, :, 2] > upper_outer_anomaly_point)
    lower_anomaly_indicies = np.where(lower_outer_array[:, :, 2] > lower_outer_anomaly_point)

    # print(f"lower_anomaly_indicies[0][0]: {lower_anomaly_indicies[0][0]}")

    lower_outer_area_row_len = lower_outer_array.shape[0]
    lower_outer_area_col_len = lower_outer_array.shape[1]

    lower_anomaly_indicies = flipAnomalyIndicies(lower_anomaly_indicies, lower_outer_area_row_len, lower_outer_area_col_len)
    # print(f"lower_anomaly_indicies[0][0]: {lower_anomaly_indicies[0][0]}")

    # 7) 얻어낸 Anomaly Point를 바탕으로 Outer Array의 Bluing Pixel에 대한 Calibration을 진행한다.
    upper_outer_array = calibrateOuterBluingPixels(upper_outer_array, upper_anomaly_indicies)
    
    # 보정된 lower_outer_array를 구하기 위하여
    # lower_outer_array를 flip하고 calibration을 진행한다.
    lower_outer_array = flipArray(lower_outer_array)
    lower_outer_array = calibrateOuterBluingPixels(lower_outer_array, lower_anomaly_indicies)
    
    # 보정이 끝났으면 다시 flip 해서 원본으로 되돌린다.
    lower_outer_array = flipArray(lower_outer_array)
    
    # 8) Bluing Pixel에 대한 Calibration이 끝났으므로,
    # Bluing Row에 대한 Calibration 실시
    
    # print(f"area_array.shape: {area_array.shape}")
    # print(f"upper_outer_array.shape: {upper_outer_array.shape}")
    # print(f"{bluing_rows_indicies[0]}:{bluing_rows_indicies[-1]}")
    # print(f"lower_outer_array.shape: {lower_outer_array.shape}")
    area_array[:bluing_rows_indicies[0], :, :] = upper_outer_array
    area_array[bluing_rows_indicies[-1]+1:, :, :] = lower_outer_array

    calibrated_bluing_rows = calibrateBluingRows(area_array, bluing_rows_indicies)
    area_array[bluing_rows_indicies[0]:bluing_rows_indicies[-1]+1, :, :] = calibrated_bluing_rows

    # 9) Calibration이 끝났으므로, 보정된 반환
    return area_array
    

# --------------------------------------------------------------------------- #

# temp_image_dir_path = "./analysis/bluing_pixel/temp_images"
# tpg_combined_dir_path = pf.tpg_combined_dir_path

temp_image_dir_path = "./analysis/bluing_pixel/temp_images"
tpg_combined_dir_path = "./image/tpg_image/combined"

tpg_code = "19-1606"
postfix = "combined"
extension = ".png"

image_file_name = f"{tpg_code}_{postfix}{extension}"
image_file_path = f"{tpg_combined_dir_path}/{image_file_name}"

image = Image.open(image_file_path)

# (2600, 2540, 3)
# print(image_array.shape)

# --------------------------------------------------------------------------- #

# 자를 영역의 사이즈
# numpy에서 이미지를 자를 때는
# image_array[area_size[0][0]:area_size[0][1], :, :]과 같이 지정할 경우,
# area_size[0][1]까지 잘리게 된다.

"""
    2024.05.15, jdk
    이미지 세팅 파일 분리를 통해 모듈화로 세팅값 가져오기
"""
image_settings_file_path = pf.image_settings_file_path
with open(image_settings_file_path, 'r', encoding='utf-8') as file:
    image_setting = json.load(file)

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

row_joint_surface_2 = height_1 + height_2
area_2_start = row_joint_surface_2 - row_margin
area_2_end = row_joint_surface_2 + row_margin

row_joint_surface_3 = height_1 + height_2 + height_3
area_3_start = row_joint_surface_3 - row_margin
area_3_end = row_joint_surface_3 + row_margin

col_joint_surface = int(width / 2)
area_4_start = col_joint_surface - col_margin
area_4_end = col_joint_surface + col_margin

area_size = [
    [area_1_start, area_1_end, 0, width],
    [area_2_start, area_2_end, 0, width],
    [area_3_start, area_3_end, 0, width],
    [0, height, area_4_start, area_4_end]
]

# 4개의 영역을 기반으로 Image를 분할하여 얻은 numpy array를 갖고 있는 list
area_array_list = getAreaArrayList(image, area_size)
area_array_num = len(area_array_list)

# --------------------------------------------------------------------------- #

# Save Images
# saveArrayImages(area_1_array, area_2_array, area_3_array, area_4_array)

# --------------------------------------------------------------------------- #

# Calibration 실시
area_array_list[0] = bluingPixelCalibration(area_array_list[0])
area_array_image = Image.fromarray(area_array_list[0])
area_array_image.save(f"{temp_image_dir_path}/bluing_calibrated.png")

# --------------------------------------------------------------------------- #
