from PIL import Image

# 이미지 열기
image = Image.open('image.png')

# 이미지 크기 얻기
width, height = image.size

# 검정색으로 만들 부분의 크기 설정
cut_width = 3
cut_height = 100

# 검정색으로 만들 부분의 위치 설정 (이미지의 상단에서 시작)
cut_area = (0, 0, cut_width, cut_height)

# 이미지의 해당 부분을 검정색으로 채우기
for x in range(cut_width):
    for y in range(cut_height):
        image.putpixel((x, y), (0, 0, 0))

# 변경된 이미지 저장
image.save('image_cut.png')
