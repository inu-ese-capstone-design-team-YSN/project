import subprocess # 외부 프로세스를 실행하고 그 결과를 다루기 위해 사용
import os # 운영체제와 상호작용을 위한 모듈, 파일 및 디렉토리 관리에 사용
import sys # 시스템 관련 파라미터와 함수를 다루기 위해 사용
import curses # 터미널 핸들링을 위해 사용, 사용자와 대화형으로 상호작용하는 텍스트 인터페이스 구성에 활용
from google.cloud import storage   # Google Cloud Storage 서비스를 사용하기 위한 클라이언트 라이브러리

# 프로젝트의 data_class 디렉토리를 모듈 검색 경로에 추가하여 해당 디렉토리의 모듈을 사용 가능하게 함

from path_finder import PathFinder  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스
from cloud_controller import CloudController  # 클라우드 관련 작업을 관리하는 클래스

# ----------------------------------------------------------------------------- #

'''
2024.04.17, kwc

SWATCH, TPG, TCX별로 촬영 모드를 선택하여
촬영 시 이미지 이름 설정, 이미지 로그 기록,
GCS 자동 업로드 기능 구현을 동작하는 총체적인 프로그램 구현
'''

'''
변수 설명:

- preview_command: 카메라 미리보기를 시작하는 외부 명령어
- capture_command: 이미지를 캡처하는 외부 명령어로, 실행 시 파일명이 동적으로 설정

- preview_stop_key: 미리보기를 중단하는데 사용되는 키
- user_input_exit: 프로그램을 종료하는데 사용되는 사용자 입력 키
- user_input_capture: 이미지 캡처를 실행하는데 사용되는 사용자 입력 키
- user_input_preview: 미리보기 모드를 시작하는데 사용되는 사용자 입력 키

- image_file_extension: 캡처된 이미지 파일의 확장자를 정의하는 문자열, 기본적으로 '.jpg'로 설정


- stdscr: curses 라이브러리에서 사용되는 표준 화면 객체로, 모든 출력과 입력은 이 객체를 통해 화면에 반영

- image_type: 사용자가 선택한 이미지 유형('1'은 Swatch, '2'는 TPG, '3'은 TCX)을 저장하는 변수
- cmd_input: 메인 루프에서 사용자로부터 받은 명령을 저장하는 변수
- directory: 선택된 이미지 유형에 따라 이미지가 저장될 경로를 저장하는 변수
- image_name: 캡처된 이미지의 이름을 저장하는 변수
- upload_path: 클라우드에 업로드할 때 사용될 경로를 저장하는 변수
- file_name: 캡처 명령을 실행할 때 사용되는 전체 파일 경로를 저장하는 변수
'''

''' 클래스 인스턴스 생성 및 사용 '''
path_finder = PathFinder() # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스
path_finder.ensureDirectoriesExist() # 
cloud_controller = CloudController(path_finder) # 클라우드 관련 작업을 관리하는 클래스

''' 카메라 커맨드 지정 '''
# 카메라 미리보기 커맨드 설정, '-t 0'은 타이머 없음을 의미
preview_command = ['rpicam-hello', '-t', '0'] 
# 이미지 캡처 커맨드, '-o'는 출력 파일 경로, '-t 1000'은 1000ms 동안 실행
capture_command = ['rpicam-still', '-o', '','-t', '100', '-n'] 
# libcamera-still -o output3.jpg --awb daylight --shutter 14000 --gain 1 --contrast 1 --saturation 1 --nopreview

''' 사용자 입력에 대한 처리를 위한 키 설정 '''
preview_stop_key = '0'  # 미리보기 종료 키
user_input_exit = '9'  # 사용자 입력: 종료
user_input_capture = '2'  # 사용자 입력: 이미지 캡처
user_input_preview = '1'  # 사용자 입력: 미리보기 시작
user_input_delete = '5'  # 사용자 입력: 이미지 삭제

# 저장될 이미지의 파일 확장자 설정
image_file_extension = '.png'  # 이미지 파일 확장자로 '.jpg'를 기본값으로 설정

# ----------------------------------------------------------------------------- #

def readImageInfo():
    """
    이미지 번호 파일에서 마지막 번호를 읽어오는 함수
    파일이 존재하지 않을 경우 초기 값을 설정하여 반환
    """
    try:
        # 이미지 번호 파일을 읽기 모드로 열어 처리
        with open(path_finder.image_number_file_path, 'r') as file:
            data = file.readline().strip()
            if data:
                parts = data.split(',')
                # 부족한 요소가 있으면 기본 값을 추가
                while len(parts) < 3:
                    parts.append('')
                return parts
            else:
                return ['1', '', '']  # Swatch는 숫자를, TPG와 TCX는 코드를 사용
        # 파일이 존재하지 않을 경우 예외 처리
    except FileNotFoundError:
        with open(path_finder.image_number_file_path, 'w') as file:
            file.write('1,,')
        return ['1', '', '']


def saveImageInfo(swatch_number, tpg_code, tcx_code):
    """
    캡처된 이미지의 번호를 파일에 저장하는 함수
    캡처마다 번호를 증가시켜 파일에 저장
    """
    
    # 이미지 번호 파일을 쓰기 모드로 열어 처리
    with open(path_finder.image_number_file_path, 'w') as file:
        file.write(f"{swatch_number},{tpg_code},{tcx_code}\n") 
        # 전달받은 이미지 번호를 문자열로 변환하여 파일에 기록

def validate_and_format_image_name(stdscr, image_type, last_tpg_code, tpg_count):

    """
    사용자로부터 이미지 이름을 입력 받고, 적절한 형식으로 변환하는 함수
    입력 형식은 '00-0000' 이며, curses 라이브러리를 사용하여 터미널에서 입력을 받음
    """
    curses.noecho()  # 사용자 입력을 화면에 바로 표시하지 않도록 설정
    stdscr.clear() # 화면을 초기화

    if image_type == '2' and tpg_count % 8 != 1:
        formatted_name = f"{last_tpg_code}_{tpg_count}"
    else:
        stdscr.addstr("이미지 파일 이름을 입력하시오 (형식 '00-0000'):\n")
        # 입력 형식 초기 설정, 하이픈 포함
        formatted_input = [" ", " ", "-", " ", " ", " ", " "] 
        stdscr.addstr("".join(formatted_input) + "\r") # 초기 형식 화면에 표시
        stdscr.refresh()

        # 하이픈을 제외한 실제 입력 받을 위치
        cursor_positions = [0, 1, 3, 4, 5, 6]  # 하이픈을 제외한 입력 위치
        position_index = 0  # cursor_positions에서의 위치 인덱스

        while position_index < 6:  # 총 6개의 숫자 입력
            ch = stdscr.getch() # 키 입력 받기
            # 백스페이스 처리
            if ch == curses.KEY_BACKSPACE or ch == 127 or ch == 8:
                if position_index > 0:  # 입력된 위치가 있을 경우에만 백스페이스 처리
                    # 커서 위치를 하나 뒤로 이동
                    position_index -= 1  
                    # 커서 위치가 하이픈 바로 다음이면 하이픈 위치로 추가 조정
                    if cursor_positions[position_index] == 3:
                        position_index -= 1
                    # 이동한 위치의 문자를 공백으로 설정
                    formatted_input[cursor_positions[position_index]] = ' ' 
                    stdscr.addstr(1, 0, "".join(formatted_input) + "\r")  # 화면 갱신
                    stdscr.move(1, cursor_positions[position_index])  # 커서 위치 조정
            # 숫자 입력 처리
            elif ch >= ord('0') and ch <= ord('9'):
                # 입력 위치에 숫자 저장
                formatted_input[cursor_positions[position_index]] = chr(ch)
                position_index += 1
                # 화면에 현재 형식 다시 표시
                stdscr.addstr(1, 0, "".join(formatted_input) + "\r")
                # 다음 입력 위치로 커서 이동
                if position_index < 6:
                    stdscr.move(1, cursor_positions[position_index])

            if position_index >= 6:  # 모든 숫자 입력 완료
                break

        stdscr.refresh() # 화면 최종 갱신
        # 최종적으로 형성된 문자열 추출
        formatted_name = "".join(formatted_input).strip() 
        # 최종 입력된 이름 출력
        stdscr.addstr(3, 0, "입력된 이름: " + formatted_name + "\n")
        formatted_name = "".join(formatted_input).strip() + "_1"
        
    # 처리된 이름 반환
    return formatted_name

def delete_last_image(code, directory, image_type):
    """
    가장 최근에 캡처된 이미지 파일을 삭제하는 함수
    """
    index = int(image_type) - 1
    last_image_code = code[index]
    if last_image_code:
        last_image_path = os.path.join(directory, f"{last_image_code}{image_file_extension}")
        if os.path.exists(last_image_path):
            os.remove(last_image_path)
            code[index] = ''  # 해당 코드 리셋
            saveImageInfo(*code)
            return True, f"이미지 삭제: {last_image_code}"
        else:
            return False, "이미지 파일이 존재하지 않습니다."
    return False, "삭제할 이미지 코드가 없습니다."

def capture_image_with_code(stdscr, code, directory, image_type):
    """
    특정 코드를 입력 받아 이미지 촬영을 수행하는 함수
    """
    curses.noecho()  
    stdscr.clear()
    stdscr.addstr("코드를 입력하십시오 (형식 '00-0000_0'):\n")
    formatted_input = [" ", " ", "-", " ", " ", " ", " ", "_", " "] 
    stdscr.addstr("".join(formatted_input) + "\r") 
    stdscr.refresh()

    cursor_positions = [0, 1, 3, 4, 5, 6, 8]  
    position_index = 0  

    while position_index < 7:  
        ch = stdscr.getch()
        if ch == curses.KEY_BACKSPACE or ch == 127 or ch == 8:
            if position_index > 0:  
                position_index -= 1  
                if cursor_positions[position_index] == 3 or cursor_positions[position_index] == 7:
                    position_index -= 1
                formatted_input[cursor_positions[position_index]] = ' ' 
                stdscr.addstr(1, 0, "".join(formatted_input) + "\r")  
                stdscr.move(1, cursor_positions[position_index])  
        elif ch >= ord('0') and ch <= ord('9'):
            formatted_input[cursor_positions[position_index]] = chr(ch)
            position_index += 1
            stdscr.addstr(1, 0, "".join(formatted_input) + "\r")
            if position_index < 7:
                stdscr.move(1, cursor_positions[position_index])

        if position_index >= 7:
            break

    stdscr.refresh()
    formatted_name = "".join(formatted_input).strip()
    stdscr.addstr(3, 0, "입력된 코드: " + formatted_name + "\n")
    stdscr.refresh()

    file_name = os.path.join(directory, f"{formatted_name}{image_file_extension}")
    capture_command[2] = file_name

    process = subprocess.Popen(preview_command)
    stdscr.addstr("'0'을 눌러 프리뷰를 종료.\n")
    stdscr.refresh()
    
    while True:
        key = stdscr.getch()
        if key == ord('0'):  # '0'을 눌러 캡처
            process.terminate()  # 프리뷰 종료
            process.wait()
            break
    
    subprocess.run(capture_command)

    if image_type == '1':
        code[0] = formatted_name
    elif image_type == '2':
        code[1] = formatted_name
    elif image_type == '3':
        code[2] = formatted_name
    saveImageInfo(*code)
    

def main_loop(stdscr):
    """
        메인 루프를 구성하여 사용자 입력에 따라 다양한 기능을 수행
    """

    global preview_command
    
    code = readImageInfo()  # 현재 이미지 정보 읽기
    last_tpg_code = ""
    while True:
        stdscr.clear()
        # 이미지 유형 선택 안내 메시지 출력
        stdscr.addstr("촬영할 이미지 종류를 선택하시오:\n")
        # 각 키에 대한 설명 출력
        stdscr.addstr("스와치: '1' \nTPG: '2'\nTCX: '3'\n프로그램 종료: '9'\n")
        stdscr.refresh()
       
        # 사용자로부터 입력 받음
        image_type = stdscr.getkey()

        # 프로그램 종료 조건
        if image_type == '9':
            stdscr.addstr("프로그램을 종료합니다.\n")
            stdscr.refresh()
            break
        
        directory = path_finder.get_directory(image_type)
        
        # 내부 루프 시작
        while True:
            tpg_count = 1  # TPG 이미지 카운터
            stdscr.clear()
            stdscr.addstr("미리보기: '1'\n이미지 캡처: '2'\n이전 이미지 삭제: '5'\n특정 이미지 다시 촬영: '6'\n이미지 종류 재선택: '8'\n종료: '9'\n")
            stdscr.refresh()

            # 사용자 입력 받기
            cmd_input = stdscr.getkey()
            # 입력받은 이미지 유형에 따라 저장 경로 설정
            if cmd_input == '6' and image_type == '2':
                capture_image_with_code(stdscr, code, directory, image_type)
                continue
            
            
            if cmd_input == '1':
                # 미리보기 시작
                process = subprocess.Popen(preview_command)
                stdscr.addstr("미리보기가 시작되었습니다. '0'를 눌러 중지하십시오.\n")
                stdscr.refresh()
                
                
                # 잠시 대기하여 미리보기 창이 뜨도록 함
                curses.napms(5000)
                # 터미널로 포커스 이동
                terminal_window_id = subprocess.check_output(['xdotool', 'getactivewindow']).strip()
                subprocess.run(['xdotool', 'windowactivate', terminal_window_id])
                
                # 미리보기 중지 대기 루프
                while True:
                    stop = stdscr.getkey()
                    # 중지 키 입력 처리
                    if stop == preview_stop_key:
                        # 미리보기 프로세스 종료
                        process.terminate()
                        # 프로세스 종료 대기
                        process.wait()
                        # 중지 안내 메시지 출력
                        stdscr.addstr("미리보기가 중지되었습니다.\n")
                        stdscr.refresh()
                        break

            # 이미지 캡처 기능 선택
            elif cmd_input == '2':
                if image_type == '2': # tpg    
                    while tpg_count < 9:  # tpg_count가 9보다 작을 동안 반복
                        
                        last_tpg_code = code[1].split('_')[0] if len(code) > 1 and code[1] else "" 
                        image_name = validate_and_format_image_name(stdscr, image_type, last_tpg_code, tpg_count)
                        file_name = os.path.join(directory, f"{image_name}{image_file_extension}")
                        
                        process = subprocess.Popen(preview_command)
                        stdscr.addstr("'0'을 눌러 프리뷰를 종료.\n")
                        stdscr.refresh()
                    
                        while True:
                            key = stdscr.getch()
                            if key == ord('0') or key == ord('+'):  # 스페이스바를 눌러 캡처
                                process.terminate()  # 프리뷰 종료
                                process.wait()
                                break
                        
                        
                        capture_command[2] = file_name
                        subprocess.run(capture_command)
                        code[1] = image_name
                        tpg_count = tpg_count + 1
                        if tpg_count >= 9:
                            break
                
                else:
                    
                    process = subprocess.Popen(preview_command)
                    stdscr.addstr("'0'을 눌러 프리뷰를 종료.\n")
                    stdscr.refresh()
                
                    while True:
                        key = stdscr.getch()
                        if key == ord('0') or key == ord('+'):  # 스페이스바를 눌러 캡처
                            process.terminate()  # 프리뷰 종료
                            process.wait()
                            break
                      
                    
                    image_name = validate_and_format_image_name(stdscr, image_type, last_tpg_code, tpg_count)
                    file_name = os.path.join(directory, f"{image_name}{image_file_extension}")
                    capture_command[2] = file_name
                    subprocess.run(capture_command)
                    if image_type == '1':
                        code[0] = image_name
                    elif image_type == '2':
                        code[1] = image_name
                    elif image_type == '3':
                        code[2] = image_name
                    saveImageInfo(*code)
                
               
               # 클라우드에 이미지 업로드
                # cloud_controller.upload_file(file_name, os.path.basename(file_name), image_type)
              
                
            elif cmd_input == '5':
                if code[int(image_type) - 1]:  # 유형에 따른 코드 존재 여부 확인
                    stdscr.addstr("이전 이미지를 삭제하는 중...\n")
                    success, message = delete_last_image(code, directory, image_type)
                    stdscr.addstr(message + "\n")
                else:
                    stdscr.addstr("삭제할 이미지가 없습니다.\n")
                stdscr.refresh()
                
            elif cmd_input == '8':
                break  # 다시 유형 선택으로 돌아가기

            elif cmd_input == '9':
                stdscr.addstr("프로그램을 종료합니다.\n")
                stdscr.refresh()
                return  # 프로그램 종료

if __name__ == "__main__":
    curses.wrapper(main_loop)  # curses를 사용하여 메인 루프 실행

# ----------------------------------------------------------------------------- #
