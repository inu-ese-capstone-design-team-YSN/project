import tkinter as tk
from config.configs import Wd_config
from page.page import *
from path_finder import PathFinder  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스

class MainScreen(tk.Tk):
    """
    Main application window class that sets up the window size, position, and background image.
    """
    def __init__(self):
        super().__init__()
        # Window configuration instance created
        self.config = Wd_config()
        # erase Window title bar
        self.config.erase_title_bar(self)
        # Initial position setting
        self.geometry(self.config.set_center_position(self))
        self.main_frame = MainFrame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

def main():
    app = MainScreen()
    app.mainloop()

if __name__ == "__main__":
    main()