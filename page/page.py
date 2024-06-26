import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import subprocess
import cv2
import os
from config.configs import Wd_config, Bt_config, Txt_config
import numpy as np
import pandas as pd
import ast
from path_finder import PathFinder  # 파일 경로를 쉽게 관리하기 위해 사용하는 클래스
#from function.swatch_capture import CameraCapture
from scripts.swatch_capture import CameraCapture
import threading
import time

# Main application frame
class MainFrame(tk.Frame):
    """
    Main frame class that sets up the background image and creates and positions buttons within the main application window.
    """
    def __init__(self, master):
        super().__init__(master)
        self.master=master
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        
        # Set background image  
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Create and position buttons
        button1 = tk.Button(self, text="Color confirmation", command=self.open_analysis_page, font=self.window_config.main_font)
        button2 = tk.Button(self, text="Past Records", command=self.open_image_page, font=self.window_config.main_font)
        exit_button = tk.Button(self, text="Exit", command=self.close_window)
        
        # Place buttons
        button1.place(x=self.button_config.button_pos1_x, y=self.button_config.button_pos1_y, width=self.button_config.button_type1_width, height=self.button_config.button_type1_height)
        button2.place(x=self.button_config.button_pos2_x, y=self.button_config.button_pos2_y, width=self.button_config.button_type1_width, height=self.button_config.button_type1_height)
        exit_button.place(x=self.button_config.exit_button_x, y=self.button_config.exit_button_y, width=self.button_config.exit_button_width, height=self.button_config.exit_button_height)
    
    def open_analysis_page(self):
        """Open the analysis page and handle it when fully loaded."""
        self.destroy_all_toplevels()
        self.analysis_page = AnalysisPage0(self, self.master, lambda: self.master.withdraw())

    def open_image_page(self):
        """Open the past data page and handle it when fully loaded."""
        self.destroy_all_toplevels()
        self.image_page = ImagePage(self, self.master, lambda: self.master.withdraw())
    
    def close_window(self):
        """Close the application."""
        self.master.destroy()
    
    def destroy_all_toplevels(self):
        """Destroy all toplevel windows."""
        for child in self.master.winfo_children():
            if isinstance(child, tk.Toplevel):
                child.destroy()

# Analysis mode selection page
class AnalysisPage0(tk.Toplevel):
    """
    Initial analysis page class that provides buttons for choosing between color inference and similarity analysis modes.
    """
    def __init__(self, master,main_window, ready_callback=None):
        super().__init__(master)
        self.ready_callback = ready_callback
        self.main_window = main_window
        
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Set initial window position
        self.geometry(self.window_config.set_center_position(self))
        
        # Create buttons for analysis options
        #analysis_button1 = tk.Button(self, text="Color inference", command=self.open_analysis_page1, font=self.window_config.main_font)
        analysis_button2 = tk.Button(self, text="Analysis similarity", command=self.open_analysis_page2, font=self.window_config.main_font)
        back_button = tk.Button(self, text="Back", command=self.go_back, font=("Arial", 8))
        
        # Place buttons
        #analysis_button1.place(x=self.button_config.button_pos1_x, y=self.button_config.button_pos1_y, width=self.button_config.button_type1_width, height=self.button_config.button_type1_height)
        analysis_button2.place(x=self.button_config.button_pos2_x, y=self.button_config.button_pos2_y, width=self.button_config.button_type1_width, height=self.button_config.button_type1_height)
        back_button.place(x=self.button_config.back_button_x, y=self.button_config.back_button_y, width=self.button_config.back_button_width, height=self.button_config.back_button_height)

    def prepare_page(self):
        """Prepare the page by arranging widgets and loading data, then call the ready callback."""
        if self.ready_callback:
            self.ready_callback()

    def open_analysis_page1(self):
        """Open the color inference page and handle it when fully loaded."""
        self.analysis_page = AnalysisPage1_0(self, self.main_window, lambda: self.destroy())

    def open_analysis_page2(self):
        """Open the analysis similarity page and handle it when fully loaded."""
        self.analysis_page = AnalysisPage2_0(self.master, self.main_window, lambda: self.master.destroy())

    def go_back(self):
        """Return to the main screen."""
        self.destroy()  # Close the current window
        self.main_window.deiconify()  # Restore the previous main window
        self.main_window.lift()

# Color inference analysis page
class AnalysisPage1_0(tk.Toplevel):
    """
    Page class for performing color inference analysis.
    """
    def __init__(self, master, main_window, ready_callback=None):
        super().__init__(master)
        self.ready_callback = ready_callback
        self.main_window = main_window
        
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Set initial window position
        self.geometry(self.window_config.set_center_position(self))
        
        # Create text label
        text_label = tk.Label(self, text="Adjust to the precise position and press the next button.", font=self.txt_config.f12)
        text_label.place(x=self.txt_config.pos_x, y=self.txt_config.pos_y, width=self.txt_config.size_x, height=self.txt_config.size_y)

        # Create buttons for running the analysis and going back
        run_button = tk.Button(self, text="Next", command=self.go_next, font=self.window_config.main_font)
        check_button = tk.Button(self, text="Check", command=self.update_preview_new,font=self.window_config.main_font)
        back_button = tk.Button(self, text="Back", command=self.go_back)
        
        #now editing
        
        # Place buttons
        run_button.place(x=self.button_config.next_button_x, y=self.button_config.next_button_y, width=self.button_config.next_button_width, height=self.button_config.next_button_height)
        check_button.place(x=self.button_config.check_button_x, y=self.button_config.check_button_y, width=self.button_config.check_button_width, height=self.button_config.check_button_height)
        back_button.place(x=self.button_config.back_button_x, y=self.button_config.back_button_y, width=self.button_config.back_button_width, height=self.button_config.back_button_height)

        # new camera test
        self.image_label = tk.Label(self)
        self.image_label.place(x=20, y=self.txt_config.pos_y+self.txt_config.size_y+self.button_config.p_y, width=480, height=360)

    def prepare_page(self):
        """Prepare the page by arranging widgets and loading data, then call the ready callback."""
        if self.ready_callback:
            self.ready_callback()

    def update_preview_new(self):
        # 카메라로 사진 찍기
        command = "rpicam-jpeg -o /home/pi/project/GUI/tempPreview/img.jpg -t 100 --width 480 --height 360 -n"
        subprocess.run(command, shell=True)

        # 이미지 파일 로드
        if os.path.exists("/home/pi/project/GUI/tempPreview/img.jpg"):
            image = Image.open("/home/pi/project/GUI/tempPreview/img.jpg")
            photo = ImageTk.PhotoImage(image)

            # 레이블에 이미지 표시
            self.image_label.image = photo
            self.image_label.configure(image=photo)

    def go_next(self):
        """Open CI waiting status page and handle it when fully loaded."""
        self.analysis_page = AnalysisPage1_1(self, self.main_window, lambda: self.destroy())

    def go_back(self):
        """Return to the main screen."""
        self.destroy()  # Close the current window
        self.main_window.deiconify()  # Restore the previous main window
        self.main_window.lift()

# CI waiting status page 여기서 실행
class AnalysisPage1_1(tk.Toplevel):
    """
    Page class for CI waiting status.
    """
    def __init__(self, master, main_window, ready_callback=None):
        super().__init__(master)
        self.ready_callback = ready_callback
        self.main_window = main_window
        
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Set initial window position
        self.geometry(self.window_config.set_center_position(self))
        
        self.canvas = tk.Canvas(self, width=800, height=480)
        self.canvas.pack()

        self.load_video("src/loading.mp4")
        self.run_CI()

    def ready_page(self):
        """Call the ready callback once the page is prepared."""
        if self.ready_callback:
            self.ready_callback()

    def load_video(self, video_path):
        """Load and display video from the specified path."""
        self.cap = cv2.VideoCapture(video_path)
        self.display_video()

    def display_video(self):
        """Display the video frame by frame."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(400, 240, image=self.photo, anchor=tk.CENTER)
            self.after(33, self.display_video)  # Play video at approximately 30fps
        else:
            self.cap.release()

    def run_CI(self):
        """Run the CI process by executing a script."""
        self.script_paths = [
            os.path.join(os.path.dirname(__file__), "../scripts", "capture.py"),
            os.path.join(os.path.dirname(__file__), "../scripts", "inference_swatch_color.py")
        ]
        self.current_script_index = 0
        self.run_next_script()

    def run_next_script(self):
        if self.current_script_index < len(self.script_paths):
            script_path = self.script_paths[self.current_script_index]
            if os.path.exists(script_path):
                if self.current_script_index == 0:
                    self.process = subprocess.Popen(["python3", script_path, "--n", str(0)], cwd=os.path.dirname(script_path))
                else:
                    self.process = subprocess.Popen(["python3", script_path], cwd=os.path.dirname(script_path))
                self.after(100, self.check_process)
            else:
                messagebox.showerror("Error", f"Script not found: {script_path}")
        else:
            self.analysis_page = AnalysisPage1_2(self, self.main_window, lambda: self.destroy())

    def check_process(self):
        """Check the status of the CI process and handle completion."""
        if self.process.poll() is None:
            self.after(100, self.check_process)
        else:
            if self.process.returncode == 0:
                self.current_script_index += 1
                self.run_next_script()
            else:
                messagebox.showerror("Error", f"Script failed with return code {self.process.returncode}")

# Feedback page for one sample analysis
class AnalysisPage1_2(tk.Toplevel):
    """
    Page class for showing feedback after color inference analysis.
    """
    def __init__(self, master, main_window, ready_callback=None):
        super().__init__(master)
        self.ready_callback = ready_callback
        self.main_window = main_window
        
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Set initial window position
        self.geometry(self.window_config.set_center_position(self))
        
        # Create a large text label for displaying results
        text_label_1 = tk.Label(self, text="", font=self.txt_config.f12, borderwidth=2, relief="solid", highlightbackground="gray", highlightcolor="gray", highlightthickness=2)
        text_label_1.place(x=self.txt_config.p_x, y=self.txt_config.p_y, width=self.txt_config.big_size_x, height=self.txt_config.big_size_y)
        
        # Create two middle text labels for displaying images
        text_label_2 = tk.Label(self, text="testing...", font=self.txt_config.f12)
        text_label_2.place(x=self.txt_config.middle1_pos_x, y=self.txt_config.middle1_pos_y, width=self.txt_config.middle1_size_x, height=self.txt_config.middle1_size_y)
        image = Image.open("/home/pi/project/img/CI/image_1.png")
        image = image.resize((300, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        text_label_2.image = photo
        text_label_2.configure(image=photo)
        
        # Add Lab plot
        self.add_lab_plot()
        
        # Create labels for displaying color information
        bottom_bar1 = tk.Label(self, text="sRGB:(0,0,0)", font=self.txt_config.f12)
        bottom_bar1.place(x=self.txt_config.bb_1_pos_x, y=self.txt_config.bb_1_pos_y-35, width=self.txt_config.bb_1_size_x, height=self.txt_config.bb_1_size_y)

        bottom_bar2 = tk.Label(self, text="LAB:(0,0,0)", font=self.txt_config.f12)
        bottom_bar2.place(x=self.txt_config.bb_2_pos_x, y=self.txt_config.bb_2_pos_y-35, width=self.txt_config.bb_1_size_x, height=self.txt_config.bb_1_size_y)

        bottom_bar3 = tk.Label(self, text="Hex: #000000", font=self.txt_config.f12)
        bottom_bar3.place(x=self.txt_config.bb_3_pos_x, y=self.txt_config.bb_3_pos_y-35, width=self.txt_config.bb_1_size_x, height=self.txt_config.bb_1_size_y)

        bottom_bar4 = tk.Label(self, text="CMYK: (0,0,0,0)", font=self.txt_config.f12)
        bottom_bar4.place(x=self.txt_config.bb_4_pos_x, y=self.txt_config.bb_4_pos_y-35, width=self.txt_config.bb_1_size_x, height=self.txt_config.bb_1_size_y)

        # Create buttons for navigation
        home_button = tk.Button(self, text="Home", command=self.go_back)
        home_button.place(x=self.txt_config.middle3_pos_x, y=self.txt_config.middle3_pos_y, width=self.txt_config.middle3_size_x, height=self.txt_config.middle3_size_y)

        save_button = tk.Button(self, text="Save", command=self.save_data)
        save_button.place(x=self.txt_config.middle4_pos_x, y=self.txt_config.middle4_pos_y, width=self.txt_config.middle4_size_x, height=self.txt_config.middle4_size_y)    

    def add_lab_plot(self):
        # Create a figure
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Example Lab data
        lab_1 = [48, 10.2, -7.1]
        lab_2 = [48.3, 10.3, -7.2]
        lab_3 = [47.6, 10.4, -7.3]
    
        # Extract L, a, b values
        L = [lab[0] for lab in [lab_1, lab_2, lab_3]]
        a = [lab[1] for lab in [lab_1, lab_2, lab_3]]
        b = [lab[2] for lab in [lab_1, lab_2, lab_3]]
        
        # Plotting the Lab data
        ax.scatter(L, a, b, c='r', marker='o')
        ax.set_xlabel('L')
        ax.set_ylabel('a')
        ax.set_zlabel('b')
        
        # Adjust layout
        fig.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
        
        # Create a canvas and add the figure to it
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().place(x=self.txt_config.middle2_pos_x, y=self.txt_config.middle2_pos_y, width=self.txt_config.middle2_size_x, height=self.txt_config.middle2_size_y)
 
    def ready_page(self):
        """Call the ready callback once the page is prepared."""
        if self.ready_callback:
            self.ready_callback()

    def go_back(self):
        """Return to the main screen."""
        self.destroy()  # Close the current window
        self.main_window.deiconify()  # Restore the previous main window
        self.main_window.lift()
    
    def save_data(self):
        """Save the analysis data."""
        pass

# Two sample capture page for the first sample
class AnalysisPage2_0(tk.Toplevel):
    """
    Page class for performing similarity analysis between two samples, starting with the first sample.
    """
    def __init__(self, master, main_window, ready_callback=None):
        super().__init__(master)
        self.ready_callback = ready_callback
        self.main_window = main_window
        
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Set initial window position
        self.geometry(self.window_config.set_center_position(self))
        
        # Create text label
        text_label = tk.Label(self, text="Adjust the first one to the precise position and press the next button.", font=self.txt_config.f12)
        text_label.place(x=self.txt_config.pos_x, y=self.txt_config.pos_y, width=self.txt_config.size_x, height=self.txt_config.size_y)

        # Create buttons for running the analysis and going back
        run_button = tk.Button(self, text="Next", command=self.go_next, font=self.window_config.main_font)
        check_button = tk.Button(self, text="Check", command=self.update_preview_new,font=self.window_config.main_font)
        back_button = tk.Button(self, text="Back", command=self.go_back)
        
        #now editing
        
        # Place buttons
        run_button.place(x=self.button_config.next_button_x, y=self.button_config.next_button_y, width=self.button_config.next_button_width, height=self.button_config.next_button_height)
        check_button.place(x=self.button_config.check_button_x, y=self.button_config.check_button_y, width=self.button_config.check_button_width, height=self.button_config.check_button_height)
        back_button.place(x=self.button_config.back_button_x, y=self.button_config.back_button_y, width=self.button_config.back_button_width, height=self.button_config.back_button_height)

        # new camera test
        self.image_label = tk.Label(self)
        self.image_label.place(x=20, y=self.txt_config.pos_y+self.txt_config.size_y+self.button_config.p_y, width=480, height=360)

    def prepare_page(self):
        """Prepare the page by arranging widgets and loading data, then call the ready callback."""
        if self.ready_callback:
            self.ready_callback()

    def update_preview_new(self):
        # 카메라로 사진 찍기
        command = "rpicam-jpeg -o /home/pi/project/img/temp_img/img.jpg -t 100 --width 480 --height 360 -n"
        subprocess.run(command, shell=True)

        # 이미지 파일 로드
        if os.path.exists("/home/pi/project/img/temp_img/img.jpg"):
            image = Image.open("/home/pi/project/img/temp_img/img.jpg")
            photo = ImageTk.PhotoImage(image)

            # 레이블에 이미지 표시
            self.image_label.image = photo
            self.image_label.configure(image=photo)

    def go_next(self):
        """Open CI waiting status page and handle it when fully loaded."""
        self.analysis_page = AnalysisPage2_0_w(self, self.main_window, lambda: self.master.destroy())

    def go_back(self):
        """Return to the main screen."""
        self.destroy()  # Close the current window
        self.main_window.deiconify()  # Restore the previous main window
        self.main_window.lift()

class AnalysisPage2_0_w(tk.Toplevel):
    """
    Page class for CI waiting status.
    """
    def __init__(self, master, main_window, ready_callback=None):
        super().__init__(master)
        self.ready_callback = ready_callback
        self.main_window = main_window
        
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Set initial window position
        self.geometry(self.window_config.set_center_position(self))
        
        self.canvas = tk.Canvas(self, width=800, height=480)
        self.canvas.pack()

        self.load_video("src/loading.mp4")
        self.run_CI()

    def ready_page(self):
        """Call the ready callback once the page is prepared."""
        if self.ready_callback:
            self.ready_callback()

    def load_video(self, video_path):
        """Load and display video from the specified path."""
        self.cap = cv2.VideoCapture(video_path)
        self.display_video()

    def display_video(self):
        """Display the video frame by frame."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(400, 240, image=self.photo, anchor=tk.CENTER)
            self.after(33, self.display_video)  # Play video at approximately 30fps
        else:
            self.cap.release()

    def run_CI(self):
        """Run the CI process by executing a script with -n option."""
        script_path = os.path.join(os.path.dirname(__file__), "../scripts", "capture.py")
        if os.path.exists(script_path):
            self.process = subprocess.Popen(["python3", script_path, "--n", str(1)], cwd=os.path.dirname(script_path))
            self.after(100, self.check_process)

    def check_process(self):
        """Check the status of the CI process and handle completion."""
        if self.process.poll() is None:
            self.after(100, self.check_process)
        else:
            if self.process.returncode == 0:
                self.analysis_page = AnalysisPage2_1(self, self.main_window, lambda: self.destroy())
            else:
                messagebox.showerror("Error", f"Script failed with return code {self.process.returncode}")

# Two sample capture page for the second sample
class AnalysisPage2_1(tk.Toplevel):
    """
    Page class for performing similarity analysis between two samples, focusing on the second sample.
    """
    def __init__(self, master, main_window, ready_callback=None):
        super().__init__(master)
        self.ready_callback = ready_callback
        self.main_window = main_window
        
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Set initial window position
        self.geometry(self.window_config.set_center_position(self))
        
        # Create text label
        text_label = tk.Label(self, text="Adjust the second one to the precise position and press the next button.", font=self.txt_config.f12)
        text_label.place(x=self.txt_config.pos_x, y=self.txt_config.pos_y, width=self.txt_config.size_x, height=self.txt_config.size_y)

        # Create buttons for running the analysis and going back
        run_button = tk.Button(self, text="Next", command=self.go_next, font=self.window_config.main_font)
        check_button = tk.Button(self, text="Check", command=self.update_preview_new,font=self.window_config.main_font)
        back_button = tk.Button(self, text="Back", command=self.go_back)
        
        #now editing
        
        # Place buttons
        run_button.place(x=self.button_config.next_button_x, y=self.button_config.next_button_y, width=self.button_config.next_button_width, height=self.button_config.next_button_height)
        check_button.place(x=self.button_config.check_button_x, y=self.button_config.check_button_y, width=self.button_config.check_button_width, height=self.button_config.check_button_height)
        back_button.place(x=self.button_config.back_button_x, y=self.button_config.back_button_y, width=self.button_config.back_button_width, height=self.button_config.back_button_height)

        # new camera test
        self.image_label = tk.Label(self)
        self.image_label.place(x=20, y=self.txt_config.pos_y+self.txt_config.size_y+self.button_config.p_y, width=480, height=360)

    def prepare_page(self):
        """Prepare the page by arranging widgets and loading data, then call the ready callback."""
        if self.ready_callback:
            self.ready_callback()

    def update_preview_new(self):
        # 카메라로 사진 찍기
        command = "rpicam-jpeg -o /home/pi/project/img/temp_img/img.jpg -t 100 --width 480 --height 360 -n"
        subprocess.run(command, shell=True)

        # 이미지 파일 로드
        if os.path.exists("/home/pi/project/img/temp_img/img.jpg"):
            image = Image.open("/home/pi/project/img/temp_img/img.jpg")
            photo = ImageTk.PhotoImage(image)

            # 레이블에 이미지 표시
            self.image_label.image = photo
            self.image_label.configure(image=photo)

    def go_next(self):
        """Open CI waiting status page and handle it when fully loaded."""
        self.analysis_page = AnalysisPage2_2(self, self.main_window, lambda: self.master.destroy())

    def go_back(self):
        """Return to the main screen."""
        self.destroy()  # Close the current window
        self.main_window.deiconify()  # Restore the previous main window
        self.main_window.lift()

# CI waiting status page for two sample analysis 여기서 스크립트 실행
class AnalysisPage2_2(tk.Toplevel):
    """
    Page class for CI waiting status during similarity analysis between two samples.
    """
    def __init__(self, master, main_window, ready_callback=None):
        super().__init__(master)
        self.ready_callback = ready_callback
        self.main_window = main_window
        
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Set initial window position
        self.geometry(self.window_config.set_center_position(self))
        
        self.canvas = tk.Canvas(self, width=800, height=480)
        self.canvas.pack()

        self.load_video("src/loading.mp4")
        self.run_CI()

    def ready_page(self):
        """Call the ready callback once the page is prepared."""
        if self.ready_callback:
            self.ready_callback()

    def load_video(self, video_path):
        """Load and display video from the specified path."""
        self.cap = cv2.VideoCapture(video_path)
        self.display_video()

    def display_video(self):
        """Display the video frame by frame."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(400, 240, image=self.photo, anchor=tk.CENTER)
            self.master.after(33, self.display_video)  # Play video at approximately 30fps
        else:
            self.cap.release()

    def run_CI(self):
        """Run the CI process by executing a script."""
        self.script_paths = [
            os.path.join(os.path.dirname(__file__), "../scripts", "capture.py"),
            os.path.join(os.path.dirname(__file__), "../scripts", "calculate_color_similarity.py")
        ]
        self.current_script_index = 0
        self.run_next_script()

    def run_next_script(self):
        if self.current_script_index < len(self.script_paths):
            script_path = self.script_paths[self.current_script_index]
            if os.path.exists(script_path):
                if self.current_script_index == 0:
                    self.process = subprocess.Popen(["python3", script_path, "--n", str(2)], cwd=os.path.dirname(script_path))
                else:
                    self.process = subprocess.Popen(["python3", script_path], cwd=os.path.dirname(script_path))
                self.master.after(100, self.check_process)
            else:
                messagebox.showerror("Error", f"Script not found: {script_path}")
        else:
            self.analysis_page = AnalysisPage2_3(self, self.master, lambda: self.destroy())

    def check_process(self):
        """Check the status of the CI process and handle completion."""
        if self.process.poll() is None:
            self.master.after(100, self.check_process)
        else:
            if self.process.returncode == 0:
                self.current_script_index += 1
                self.run_next_script()
            else:
                messagebox.showerror("Error", f"Script failed with return code {self.process.returncode}")

# Feedback page for two sample analysis
class AnalysisPage2_3(tk.Toplevel):
    """
    Page class for showing feedback after similarity analysis between two samples.
    """
    def __init__(self, master, main_window, ready_callback=None):
        super().__init__(master)
        self.ready_callback = ready_callback
        self.main_window = main_window
        
        # Configuration instances
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Set initial window position
        self.geometry(self.window_config.set_center_position(self))
        
        self.Lab_distance = []
        self.L_distance = []
        self.a_distance = []
        self.b_distance = []
        self.Lab1=[]
        self.RGB1=[]
        self.HEX1=[]
        self.CMYK1=[]
        self.Lab2=[]
        self.RGB2=[]
        self.HEX2=[]
        self.CMYK2=[]
        self.idx=0
        self.load_data()
        self.idx2=self.idx+len(self.Lab_distance)
        print('===============================')
        
        # Create a large text label for displaying results
        text_label_1 = tk.Label(self, text="", font=self.txt_config.f12)
        text_label_1.place(x=self.txt_config.p_x, y=self.txt_config.p_y, width=self.txt_config.big_size_x, height=self.txt_config.big_size_y)

        # create info name container
        info1_name_container = tk.Label(self, text="First Sample", font=self.txt_config.f10, borderwidth=1, relief="solid")
        info1_name_container.place(x=self.txt_config.info1_name_pos_x, y=self.txt_config.info1_name_pos_y, width=self.txt_config.info1_name_size_x, height=self.txt_config.info1_name_size_y)
        info2_name_container = tk.Label(self, text="Second Sample", font=self.txt_config.f10, borderwidth=1, relief="solid")
        info2_name_container.place(x=self.txt_config.info2_name_pos_x, y=self.txt_config.info2_name_pos_y, width=self.txt_config.info2_name_size_x, height=self.txt_config.info2_name_size_y)

        # Create labels for displaying color information
        self.bottom_bar1 = tk.Label(self, text=f"RGB {self.RGB1[self.idx]}", font=self.txt_config.f10)
        self.bottom_bar1.place(x=self.txt_config.bottom_bar_1_pos_x, y=self.txt_config.bottom_bar_1_pos_y, width=self.txt_config.bottom_bar_1_size_x, height=self.txt_config.bottom_bar_1_size_y)

        self.bottom_bar2 = tk.Label(self, text=f"LAB {self.Lab1[self.idx]}", font=self.txt_config.f10)
        self.bottom_bar2.place(x=self.txt_config.bottom_bar_1_pos_x+self.txt_config.bottom_bar_1_size_x, y=self.txt_config.bottom_bar_2_pos_y, width=self.txt_config.bottom_bar_2_size_x, height=self.txt_config.bottom_bar_2_size_y)

        # Create labels for displaying color information
        self.bottom_bar2_1 = tk.Label(self, text=f"RGB {self.RGB2[self.idx]}", font=self.txt_config.f10)
        self.bottom_bar2_1.place(x=self.txt_config.bottom_bar_2_pos_x, y=self.txt_config.bottom_bar_1_pos_y, width=self.txt_config.bottom_bar_1_size_x, height=self.txt_config.bottom_bar_1_size_y)

        self.bottom_bar2_2 = tk.Label(self, text=f"LAB {self.Lab2[self.idx]}", font=self.txt_config.f10)
        self.bottom_bar2_2.place(x=self.txt_config.bottom_bar_2_pos_x+self.txt_config.bottom_bar_2_size_x, y=self.txt_config.bottom_bar_2_pos_y, width=self.txt_config.bottom_bar_2_size_x, height=self.txt_config.bottom_bar_2_size_y)

        self.add_lab_plot()
        
        # show diffrence
        self.diff_label = tk.Label(self, text=f"ΔE : {self.Lab_distance[self.idx]}  ΔL : {self.L_distance[self.idx]}\n\nΔa : {self.a_distance[self.idx]}  Δb : {self.b_distance[self.idx]}", font=self.txt_config.f16)
        self.diff_label.place(x=self.txt_config.middle1_pos_x+self.txt_config.middle1_size_x+60, y=self.txt_config.middle1_pos_y+180, width=self.txt_config.middle1_size_x, height=self.txt_config.middle1_size_y-200)

        # Create two middle labels for displaying images
        self.image_dir = "/home/pi/project/img/SM_clusters"
        self.image_files = sorted(os.listdir(self.image_dir))
        
        
        self.image_label_1 = tk.Label(self,bg=self.rgb_to_hex(self.RGB1[self.idx]), font=self.txt_config.f12,bd=2, relief="solid")
        self.image_label_1.place(x=self.txt_config.middle1_pos_x+self.txt_config.middle1_size_x+60+10, y=self.txt_config.middle1_pos_y+20, width=self.txt_config.middle3_size_x, height=self.txt_config.middle3_size_y)
        
        self.image_label_2 = tk.Label(self,bg=self.rgb_to_hex(self.RGB2[self.idx]), font=self.txt_config.f12,bd=2, relief="solid")
        self.image_label_2.place(x=self.txt_config.middle1_pos_x+self.txt_config.middle1_size_x+60+self.txt_config.middle3_size_x+20, y=self.txt_config.middle1_pos_y+20, width=self.txt_config.middle3_size_x, height=self.txt_config.middle3_size_y)
        
        print(f'idx1:{self.idx} idx2:{self.idx2}')
        
        # Create buttons for navigation
        home_button = tk.Button(self, text="Home", command=self.go_back)
        home_button.place(x=self.txt_config.middle3_pos_x+2*self.txt_config.ip_x, y=self.txt_config.middle3_pos_y, width=self.txt_config.middle3_size_x, height=self.txt_config.middle3_size_y)

        next_color_button = tk.Button(self, text="Next Color", command=self.next_color)
        next_color_button.place(x=self.txt_config.middle4_pos_x+2*self.txt_config.ip_x, y=self.txt_config.middle4_pos_y, width=self.txt_config.middle4_size_x, height=self.txt_config.middle4_size_y)    

    def add_lab_plot(self):
        # Create a figure
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract L, a, b values
        L = [lab[0] for lab in [self.Lab1[self.idx], self.Lab2[self.idx]]]
        a = [lab[1] for lab in [self.Lab1[self.idx], self.Lab2[self.idx]]]
        b = [lab[2] for lab in [self.Lab1[self.idx], self.Lab2[self.idx]]]
        RGB1=self.RGB1[self.idx]
        RGB2=self.RGB2[self.idx]
        rgb_1 = [c / 255.0 for c in RGB1]
        rgb_2 = [c / 255.0 for c in RGB2]
        # Plotting the Lab data
        ax.scatter(L[0], a[0], b[0], c=[rgb_1], marker='o')
        ax.scatter(L[1], a[1], b[1], c=[rgb_2], marker='o')
        # Plotting the vertical lines and bottom points
        # for l, a_val, b_val in zip(L, a, b):
        #     ax.plot([l, l], [a_val, a_val], [0, b_val], color='r', linestyle='dashed')
        #     ax.scatter([l], [a_val], [0], c='r', marker='o')
        ax.set_xlabel('L')
        ax.set_ylabel('a')
        ax.set_zlabel('b')
        
        # Adjust layout
        fig.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
        
        # Create a canvas and add the figure to it
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().place(x=self.txt_config.middle1_pos_x, y=self.txt_config.middle1_pos_y, width=self.txt_config.middle1_size_x+60, height=self.txt_config.middle1_size_y+40)

    def ready_page(self):
        """Call the ready callback once the page is prepared."""
        if self.ready_callback:
            self.ready_callback()

    def go_back(self):
        """Return to the main screen."""
        self.destroy()  # Close the current window
        self.main_window.deiconify()  # Restore the previous main window
        self.main_window.lift()
    
    def parse_array_string(self,array_str):
        """문자열을 파싱하여 리스트로 변환"""
        array_str = array_str.replace(' ', ',')  # 공백을 쉼표로 대체
        return ast.literal_eval(array_str)

    def parse_space_separated_array(self, array_str):
        """Parse a space-separated array string into a list of floats."""
        return [float(x) for x in array_str.strip('[]').split()]

    def parse_comma_separated_array(self, array_str):
        """Parse a comma-separated array string into a list of floats."""
        return [float(x) for x in array_str.strip('[]').split(',')]
    
    def round_array(self, array, decimals=2):
        """Round each element in the array to the specified number of decimals."""
        return [round(x, decimals) for x in array]
    
    def remove_trailing_zeros(self, array):
        """Convert floats to integers if they are whole numbers."""
        return [int(x) if x.is_integer() else x for x in array]
    
    def rgb_to_hex(self,rgb):
        """Convert an RGB tuple to a hex string."""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def load_data(self):
        """Load the analysis data."""
        df = pd.read_csv("/home/pi/project/result_data/analysis.csv")
        df = df.round(2)
        for index, row in df.iterrows():
            self.RGB1.append(self.remove_trailing_zeros(self.parse_space_separated_array(row['RGB1'])))
            self.RGB2.append(self.remove_trailing_zeros(self.parse_space_separated_array(row['RGB2'])))
            self.Lab_distance.append(float(row['DeltaE']))
            self.Lab1.append(self.round_array(self.parse_comma_separated_array(row['Lab1'])))
            self.Lab2.append(self.round_array(self.parse_comma_separated_array(row['Lab2'])))
            
            Lab1 = self.parse_comma_separated_array(row['Lab1'])
            Lab2 = self.parse_comma_separated_array(row['Lab2'])
            
            dL = Lab1[0] - Lab2[0]
            da = Lab1[1] - Lab2[1]
            db = Lab1[2] - Lab2[2]
            
            self.L_distance.append(round(dL, 2))
            self.a_distance.append(round(da, 2))
            self.b_distance.append(round(db, 2))
    
    def next_color(self):
        """Save the analysis data."""
        self.idx += 1
        self.idx2 += 1
        if self.idx >= len(self.Lab_distance):
            self.idx = 0
        if self.idx2 >= 2*len(self.Lab_distance):
            self.idx2 = len(self.Lab_distance)
            
        self.bottom_bar1.config(text=f"RGB {self.RGB1[self.idx]}", font=self.txt_config.f10)
        self.bottom_bar2.config(text=f"LAB {self.Lab1[self.idx]}", font=self.txt_config.f10)
        self.bottom_bar2_1.config(text=f"RGB {self.RGB2[self.idx]}", font=self.txt_config.f10)
        self.bottom_bar2_2.config(text=f"LAB {self.Lab2[self.idx]}", font=self.txt_config.f10)
        self.diff_label.config(text=f"ΔE : {self.Lab_distance[self.idx]}  ΔL : {self.L_distance[self.idx]}\n\nΔa : {self.a_distance[self.idx]}  Δb : {self.b_distance[self.idx]}", font=self.txt_config.f16)
        self.image_label_1.config(bg=self.rgb_to_hex(self.RGB1[self.idx]))
        self.image_label_2.config(bg=self.rgb_to_hex(self.RGB2[self.idx]))
        self.add_lab_plot()

# Image view page (temp)
class ImagePage(tk.Toplevel):
    def __init__(self, master, ready_callback=None):
        super().__init__(master)
        self.ready_callback = ready_callback
        
        # Configs
        self.window_config = Wd_config()
        self.button_config = Bt_config()
        self.txt_config = Txt_config()
        
        # Remove window title bar
        self.window_config.erase_title_bar(self)
        
        # Set background image
        self.bg_label = self.window_config.set_bg_img(self)
        
        # Set initial window position
        self.geometry(self.window_config.set_center_position(self))

        # Image file directory set (directory where image files are stored)
        self.image_dir = "img"  # You need to set the image directory path here.
        self.image_files = sorted(os.listdir(self.image_dir))
        self.current_image_index = 0

        # Back button created
        back_button = tk.Button(self, text="Back", command=self.go_back)
        back_button.place(x=10, y=10)

        # Next photo view button created
        next_button = tk.Button(self, text="Next Photo", command=self.show_next_image)
        next_button.place(x=100, y=10)

        # Previous photo view button created
        prev_button = tk.Button(self, text="Previous Photo", command=self.show_previous_image)
        prev_button.place(x=200, y=10)

        # Frame created for image display
        self.image_frame = tk.Frame(self)
        self.image_frame.place(x=200, y=50, width=400, height=400)

        # Image label created (centered 400x400)
        self.image_label = tk.Label(self.image_frame, width=400, height=400)
        self.image_label.pack()

        # Window close event handling
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initial image displayed
        self.show_image()

    def page_ready(self):
        """Call the ready callback once the page is prepared."""
        if self.ready_callback:
            self.ready_callback()

    def show_image(self):
        """Display the current image from the directory."""
        image_path = os.path.join(self.image_dir, self.image_files[self.current_image_index])
        image = Image.open(image_path)
        image = image.resize((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def show_next_image(self):
        """Display the next image in the directory."""
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_files):
            self.current_image_index = 0
        self.show_image()

    def show_previous_image(self):
        """Display the previous image in the directory."""
        self.current_image_index -= 1
        if self.current_image_index < 0:
            self.current_image_index = len(self.image_files) - 1
        self.show_image()

    def go_back(self):
        """Return to the main screen."""
        self.destroy()  # Close the current window
        self.master.deiconify()  # Restore the previous main window

    def on_closing(self):
        """Handle the window close event."""
        self.destroy()  # Close the current window
        self.master.destroy()  # Also close the main window

# Plot page (temp)
class PlotPage(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Plot Page")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = int((screen_width - 800) / 2)  # Window placed in the center of the screen
        y = int((screen_height - 480) / 2)
        self.geometry(f"800x480+{x}+{y}")

        # Back button created
        back_button = tk.Button(self, text="Back", command=self.go_back)
        back_button.pack(side=tk.TOP, pady=10)

        # Matplotlib graph created
        fig = Figure(figsize=(5, 4), dpi=100)
        t = [0.1 * i for i in range(100)]
        s = [i ** 2 for i in t]
        ax = fig.add_subplot(111)
        ax.plot(t, s)
        ax.set_title("matplotlib test")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")

        # FigureCanvasTkAgg used to insert Matplotlib Figure into Tkinter widget
        canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def go_back(self):
        self.destroy()  # Current window closed
        self.master.deiconify()  # Previous main window restored