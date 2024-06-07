import tkinter as tk
from PIL import Image, ImageTk

#Window config
class Wd_config:
    """
    Class to manage window configurations such as window size, font settings, and background image settings.
    """
    def __init__(self) -> None:
        # Window size
        self.width = 800
        self.height = 480
        # Font
        self.main_font = ("Arial", 26)
        self.sub_font = ("Arial", 16)
        self.sub_font_small = ("Arial", 12)
    
    def erase_title_bar(self, master):
        # erase Window title bar
        master.overrideredirect(True)   

    def set_center_position(self, master):
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        self.position_x = int((screen_width - self.width) / 2)
        self.position_y = int((screen_height - self.height) / 2)
        return f"{self.width}x{self.height}+{self.position_x}+{self.position_y}"

    def set_bg_img(self, parent):
        # Background image config
        self.bg_img_width = self.width
        self.bg_img_height = self.height
        # Open the background image
        self.bg_img_path = "src/bg_img.jpg"
        bg_image = Image.open(self.bg_img_path)
        # Resize the image to fit the window
        bg_image = bg_image.resize((self.bg_img_width, self.bg_img_height), Image.Resampling.LANCZOS)
        # Create a label with the image
        bg_photo = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(parent, image=bg_photo)
        bg_label.image = bg_photo  # Keep a reference to avoid garbage collection
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        #bg_label.lower()
        return bg_label

#Button config
class Bt_config:
    """
    Class to manage button configurations including button sizes and positions.
    """
    def __init__(self) -> None:
        # Window size
        self.width = 800
        self.height = 480
        # Padding
        self.p_x = int(20)
        self.p_y = int(20)
        #region Button size
        # Back button
        self.back_button_width = 60
        self.back_button_height = 60
        # Next button
        self.next_button_width = 240
        self.next_button_height = 160
        # Check button
        self.check_button_width = 240
        self.check_button_height = 160
        # Button type1 
        self.button_type1_width = 320
        self.button_type1_height = 210
        # Button type2
        self.button_type2_width = 300
        self.button_type2_height = 360
        # Button3-1
        self.button3_width = 60
        self.button3_height = 60
        # exit button
        self.exit_button_width = 60
        self.exit_button_height = 60
        #endregion
        
        #region Button position 
        # Back button
        self.back_button_x = int(self.p_x)
        self.back_button_y = int(self.p_y)
        # Next button
        self.next_button_x = int(self.width - self.next_button_width - self.p_x)
        self.next_button_y = int(2.5*self.p_y+self.back_button_height)
        # Check button
        self.check_button_x = int(self.next_button_x)
        self.check_button_y = int(self.next_button_y+self.next_button_height+self.p_y)
        # Button1 - right up side
        self.button_pos1_x = int(self.width - self.button_type1_width - self.p_x)
        self.button_pos1_y = int(self.p_y)
        # Button2 - right down side
        self.button_pos2_x = int(self.width - self.button_type1_width - self.p_x)
        self.button_pos2_y = int(2*self.p_y+self.button_type1_height)
        # Buttone3 - left up side
        self.button3_x = int(100)
        self.button3_y = int(self.p_y)
        # Button4 - left down side
        self.button4_x = int(100)
        self.button4_y = int(240)
        # Button right big one
        self.button_right_big_x = int(480)
        self.button_right_big_y = int(100)
        # exit button
        self.exit_button_x = int(self.p_x)
        self.exit_button_y = int(self.height - self.exit_button_height - self.p_y)
        # Button_t - matplot test
        self.buttont_x = int(self.p_x+60)
        self.buttont_y = int(self.exit_button_y)
        #endregion

#Text config
class Txt_config:
    """
    Class to manage text configurations including font settings.
    """
    def __init__(self) -> None:
        # Window size
        self.width = 800
        self.height = 480
        # Padding
        self.p_x = int(20)
        self.p_y = int(20)
        # innner Padding
        self.ip_x = int(10)
        self.ip_y = int(10)
        
        # Font
        self.f24 = ("Arial", 24)
        self.f20 = ("Arial", 20)
        self.f16 = ("Arial", 16)
        self.f12 = ("Arial", 12)
        self.f10 = ("Arial", 10)
        self.f8 = ("Arial", 8)
        self.f4 = ("Arial", 4)
        
        # size
        self.size_x = int(680)
        self.size_y = int(60)
        self.big_size_x = int(760)
        self.big_size_y = int(440)
        self.middle1_size_x = int(280)
        self.middle1_size_y = int(280)
        self.middle2_size_x = int(self.middle1_size_x)
        self.middle2_size_y = int(self.middle1_size_y)
        self.middle3_size_x = int(120)
        self.middle3_size_y = int(120)  
        self.middle4_size_x = int(self.middle3_size_x)
        self.middle4_size_y = int(self.middle3_size_y)
        
        #info container
        self.bottom_bar_size_x = int(740)
        self.bottom_bar_size_y = int(130)
        #info name container
        self.info1_name_size_x = int(self.bottom_bar_size_x/2)
        self.info1_name_size_y = int(self.bottom_bar_size_y/4)
        self.info2_name_size_x = int(self.bottom_bar_size_x/2)
        self.info2_name_size_y = int(self.bottom_bar_size_y/4)
        
        #color info
        self.bb_1_size_x = int(self.bottom_bar_size_x/2)
        self.bb_1_size_y = int(self.bottom_bar_size_y/2)
        
        self.bottom_bar_1_size_x = int(self.bottom_bar_size_x/4)
        self.bottom_bar_1_size_y = int(self.bottom_bar_size_y/4)
        self.bottom_bar_2_size_x = int(self.bottom_bar_size_x/4)
        self.bottom_bar_2_size_y = int(self.bottom_bar_size_y/4)
        self.bottom_bar_3_size_x = int(self.bottom_bar_size_x/4)
        self.bottom_bar_3_size_y = int(self.bottom_bar_size_y/4)
        self.bottom_bar_4_size_x = int(self.bottom_bar_size_x/4)
        self.bottom_bar_4_size_y = int(self.bottom_bar_size_y/4)
        
        # position
        self.pos_x = int(100)
        self.pos_y = int(self.p_y)
        
        self.middle1_pos_x = int(self.p_x + self.ip_x)
        self.middle1_pos_y = int(self.p_y + self.ip_y)
        self.middle2_pos_x = int(self.middle1_pos_x + self.middle1_size_x + self.ip_x)
        self.middle2_pos_y = int(self.middle1_pos_y)
        
        self.middle3_pos_x = int(self.middle2_pos_x+self.middle2_size_x+3*self.ip_x)
        self.middle3_pos_y = int(self.middle1_pos_y+20)
        self.middle4_pos_x = int(self.middle2_pos_x+self.middle2_size_x+3*self.ip_x)
        self.middle4_pos_y = int(self.middle3_pos_y+self.middle3_size_y+self.ip_y)
        
        self.bottom_bar_pos_x = int(self.p_x+self.ip_x)
        self.bottom_bar_pos_y = int(self.p_y+self.middle2_size_y+4*self.ip_y)
        
        self.bb_1_pos_x = int(self.bottom_bar_pos_x)
        self.bb_1_pos_y = int(self.bottom_bar_pos_y+self.ip_y)
        self.bb_2_pos_x = int(self.bb_1_pos_x+self.bb_1_size_x)
        self.bb_2_pos_y = int(self.bb_1_pos_y)
        self.bb_3_pos_x = int(self.bb_1_pos_x)
        self.bb_3_pos_y = int(self.bb_1_pos_y+self.bb_1_size_y+self.ip_y)
        self.bb_4_pos_x = int(self.bb_1_pos_x+self.bb_1_size_x)
        self.bb_4_pos_y = int(self.bb_1_pos_y+self.bb_1_size_y+self.ip_y)
        
        self.bottom_bar_1_pos_x = int(self.bottom_bar_pos_x)
        self.bottom_bar_1_pos_y = int(self.bottom_bar_pos_y+5*self.ip_y)
        self.bottom_bar_2_pos_x = int(self.bottom_bar_pos_x+2*self.bottom_bar_1_size_x)
        self.bottom_bar_2_pos_y = int(self.bottom_bar_pos_y+5*self.ip_y)
        self.bottom_bar_3_pos_x = int(self.bottom_bar_pos_x)
        self.bottom_bar_3_pos_y = int(self.bottom_bar_pos_y+self.bottom_bar_1_size_y+5*self.ip_y)
        self.bottom_bar_4_pos_x = int(self.bottom_bar_pos_x+2*self.bottom_bar_1_size_x)
        self.bottom_bar_4_pos_y = int(self.bottom_bar_pos_y+self.bottom_bar_1_size_y+5*self.ip_y)
        
        self.info1_name_pos_x = int(self.p_x+self.ip_x)
        self.info1_name_pos_y = int(self.p_y+self.middle2_size_y+5*self.ip_y)
        self.info2_name_pos_x = int(self.info1_name_pos_x+self.info1_name_size_x)
        self.info2_name_pos_y = int(self.p_y+self.middle2_size_y+5*self.ip_y)
        
