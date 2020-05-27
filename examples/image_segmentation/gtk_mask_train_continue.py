import os 
import os.path 
import shutil 
import hashlib
import time
import random
import threading

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from skimage import io as skio
from skimage import exposure as skie
from skimage import transform as sktr

import gi 
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib

import matplotlib.cm as cm

from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

from  auto_encoder  import  AutoEncoder, Encoder_Thread

class BpcvMain:
    def __init__(self):
        imgs_dir = "./imgs_comp_box"
        imgs_mask_dir = "./imgs_mask_box"

        self.str_imgs_fns = []
        self.str_mask_fns = []

        dirs = []

        for parent,dirnames,filenames in os.walk(imgs_mask_dir):
            for dirname in dirnames:
                dirs.append(dirname)

        for str_dir in dirs:
            str_dir_path = imgs_mask_dir + "/" + str_dir
            for parent,dirnames,filenames in os.walk(str_dir_path):
                for filename in filenames:
                    str_path = str_dir_path + "/" + filename
                    self.str_mask_fns.append(str_path)
                    idx = filename.find(".png")
                    str_img_path = imgs_dir + "/" + str_dir + "/" + filename[:idx] + ".png"
                    self.str_imgs_fns.append(str_img_path)

        #str_pth_fn = "./models/bpcv_encoder_06000.pth"   
        str_pth_fn = "./models/bpcv_encoder_12000.pth"   

        self.autoencoder = AutoEncoder()

        bpcv_dict = torch.load(str_pth_fn)
        self.autoencoder.load_state_dict(bpcv_dict["net_state"])

        print("continue: ...  n_loop: {0:0>5d}  idx_loop: {1:0>5d}".format(bpcv_dict["n_loop"], bpcv_dict["idx_loop"]) )
        print("..............................................................................")      

        self.win = Gtk.Window()
        self.win.connect("delete-event", self.win_quit)
        self.win.set_default_size(1000, 600)
        self.win.set_title("show imgs")

        self.sw = Gtk.ScrolledWindow()
        self.win.add(self.sw)
        self.sw.set_border_width(2)

        fig = Figure(figsize=(8, 8), dpi=80)
        self.canvas = FigureCanvas(fig) 
        self.canvas.set_size_request(1000, 600)
        self.sw.add(self.canvas)
        self.win.show_all()

        self.torch_lock = threading.Lock()
        self.torch_show_data = {}
        self.n_test_imgs = 5
        self.torch_show_data["mess_quit"] = False  

        thread_torch = Encoder_Thread(self.update_torch_data, self.torch_lock, self.autoencoder, 
                                      self.str_imgs_fns, self.str_mask_fns, self.torch_show_data,
                                      wh = 97, max_n_loop = 3, 
                                      n_loop = bpcv_dict["n_loop"], idx_segment = bpcv_dict["idx_loop"])

        thread_torch.start()    


    def update_torch_data(self, str_txt):

        self.torch_lock.acquire()
        np_imgs = self.torch_show_data["np_imgs"]
        np_mask_imgs = self.torch_show_data["np_mask_imgs"] 
        np_decoded = self.torch_show_data["np_decoded"]
        self.torch_lock.release()
        
        np_imgs = np_imgs.transpose((0, 2, 3, 1))

        self.sw.remove(self.canvas)

        axs = [[0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,0,0,0]]

        fig = Figure(figsize=(8, 8), dpi=80)

        for n in range(3):
            for i in range(self.n_test_imgs):
                axs[n][i] = fig.add_subplot(3, self.n_test_imgs, n*self.n_test_imgs+i+1)

        for i in range(self.n_test_imgs):
            axs[0][i].imshow(np_imgs[i] * 0.5 + 0.5, cmap='gray')   
            axs[1][i].imshow(np_mask_imgs[i][0], cmap='gray')
            axs[2][i].imshow(np_decoded[i][0], cmap='gray')

        self.canvas = FigureCanvas(fig)
        self.canvas.set_size_request(1000, 600)
        self.sw.add(self.canvas)
        self.sw.show_all()

    def win_quit(self,  a,  b):     
        self.torch_lock.acquire()
        self.torch_show_data["mess_quit"] = True   
        self.torch_lock.release()
        Gtk.main_quit()
      

if __name__ == "__main__":
    win_main = BpcvMain()

    Gtk.main()


