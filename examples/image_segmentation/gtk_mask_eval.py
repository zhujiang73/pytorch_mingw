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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,      out_planes, 
                     kernel_size=3,  stride=stride,
                     padding=1,      bias=False)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class C2Block(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super(C2Block, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(inplanes, inplanes)
        self.bn2 = nn.BatchNorm2d(inplanes)

        self.conv = nn.Conv2d(inplanes*2, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(outplanes)

        self.stride = stride

    def forward(self, x):
        residual = x

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = torch.cat([x1, x2], 1)
        out = self.conv(x3)
        out = self.norm(out)

        out += residual
        out = self.relu(out)

        return out

class C4Block(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super(C4Block, self).__init__()
        self.block01 = C2Block(inplanes, inplanes)
        self.block02 = C2Block(inplanes, inplanes)
        self.conv = nn.Conv2d(inplanes*2, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        x1 = self.block01(x)
        x2 = self.block02(x1)

        x3 = torch.cat([x1, x2], 1)
        out = self.conv(x3)
        out = self.norm(out)

        out += residual
        out = self.relu(out)

        return out

class C8Block(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super(C8Block, self).__init__()
        self.block01 = C4Block(inplanes, inplanes)
        self.block02 = C4Block(inplanes, inplanes)
        self.conv = nn.Conv2d(inplanes*2, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        x1 = self.block01(x)
        x2 = self.block02(x1)

        x3 = torch.cat([x1, x2], 1)
        out = self.conv(x3)
        out = self.norm(out)

        out += residual
        out = self.relu(out)

        return out

class Cat3C8Block(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super(Cat3C8Block, self).__init__()
        self.layer01 = C8Block(inplanes, inplanes)
        self.layer02 = C8Block(inplanes, inplanes)
        self.layer03 = C8Block(inplanes, inplanes)

        self.conv_cat = nn.Conv2d(inplanes*3, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_cat = nn.BatchNorm2d(outplanes)
        self.relu_cat = nn.ReLU(inplace=True)

    def forward(self, x):

        x_layer01 = self.layer01(x)
        x_layer02 = self.layer02(x_layer01)
        x_layer03 = self.layer03(x_layer02)

        x_layer_cat = torch.cat([x_layer01, x_layer02, x_layer03], 1)

        x_out = self.conv_cat(x_layer_cat)
        x_out = self.norm_cat(x_out)
        x_out = self.relu_cat(x_out)

        return  x_out


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   
            nn.BatchNorm2d(32),         
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, stride=2, padding=1),           
            nn.BatchNorm2d(32),         
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(Cat3C8Block(32, 64),)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),  
            nn.BatchNorm2d(32), 
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1),   
            nn.Tanh(),
        )

    def forward(self, x):

        x_features = self.features(x)

        x_encoded = self.encoder(x_features)

        x_decoded = self.decoder(x_encoded)

        return x_encoded, x_decoded


class Encoder_Thread(threading.Thread):
    def __init__(self, p_caller_slot, p_lock, p_model, p_str_fns, p_show_data):
        super(Encoder_Thread, self).__init__()
        self.caller_slot = p_caller_slot
        self.lock = p_lock
        self.model = p_model
        self.str_fns = p_str_fns
        self.show_data = p_show_data
        self.wh = 97
        #self.wh = 129
        self.batch_size = 5
        self.n_loops = 1  #100
        self.len_fns = len(self.str_fns)
        self.imgs_ten = torch.Tensor(self.batch_size,1,self.wh,self.wh)

    def run(self):
        for idx_loop in range(self.n_loops):

            idx_batch = idx_loop*self.batch_size

            for idx in range(idx_batch, idx_batch+self.batch_size):

                img_gray = skio.imread(self.str_fns[idx], as_gray=True)
                img_regray = sktr.resize(img_gray, (self.wh, self.wh), mode = 'reflect', anti_aliasing = True )
                img_rescale = skie.rescale_intensity(img_regray, in_range="image", out_range=(0.0, 1.0) )        
                img_data = img_rescale - 0.5
                img_data = img_data / 0.5

                self.imgs_ten[idx-idx_batch][0] = torch.from_numpy(img_data) 

            x_encoded, x_decoded = self.model(self.imgs_ten) 

            print("loop_{0:0>5d} idx:{1:0>5d}  x_encoded.shape:{2}".format(idx_loop, idx_batch, x_encoded.shape))

            np_imgs = self.imgs_ten.detach().numpy()
            np_decoded = x_decoded.detach().numpy()

            self.lock.acquire()
            mess_quit = self.show_data["mess_quit"]
            self.show_data["np_imgs"] = np_imgs.copy()
            self.show_data["np_decoded"] = np_decoded.copy()
            self.lock.release()

            if (mess_quit == True):
                    break;
            
            GLib.idle_add(self.caller_slot, "torch data")
            #time.sleep(5.0)

class main:
    def __init__(self):
        self.imgs_dir = "./imgs"

        self.str_fns = []

        for parent,dirnames,filenames in os.walk(self.imgs_dir): 
            for filename in filenames: 
                fn = self.imgs_dir + "/" + filename
                self.str_fns.append(fn)

        #self.str_pth_fn = "./models/bpcv_encoder_12000.pth"
        self.str_pth_fn = "./models/bpcv_encoder_24000.pth"

        self.autoencoder = torch.load(self.str_pth_fn)
        self.autoencoder.eval()

        self.win = Gtk.Window()
        self.win.connect("delete-event", self.win_quit)
        self.win.set_default_size(1000, 600)
        self.win.set_title("show imgs")

        self.sw = Gtk.ScrolledWindow()
        self.win.add(self.sw)
        self.sw.set_border_width(2)
        self.win.show_all()

        self.axs = [[0,0,0,0,0],[0,0,0,0,0]]

        self.torch_lock = threading.Lock()
        self.torch_show_data = {}
        self.n_test_imgs = 5
        self.torch_show_data["mess_quit"] = False  

        fig = Figure(figsize=(8, 8), dpi=80)

        for n in range(2):
            for i in range(self.n_test_imgs):
                self.axs[n][i] = fig.add_subplot(2, self.n_test_imgs, n*self.n_test_imgs+i+1)

        self.canvas = FigureCanvas(fig)
        self.canvas.set_size_request(800, 620)
        self.sw.add(self.canvas)

        thread_torch = Encoder_Thread(self.update_torch_data, self.torch_lock, self.autoencoder, 
                                        self.str_fns, self.torch_show_data)
        thread_torch.start()    


    def update_torch_data(self, str_txt):
        self.torch_lock.acquire()
        np_imgs = self.torch_show_data["np_imgs"]
        np_decoded = self.torch_show_data["np_decoded"]
        #self.torch_show_data.clear()
        self.torch_lock.release()

        self.sw.hide()

        for i in range(self.n_test_imgs):
            self.axs[0][i].imshow(np_imgs[i][0], cmap='gray')
            self.axs[1][i].imshow(np_decoded[i][0], cmap='gray')

        self.sw.show_all()


    def win_quit(self,  a,  b):     
        self.torch_lock.acquire()
        self.torch_show_data["mess_quit"] = True   
        self.torch_lock.release()
        Gtk.main_quit()
 

if __name__ == "__main__":
    win_main = main()

    Gtk.main()


