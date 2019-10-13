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
    def __init__(self, p_caller_slot, p_lock, p_model, p_str_fns, p_str_mask_fns, p_show_data):
        super(Encoder_Thread, self).__init__()
        self.caller_slot = p_caller_slot
        self.lock = p_lock
        self.model = p_model
        self.str_fns = p_str_fns
        self.str_mask_fns = p_str_mask_fns
        self.show_data = p_show_data
        self.wh = 97
        #self.wh = 129
        self.batch_size = 10
        self.len_fns = len(self.str_fns)
        self.imgs_ten = torch.Tensor(self.batch_size,1,self.wh,self.wh)
        self.imgs_mask_ten = torch.Tensor(self.batch_size,1,self.wh,self.wh)

    def run(self):
        lrs = [0.01*0.1, 0.01*0.1*0.5]

        m_loops = [6000*4, 6000*5]  

        mess_quit = False

        for n in [0,1]:

            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lrs[n])
            loss_func = nn.MSELoss()

            for idx_loop in range(m_loops[n], m_loops[n]+6000):

                idx_batch = random.randint(0, self.len_fns-self.batch_size-1)

                for idx in range(idx_batch, idx_batch+self.batch_size):

                    img_gray = skio.imread(self.str_fns[idx], as_gray=True)
                    img_regray = sktr.resize(img_gray, (self.wh, self.wh), mode = 'reflect', anti_aliasing = True )
                    img_rescale = skie.rescale_intensity(img_regray, in_range="image", out_range=(0.0, 1.0) )        
                    img_data = img_rescale - 0.5
                    img_data = img_data / 0.5

                    img_mask_gray = skio.imread(self.str_mask_fns[idx], as_gray=True)
                    img_mask_regray = sktr.resize(img_mask_gray, (self.wh, self.wh), mode = 'reflect', anti_aliasing = True )
                    img_mask_rescale = skie.rescale_intensity(img_mask_regray, in_range="image", out_range=(0.0, 1.0) )        
                    img_mask_data = img_mask_rescale - 0.5
                    img_mask_data = img_mask_data / 0.5

                    self.imgs_ten[idx-idx_batch][0] = torch.from_numpy(img_data) 
                    self.imgs_mask_ten[idx-idx_batch][0] = torch.from_numpy(img_mask_data) 

                x_encoded, x_decoded = self.model(self.imgs_ten)   
                loss = loss_func(x_decoded, self.imgs_mask_ten)      # mean square error
                optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                     # backpropagation, compute gradients
                optimizer.step()                    # apply gradients   

                np_imgs = self.imgs_ten.detach().numpy()
                np_mask_imgs = self.imgs_mask_ten.detach().numpy()
                np_decoded = x_decoded.detach().numpy()

                self.lock.acquire()
                mess_quit = self.show_data["mess_quit"]
                self.lock.release()

                if (mess_quit == True):
                    return  

                if (idx_loop % 10 == 0):
                    print("loop_{0:0>5d} idx:{1:0>5d}  train loss: {2:.4f}".format(idx_loop, idx_batch, loss.data.numpy()))

                if (idx_loop % 10 == 0):
                    self.lock.acquire()
                    self.show_data["np_imgs"] = np_imgs.copy()
                    self.show_data["np_mask_imgs"] = np_mask_imgs.copy()
                    self.show_data["np_decoded"] = np_decoded.copy()
                    self.lock.release()
                    GLib.idle_add(self.caller_slot, "torch data")

                if ((idx_loop+1) % 500 == 0):
                    str_pth_fn = "./models/bpcv_encoder_{0:0>5d}.pth".format(idx_loop+1)
                    print("model save to : {}".format(str_pth_fn))
                    torch.save(self.model, str_pth_fn)


class main:
    def __init__(self):
        imgs_dir = "e:/cv_datasets/CUB_200_2011/images"
        imgs_mask_dir = "e:/cv_datasets/CUB_200_2011/segmentations"

        self.str_imgs_fns = []
        self.str_mask_fns = []

        dirs = []

        for parent,dirnames,filenames in os.walk(imgs_dir):
            for dirname in dirnames:
                dirs.append(dirname)

        for str_dir in dirs:
            str_dir_path = imgs_dir + "/" + str_dir
            for parent,dirnames,filenames in os.walk(str_dir_path):
                for filename in filenames:
                    str_path = str_dir_path + "/" + filename
                    self.str_imgs_fns.append(str_path)
                    idx = filename.find(".jpg")
                    str_mask_path = imgs_mask_dir + "/" + str_dir + "/" + filename[:idx] + ".png"
                    self.str_mask_fns.append(str_mask_path)

        str_pth_fn = "./models/bpcv_encoder_24000.pth"
        self.autoencoder = torch.load(str_pth_fn)

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
                                        self.str_imgs_fns, self.str_mask_fns, self.torch_show_data)

        thread_torch.start()    


    def update_torch_data(self, str_txt):

        self.torch_lock.acquire()
        np_imgs = self.torch_show_data["np_imgs"]
        np_mask_imgs = self.torch_show_data["np_mask_imgs"] 
        np_decoded = self.torch_show_data["np_decoded"]
        self.torch_lock.release()

        self.sw.remove(self.canvas)

        axs = [[0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,0,0,0]]

        fig = Figure(figsize=(8, 8), dpi=80)

        for n in range(3):
            for i in range(self.n_test_imgs):
                axs[n][i] = fig.add_subplot(3, self.n_test_imgs, n*self.n_test_imgs+i+1)

        for i in range(self.n_test_imgs):
            axs[0][i].imshow(np_imgs[i][0], cmap='gray')
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
    win_main = main()

    Gtk.main()


