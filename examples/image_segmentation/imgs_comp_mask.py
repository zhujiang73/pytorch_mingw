import os 
import os.path 
import shutil 
import hashlib
import time
import random

import numpy as np

from skimage import io as skio
from skimage import exposure as skie
from skimage import transform as sktr
from skimage import color as skco
from skimage import img_as_float, img_as_ubyte
from skimage import feature as skft
from skimage import filters 
from skimage import draw,data

from  scipy import spatial
 
bg_wh = 150

def fun_img_comp(img_src_fg, img_src_mask):
    img_bg = np.zeros((bg_wh, bg_wh, 3), dtype=float)
    img_fg = np.zeros((bg_wh, bg_wh, 3), dtype=float)
    img_comp = np.zeros((bg_wh, bg_wh, 3), dtype=float)
    img_mask = np.zeros((bg_wh, bg_wh), dtype=float)
    
    h = img_src_fg.shape[0]
    w = img_src_fg.shape[1]

    wh = bg_wh

    if (w > h):
        zoom = wh*1.0 / w
        w1 = wh
        h1 = int(h * zoom + 0.5)
    else:
        zoom = wh*1.0 / h
        h1 = wh
        w1 = int(w * zoom + 0.5)
        
    img_src_fg = sktr.resize(img_src_fg, (h1, w1), mode = 'reflect', anti_aliasing = True )
    img_src_mask = sktr.resize(img_src_mask, (h1, w1), mode = 'reflect', anti_aliasing = True )
        
    h = img_src_fg.shape[0]
    w = img_src_fg.shape[1]
    
    y_idx = random.randint(0, bg_wh-h)
    x_idx = random.randint(0, bg_wh-w)

    img_comp = img_bg  
    img_comp[y_idx:y_idx+h,x_idx:x_idx+w] = img_src_fg
    img_mask[y_idx:y_idx+h,x_idx:x_idx+w] = img_src_mask
    
    return img_comp, img_mask

imgs_dir = "d:/cv_datasets/CUB_200_2011/images"
imgs_mask_dir = "d:/cv_datasets/CUB_200_2011/segmentations"
comp_dir = "./imgs_comp_box/0001"
comp_mask_dir = "./imgs_mask_box/0001"

str_imgs_fns = []
str_imgs_mask_fns = []
str_comp_fns = []
str_comp_mask_fns = []
        
dirs = []

for parent,dirnames,filenames in os.walk(imgs_mask_dir):
    for dirname in dirnames:
        dirs.append(dirname)

for str_dir in dirs:
    str_dir_path = imgs_mask_dir + "/" + str_dir
    for parent,dirnames,filenames in os.walk(str_dir_path):
        for filename in filenames:
            str_path = str_dir_path + "/" + filename
            str_imgs_mask_fns.append(str_path)
            idx = filename.find(".png")
            str_img_path = imgs_dir + "/" + str_dir + "/" + filename[:idx] + ".jpg"
            str_imgs_fns.append(str_img_path)

num_imgs = len(str_imgs_fns)

for idx_img in range(0, num_imgs):

    img_fg = img_as_float(skio.imread(str_imgs_fns[idx_img]))
    if (len(img_fg.shape)<3):
        img_fg = skco.gray2rgb(img_fg)
    img_fg = skie.rescale_intensity(img_fg, in_range="image", out_range=(0.0, 1.0) ) 

    img_mask = img_as_float(skio.imread(str_imgs_mask_fns[idx_img], as_gray=True))
    img_mask = skie.rescale_intensity(img_mask, in_range="image", out_range=(0.0, 1.0) ) 

    str_comp_fn = comp_dir + "/img_" + "{0:0>6d}".format(idx_img+1) + "_.png"
    str_comp_mask_fn = comp_mask_dir + "/img_" + "{0:0>6d}".format(idx_img+1) + "_.png"

    img_comp, img_comp_mask = fun_img_comp(img_fg, img_mask)  

    img_comp = skie.rescale_intensity(img_comp, in_range="image", out_range=(0.0, 1.0) ) 
    img_comp_mask = skie.rescale_intensity(img_comp_mask, in_range="image", out_range=(0.0, 1.0) ) 

    try:
        skio.imsave(str_comp_fn, img_as_ubyte(img_comp))
    except:
        print(str_comp_fn , "error !!!")
    else:
        print(str_comp_fn)
        
    try:
        skio.imsave(str_comp_mask_fn, img_as_ubyte(img_comp_mask))
    except:
        print(str_comp_mask_fn , "error !!!")
    else:
        print(str_comp_mask_fn)



