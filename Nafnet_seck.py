#!/usr/bin/env python
# coding: utf-8

# # SECK Mouhamadou Abdoulaye


import os
os.chdir('NAFNet')



"""!python -m pip install --upgrade pip
!python -m pip install --upgrade Pillow
!pip install nes-py --no-cache-dir
!pip install --upgrade pip setuptools wheel"""


# In[6]:


#!pip install -r requirements.txt
#!pip install --upgrade --no-cache-dir gdown
#!python setup.py develop --no_cuda_ext


# In[7]:


#gdown.download('https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR', "./experiments/pretrained_models/", quiet=False)


# In[9]:


from skimage import io, color
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import torch

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import shutil
import glob
import image_slicer


# In[ ]:





# In[33]:


def regroupe(dec):
    res=[]
    az = list(range(len(dec)))
    j=0
    aa=az[::int(np.sqrt(len(dec)))]
    aa.append(len(dec))
    for i in range(len(aa)-1):
        res.append(np.concatenate((dec[j:aa[i+1]]),axis =1))
        j=aa[i+1]
    image = np.concatenate((res),axis = 0)
    return image


def recup(img):
    out_t = img.copy()
    new = color.rgb2gray(out_t)
    return new


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def display(img1, img2):
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1) 
    plt.title('Input image', fontsize=16)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.title('NAFNet output', fontsize=16)
    ax2.axis('off')
    ax1.imshow(img1)
    ax2.imshow(img2)

def single_image_inference(model, img, save_path):
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, save_path)


# In[34]:


opt_path = 'options/test/SIDD/NAFNet-width64.yml'
opt = parse(opt_path, is_train=False)
opt['dist'] = False
NAFNet = create_model(opt)


# In[35]:


input_path=input("L'adresse du fichier : ")
im = plt.imread(input_path)

# In[36]:



uploadd_folder = 'seck/image'
upload_folder = 'seck/input'
result_folder = 'seck/output'
result_inp = 'seck/dec_inp'
result_out = 'seck/dec_out'

if os.path.isdir(upload_folder):
    shutil.rmtree(upload_folder)
if os.path.isdir(result_folder):
    shutil.rmtree(result_folder)
if os.path.isdir(uploadd_folder):
    shutil.rmtree(uploadd_folder)
if os.path.isdir(result_inp):
    shutil.rmtree(result_inp)
if os.path.isdir(result_out):
    shutil.rmtree(result_out)
os.makedirs(upload_folder)
os.makedirs(uploadd_folder)
os.makedirs(result_folder)
os.makedirs(result_inp)
os.makedirs(result_out)


# In[37]:


input = "seck/image/im_jpg.jpg"
plt.imsave(input, im)

b=image_slicer.slice(input,4)

a=[]
for i in b:
    a.append(plt.imread(i.filename))

for j,i in zip("abcdefghijklmnopqrstuvwxyz",range(len(a))):
    mpimg.imsave(f"seck/input/resultat{j}.png", a[i])



# In[38]:


import glob
input_list = sorted(glob.glob(os.path.join(upload_folder, '*')))
for input_path in input_list:
    img_input = imread(input_path)
    inp = img2tensor(img_input)
    output_path = os.path.join(result_folder, os.path.basename(input_path))
    single_image_inference(NAFNet, inp, output_path)


# In[39]:


# visualize
inp=[]
out=[]
input_list = sorted(glob.glob(os.path.join(upload_folder, '*')))
output_list = sorted(glob.glob(os.path.join(result_folder, '*')))
for input_path, output_path in zip(input_list, output_list):
    img_input = imread(input_path)
    img_output = imread(output_path)
    out.append(img_output)
    inp.append(img_input)

# In[40]:


for j,i in zip("abcdefghijklmnopqrstuvwxyz",range(len(inp))):
    mpimg.imsave(f"/content/NAFNet/seck/dec_inp/inp{j}.tiff", inp[i])
    mpimg.imsave(f"/content/NAFNet/seck/dec_out/out{j}.tiff", out[i])


# In[41]:


dec=[]
for i in range(len(output_list)):
    dec.append(plt.imread(output_list[i]))


# In[42]:


images = regroupe(dec)
new_out_np = recup(images) 


# In[44]:


plt.imsave(f"seck/output/resultat.tiff", new_out_np,cmap='gray')
plt.imsave(f"seck/output/orig.tiff", im,cmap='gray')


# In[44]:




