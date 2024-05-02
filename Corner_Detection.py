#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import requests
from PIL import Image


# In[4]:


def load_image(name, no_alpha=True):
    url = 'Image URL'
    image = np.asarray(Image.open(requests.get(url, stream=True).raw))
    if no_alpha and len(image) > 2 and image.shape[2] == 4:
        image = image[:,:,:3]
    return image[:,:,::-1].copy()

def resize(image, scale):
    return cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))

def show(*images, titles=None, figsize=None, **kwargs):
    ROWS, COLS = 1, len(images)
    if figsize is not None:
        plt.figure(figsize=(18,6))
    for i, img in enumerate(images):
        plt.subplot(ROWS, COLS, i+1)
        if titles is not None:
            plt.title(titles[i])
        if len(img.shape) == 3:
            plt.imshow(img[:,:,::-1], **kwargs)
        else:
            plt.imshow(img, **kwargs)
    plt.show()
    
def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
def conv(image, kernel):
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

def gaussian_blur(image, size):
    return cv2.GaussianBlur(image, (size, size), 0)

def visualize_corners(I, R, threshold=0.6):
    I = I.copy()
    loc = np.where(R >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.circle(I, pt, 3, (0, 0, 255), -1)
    return I


# In[6]:


SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)


# In[7]:


I = load_image("img1.png")
I_gray = gray(I) / 255.0

def apply_sobel(I_gray):
    I_x = conv(I_gray,SOBEL_X)
    I_y = conv(I_gray,SOBEL_Y)
    dx = I_x  
    dy = I_y  
    return dx, dy

sobel = apply_sobel(I_gray)
show(
    *sobel,
    titles=["dx", "dy"],
    figsize=(16,4),
    cmap="gray"
)


# In[8]:


def compute_components_of_H(dx, dy):
    dx2 = np.square(dx)  
    dy2 = np.square(dy)  
    dxdy = dx*dy  
    dx2 = gaussian_blur(dx2,5)
    dy2 = gaussian_blur(dy2,5)
    dxdy = gaussian_blur(dxdy,5)

    return dxdy, dx2, dy2

H = compute_components_of_H(*sobel)
show(
    *H,
    titles=["dxdy", "dx2", "dy2"],
    figsize=(16,4),
    cmap="gray"
)


# In[9]:


def compute_R(dxdy, dx2, dy2, k=0.06):
    H=np.array([[dx2,dxdy],[dxdy,dy2]])
    det=dx2*dy2-np.square(dxdy)
    tr=np.matrix.trace(H)
    R = det - k*(tr**2)  
    norm = (R-np.min(R))/(np.max(R)-np.min(R))
    return norm

R = compute_R(*H)
show(
    R,
    titles=["R matrix"],
    figsize=(16,4),
    cmap="gray"
)


# In[10]:


def harris(I_gray, k=0.06):
    R = I_gray  
    sobel = apply_sobel(R)
    H = compute_components_of_H(*sobel)
    R = compute_R(*H)
    return R

I = load_image("img1.png")
I_gray = gray(I) / 255.0
R = harris(I_gray)
show(
    R, visualize_corners(I, R),
    titles=["R matrix", "Corners"],
    figsize=(16,4),
    cmap="gray"
)


# In[ ]:




