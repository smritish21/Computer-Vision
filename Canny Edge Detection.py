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


# In[2]:


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


# In[3]:


I = gray(load_image("img1.png"))


I_blured = gaussian_blur(I,9)  # make it "compile"

show(I_blured, cmap="gray")


# In[4]:


SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

def compute_grads(I):
    I_x = conv(I,SOBEL_X)
    I_y = conv(I,SOBEL_Y) 
    magnitude = np.hypot(I_x,I_y) 
    #normalization
    magnitude = magnitude / magnitude.max() * 255
    orientation = np.arctan2(I_x,I_y)  
    return magnitude, orientation

magnitude, orientation = compute_grads(I_blured)
show(
    magnitude, orientation,
    titles=["magnitude", "orientation"],
    figsize=(16,4)
)


# In[5]:


from math import radians

def non_max_suppression(magnitude, orientation):
    m, n = magnitude.shape
    z = np.zeros((m,n), dtype=np.int32)
    angle = orientation * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1,m-1):
        for j in range(1,n-1):
            q = 255
            r = 255
            
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
                #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
                #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
                #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                z[i,j] = magnitude[i,j]
            else:
                z[i,j] = 0
    
    return z



magnitude_nms = non_max_suppression(magnitude, orientation)
show(magnitude_nms)


# In[6]:


WEAK = 64
STRONG = 255

def double_threshold(magnitude, alpha=0.09, beta=0.05):
        
    high = magnitude.max() * alpha;
    low = high * beta;
    
    m, n = magnitude.shape
    res = np.zeros((m,n), dtype=np.int32)
    
    
    strong_i, strong_j = np.where(magnitude >= high)
    zeros_i, zeros_j = np.where(magnitude < low)
    
    weak_i, weak_j = np.where((magnitude <= high) & (magnitude >= low))
    
    res[strong_i, strong_j] = STRONG
    res[weak_i, weak_j] = WEAK
    
    
    return res

magnitude_threshold = double_threshold(magnitude_nms)
show(magnitude_threshold)


# In[7]:


def hysteresis(magnitude_threshold):
    m, n = magnitude_threshold.shape  
    for i in range(1, m-1):
        for j in range(1, n-1):
            if (magnitude_threshold[i,j] == WEAK):
                try:
                    if ((magnitude_threshold[i+1, j-1] == STRONG) or (magnitude_threshold[i+1, j] == STRONG) or (magnitude_threshold[i+1, j+1] == STRONG)
                        or (magnitude_threshold[i, j-1] == STRONG) or (magnitude_threshold[i, j+1] == STRONG)
                        or (magnitude_threshold[i-1, j-1] == STRONG) or (magnitude_threshold[i-1, j] == STRONG) or (magnitude_threshold[i-1, j+1] == STRONG)):
                        magnitude_threshold[i, j] = STRONG
                    else:
                        magnitude_threshold[i, j] = 0
                except IndexError as e:
                    pass
    return magnitude_threshold

edges = hysteresis(magnitude_threshold)
show(edges)


# In[8]:


def canny_edge(I):
    I_blur = gaussian_blur(I,9)
    magnitude, orientation = compute_grads(I_blur)
    magnitude_nms = non_max_suppression(magnitude, orientation)
    magnitude_threshold = double_threshold(magnitude_nms)
    edges = hysteresis(magnitude_threshold)
    return edges

I = load_image("img1.png")
show(
    I, canny_edge(gray(I)),
    titles=["original", "edges"],
    figsize=(16,4),
    cmap="gray"
)
I = load_image("park.png")
show(
    I, canny_edge(gray(I)),
    titles=["original", "edges"],
    figsize=(16,4),
    cmap="gray"
)


# In[ ]:




