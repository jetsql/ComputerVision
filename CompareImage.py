#!/usr/bin/env python
# coding: utf-8

# ## import

# In[1]:


import cv2
import numpy as np
import PIL
from PIL import Image
import shutil,os,sys
import glob
from time import sleep
import time
import matplotlib.pyplot as plt
#import matplotlib.image as ImagePG


# ## 統一大小

# In[2]:


#單改一張大小
def ChangeSize(filename):
        img=Image.open(filename)
        #尺寸轉換200x200
        img_new=img.resize((500,500),PIL.Image.ANTIALIAS)
        img_new.save('Screen/transport/'+filename.split("/")[-1] )


# ## 前處理

# In[3]:


#圖片,要不要存檔
def ImageProcess(imgpath,switch):
    img=cv2.imread(imgpath)
    #/////Laplacian//////
    gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(gray_lap) # 轉回uint8
    #/////Laplacian//////
    
    #/////存檔/////
    if switch == 1:
        #取檔案名
        #split處理檔名
        cv2.imwrite('Screen/transport/'+imgpath.split("/")[-1],dst)


# ## CheckSum-計算單通道的直方圖的相似值

# In[4]:


# 計算單通道的直方圖的相似值 
def calculate(image1,image2): 
    #cv2.calcHist(影像, 通道, 遮罩, 區間數量, 數值範圍)
    hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0]) 
    hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0]) 
    # 畫圖
    '''
    plt.plot(range(256),hist1,'r') 
    plt.plot(range(256),hist2,'b') 
    plt.show() 
    '''
    
    # 計算直方圖的重合度 
    degree = 0 
    for i in range(len(hist1)): 
        if hist1[i] != hist2[i]: 
            degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i])) 
        else: 
            degree = degree + 1 
    degree = degree/len(hist1)
    
    return degree 


# ## 設定門檻

# In[6]:


def CompareThreshold():
    #用兩兩比較出的數值當作門檻
    #設門檻0.969
    '''
    rate=calculate(cv2.imread('Screen/compare/c003.jpg'),
                    cv2.imread('Screen/compare/c004.jpg'))
    '''
    #0.6881
    rate=calculate(cv2.imread('Screen/compare/c001.jpg'),cv2.imread('Screen/compare/c002.jpg'))
    return rate


# ## 合併比對

# In[11]:


#(原圖,比較圖)
def CompareAlgo(file,file2):
    start = time.time()

    #統一大小 function
    ChangeSize(file)
    ChangeSize(file2)
    
    #前處理影像 function 
    ImageProcess(file,1)
    ImageProcess(file2,1)
    
    #設定門檻 function
    threshold=CompareThreshold()
    
    #比對兩張相似度 function
    rate=calculate(cv2.imread('Screen/transport/'+file.split("/")[-1]),cv2.imread('Screen/transport/'+file2.split("/")[-1]))
    print(rate)
    #相似度再比對門檻值
    if rate<threshold:
        return False
    if rate>=threshold:
        return True

    end=time.time()
    '''
    print("執行時間：%f 秒" % (end - start))

    print(threshold)
    print(rate)
    '''


# ## 使用

# In[12]:


'''
file='Screen/scan228.jpg'
file2='Screen/scan226.jpg'
CompareAlgo(file,file2)
'''


# In[ ]:




