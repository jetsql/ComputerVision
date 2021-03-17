#!/usr/bin/env python
# coding: utf-8

# ## import

# In[20]:


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


# ## 確認檔案

# In[21]:


def CheckFile(path):
    filepath='Screen\\transport\\'+path.split("\\")[-1] 
    if os.path.isfile(filepath):
        os.remove(filepath)


# ## 統一大小

# In[215]:





# In[22]:


#單改一張大小
def ChangeSize(filename):
        img=Image.open(filename)
        #尺寸轉換200x200
        img_new=img.resize((500,500),PIL.Image.ANTIALIAS)
        #注意斜線方向
        img_new.save('Screen\\transport\\'+filename.split("\\")[-1] )


# In[187]:


#去紅線print 圖結果
def Dilate(path):
    #灰階讀進
    image = cv2.imread(path, 0)
    #膨脹
    kernel = np.ones((3,3), np.uint8)
    #(圖檔案,捲機大小,迭代次數)
    dilate = cv2.dilate(image, kernel, iterations = 2)

    #降噪(中值濾波)，彌平紅線條
    image_mid_blur =cv2.medianBlur(dilate, 9) 
    #cv2.imwrite('Screen/transport/Test/1-'+path.split("/")[-1],image_mid_blur)
    #細線化(凸顯切割線)
    kernelN=np.ones((11,11),np.uint8)
    Negtive=cv2.erode(image_mid_blur,kernelN,iterations=1)
    
    #混和拉普拉斯(銳利化)
    '''一般銳利效果
    Kernel_Laplacian=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    temp=cv2.filter2D(Negtive,cv2.CV_32F,Kernel_Laplacian)
    Laplacian_Mix=np.uint8(np.clip(temp,0,255))
    '''
    Kernel_Laplacian2=np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    temp2=cv2.filter2D(Negtive,cv2.CV_32F,Kernel_Laplacian2)
    Laplacian_Mix2=np.uint8(np.clip(temp2,0,255))
    

    
    #雙邊濾波器
    #TwoSide_Process=cv2.bilateralFilter(image_mid_blur,3,50,50)
    
    #影像疊加(增加畫素，變清晰
    #先高斯濾波(影像,捲積核,標準差)，模糊鋸齒狀
    Gaussian=cv2.GaussianBlur(Negtive,(0,0),50)
    usm = cv2.addWeighted(Negtive, 1.5, Gaussian, -0.5, 0)
    
    Gaussian2=cv2.GaussianBlur(Negtive,(0,0),100)
    usm2 = cv2.addWeighted(Negtive, 1.5, Gaussian2, -0.5, 0)

    #取邊緣(下線數值,上線數值)
    #canny=cv2.Canny(Gaussian,10,80)
    
    #先膨脹再減侵蝕
    #侵蝕Kernel
    #kernelN=np.ones((3,3),np.uint8)
    #Negtive=cv2.erode(image_mid_blur,kernelN,iterations=1)
    #相減(前面膨脹-侵蝕)
    #result=cv2.absdiff(image_mid_blur,Negtive)
    
    #反二值化更清楚(輸出rst)
    #thr,reverse=cv2.threshold(result,4,10,cv2.THRESH_BINARY_INV)

    #二值化(輸出forward)
    #thr,forward=cv2.threshold(result,4,10,cv2.THRESH_BINARY)
    
    #截斷設定
    #thr,capture=cv2.threshold(result,50,120,cv2.THRESH_TRUNC)
    
    #超零定址(大於歸0，小於保持)
    #thr,supzeort=cv2.threshold(result,100,5,cv2.THRESH_TOZERO_INV)
    
    #定值設定(大於保持,小於歸0)
    #thr,zeort=cv2.threshold(result,4,100,cv2.THRESH_TOZERO)
    
    #方框濾波
    
    #連接圖片
    #Hstack_Image=np.hstack([Laplacian_Mix,Laplacian_Mix2])
    #Hstack_Image2=np.hstack([usm,Negtive])
    ''' 
    #原圖laplase
    gray_lap1 = cv2.Laplacian(image, cv2.CV_16S, ksize=3)
    dst1 = cv2.convertScaleAbs(gray_lap1) # 轉回uint8

    #去除紅線laplase
    gray_lap2 = cv2.Laplacian(dilate, cv2.CV_16S, ksize=3)
    dst2= cv2.convertScaleAbs(gray_lap2) # 轉回uint8
    
    #去除紅線+降噪laplase
    gray_lap3 = cv2.Laplacian(image_mid_blur, cv2.CV_16S, ksize=3)
    dst3 = cv2.convertScaleAbs(gray_lap3) # 轉回uint8
    cv2.imwrite('Screen/transport/Test/2-'+path.split("/")[-1],dst3)
    '''
    
    while True:
        #cv2.imshow('Input 1',Hstack_Image)
        #cv2.imshow('Input 2',Hstack_Image2)
        
        #cv2.imshow('sharp 1',Laplacian_Mix2)
        cv2.imshow('sharp 2',usm)
        cv2.imshow('sharp 3',usm2)
        
        #cv2.imshow('Laplacian 1', dst1)
        #cv2.imshow('Laplacian 2', dst2)
        #cv2.imshow('Laplacian 3', dst3)

        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break


# In[43]:


'''
rate=calculate(cv2.imread('Screen/transport/OG/OG001.jpg'),cv2.imread('Screen/transport/OG/OG008.jpg'))
print("OG",rate)
rate=calculate(cv2.imread('Screen/transport/Test/1-OG003.jpg'),cv2.imread('Screen/transport/Test/1-OG008.jpg'))
print("去紅線",rate)
rate=calculate(cv2.imread('Screen/transport/Test/2-OG003.jpg'),cv2.imread('Screen/transport/OG/2-OG008.jpg'))
print("濾波B",rate)
file1='Screen/transport/OG/OG003.jpg'
file2='Screen/transport/OG/OG008.jpg'
CompareAlgo(file1,file2)
print("濾波A")
'''


# ## 前處理(去紅線，凸顯特徵)

# In[203]:


#圖片,要不要存檔
def ImageProcess(imgpath,switch):
    #灰階讀進
    img=cv2.imread(imgpath,0)
    
    start1 = time.time()
    
    #膨脹
    kernel = np.ones((3,3), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations = 1)
    
    #降噪(中值濾波)，彌平紅線條
    image_mid_blur =cv2.medianBlur(dilate, 9) 
    end1=time.time()
    print("去紅線時間：%f 秒" % (end1 - start1))
    
    start2 = time.time()
    #細線化(凸顯切割線)
    kernelN=np.ones((11,11),np.uint8)
    Negtive=cv2.erode(image_mid_blur,kernelN,iterations=1)

    end2=time.time()
    print("細線化時間：%f 秒" % (end2 - start2))
    
    #影像疊加(增加畫素，變清晰
    #先高斯濾波(影像,捲積核,標準差)，模糊鋸齒狀
    start3 = time.time()
    Gaussian=cv2.GaussianBlur(Negtive,(0,0),50)
    end3=time.time()
    print("高斯時間：%f 秒" % (end3 - start3))
    
    start4 = time.time()
    usm = cv2.addWeighted(Negtive, 1.5, Gaussian, -0.5, 0)
    end4=time.time()
    print("疊圖時間：%f 秒" % (end4 - start4))
    
    
    if switch == 1:
        #取檔案名
        #split處理檔名
        #注意斜線方向
        cv2.imwrite('Screen\\transport\\'+imgpath.split("\\")[-1],usm)


# ## CheckSum-計算單通道的直方圖的相似值

# In[30]:


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

# In[27]:


def CompareThreshold():
    #用兩兩比較出的數值當作門檻
    #0.6881
    rate=calculate(cv2.imread('Screen/compare/c001.jpg'),cv2.imread('Screen/compare/c002.jpg'))
    return rate


# ## 合併比對

# In[184]:


#(原圖,比較圖)
def CompareAlgo(file,file2):
    
    #確認重複檔名，避免比到舊的
    CheckFile(file)
    CheckFile(file2)
    
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
    #相似度再比對門檻值0.6881

    if rate<threshold:
        return False
    if rate>=threshold:
        return True

    
    '''
    start = time.time()
    end=time.time()
    print("執行時間：%f 秒" % (end - start))

    print(threshold)
    print(rate)
    '''


# ## 使用

# In[210]:


'''
#相同(都沒切)
fileNO0='Screen/transport/OG/OG001.jpg'
fileNO1='Screen/transport/OG/OG002.jpg'
#相同(都有切)
file1='Screen/transport/OG/OG003.jpg'
file2='Screen/transport/OG/OG004.jpg'
#不同(有切沒切_直條狀)
file3='Screen/transport/OG/OG005.jpg'
file4='Screen/transport/OG/OG006.jpg'
#不同(有切沒切_橫條狀)
file5='Screen/transport/OG/OG007.jpg'
file6='Screen/transport/OG/OG008.jpg'
print("相同都沒切 ",CompareAlgo(fileNO0,fileNO1))
print("相同都有切 ",CompareAlgo(file1,file2))
print("不同有切沒切_直條狀 ",CompareAlgo(file3,file4))
print("不同有切沒切_橫條狀 ",CompareAlgo(file5,file6))
'''


# In[ ]:




