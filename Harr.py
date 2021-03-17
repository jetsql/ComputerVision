#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image,ImageDraw
import PIL
import shutil,os,sys
import glob
from time import sleep


# ## 特徵分類(function)

# In[2]:


#建立訓練資料
#建立資料夾(刪掉重建)
def emptydir(dirname):
    if os.path.isdir(dirname):
        #資料夾存在就刪除
        shutil.rmtree(dirname)
        #延遲處理
        sleep(2)
    #建立資料夾
    os.mkdir(dirname)


# ## 單一圖片改大小(function)

# In[15]:


#單改一張
def ChangeSize(filename):
        img=Image.open(filename)
        #尺寸轉換200x200
        img_new=img.resize((500,500),PIL.Image.ANTIALIAS)
        img_new.save('Screen/transport/'+filename.split("/")[-1] )


# ## 批次圖片改大小(function)

# In[4]:


#轉換尺寸
def dirResize(src,dst):
    #讀取資料夾中所有jpg
    #myfiles=get_all
    myfiles=glob.glob(src+'/*.JPG')
    emptydir(dst)
    print(src+'資料夾')
    print('Start')
    #逐一將檔案轉換尺吋
    for i,f in enumerate(myfiles):
        img=Image.open(f)
        #尺寸轉換200x200
        img_new=img.resize((500,500),PIL.Image.ANTIALIAS)
        #批次從000~100(0>3d)的檔名.jpg
        outname=str("scan")+str('{:0>3d}').format(i+1)+'.jpg'
        img_new.save(dst+'/'+outname)
    print('End \n')


# ## 統計檔案數量(function)

# In[7]:


#統計檔案數量
def CountPostive(filename):
    #取路徑
    path = os.getcwd() 
    num_files=0
    for fn in os.listdir(filename):
            num_files += 1
    return num_files


# ## 轉檔按PNG轉JPG處理(非必要function)

# In[8]:


#轉檔按PNG轉JPG處理(非必要)
#讀取目錄
def TransportFile(path,newpath):
    #目前目錄
    print(os.getcwd())
    print("Start   "+path)
    for filename in os.listdir(path):
        #取檔名
        if os.path.splitext(filename)[1]=='.png':
            print("原檔"+os.path.splitext(filename)[0]+os.path.splitext(filename)[1])
            img=cv2.imread(path+filename)
            print("改為"+filename.replace(".png",".jpg"))
            newfilename=filename.replace(".png",".jpg")
            #另存另一個資料夾
            #imwrite(檔名,圖檔)
            cv2.imwrite(newpath+newfilename,img)
    print("End")


# ## 截圖(全螢幕)Function

# In[9]:


#螢幕截圖function
#/////////測試截圖存檔///////////
#(路徑,檔案數,控制要不要執行)
def Scann_All(path,filename,switch):
    if(switch==True):
        #擷取全螢幕
        image=ImageGrab.grab()
        #存檔
        #path='Screen\\'
        outname=str("scan")+filename+'.jpg'
        print('截圖存檔'+outname)
        image.save(path+outname)
    else:
         None


# ## 截圖(特定區域)Function

# In[11]:


#框區域截圖function
#(存檔路徑,存檔檔名,要不要存檔TF,左上x1,y1,右下x2,y2)
def Scann_Aera(path,filename,switch,x1,y1,x2,y2):
    if(switch==True):
        #0,0,500,500
        image=ImageGrab.grab(bbox=(x1,y1,x2,y2))

        #存檔
        #path='Screen\\'
        outname=str("scan")+filename+'.jpg'
        print('截圖存檔'+outname)
        image.save(path+outname)
    else:
         None


# In[5]:


#分類器resize
#resize正樣本到200*200大小
#(讀進、另外寫入)
'''
#正向樣本
dirResize('postive','carpost')
#負向樣本
dirResize('negtive','carnegt')
'''


# In[6]:


'''
dirResize('cutImage','Haar-Training-Plate-cut/training/positive/rawdata')
'''


# ## JPG轉BPM

# In[12]:


#jpg轉換成bmp格式
#從carplate中將原始檔案jpg轉bmp後刪除
#filename="Haar-Training-Plate-cut/training/positive/rawdata"
def JPG2BPM(filename):
    myfiles=glob.glob(filename+"/*.jpg")
    print('Start bmp')
    #處理前目錄檔案
    print(os.listdir(filename))
    for f in myfiles:
        namespilt=f.split("\\")
        img=Image.open(f)
        #置換檔名
        outname=namespilt[1].replace('rrjpg','bmpraw')
        #置換副檔名
        outname=outname.replace('.jpg','.bmp')
        #存為bmp
        img.save(filename+'/'+outname,'bmp')
        #刪除原始檔案
        os.remove(f)
    print('end')
    #處理後目錄檔案
    print(os.listdir(filename))
    print(os.walk(filename))
    print(os.walk)


# ## 標記負樣本

# In[13]:


#標記負樣本
def LabelNegtive():
    #open()，w寫入，r讀取，x新建立
    #寫入標記txt
    fp=open('Haar-Training-Plate-cut/training/negative/bg.txt','w')
    #讀出根目錄檔案
    files=glob.glob("Haar-Training-Plate-cut/training/negative/*.jpg")
    print('寫入bg.txt')
    text=""
    for file in files:
        #取檔名
        basename=os.path.basename(file)
        filename="negative/"+basename
        text+=filename+"\n"

    print(text)
    fp.write(text)
    fp.close()
    print('完成bg.txt')


# ### 標記圖案

# In[14]:


#顯示前面手動標區的標框
#open()，r讀取
def LabelImage():
    path='C:/Users/2102048/pythonCV/Haar-Training-Plate-cut2/training/picMark'
    fplabel=open('Haar-Training-Plate-cut2/training/positive/info.txt','r')
    lines=fplabel.readlines()
    emptydir('picMark')
    print('Start繪製方框')
    for line in lines:
        #一個array讀取所有筆數
        data=line.split(' ')
        #C:\Users\2102048\pythonCV\Haar-Training-Plate-cut\training\positive
        img=Image.open('Haar-Training-Plate-cut2/training/positive/'+data[0])
        draw=ImageDraw.Draw(img)
        #方框數量
        n=data[1]
        #繪製方框
        for i in range(int(n)):
            x=int(data[2+i*4])
            y=int(data[3+i*4])
            w=int(data[4+i*4])
            h=int(data[5+i*4])
            draw.rectangle((x,y,x+w,y+h),outline='red')
        filename=(data[0].split('/'))[-1]
        #存檔路徑
        img.save('Haar-Training-Plate-cut2/training/picMark/'+filename)
    fplabel.close()
    print('End繪製結束')
    CountPostive(path)


# #### 修改samples_creation.bat(執行並產生向量檔)
# #### 修改 haarTraining.bat(執行並開始訓練)

# In[ ]:


#打包分類器(產生facevector.vec向量檔)
#info 正樣本標記檔路徑
#vec向量檔路徑
#num正向樣本圖片數量
#w 偵測物件寬度
#h偵測物件高度

#在資料夾下新建samples_creation.bat，批處理檔案
#opencv_createsamples.exe -info positive/info.txt -vec vector/facevector.vec -num 497 -w 76 -h 20


# In[ ]:


#訓練(產生訓練檔案)
#data 儲存訓練完的結果路徑
#vec 正向樣本向量檔
#bg 負向樣本檔路徑
#numPos 正向樣本數量
#numNeg 負向樣本數量
#numStage 訓練級數(等級15~25)
#minHitRate 每一個訓練及數所需要命中的次數
#precalcValBufSize 配置訓練記憶體大小
#precalcIdxBufSize 訓練使用記憶體大小
#model 訓練的模型 ALL(所有) BASIC(線性) CORE(線性+中心)
#w 偵測物件的寬
#h 偵測物件的高
# opencv_traincascade.exe -data cascades -vec vector/facevector.vec -bg negative/bg.txt -numPos 497 -numNeg 293 -numStages 15 -w 76 -h 20 -minHitRate 0.9999 -precalcValBufSize 3000 -precalcIdxBufSize 3000 -mode ALL

