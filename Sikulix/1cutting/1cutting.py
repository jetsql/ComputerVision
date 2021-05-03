import datetime,time
from  datetime import datetime, timedelta
import shutil
import os.path


#"U:/L3C_AI_Project/X-Ray_cut_image/"
def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        print("已刪除重建",dirname,"  資料夾")
        sleep(2)
    #建立資料夾
    os.mkdir(dirname)

#測試資料夾
store_path_base="C:/Users/2102048/pythonCV/Screen/transport/Test/"

#Type1
store_path_type1="C:/Users/2102048/pythonCV/Screen/transport/Type1/"
#Type2
store_path_type2="C:/Users/2102048/pythonCV/Screen/transport/Type2/"
#Type3
store_path_type3="C:/Users/2102048/pythonCV/Screen/transport/Type3/"
#Type4
store_path_type4="C:/Users/2102048/pythonCV/Screen/transport/Type4/"

#不同Type放不用資料夾，在讀進判斷篩選出只有一張
#global varrible
key=0
_t1=0
_t2=0
_t3=0
_t4=0
#Type1、3辨識效果不太好
while 1:
    Type1_name=str("Type1_")+str('{:0>3d}').format(_t1)+".jpg"
    Type2_name=str("Type2_")+str('{:0>3d}').format(_t2)+".jpg"
    Type3_name=str("Type3_")+str('{:0>3d}').format(_t3)+".jpg"
    Type4_name=str("Type4_")+str('{:0>3d}').format(_t4)+".jpg"
    #開始時間
    start=datetime.now()
    key=key+1
    print('key',key)
    #如果有存在(exists)圖片就執行
    if  exists ("1617075661849.png",0.3)or exists("1617075729984.png",0.3) or exists ("1617075802855.png",0.3):
        try:
            x1=find("1617075661849.png")or find("1617075729984.png") or find ("1617075802855.png") or find("1616729306368.png") or find("1616729456920.png") or find("1616996612905.png")
           #在script內抓圖，抓取reg(事先定義好的Region物件)內的圖像並存檔
            y1=capture(x1)
            #產生存檔檔名
            _t1=_t1+1
            new_path = store_path_type1 +Type1_name
            #移動至目錄(圖,路徑)
            #os.path.join() 將多個路徑組合
            shutil.move(y1,os.path.join(new_path)) #move img to destination
            print("get type1")
        except:
            print("type1 Error!!")
    if exists ("1616729344286.png",0.3)or exists("1616729469994.png",0.3) or exists("1616996579668.png",0.3):
        try:
            x2=find("1616729344286.png")or fine("1616729469994.png") or find("1616996579668.png")
            y2=capture(x2)
            _t2=_t2+1
            new_path = store_path_type2 +Type2_name
            shutil.move(y2,os.path.join(new_path)) #move img to destination   
            print("get_type2")
        except:
            print("type2 Error!")
    if exists ("1617075687049.png",0.3)or exists("1617075741359.png",0.3)or exists("1617075811837.png",0.3):
        try:
            x3=find("1617075687049.png")or find("1617075741359.png")or find("1617075811837.png")
            y3=capture(x3)
            _t3=_t3+1
            new_path = store_path_type3 +Type3_name
            shutil.move(y3,os.path.join(new_path)) #move img to destination   
            print("get_type3")
        except:
            print("type3 Error!")
    if exists ("1616729412049.png",0.3) or exists("1616729503016.png",0.3) or exists("1616996658209.png",0.3):
        try:
            x4=find("1616729412049.png")or find("1616729503016.png") or find( "1616996658209.png")
            y4=capture(x4)
            _t4=_t4+1
            new_path = store_path_type4 +Type4_name
            shutil.move(y4,os.path.join(new_path)) #move img to destination   
            print("get_type4")
        except:
            print("type4 Error!")
    ##Showin測試

    
    ###
    else:
        print("NoneImage")
    ####
    end=datetime.now()
    gap=(end-start)
    print("detect_time:",gap.seconds)
    print("========")
'''
type1= "1616729306368.png" or "1616729456920.png" or "1616996612905.png"
type2= "1616729344286.png" or "1616729469994.png" or "1616996579668.png"
type3= "1616729396682.png" or "1616729493311.png" or "1616996649154.png"
type4= "1616729412049.png" or "1616729503016.png" or "1616996658209.png"
'''
'''type1、type3改良
type1= "1617075661849.png" or  "1617075729984.png" or "1617075802855.png"
type3= "1617075687049.png" or  "1617075741359.png" or "1617075811837.png"

'''




