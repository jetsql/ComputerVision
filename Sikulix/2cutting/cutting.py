import datetime,time
from datetime import datetime,timedelta
import shutil
import os.path

store_path_one="C:/Users/2102048/pythonCV/Screen/one/"
store_path_two="C:/Users/2102048/pythonCV/Screen/two/"

key=0
_one=0
_two=0
#辨識區域
area=Region(363,213,1212,422)
while 1:
    one_name="One_"+str('{:0>3d}').format(_one)+".jpg"
    two_name="Two_"+str('{:0>3d}').format(_two)+".jpg"

    #開始計時
    start=datetime.now()
    key=key+1
    print('Key',key)
    if (exists("1617779495031.png",0.3)and exists("1617779566373.png",0.1) and exists("1617779588907.png",0.1)) or (exists("1617779644568.png",0.3) and exists("1617779658962.png",0.1) and exists("1617779675147.png",0.1)):
        try:
            x1=find("1617779509616.png") or find("1617779687301.png") or find("1617779839019.png") 
            #截圖抓取範圍
            y1=capture(x1)
            #產生檔名
            _one=_one+1
            new_path=store_path_one+one_name
            #移動目錄資料夾
            shutil.move(y1,os.path.join(new_path))
            print("get ONE")
        except:
           print("ONE Error!!!")

    if (exists ("1617779715732.png",0.3)and exists("1617779743308.png",0.1)and exists("1617777547927.png",0.1)) or (exists("1617779784372.png",0.3) and exists("1617779806033.png",0.1)and exists("1617779818154.png",0.1)):
        try:
            x2=find("1617779723683.png") or find("1617779793941.png") or find("1617779859211.png")
            #抓取範圍
            y2=capture(x2)
            #產生檔名
            _two=_two+1
            new_path=store_path_two+two_name
            #移動目錄資料夾
            shutil.move(y2,os.path.join(new_path))
            print("get TWO")
        except:
            print("TWO error!!!")
    else:
        print("NoneImage")
    end=datetime.now()
    gap=(end-start)
    print("detect_time:",gap.seconds)
    print("===========================")
                    
'''
#截圖位置
Location(967, 424)
#辨識區域
Region(363,213,1212,422)
Offset(1212, 423)
'''