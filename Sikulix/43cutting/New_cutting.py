import datetime,time
from datetime import datetime,timedelta
import shutil
import os.path
#1714AS04，髒版測試
store_path_one="C:/Users/2102048/pythonCV/Screen/one/"
store_path_two="C:/Users/2102048/pythonCV/Screen/two/"

key=0
_one=0
_two=0
Location(960, 532)
while 1:
    one_name="One_"+str('{:0>3d}').format(_one)+".jpg"
    two_name="Two_"+str('{:0>3d}').format(_two)+".jpg"

    #開始計時
    start=datetime.now()
    key=key+1
    print('Key',key)
    #第一次出現切割
    try:
        wait("1621566051168.png",1)
        control=1
        while control:

            x1=Region(361,105,1210,568)
            #截圖抓取範圍
            y1=capture(x1)
            #產生檔名
            _one=_one+1
            new_path=store_path_one+one_name
            #移動目錄資料夾
            shutil.move(y1,os.path.join(new_path))
            print("get ONE")
            if waitVanish("1621564331449.png"):
                control=0
    except:
       print("ONE Error!!!")
    #第二此出現切割
    if (exists ("1621564428366.png",0.5)and exists(Pattern("1621564437449.png").similar(0.90),0.5)and exists(Pattern("1621564461422.png").similar(0.85),1)) or (exists("1621564571782.png",0.5) and exists(Pattern("1621564579243.png").similar(0.89),0.5)and exists(Pattern("1621564597437.png").similar(0.81),1)):
            try:
                x2=Region(361,105,1210,568)
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

Offset(1212, 423)

    ###///測試區///###
    
    #對圖片向下找220為圈選範圍，找尋圖片
    SearchRegion=find().below(220).find() or find() or find() 
    save_s=capture(SearchRegion) 
    #在範圍內尋找圖片
    
    X=.find()
    y=capture(x)
    
    #等待指定圖片出現
    if wait():
        print("start ok")

    #找到指定圖片就停止
    while not exists()or exists():
        type(Key.PAGE_DOWN)
'''
    ###///測試區///###