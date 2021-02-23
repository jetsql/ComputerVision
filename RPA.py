store_path_base = "C:\Users\2102048\pythonCV\AI_Project"
key=0

while 1:
    start=datetime.now()
    key=key+1
    print(key)
    if exists ("1614051150400.png",0.5)or exists("1614065756317.png",0.5):
        try:
            x1=find("1614051150400.png")or find("1614065756317.png")
            y1=capture(x1)
            new_path = store_path_base +"Type1_"+time.strftime('%Y-%m-%d_%H-%M-%S') +".png"
            shutil.move(y1,os.path.join(new_path)) #move img to destination
            print("get type1")
        except:
            print("type1 Error!!")
    if exists ("1614051186910.png",0.5)or exists("1614065807023.png",0.5):
        try:
            x2=find("1614051186910.png")or fine("1614065807023.png")
            y2=capture(x2)
            new_path = store_path_base +"Type2_"+time.strftime('%Y-%m-%d_%H-%M-%S') +".png"
            shutil.move(y2,os.path.join(new_path)) #move img to destination   
            print("get_type2")
        except:
            print("type2 Error!")
    if exists ("1614051239543.png",0.5)or exists("1614065237011.png",0.5)or exists("1614065996994.png",0.5):
        try:
            x3=find("1614051239543.png")or find("1614065237011.png")or find("1614065996994.png")
            y3=capture(x3)
            new_path = store_path_base +"Type3_"+time.strftime('%Y-%m-%d_%H-%M-%S') +".png"
            shutil.move(y3,os.path.join(new_path)) #move img to destination   
            print("get_type3")
        except:
            print("type3 Error!")
    if exists ("1614051283199.png",0.5) or exists("1614066056104.png",0.5):
        try:
            x4=find("1614051283199.png")or find("1614066056104.png")
            y4=capture(x4)
            new_path = store_path_base +"Type4_"+time.strftime('%Y-%m-%d_%H-%M-%S') +".png"
            shutil.move(y4,os.path.join(new_path)) #move img to destination   
            print("get_type4")
        except:
            print("type4 Error!")
    end=datetime.now()
    gap=(end-start)
    print("detect_time:",gap.seconds)
    print("========")
