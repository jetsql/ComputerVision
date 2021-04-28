import numpy as np
import os
import cv2
from .colors import get_color

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None ):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1
        
       
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score      

    def get_xml(self):
        if self.xml == "":
            self.xml = xml
            
        return self.xml      
    
    
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3    

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
   
    for box in boxes:
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)
                
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1e-3 * image.shape[0],100)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32')  

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(box.xmin+13, box.ymin - 13), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1e-3 * image.shape[0], 
                        color=(0,0,0), 
                        thickness=2)
            
    
    return image

def convert_to_PascalVOC(image_path  , width  , height , bbx_label, xmin, xmax, ymin, ymax):
    
    label_str = image_path
    fileName = image_path.split('/')[-1]
    filePath = image_path
    
    path=""
    name=fileName.split('.')[0]
    
    for text in image_path.split('/',-1)[:-2]:
        path= path +text + "/"
        

    
    print("label_str = " +label_str)
    print("fileName = " + fileName)
    print("bbx_label = " + bbx_label)
    
    xml = "<?xml version='1.0'?>"
    xml = xml + "<annotation>"
    xml = xml + "<folder>" + bbx_label + "</folder>"
    xml = xml +   "<filename>" + fileName +"</filename>"
    xml = xml +     "<path>" + filePath +"</path>"
    xml = xml +     "<source><database>Unknown</database></source>"
    xml = xml +     "<size>"
    xml = xml +        "<width>" + str(width) + "</width>"
    xml = xml +        "<height>" + str(height) + "</height>"
    xml = xml +        "<depth>Unspecified</depth>"
    xml = xml +     "</size>"
    xml = xml + "<segmented>Unspecified</segmented>"
            
    
    xmin = xmin
    xmax = xmax
    ymin = ymin
    ymax = ymax
            
    xml = xml + "<object>"
    xml = xml + "<name>" + str(bbx_label) + "</name>"
    xml = xml + "<pose>Unspecified</pose>"
    xml = xml + "<truncated>Unspecified</truncated>"
    xml = xml + "<difficult>0</difficult>"
    xml = xml + "<occluded>Unspecified</occluded>"
    xml = xml + "<bndbox>"
    xml = xml +     "<xmin>" + str(xmin) + "</xmin>"
    xml = xml +     "<ymin>" + str(ymin) + "</ymin>"
    xml = xml +     "<ymax>" + str(ymax) + "</ymax>"
    xml = xml +     "<xmax>" + str(xmax) + "</xmax>"
    xml = xml + "</bndbox>"
    xml = xml + "</object>"
    
    xml = xml + "</annotation>"
    

    import xml.etree.ElementTree as ET
    tree = ET.ElementTree(ET.fromstring(xml))
    
    #./C050QTN01+C087XAN01/test/C050QTN01/
    tree.write(path+"xml/"+name+".xml")

        
    
            
    

            