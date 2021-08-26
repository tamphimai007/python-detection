#!/usr/bin/env python
# coding: utf-8

# # สิ่งที่ใช้ในคลิปนี้

# #1  https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

# #2 https://gist.github.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7

# #3 https://github.com/pjreddie/darknet/blob/master/data/coco.names

# # Library ที่จำเป็นต้องติดตั้งนะครับ

# #1 pip install opencv-python
# 
# #2 pip install matplotlib

# # ถ้าพร้อมแล้วมาลุยกันเลย

# In[1]:


#import Module
import  cv2
import matplotlib.pyplot as plt


# In[2]:


#import model ที่เทรนนิ่งเรียบร้อยแล้ว
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file)


# In[3]:


#สร้างตัวแปรว่างๆ ไว้รับค่าจาก Labels
classLabels = []
file_labels = 'Labels.txt'


# In[4]:


#เปิดไฟล์แล้วแต่ง String สักหน่อยให้อยู่ในรูปแบบ List 
with open(file_labels,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
print(classLabels)


# In[14]:


#กำหนดขนาดของ Input --- ศึกษาเพิ่มเติม https://docs.opencv.org/4.5.2/d3/df0/classcv_1_1dnn_1_1Model.html
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


# In[15]:


#เปิดภาพดูสักหน่อยครับ
img = cv2.imread('air3.JPG')
plt.imshow(img)


# In[16]:


#เปิดภาพใส่สีให้ปกติ
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[17]:


# สร้างตัวแปรมารับค่า จาก model ที่เทรน เรียบร้อยแล้ว #confThreshold คือการปรับค่าสี-- ศึกษาเพิ่มเติมได้จาก https://phyblas.hinaboshi.com/oshi10
ClassIndex, confidece , bbox  = model.detect(img,confThreshold=0.5)


# In[18]:


# สร้างกรอบรอบวัตถุ กับ ใส่ตัวหนังสือ สีแบบ BGR--- ศึกษาเพิ่มเติมได้จาก https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
colorBox = (255,0,0)
colorFont = (0,255,0)
for ClassInd, conf , boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
    cv2.rectangle(img,boxes,colorBox)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10,boxes[1]+40),font,font_scale,colorFont,thickness= 3)


# In[19]:


# เสร็จแล้วครับ ในส่วนของรูปภาพ
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# # มาลองเป็น วิดีโอกันบ้างนะครับ ขั้นตอนก็เหมือนกันเพิ่มมานิดหน่อย

# In[49]:


# ผมลากยาวเลยนะครับ 
#rtsp://freja.hiof.no:1935/rtplive/definst/hessdalen03.stream
#rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen02.stream
#4K camera example for Traffic Monitoring (Road).mp4
# 'VIVOTEK FD8372 5MP IP Camera Thailand with street 4 lanes all clear license plate from TSOLUTIONS.mkv'
video = 'test.mp4'
cap = cv2.VideoCapture(video)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot Open video')
    
                 
while True:
    ret , frame = cap.read()
    ClassIndex, confidece , bbox  = model.detect(frame,confThreshold=0.5)
    
    if(len(ClassIndex) != 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassInd <= 80):
                cv2.rectangle(frame,boxes,colorBox)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10,boxes[1]+40),font,font_scale,colorFont,thickness= 3)
    cv2.imshow('Hello Demo Video',frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
                


# In[ ]:




