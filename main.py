'''The Sparks Foundation Internship Program

Internship on Computer vision and Internet of Things 

Task 1: Object Detection

Objective: To implement an Object Detector which identifies classes of object in an Image

Author: Tamil Mani P'''


#import required dependencies
import cv2

#threshold level for detection
thres = 0.50 

#importing names of objects to be detected
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#Setting up the Model
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

image_labels=['2','4','5']

for i in range(3):

    #reading the test images
    img=cv2.imread(image_labels[i]+'.jpg')
    
    #detecting the objects in images using Model
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,0,255),thickness=2)

            #displaying the name of detected object
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

            #displaying the accuracy of detection
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    
    #displaying the image
    cv2.imshow('Output',img)
    cv2.waitKey(0)
