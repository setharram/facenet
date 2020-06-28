#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:22:37 2020

@author: Seetharam/setharram@gmail.com
"""

# import os
# os.environ['LD_PRELOAD'] = '/usr/lib/arm-linux-gnueabihf/libatomic.so.1'

import cv2 as cv
#import tflite_runtime.interpreter as tflite

import tensorflow.lite as tflite
import numpy as np
import pickle

cap = cv.VideoCapture(0)

#cap.set(3, 640) # set video width
#cap.set(4, 480) # set video height
#cap.set(5, 1) # set FPS to 20


if not cap.isOpened():
    print("Cannot open camera")
    exit()

face_cascade = cv.CascadeClassifier('lib/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('lib/haarcascade_eye.xml')

svm_model = 'lib/fac.pkl'
tf_model = 'lib/mobilenet_81.tflite'

def stad(im):
  mean = np.mean(im)
  #st = np.std(im)
  std_adj = np.std(im)
  #std_adj = np.maximum(st, 1.0/np.sqrt(im.size))
  im = np.multiply(np.subtract(im,mean), 1/std_adj)
  return im

def init_model(tf_path,cls_path):
  interpreter = tflite.Interpreter(model_path=tf_path)
  interpreter.allocate_tensors()
  
  with open(cls_path, 'rb') as infile:
      (cls_model, class_names) = pickle.load(infile)
    
  return interpreter,cls_model,class_names

def classify_image(image,model,clasify,labels):
     # Load TFLite model and allocate tensors.
     
     input_details = model.get_input_details()
     output_details = model.get_output_details()
     # standardize input image
     tf_input = cv.resize(image,(224,224))/255.0
     #image = stad(image)
     tf_input = np.array(tf_input,dtype=np.float32)
     model.set_tensor(input_details[0]['index'], [tf_input])
     model.invoke()

     pred = model.get_tensor(output_details[0]['index'])
     
     cls_out = clasify.predict_proba(pred)
     # calculating probabilities
     # cls_pro = np.exp(cls_out - np.max(cls_out))
     # prob = cls_pro/cls_pro.sum()
     # extract text labels
     predicted_ids = np.argmax(cls_out, axis=-1)

     return labels[predicted_ids[0]],round(cls_out[0][predicted_ids[0]],2)

fac = 0.2
model,cls_model,classname = init_model(tf_model,svm_model)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    image = cv.flip(frame,1)
    # covert into grayscaler for eye detection
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in faces:
        cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

        if len(faces) == 1:
             roi_gray = gray[y:y+h, x:x+w]
             roi_color = image[y:y+h, x:x+w]
             prob = 0
             # crop face for model input
             if ((x-(w*fac))>0) & ((y-(h*fac)>0)):
              md_x = x-int(w*fac)
              md_y = y-int(w*fac)
              md_w = w+int((w*fac)*2)
              md_h = h+int((h*fac)*2)
              
              tf_image = image[md_y:md_y+md_h,md_x:md_x+md_w]
              
              name,prob = classify_image(tf_image, model, \
                              cls_model, classname)
              sprob = str(prob)
              print(name,sprob)
              
              eyes = eye_cascade.detectMultiScale(roi_gray)
              for (ex,ey,ew,eh) in eyes:
                  cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
              if prob > 0.55:
                cv.putText(image, name,\
                     (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv.putText(image, sprob,\
                     (x, y+h), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
             
              image = tf_image
             else:
                  prob = 0.2
                  #display text

             
    # Display the resulting frame
    cv.imshow('face detection', image)
    if cv.waitKey(200) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()