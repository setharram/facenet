#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:50:45 2020

@author: Seetharam/setharram@gmail.com
"""

import cv2 as cv
import tensorflow as tf
import numpy as np
import facenet
import pickle

cap = cv.VideoCapture(0)

cap.set(3, 640) # set video width
cap.set(4, 480) # set video height
cap.set(5, 1) # set FPS to 20


if not cap.isOpened():
    print("Cannot open camera")
    exit()

face_cascade = cv.CascadeClassifier('lib/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('lib/haarcascade_eye.xml')

tf_model = 'lib/20180402-114759.pb'
clasfier = '/home/ram/Documents/opencv/facenet/jun.pkl'

def classify_image(image,sess,model):
  im = cv.resize(image,(160,160))
  mean = np.mean(im)
  st = np.std(im)
  std_adj = np.maximum(st, 1.0/np.sqrt(im.size))
  
  im = np.multiply(np.subtract(im,mean), 1/std_adj)
  img = np.zeros((1, 160, 160, 3))
  img[0,:,:,:] = im
  
  feed_dict = { images_placeholder:img, phase_train_placeholder:False }
  arr = sess.run(embeddings,feed_dict=feed_dict)
  
  pred = model.predict(arr)
  pred_pro = model.predict_proba(arr)
  #return class and confidence
  return pred[0], pred_pro[0][pred[0]]

# Load SVM Classifier.
with open(clasfier, 'rb') as infile:
     (model, class_names) = pickle.load(infile)

with tf.compat.v1.Graph().as_default():
 with tf.compat.v1.Session() as sess:
   facenet.load_model(tf_model)
   embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
   images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
   phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
   embedding_size = embeddings.get_shape()[1]
   
   while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    image = cv.flip(frame, 1)
    # covert into grayscaler for eye detection
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x,y,w,h) in faces:
        cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        
        if len(faces) == 1:
             (x,y,w,h) = faces[0,:]
             roi_gray = gray[y:y+h, x:x+w]
             roi_color = image[y:y+h, x:x+w]
             
             fac = 0.1
             # crop face for model input
             if ((x-(w*fac))>0) & ((y-(h*fac)>0)):
               md_x = x-int(w*fac)
               md_y = y-int(h*fac)
               md_w = w+int((w*fac)*2)
               md_h = h+int((h*fac)*2) 
               
               cr_image = image[md_y:md_y+md_h,md_x:md_x+md_w]
               # classify dataset
               name, prob = classify_image(image,sess,model)
               name = class_names[name]
               prob = round(prob,2)
               sprob = str(prob)
               print(name,sprob)

             eyes = eye_cascade.detectMultiScale(roi_gray)
             for (ex,ey,ew,eh) in eyes:
                  cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
             if prob > 0.3:
                cv.putText(image, name,\
                     (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv.putText(image, sprob,\
                     (x, y+h), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # Display the resulting frame
    cv.imshow('face detection', image)
    if cv.waitKey(200) == ord('q'):
        break
# When everything done, release the capture
cap.release()   
cv.destroyAllWindows()