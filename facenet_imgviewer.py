#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:48:00 2020

@author: Seetharam/setharram@gmail.com
"""

import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import facenet
import imutils
import pickle

datst_pt = '/home/ram/Documents/opencv/face-recognition-opencv/dataset'
#datst_pt = '/home/ram/Documents/opencv/facenet/dataset_clean/dataset_clean'
face_cascade = cv.CascadeClassifier('lib/haarcascade_frontalface_default.xml')

tf_model = 'lib/20180402-114759.pb'
#clasfier = 'lib/svm_classifier.pkl'
clasfier = '/home/ram/Documents/opencv/facenet/jun.pkl'

arr = []
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

# Load TFLite model and allocate tensors.
with open(clasfier, 'rb') as infile:
     (model, class_names) = pickle.load(infile)

with tf.compat.v1.Graph().as_default():
 with tf.compat.v1.Session() as sess:
   facenet.load_model(tf_model)
   embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
   images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
   phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
   embedding_size = embeddings.get_shape()[1]     
   
   for fil in os.listdir(datst_pt):
     for imag in os.listdir(datst_pt+'/'+fil):
          
          # load the input image and convert it from BGR to RGB
          image = cv.imread(datst_pt+'/'+fil+'/'+imag)
          gray = cv.cvtColor(image, cv.COLOR_BGR2RGB)
          
          faces = face_cascade.detectMultiScale(image, 1.1, 4)
          
          # name, prob = classify_image(image,sess,model)
          # print(class_names[name])
          # print(round(prob,2))

          if (len(faces) == 1) :     
               (x,y,w,h) = faces[0,:]
               fac = 0.1
               print(x,y,w,h)
               if ((x-(w*fac))>0) & ((y-(h*fac)>0)):
                    md_x = x-int(w*fac)
                    md_y = y-int(h*fac)
                    md_w = w+int((w*fac)*2)
                    md_h = h+int((h*fac)*2)

                    print(md_x,md_y,md_w,md_h)
                    cr_image = image[md_y:md_y+md_h,md_x:md_x+md_w]
                    cv.rectangle(image,(md_x,md_y),(md_x+md_w,md_y+md_h),(0,255,0),2)
                    x,y,_ = cr_image.shape
                    print(x,y)
                    # print(cr_image.shape)
                    name, prob = classify_image(image,sess,model)
                    name = class_names[name]
                    prob = str(round(prob,2))

                    # print(name,prob)
                    # image = cr_image
                    
                    # print infomation on the image
                    cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                    cv.putText(image, name,\
                             (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    cv.putText(image, prob,\
                             (x, y+h), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
               
          image = cv.resize(image,(512,512))
          cv.imshow("Image", image)
          
          if cv.waitKey(0) == ord('q'):
               break

cv.destroyAllWindows()


