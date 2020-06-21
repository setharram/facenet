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
face_cascade = cv.CascadeClassifier('lib/haarcascade_frontalface_default.xml')

tf_model = 'lib/20180402-114759.pb'
clasfier = 'lib/svm_classifier.pkl'

def classify_image(image):

     #im = cv.imread(image)
     im = imutils.resize(image,160,160)
     image = np.zeros((1, 160, 160, 3))
     image[0,:,:,:] = im
     
     feed_dict = { images_placeholder:image, phase_train_placeholder:False }
     arr = sess.run(embeddings,feed_dict=feed_dict)
     
     with open(clasfier, 'rb') as infile:
          (model, class_names) = pickle.load(infile)
    
     pred = model.predict(arr)
     print(pred)
     # # calculating probabilities
     # tf_exp = np.exp(tflite_results - np.max(tflite_results))
     # prob = tf_exp/tf_exp.sum()
     # # extract text labels
     # predicted_ids = np.argmax(tflite_results, axis=-1)
     # # print(prob[0][predicted_ids[0]])
     # # print(labels[predicted_ids[0]])
     # return labels[predicted_ids[0]],prob[0][predicted_ids[0]]
     return 0 , 0

# Load TFLite model and allocate tensors.
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

          if (len(faces) == 1) :     
               (x,y,w,h) = faces[0,:]
               fac = 0.1
               #print(x,y,w,h)
               if ((x-(w*fac))>0) & ((y-(h*fac)>0)):
                    md_x = x-int(w*fac)
                    md_y = y-int(h*fac)
                    md_w = w+int((w*fac)*2)
                    md_h = h+int((h*fac)*2)

                    #print(md_x,md_y,md_w,md_h)
                    cr_image = image[md_y:md_y+md_h,md_x:md_x+md_w]
                    cv.rectangle(image,(md_x,md_y),(md_x+md_w,md_y+md_h),(0,255,0),2)
                    x,y,_ = cr_image.shape
                    # print(x,y)
                    # print(cr_image.shape)
                    name, prob = classify_image(cr_image)
                              
                    # prob = str(round(prob,2))
                    # print(name,prob)
                    image = cr_image
                    
               # print infomation on the image
               cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
               # cv.putText(image, name,\
               #         (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
               # cv.putText(image, prob,\
               #         (x, y+h), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
               
          image = cv.resize(image,(512,512))
          cv.imshow("Image", image)
          
          if cv.waitKey(0) == ord('q'):
               break

cv.destroyAllWindows()


