

import numpy as np 
import tensorflow as tf 
import cv2 
from glob import glob as glb
import re

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_bool('debug', False,
                            """ this will show the images while generating records. """)

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# create tf writer
record_filename = '../data/tfrecords/train.tfrecords'

writer = tf.python_io.TFRecordWriter(record_filename)

# the stored frames
shape = (420, 580)
frames = np.zeros((shape[0], shape[1], 1))

# list of files
train_filename = glb('../data/train/*') 
mask_filename = [s for s in train_filename if "mask" in s]
image_filename = [s for s in train_filename if "mask" not in s]

pair_filename = []

for image in image_filename:
  key = image[:-4] 
  mask = [s for s in mask_filename if key in s][0]
  pair_filename.append((image, mask))

for pair in pair_filename:
  # read in images
  image = cv2.imread(pair[0], 0) 
  mask = cv2.imread(pair[1], 0) 
  
  # Display the resulting frame
  if FLAGS.debug == True:
    cv2.imshow('image', image) 
    cv2.waitKey(0)
    cv2.imshow('image', mask) 
    cv2.waitKey(0)
   
  # process frame for saving
  image = np.uint8(image)
  mask = np.uint8(mask)
  image = image.reshape([1,shape[0]*shape[1]])
  mask = mask.reshape([1,shape[0]*shape[1]])
  image = image.tostring()
  mask = mask.tostring()
  
  # create example and write it
  example = tf.train.Example(features=tf.train.Features(feature={
    'image': _bytes_feature(image),
    'mask': _bytes_feature(mask)})) 
  writer.write(example.SerializeToString()) 


