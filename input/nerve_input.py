
import os
import numpy as np
import tensorflow as tf
from glob import glob as glb


FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_integer('min_queue_examples', 1000,
                           """ min examples to queue up""")

def read_data(filename_queue, shape):
  """ reads data from tfrecord files.

  Args: 
    filename_queue: A que of strings with filenames 
    shape: image shape 

  Returns:
    frames: the frame data in size (batch_size, image height, image width, frames)
  """
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image':tf.FixedLenFeature([],tf.string),
      'mask':tf.FixedLenFeature([],tf.string)
    }) 
  image = tf.decode_raw(features['image'], tf.uint8)
  mask = tf.decode_raw(features['mask'], tf.uint8)
  image = tf.reshape(image, [shape[0], shape[1], 1])
  mask = tf.reshape(mask, [shape[0], shape[1], 1])
  image = tf.to_float(image)
  mask = tf.to_float(mask) 
  image = image / 255.0
  mask = mask / 255.0
  return image, mask

def _generate_image_label_batch(image, mask, batch_size, shuffle=True):
  """Construct a queued batch of images.
  Args:
    image: 3-D Tensor of [height, width, frame_num] 
    mask: 3-D Tensor of [height, width, frame_num] 
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 1] size.
  """

  num_preprocess_threads = 1
  if shuffle:
    #Create a queue that shuffles the examples, and then
    #read 'batch_size' images + labels from the example queue.
    images, masks = tf.train.shuffle_batch(
      [image, mask],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_examples + 3 * batch_size,
      min_after_dequeue=FLAGS.min_queue_examples)
  else:
     images, masks = tf.train.batch(
      [image, mask],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_examples + 3 * batch_size)
  return images, masks

def nerve_inputs(batch_size):
  """ Construct nerve input net.
  Args:
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 84x84x4].
    mask: Images. 4D tensor. Possible of size [batch_size, 84x84x4].
  """

  shape = (420,580)

  tfrecord_filename = glb('../data/tfrecords/*') 
  print(tfrecord_filename)
  
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  image, mask = read_data(filename_queue, shape)

  images, masks = _generate_image_label_batch(image, mask, batch_size)
 
  # display in tf summary page 
  tf.image_summary('images', images)
  tf.image_summary('mask', masks)

  return images, masks 

