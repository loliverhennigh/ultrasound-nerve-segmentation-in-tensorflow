
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import nerve_architecture
import input.nerve_input as nerve_input

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_string('model', 'ced',
                           """ model name to train """)
tf.app.flags.DEFINE_string('output_type', 'mask_image',
                           """ What kind of output, possibly image. Maybe other in future """)
tf.app.flags.DEFINE_integer('nr_res_blocks', 1,
                           """ nr res blocks """)
tf.app.flags.DEFINE_bool('gated_res', True,
                           """ gated resnet or not """)
tf.app.flags.DEFINE_string('nonlinearity', 'concat_elu',
                           """ nonlinearity used such as concat_elu, elu, concat_relu, relu """)

def inputs(batch_size):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  x, mask = nerve_input.nerve_inputs(batch_size)
  return x, mask

def inference(inputs, keep_prob):
  """Builds network.
  Args:
    inputs: input to network 
    keep_prob: dropout layer
  """
  if FLAGS.model == "ced": 
    prediction = nerve_architecture.conv_ced(inputs, nr_res_blocks=FLAGS.nr_res_blocks, keep_prob=keep_prob, nonlinearity_name=FLAGS.nonlinearity, gated=FLAGS.gated_res)

  return prediction 

def loss_image(prediction, mask):
  """Calc loss for predition on image of mask.
  Args.
    inputs: prediction image 
    mask: true image 

  Return:
    error: loss value
  """
  print(prediction.get_shape())
  print(mask.get_shape())
  #mask = tf.flatten(mask)
  #prediction = tf.flatten(prediction)
  intersection = tf.reduce_sum(prediction * mask)
  loss = -(2. * intersection + 1.) / (tf.reduce_sum(mask) + tf.reduce_sum(prediction) + 1.)
  tf.scalar_summary('loss', loss)
  return loss

def train(total_loss, lr):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
   return train_op

