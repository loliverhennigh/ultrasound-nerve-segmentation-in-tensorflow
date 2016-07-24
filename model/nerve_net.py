
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
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/run_0001',
                           """ checkpoint file to save to """)
tf.app.flags.DEFINE_string('model', 'ced',
                           """ model name to train """)
tf.app.flags.DEFINE_string('output_type', 'mask_image',
                           """ What kind of output, possibly image. Maybe other in future """)
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """The decay to use for the moving average""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """momentum of learning rate""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)
tf.app.flags.DEFINE_float('dropout_hidden', 0.5,
                          """ dropout on hidden """)
tf.app.flags.DEFINE_float('dropout_input', 0.8,
                          """ dropout on input """)
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
    conv_peice = nerve_architecture.conv_ced(inputs, keep_prob)
    prediction = nerve_architecture.trans_conv_ced(conv_peice)

  return prediction 

def loss_image(prediction, mask):
  """Calc loss for predition on image of mask.
  Args.
    inputs: prediction image 
    mask: true image 

  Return:
    error: loss value
  """
  error = tf.nn.l2_loss(prediction - mask)
  tf.scalar_summary('error', error)
  error.set_shape([])
  tf.add_to_collection('losses', error)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, lr):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
   return train_op

