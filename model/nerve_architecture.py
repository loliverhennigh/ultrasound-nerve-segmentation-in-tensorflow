
"""functions used to construct different architectures  
"""


import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  #if wd:
  if False:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var

def _conv_layer(inputs, kernel_size, stride, num_features, pad_size, idx):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    print(inputs.get_shape())
    pad_mat = np.array([[0,0],[0,pad_size[0]],[0,pad_size[1]],[0,0]])
    input_pad = tf.pad(inputs, pad_mat)

    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,input_channels,num_features],stddev=0.001, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.001))

    conv = tf.nn.conv2d(input_pad, weights, strides=[1, stride, stride, 1], padding='VALID')
    conv_biased = tf.nn.bias_add(conv, biases)
    #elu
    conv_rect = tf.nn.elu(conv_biased,name='{0}_conv'.format(idx))
    return conv_rect

def _transpose_conv_layer(inputs, kernel_size, stride, num_features, idx):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    print(inputs.get_shape())
    #pad_mat = np.array([[0,0],[0,pad_size[0]],[0,pad_size[1]],[0,0]])
    #input_pad = tf.pad(inputs, pad_mat)
    
    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_features,input_channels], stddev=0.001, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.001))
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.pack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    #elu
    conv_rect = tf.nn.elu(conv_biased,name='{0}_conv'.format(idx))
    return conv_rect
     
def _pool_layer(inputs, kernel_size, stride, pad_size, idx):
  print(inputs.get_shape())
  pad_mat = np.array([[0,0],[0,pad_size[0]],[0,pad_size[1]],[0,0]])
  input_pad = tf.pad(inputs, pad_mat)
  max_pool = tf.nn.max_pool(input_pad,  kernel_size, stride, padding='VALID', name='{0}_max_pool'.format(idx))
  average_pool = tf.nn.avg_pool(input_pad,  kernel_size, stride, padding='VALID', name='{0}_average_pool'.format(idx))
  pool = tf.add(average_pool, max_pool) 
  return pool

def _fc_layer(inputs, hiddens, idx, flat = False, linear = False):
  with tf.variable_scope('fc{0}'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    print(inputs.get_shape())
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(0.01))
    if linear:
      return tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
  
    ip = tf.add(tf.matmul(inputs_processed,weights),biases)
    return tf.nn.elu(ip,name=str(idx)+'_fc')

def conv_ced(inputs, keep_prob):
  """Builds conv part of net.
  Args:
    inputs: input images
    keep_prob: dropout layer
  """
  # conv1
  conv1 = _conv_layer(inputs, 7, 2, 64, [6,6], 1)
  # pool1
  pool1 = _pool_layer(conv1, [1, 2, 2, 1], [1, 2, 2, 1], [2,2], 2)
  # conv2
  conv2 = _conv_layer(pool1, 3, 1, 128, [2,2], 3)
  # pool2
  pool2 = _pool_layer(conv2, [1, 2, 2, 1], [1, 2, 2, 1], [1,1], 4)
  # conv3 
  conv3 = _conv_layer(pool2, 1, 1, 128, [0,0], 5)
  # conv4 
  conv4 = _conv_layer(conv3, 3, 1, 256, [2,2], 6)
  # conv5 
  conv5 = _conv_layer(conv4, 1, 1, 256, [0,0], 7)
  # conv6 
  conv6 = _conv_layer(conv5, 3, 1, 512, [2,2], 8)
  # pool3
  pool3 = _pool_layer(conv6, [1, 2, 2, 1], [1, 2, 2, 1], [1,1], 9)
  # conv7 
  conv7 = _conv_layer(pool3, 1, 1, 256, [0,0], 10)
  # conv8 
  conv8 = _conv_layer(conv7, 3, 1, 512, [2,2], 11)
  # conv9 
  conv9 = _conv_layer(conv8, 1, 1, 256, [0,0], 12)
  # conv10 
  conv10 = _conv_layer(conv9, 3, 1, 512, [1,1], 13)
  # conv11 
  conv11 = _conv_layer(conv10, 1, 1, 256, [0,0], 14)

  return conv11 

def trans_conv_ced(inputs):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  # conv23
  #conv23 = _transpose_conv_layer(inputs, 3, 2, 128, [0,0], 23)
  # conv24
  #conv24 = _transpose_conv_layer(conv23, 3, 2, 128, [0,0], 24)
  # conv25
  #conv25 = _transpose_conv_layer(conv24, 3, 1, 128, [0,0], 26)
  # conv26
  #conv26 = _transpose_conv_layer(conv25, 3, 2, 256, [0,0], 27)
  # mask
  #mask = _transpose_conv_layer(conv26, 4, 2, 1, [0,0], 28)
  #_activation_summary(mask)

  # conv23
  conv23 = _transpose_conv_layer(inputs, 3, 2, 128, 23)
  # conv24
  conv24 = _transpose_conv_layer(conv23, 3, 2, 128, 24)
  # conv25
  conv25 = _transpose_conv_layer(conv24, 3, 1, 128, 26)
  # conv26
  conv26 = _transpose_conv_layer(conv25, 3, 2, 256, 27)
  # mask
  mask = _transpose_conv_layer(conv26, 4, 2, 1, 28)
  pad_mat = np.array([[0,0],[0,4],[0,4],[0,0]])
  mask_pad = tf.pad(mask, pad_mat)
  # pad mask a little :-( 
  print("mask is shape")
  print(mask_pad.get_shape())
  _activation_summary(mask_pad)

  # display output
  tf.image_summary('predicted_mask', mask_pad)

  return mask_pad 

