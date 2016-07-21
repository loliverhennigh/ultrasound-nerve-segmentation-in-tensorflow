
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

def _conv_layer(inputs, kernel_size, stride, num_features, idx):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    print(inputs.get_shape())

    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,input_channels,num_features],stddev=0.1, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.1))

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    #elu
    conv_rect = tf.nn.elu(conv_biased,name='{0}_conv'.format(idx))
    return conv_rect

def _transpose_conv_layer(inputs, kernel_size, stride, num_features, idx):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    print(inputs.get_shape())
    
    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_features,input_channels], stddev=0.1, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.1))
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.pack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    #elu
    conv_rect = tf.nn.elu(conv_biased,name='{0}_conv'.format(idx))
    return conv_rect
     
def _pool_layer(inputs, kernel_size, stride, idx):
  max_pool = tf.nn.max_pool(inputs_pad,  kernel_size, stride, padding='SAME', name='{0}_max_pool'.format(idx))
  average_pool = tf.nn.avg_pool(inputs_pad,  kernel_size, stride, padding='SAME', name='{0}_average_pool'.format(idx))
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
  conv1 = _conv_layer(inputs, 7, 2, 64, 1)
  # pool1
  pool1 = _pool_layer(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 2)
  # conv2
  conv2 = _conv_layer(pool1, 3, 1, 128, 3)
  # pool2
  pool2 = _pool_layer(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 4)
  # conv3 
  conv3 = _conv_layer(pool2, 1, 1, 128, 5)
  # conv4 
  conv4 = _conv_layer(conv3, 3, 1, 256, 6)
  # conv5 
  conv5 = _conv_layer(conv4, 1, 1, 256, 7)
  # conv6 
  conv6 = _conv_layer(conv5, 3, 1, 512, 8)
  # pool3
  pool3 = _pool_layer(conv6, [1, 2, 2, 1], [1, 2, 2, 1], 9)
  # conv7 
  conv7 = _conv_layer(pool3, 1, 1, 256, 10)
  # conv8 
  conv8 = _conv_layer(conv7, 3, 1, 512, 10)
  # conv9 
  conv9 = _conv_layer(conv8, 1, 1, 256, 10)
  # conv10 
  conv10 = _conv_layer(conv9, 3, 1, 512, 10)
  # conv11 
  conv11 = _conv_layer(conv10, 1, 1, 256, 10)

  return conv11 

def decoding_84x84x3(inputs):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  y_2 = inputs 
 
  # fc21
  fc21 = _fc_layer(y_2, 64*14*14, 21, False, False)
  conv22 = tf.reshape(fc21, [-1, 14, 14, 64])
  # conv23
  conv23 = _transpose_conv_layer(conv22, 1, 1, 128, 23)
  # conv24
  conv24 = _transpose_conv_layer(conv23, 3, 1, 64, 24)
  # conv25
  conv25 = _transpose_conv_layer(conv24, 1, 1, 128, 25)
  # conv26
  conv26 = _transpose_conv_layer(conv25, 3, 1, 128, 26)
  # conv25
  conv27 = _transpose_conv_layer(conv26, 4, 2, 256, 27)
  # conv26
  x_2 = _transpose_conv_layer(conv27, 8, 3, 3, 28)
  # x_2 
  _activation_summary(x_2)

  return x_2 

