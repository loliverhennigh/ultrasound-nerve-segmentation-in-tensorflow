from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import cv2
from glob import glob as glb

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import model.nerve_net as nerve_net 
import input.nerve_input as nerve_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../checkpoints/eval_run',
                            """dir to store eval run """)
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/train_store_run_0001',
                            """dir to load trained net """)

def evaluate():
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  # get a list of image filenames
  filenames = glb('../data/test/*')
  #num_files = len(filename)

  with tf.Graph().as_default():
    # Make image placeholder
    images_op = tf.placeholder(tf.float32, [1, 420, 580, 1])
    keep_prob = tf.placeholder("float")

    # Build a Graph that computes the logits predictions from the
    # inference model.
    mask = nerve_net.inference(images_op,keep_prob)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    
    sess = tf.Session()

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    for f in filenames:
      # read in image
      img = cv2.imread(f, 0)
 
      # format image for network
      img = np.expand_dims(img, axis=0)
      img = np.expand_dims(img, axis=3)
  
      # calc logits 
      generated_mask = sess.run([mask],feed_dict={images_op: img, keep_prob: 1.0})
      generated_mask = generated_mask[0]
      generated_mask = generated_mask[0, :, :, :]
      print(generated_mask.shape)
 
      # display image
      cv2.imshow('mask', generated_mask)
      cv2.waitKey(0)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()
