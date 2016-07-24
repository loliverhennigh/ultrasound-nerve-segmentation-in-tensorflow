
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import model.nerve_net as nerve_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../checkpoints/train_store_run_0001',
                            """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 500000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('input_keep_prob', .9,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('keep_prob', .5,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate', .0001,
                            """ keep probability for dropout """)


def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    image, mask = nerve_net.inputs(FLAGS.batch_size) 
    # possible input dropout 
    input_keep_prob = tf.placeholder("float")
    image_drop = tf.nn.dropout(image, input_keep_prob)
    # possible dropout inside
    keep_prob = tf.placeholder("float")
    # create and unrap network
    prediction = nerve_net.inference(image_drop, keep_prob) 
    # calc error
    error = nerve_net.loss_image(mask, prediction) 
    # train hopefuly 
    train_op = nerve_net.train(error, FLAGS.learning_rate)
    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   
    #for i, variable in enumerate(variables):
    #  print '----------------------------------------------'
    #  print variable.name[:variable.name.index(':')]

    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)

    for step in xrange(FLAGS.max_steps):
      t = time.time()
      _ , loss_value = sess.run([train_op, error],feed_dict={keep_prob:0.9, input_keep_prob:.8})
      print(loss_value)
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%5 == 0:
        summary_str = sess.run(summary_op, feed_dict={keep_prob:0.9, input_keep_prob:.8})
        summary_writer.add_summary(summary_str, step) 

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
