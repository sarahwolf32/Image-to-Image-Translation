import tensorflow as tf
import os

class Logger:

    '''
    Logs progress to an event file for Tensorboard.
    Prints out losses.
    Saves checkpoints.
    '''

    def __init__(self, config, sess, ops):
        self.config = config
        self.sess = sess
        self.writer = tf.summary.FileWriter(config.summary_dir, graph=sess.graph)
        self.saver = tf.train.Saver()
        self.ops = ops

    def log(self, feed_dict):

        # write tensorboard summary
        global_step = self.sess.run(self.ops.global_step)
        if global_step % self.config.log_freq == 0:
            summary = self.sess.run(self.ops.summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, global_step=global_step)

        # print
        loss_g, loss_d, prob_x, prob_g, loss_L1 = self.sess.run([self.ops.loss_g, self.ops.loss_d, self.ops.prob_x, self.ops.prob_g, self.ops.loss_L1], feed_dict=feed_dict)
        print("step: " + str(global_step))
        print("loss_g: " + str(loss_g))
        print("loss_d: " + str(loss_d))
        print("prob_real_x: " + str(prob_x))
        print("prob_real_g: " + str(prob_g))
        print("loss_L1: " + str(loss_L1))
        print(" ")
        
    def checkpoint(self, feed_dict):

        # determine if we should save checkpoint 
        step = self.sess.run(self.ops.global_step)
        if step % self.config.checkpoint_freq == 0:

            # create checkpoint directory if needed
            if not os.path.exists(self.config.checkpoint_dir):
                os.makedirs(self.config.checkpoint_dir)

            #save
            model_name = self.config.checkpoint_dir + '/model-' + str(step) + '.cptk'
            self.saver.save(self.sess, model_name, global_step=step)

        





