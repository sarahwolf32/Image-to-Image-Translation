import tensorflow as tf

class Logger:

    def __init__(self, config, sess, ops):
        self.config = config
        self.sess = sess
        self.writer = tf.summary.FileWriter(config.summary_dir, graph=sess.graph)
        self.ops = ops

    def log(self, feed_dict):
        global_step = self.sess.run(self.ops.global_step)
        if global_step % self.config.log_freq == 0:
            summary = self.sess.run(self.ops.summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, global_step=global_step)

        # print
        loss_g, loss_d = self.sess.run([self.ops.loss_g, self.ops.loss_d], feed_dict=feed_dict)
        print("step: " + str(global_step))
        print("loss_g: " + str(loss_g))
        print("loss_d: " + str(loss_d))
