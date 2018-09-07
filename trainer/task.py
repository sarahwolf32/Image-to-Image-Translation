import tensorflow as tf
import numpy as np
from data_loader import DataLoader
from model import Model
from train_ops import TrainOps
from architecture import Architecture as A
import argparse
from logger import Logger
from sampler import Sampler


def train(sess, dataset, config):

    # prepare train ops
    ops = TrainOps(sess.graph)
    logger = Logger(config, sess, ops)
    epoch = sess.run(ops.epoch)

    # loop through epochs
    while epoch < config.num_epochs:
        iterator = dataset.make_one_shot_iterator()  
        batch_var = iterator.get_next()   

        # loop through batches
        while True:
            try: 

                # get mini-batch 
                batch = sess.run(batch_var)
                x_images, y_images = DataLoader().split_images(batch)
                
                # train
                feed_dict = {ops.x_images_holder: x_images, ops.y_images_holder: y_images}
                sess.run([ops.train_g, ops.train_d], feed_dict=feed_dict)
                logger.log(feed_dict)
                logger.checkpoint(feed_dict)
                sess.run(tf.assign_add(ops.global_step, 1))
                
            except tf.errors.OutOfRangeError:
                break

        # increment epoch
        sess.run(tf.assign_add(ops.epoch, 1))
        epoch = sess.run(ops.epoch)

def start_training(config):
    Model()
    dataset = DataLoader().load_images(config.data_dir, batch_size=config.batch_size)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    train(sess, dataset, config)

def load_session(config):
    sess = tf.Session()

    # load stored graph into current graph
    graph_filename = str(tf.train.latest_checkpoint(config.checkpoint_dir)) + '.meta'
    saver = tf.train.import_meta_graph(graph_filename)

    # restore variables into graph
    saver.restore(sess, tf.train.latest_checkpoint(config.checkpoint_dir))
    return sess

def continue_training(config):
    sess = load_session(config)
    dataset = DataLoader().load_images(config.data_dir, batch_size=config.batch_size)
    train(sess, dataset, config)

def sample(config):
    sess = load_session(config)
    sampler = Sampler()
    sampler.sample(config.translate_image_dir, config.sample_dir, sess)


# MAIN
if __name__=='__main__':

    # unwrap config
    parser = argparse.ArgumentParser()

    # filepaths
    parser.add_argument('--data-dir', default='data/facades')
    parser.add_argument('--summary-dir', default='summary')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--sample-dir', default='samples')

    # training settings
    parser.add_argument('--num-epochs', type=int, default=650)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--log-freq', type=int, default=25)
    parser.add_argument('--checkpoint-freq', type=int, default=50)
    parser.add_argument('--continue-train', type=bool, default=False)
    parser.add_argument('--translate-image-dir', default=None)
    config = parser.parse_args()

    # continue training from checkpoint
    if config.continue_train:
        continue_training(config)

    # sample from saved checkpoint
    elif config.translate_image_dir:
        sample(config)

    # train from scratch
    else:
        start_training(config)





