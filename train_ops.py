import tensorflow as tf
from model import Model

class TrainOps:

    def __init__(self, graph):

        # placeholders
        self.x_images_holder = graph.get_tensor_by_name('x_images_holder:0')
        self.y_images_holder = graph.get_tensor_by_name('y_images_holder:0')

        # core training values
        self.train_d = graph.get_operation_by_name('train_d')
        self.train_g = graph.get_operation_by_name('train_g')
        self.loss_d = graph.get_tensor_by_name('loss_d:0')
        self.loss_g = graph.get_tensor_by_name('loss_g:0')
        self.generated_images = graph.get_tensor_by_name('generator/generated_images:0')
        self.prob_x = graph.get_tensor_by_name('prob_x:0')
        self.prob_g = graph.get_tensor_by_name('prob_g:0')

        # counters
        self.epoch = graph.get_tensor_by_name('epoch:0')
        self.global_step = graph.get_tensor_by_name('global_step:0')

        # tensorboard
        self.summary = graph.get_tensor_by_name('Merge/MergeSummary:0')
        


