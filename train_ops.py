import tensorflow as tf
from model import Model

class TrainOps:

    def __init__(self, graph):
        self.train_d = graph.get_operation_by_name('train_d')
        self.train_g = graph.get_operation_by_name('train_g')
        self.loss_d = graph.get_tensor_by_name('loss_d:0')
        self.loss_g = graph.get_tensor_by_name('loss_g:0')
        self.generated_images = graph.get_tensor_by_name('generator/generated_images:0')



