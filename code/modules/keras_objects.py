from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer

def log_weighted_mse(y_true, y_pred):
    
    return K.mean(K.log(y_true+1.5) + K.square(y_pred - y_true), axis=-1)

def normal_weighted_mse(y_true, y_pred):
    
    square = K.square(y_pred - y_true)
    weighted_square = y_true * square / K.sum(y_true)
    
    return K.mean(weighted_square, axis=-1)

def stellar_mass_weighted_mse(y_true, y_pred):
    
    true_stellar_masses = tf.pow(K.cast(10, 'float32'), y_true)
    square = K.square(y_pred - y_true)
    weighted_square = true_stellar_masses * square / K.sum(true_stellar_masses)
    
    return K.mean(weighted_square, axis=-1)

def halo_mass_weighted_loss_wrapper(halo_masses):
    def halo_mass_weighted_loss(y_true, y_pred):
        
        #true_halo_masses = halo_masses / 10
        
        true_halo_masses = tf.pow(K.cast(10, 'float32'), halo_masses)
        squared_diffs = K.square(y_pred - y_true)
        weighted_square = true_halo_masses * squared_diffs / K.sum(true_halo_masses)
        
        return K.mean(weighted_square, axis=-1)
    return halo_mass_weighted_loss

def tunnel_loss(y_true, y_pred):
    return y_pred

class Weighted_loss_layer(Layer):

    ### OBS!! This layer takes as input a list of three tensors: [weights to give points, correct output, output of network]
    ### it has no trainable weights and the weights are zero (do not contribute to reg loss)

    def __init__(self):
        self.output_dim = 1
        super(Weighted_loss_layer, self).__init__()

    def build(self, input_shape):
        # Create a non-trainable weight variable for this layer.
        print(input_shape)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='zero',
                                      trainable=False)
        super(Weighted_loss_layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        weights, correct_output, network_output = inputs
        squared_diffs = K.square(network_output - correct_output)
        weighted_square = weights * squared_diffs / K.sum(weights)
        mean = K.mean(weighted_square, axis=-1)
        return mean

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
class Nonweighted_loss_layer(Layer):

    ### OBS!! This layer takes as input a list of two tensors: [correct output, output of network]
    ### it has no trainable weights and the weights are zero (do not contribute to reg loss)

    def __init__(self):
        self.output_dim = 1
        super(Nonweighted_loss_layer, self).__init__()

    def build(self, input_shape):
        # Create a non-trainable weight variable for this layer.
        print(input_shape)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='zero',
                                      trainable=False)
        super(Nonweighted_loss_layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        correct_output, network_output = inputs
        squared_diffs = K.square(network_output - correct_output)
        mean = K.mean(squared_diffs, axis=-1)
        return mean

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    