import os
from os.path import expanduser
home_dir = expanduser("~")
module_path = home_dir + '/code/modules/'
import sys
sys.path.append(module_path)
import random
import numpy as np

from data_processing import *
from plotting import *
from pso_parallel_training_queue import *

np.random.seed(999)
random.seed(999)

### General parameters
total_set_size = 2.9e4 # how many examples will be used for training+validation+testing
train_size = 1.5e4
val_size = 1e4
test_size = .4e4
input_features = ['Halo_mass', 'Halo_mass_peak', 'Scale_peak_mass', 'Scale_half_mass', 'Halo_growth_rate']#, 'Redshift']
output_features = ['Stellar_mass', 'SFR']
redshifts = [0]#,.1,.2,.5,1,2,3,4,6,8]
same_n_points_per_redshift = False # if using the smf in the objective function, must be false!

reinforcement_learning = True
real_observations = False

verbatim = True

network_name = 'testing'
# network_name = '6x6_tanh_xi5_loss_point9_cutoff_no_empty_bin_punish_nbin_weighted_loss'
# network_name = '{}'.format(datetime.datetime.now().strftime("%Y-%m-%d"))
save_all_nets = True
draw_figs = True

### Network parameters
nr_hidden_layers = 6
activation_function = 'tanh'
output_activation = {'SFR': None, 'Stellar_mass': None}
nr_neurons_per_layer = 6
regularisation_strength = 1e-2
std_penalty = False
norm = {'input': 'zero_mean_unit_std',
        'output': 'zero_mean_unit_std'} # 'none',   'zero_mean_unit_std',   'zero_to_one'

### Loss parameters
loss_dict = {
    'dist_outside_punish': 'exp',
    'exp_factor': 10
}

### PSO parameters
nr_processes = 30
nr_iterations = 2000
min_std_tol = 0.01                # minimum allowed std for any parameter
pso_param_dict = {
    'nr_particles': 3 * nr_processes,
    'inertia_weight_start': 1.4,
    'inertia_weight_min': 0.3,
    'exploration_iters': 1500,
    'patience': 10000,
    'patience_parameter': 'train',
    'restart_check_interval': 200
}

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# load the selected galaxyfile
galaxies, data_keys = load_galfiles(redshifts=redshifts, equal_numbers=same_n_points_per_redshift)
    
# prepare the training data
training_data_dict = divide_train_data(galaxies, data_keys, input_features, output_features, redshifts, 
                                       total_set_size=int(total_set_size), train_size=int(train_size), val_size=int(val_size), 
                                       test_size=int(test_size), pso=True)
training_data_dict = normalise_data(training_data_dict, norm, pso=True)


# Start training
network = Feed_Forward_Neural_Network(nr_hidden_layers, nr_neurons_per_layer, input_features, output_features, 
                                      activation_function, output_activation, regularisation_strength, network_name)
network.setup_pso(pso_param_dict, reinf_learning=reinforcement_learning, real_observations=real_observations, 
                  nr_processes=nr_processes)
network.train_pso(nr_iterations, training_data_dict, std_penalty=std_penalty, verbatim=verbatim, save_all_networks=save_all_nets,
                  draw_figures=draw_figs, loss_dict=loss_dict)

















