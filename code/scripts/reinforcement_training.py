import os
from os.path import expanduser
home_dir = expanduser("~")
module_path = home_dir + '/code/modules/'
backprop_nets_dir = home_dir + '/trained_networks/backprop_trained/'
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
tot_nr_points = 'all' # how many examples will be used for training+validation+testing, either 'all' or <nr>
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1
input_features = ['Halo_mass', 'Halo_mass_peak', 'Scale_peak_mass', 'Scale_half_mass', 'Halo_growth_rate', 'Redshift']
output_features = ['Stellar_mass', 'SFR']
redshifts = [0,.1,.2,.5,1,2,3,4,6,8]
same_n_points_per_redshift = False # if using the smf in the objective function, must be false!

reinforcement_learning = True
real_observations = False

verbatim = True

test = False
use_pretrained_network = True
pretrained_network_name = '10x10_all-points_redshifts00-01-02-05-10-20-30-40-60-80_tanh_Halo_mass-Halo_mass_peak-Scale_peak_mass-Scale_half_mass-Halo_growth_rate-Redshift_to_Stellar_mass-SFR_test_score4.81e-07'
# network_name = '{}'.format(datetime.datetime.now().strftime("%Y-%m-%d"))
draw_figs = True

### Network parameters
nr_hidden_layers = 10
activation_function = 'tanh'
output_activation = {'SFR': None, 'Stellar_mass': None}
nr_neurons_per_layer = 10
regularisation_strength = 1e-2
std_penalty = False
norm = {'input': 'zero_mean_unit_std', # 'none',   'zero_mean_unit_std',   'zero_to_one'
        'output': 'none'} # output norm does not matter for pso, only bp

### Loss parameters
loss_dict = {
    'fq_weight': 1,
    'ssfr_weight': 1,
    'smf_weight': 1,
    'shm_weight': 1,
    'dist_outside_punish': 'exp',
    'dist_outside_factor': 10,
    'min_filled_bin_frac': 0,
    'nr_redshifts_per_eval': 'all' # nr, 'all'
}

### PSO parameters
nr_processes = 30
nr_iterations = 2000
min_std_tol = 0.01 # minimum allowed std for any parameter
pso_param_dict = {
    'nr_particles': 3 * nr_processes,
    'inertia_weight_start': 1.4,
    'inertia_weight_min': 0.3,
    'exploration_iters': 1500,
    'patience': 100,
    'patience_parameter': 'train',
    'restart_check_interval': 10
}
if use_pretrained_network:
    pso_param_dict['inertia_weight_start'] = 1

if test:
    network_name = 'testing'
elif use_pretrained_network:
    network_name = pretrained_network_name
else:
    redshift_string = '-'.join(['{:02.0f}'.format(red*10) for red in redshifts])
    weight_string = '-'.join([str(loss_dict['fq_weight']), str(loss_dict['ssfr_weight']), str(loss_dict['smf_weight']), 
                              str(loss_dict['shm_weight'])])
    network_name = '{:d}x{:d}_{:.1e}points_redshifts{}_{}_{}{}_loss_{}_minFilledBinFrac{:03.0f}_fq-ssfr-smf-shm_weights_{}'.format(
        nr_hidden_layers, nr_neurons_per_layer, tot_nr_points, redshift_string, activation_function, 
        loss_dict['dist_outside_punish'], loss_dict['dist_outside_factor'], '-'.join(input_features), 
        100 * loss_dict['min_filled_bin_frac'], weight_string
    )

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if use_pretrained_network:
    if os.path.exists(backprop_nets_dir + pretrained_network_name + '/training_data_dict.p'):
        training_data_dict = pickle.load(open(backprop_nets_dir + pretrained_network_name + '/training_data_dict.p', 'rb'))
        del training_data_dict["output_train_dict"]
        del training_data_dict["output_val_dict"]
        del training_data_dict["output_test_dict"]
    else:
        print('there is no pretrained network with that name.')
        print(backprop_nets_dir + pretrained_network_name + '/training_data_dict.p')
        sys.exit()
    
else:
    # load the selected galaxyfile
    galaxies, data_keys = load_galfiles(redshifts=redshifts, equal_numbers=same_n_points_per_redshift)

    # prepare the training data
    training_data_dict = divide_train_data(galaxies, data_keys, input_features, output_features, redshifts, 
                                           total_set_size=tot_nr_points, train_frac=train_frac, val_frac=val_frac, 
                                           test_frac=test_frac, pso=True)
    training_data_dict = normalise_data(training_data_dict, norm, pso=True)


# Start training
network = Feed_Forward_Neural_Network(nr_hidden_layers, nr_neurons_per_layer, input_features, output_features, 
                                      activation_function, output_activation, regularisation_strength, network_name)
network.setup_pso(pso_param_dict, reinf_learning=reinforcement_learning, real_observations=real_observations, 
                  nr_processes=nr_processes, start_from_pretrained_net=use_pretrained_network, 
                  pretrained_net_name=pretrained_network_name)
network.train_pso(nr_iterations, training_data_dict, std_penalty=std_penalty, verbatim=verbatim, draw_figures=draw_figs, 
                  loss_dict=loss_dict)

















