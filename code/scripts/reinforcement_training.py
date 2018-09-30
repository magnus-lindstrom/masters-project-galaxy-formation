import os
from os.path import expanduser
home_dir = expanduser("~")
module_path = home_dir + '/code/modules/'
pso_nets_dir = home_dir + '/trained_networks/pso_trained/'
import sys
sys.path.append(module_path)
import random
import numpy as np

from data_processing import *
from loading_datasets import *
from plotting import *
from pso_parallel_training_queue import *

np.random.seed(999)
random.seed(999)

### General parameters
tot_nr_points = 'all' # how many examples will be used for training+validation+testing, either 'all' or <nr>
input_features = ['Halo_mass_peak', 'Scale_peak_mass', 'Halo_growth_rate', 'Halo_radius', 'Redshift']
output_features = ['Stellar_mass', 'SFR']
redshifts = [0,.1,.2,.5,1,2,3,4,6,8]
same_n_points_per_redshift = False # if using the smf in the objective function, must be false!

reinforcement_learning = True
real_observations = True

verbatim = True

test = True
draw_figs = {'train': True, 'val': False} # should figures using the <mode> weights predicting on <mode> data be drawn?
plots = ['csfrd_emerge', 'ssfr_emerge', 'surf_data'] # 'csfrd', 'csfrd_emerge', 'ssfr_emerge', 'wp', 'triple_plot', 'surf_data'

### Network parameters
std_penalty = False
norm = {'input': 'zero_mean_unit_std', # 'none',   'zero_mean_unit_std',   'zero_to_one'
        'output': 'none'} # output norm does not matter for pso, only bp (script-wise)

network_args = {        
    'nr_hidden_layers': 5,
    'nr_neurons_per_lay': 5,
    'input_features': input_features,
    'output_features': output_features,
    'activation_function': 'tanh', # 'tanh', 'leaky_relu'
    'output_activation': {'SFR': None, 'Stellar_mass': None},
    'reg_strength': 1e-20
}

### Loss parameters
stellar_mass_bin_width = 0.2 # concerns smf, fq, ssfr losses
loss_dict = {
    'fq_weight': 1,
    'ssfr_weight': 1,
    'smf_weight': 1, 
    'shm_weight': 2, # is using mock observations
    'csfrd_weight': 1,
    'clustering_weight': 1,
    'dist_outside_punish': 'exp',
    'dist_outside_factor': 10,
    'no_coverage_punish': 'exp',
    'no_coverage_factor': 10,
    'min_filled_bin_frac': 0,
    'nr_redshifts_per_eval': 'all', # nr, 'all'
    'stellar_mass_bins': np.arange(7, 12.5, stellar_mass_bin_width),
    'stellar_mass_bin_width': stellar_mass_bin_width
}

### PSO parameters
nr_processes = 20
nr_iterations = 2000
min_std_tol = 0.01 # minimum allowed std for any parameter
pso_args = {
    'nr_particles': 2 * nr_processes,
    'inertia_weight_start': 1.4,
    'inertia_weight_min': 0.3,
    'exploration_iters': 200,
    'patience': 100,
    'patience_parameter': 'train',
    'patience_min_score_increase': 1e-3,
    'restart_check_interval': 1e5, # lower to start checking for low stds, not implemented for no val set atm
    'no_validation': True
}

if test:
    network_name = 'testing'
else:
    
    redshift_string = '-'.join(['{:02.0f}'.format(red*10) for red in redshifts])
    weight_string = '-'.join([str(loss_dict['fq_weight']), str(loss_dict['ssfr_weight']), str(loss_dict['smf_weight']), 
                              str(loss_dict['shm_weight'])])
    if loss_dict['nr_redshifts_per_eval'] != 'all':
        nr_redshift_per_eval_string = '{:d}'.format(loss_dict['nr_redshifts_per_eval'])
    else:
        nr_redshift_per_eval_string = loss_dict['nr_redshifts_per_eval']
    if tot_nr_points == 'all':
        tot_nr_points_str = tot_nr_points
    else:
        tot_nr_points_str = '{:.1e}'.format(tot_nr_points)

    if real_observations:
        network_name = '{:d}x{:d}_{}points_redshifts{}_{}_{}_nrRedshiftPerEval-{}_fq-ssfr-smf_weights_{}_realObs'.format(
            nr_hidden_layers, nr_neurons_per_layer, tot_nr_points_str, redshift_string, activation_function, 
            '-'.join(input_features), nr_redshift_per_eval_string, weight_string
        )
    else:
        network_name = '{:d}x{:d}_{}points_redshifts{}_{}_{}{}_loss_{}_minFilledBinFrac{:03.0f}_noCovPunish{}{:d}_nrRedshiftPerEval-{}_fq-ssfr-smf-shm_weights_{}_mockObs'.format(
            nr_hidden_layers, nr_neurons_per_layer, tot_nr_points_str, redshift_string, activation_function, 
            loss_dict['dist_outside_punish'], loss_dict['dist_outside_factor'], '-'.join(input_features), 
            100 * loss_dict['min_filled_bin_frac'], loss_dict['no_coverage_punish'], loss_dict['no_coverage_factor'], 
            nr_redshift_per_eval_string, weight_string
        )
        
        
if real_observations:
    path = pso_nets_dir + 'real_observations/' + network_name
else:
    path = pso_nets_dir + 'mock_observations/' + network_name
    
already_exists = True
while already_exists:
    if os.path.exists(path):
        path += '_new'
        network_name += '_new'
    else:
        already_exists = False

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# load the selected galaxyfile
galaxies, data_keys = load_galfiles(redshifts=redshifts, equal_numbers=same_n_points_per_redshift)

# prepare the training data
training_data_dict = divide_train_data(galaxies, data_keys, network_args, redshifts, loss_dict=loss_dict,
                                       total_set_size=tot_nr_points, real_observations=real_observations)
training_data_dict = normalise_data(training_data_dict, norm, pso=True)

# Start training    
network = Feed_Forward_Neural_Network(network_name, training_data_dict, input_features, 
                                      output_features, nr_processes=nr_processes, pso_args=pso_args, network_args=network_args)
network.train_pso(nr_iterations, std_penalty=std_penalty, verbatim=verbatim, draw_figures=draw_figs, 
                  loss_dict=loss_dict, plots=plots)













