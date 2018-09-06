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
from loading_datasets import *
from plotting import *
from pso_parallel_training_queue import *

np.random.seed(999)
random.seed(999)

### General parameters
tot_nr_points = 'all' # how many examples will be used for training+validation+testing, either 'all' or <nr>
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1
input_features = ['Halo_mass_peak', 'Scale_peak_mass', 'Halo_growth_rate', 'Halo_radius', 'Redshift']
output_features = ['Stellar_mass', 'SFR']
redshifts = [0,.1,.2,.5]#,1,2,3,4,6,8]
same_n_points_per_redshift = False # if using the smf in the objective function, must be false!

reinforcement_learning = True
real_observations = True

verbatim = True

test = True
use_pretrained_network = True
pretrained_network_name = '8x8_2.0e+05points_redshifts00-01-02-05-10_tanh_Halo_mass_peak-Scale_peak_mass-Halo_growth_rate-Halo_radius-Redshift_to_Stellar_mass-SFR_test_score7.48e-06'
# network_name = '{}'.format(datetime.datetime.now().strftime("%Y-%m-%d"))
draw_figs = True

### Network parameters
nr_hidden_layers = 8
activation_function = 'tanh'
output_activation = {'SFR': None, 'Stellar_mass': None}
nr_neurons_per_layer = 8
regularisation_strength = 1e-2
std_penalty = False
norm = {'input': 'zero_mean_unit_std', # 'none',   'zero_mean_unit_std',   'zero_to_one'
        'output': 'none'} # output norm does not matter for pso, only bp (script-wise)

### Loss parameters
loss_dict = {
    'fq_weight': 1,
    'ssfr_weight': 1,
    'smf_weight': 1,
    'shm_weight': 2,
    'csfrd_weight': 1,
    'dist_outside_punish': 'exp',
    'dist_outside_factor': 10,
    'no_coverage_punish': 'exp',
    'no_coverage_factor': 10,
    'min_filled_bin_frac': 0,
    'nr_redshifts_per_eval': 'all', # nr, 'all'
    'nr_bins_real_obs': 20
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
                                           test_frac=test_frac, pso=True, real_observations=real_observations)
    training_data_dict = normalise_data(training_data_dict, norm, pso=True)


# Start training
network = Feed_Forward_Neural_Network(nr_hidden_layers, nr_neurons_per_layer, input_features, output_features, 
                                      activation_function, output_activation, regularisation_strength, network_name)
network.setup_pso(pso_param_dict, reinf_learning=reinforcement_learning, real_observations=real_observations, 
                  nr_processes=nr_processes, start_from_pretrained_net=use_pretrained_network, 
                  pretrained_net_name=pretrained_network_name)
network.train_pso(nr_iterations, training_data_dict, std_penalty=std_penalty, verbatim=verbatim, draw_figures=draw_figs, 
                  loss_dict=loss_dict)

















