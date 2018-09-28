import os
from os.path import expanduser
home_dir = expanduser("~")
module_path = home_dir + '/code/modules/'
backprop_nets_dir = home_dir + '/trained_networks/backprop_trained/'
bp_pso_network_dir = home_dir + '/trained_networks/backprop_and_pso_trained/'
import sys
sys.path.append(module_path)
import random
import numpy as np

from data_processing import *
from observational_data_management import add_obs_data
from loading_datasets import *
from plotting import *
from pso_parallel_training_queue import *

np.random.seed(999)
random.seed(999)

real_observations = True
verbatim = True

test = False

pretrained_network_name = '6x6_all-points_redshifts00-01-02-05-10-20-30-40-60-80_train-test-val080-010-010_tanh_Halo_mass_peak-Scale_peak_mass-Halo_growth_rate-Halo_radius-Redshift_to_Stellar_mass-SFR_test_score4.37e-07'
# network_name = '{}'.format(datetime.datetime.now().strftime("%Y-%m-%d"))
draw_figs = {'train': True, 'val': False} # should figures using the <mode> weights predicting on <mode> data be drawn?
# Which plots to create while running
plots = ['csfrd', 'wp', 'triple_plot', 'triple_surf'] # 'csfrd', 'wp', 'triple_plot', 'triple_surf'

### Loss parameters
stellar_mass_bin_width = 0.2 # concerns smf, fq, ssfr losses
loss_dict = {
    'fq_weight': 1,
    'ssfr_weight': 1,
    'smf_weight': 1, 
    'shm_weight': 2, # only available when using mock observations
    'csfrd_weight': 1,
    'clustering_weight': 1,
    'dist_outside_punish': 'exp',
    'dist_outside_factor': 10,
    'no_coverage_punish': 'exp',
    'no_coverage_factor': 10,
    'min_filled_bin_frac': 0,
    'nr_redshifts_per_eval': 'all', # a nr or the string 'all'
    'stellar_mass_bins': np.arange(7, 12.5, stellar_mass_bin_width),
    'stellar_mass_bin_width': stellar_mass_bin_width
}

### PSO parameters
nr_processes = 20
nr_iterations = 2000
std_penalty = False
min_std_tol = 0.01 # minimum allowed std for any parameter
pso_args = {
    'nr_particles': 2 * nr_processes,
    'inertia_weight_start': 0.5,
    'inertia_weight_min': 0.3,
    'exploration_iters': 200, # nr of iteraions before the pso reaches the min val of the inertia weight
    'patience': 50, # nr of iterations without improvement on 'patience_parameter' before training is restarted
    'patience_parameter': 'train',
    'patience_min_score_increase': 1e-3,
    'restart_check_interval': 1e5, # lower to start checking for low stds, not implemented for no val set atm
    'no_validation': True
}
 
if test:
    network_name = 'testing'
else:
    network_name = pretrained_network_name
    weight_string = 'fq-ssfr-smf-csfrd-wp_' + '-'.join([str(loss_dict['fq_weight']), str(loss_dict['ssfr_weight']), 
                                                        str(loss_dict['smf_weight']), str(loss_dict['csfrd_weight']), 
                                                        str(loss_dict['clustering_weight'])])
    network_name += '__' + weight_string
    network_name += '_inertiaStart{:.0f}_{:02.0f}Explore'.format(10*pso_args['inertia_weight_start'], 
                                                                 pso_args['exploration_iters'])
#     if pso_args['no_validation']:
#         network_name += '_noVal'
#     else:
#         network_name += '_withVal'
#     if real_observations:
#         network_name += '_real_obs'
#     else:
#         network_name += '_mock_obs'

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if real_observations:
    path = bp_pso_network_dir + 'real_observations/' + network_name
else:
    path = bp_pso_network_dir + 'mock_observations/' + network_name
    
already_exists = True
while already_exists:
    if os.path.exists(path):
        path += '_new'
        network_name += '_new'
    else:
        already_exists = False

if os.path.exists(backprop_nets_dir + pretrained_network_name + '/training_data_dict.p'):
    training_data_dict = pickle.load(open(backprop_nets_dir + pretrained_network_name + '/training_data_dict.p', 'rb'))
    print('adding observational data...\n')
    training_data_dict = add_obs_data(training_data_dict, loss_dict, h_0=0.6781, real_obs=real_observations)
    # save complete training_data_dict for later plotting use
    if real_observations:
        full_train_dict_path = bp_pso_network_dir + 'real_observations/' + network_name + '/full_training_data_dict.p'
    else:
        full_train_dict_path = bp_pso_network_dir + 'mock_observations/' + network_name + '/full_training_data_dict.p'
    os.makedirs(os.path.dirname(full_train_dict_path), exist_ok=True)
    pickle.dump(training_data_dict, open(full_train_dict_path, 'wb'))
    # now prune training_data_dict to make it lighter for the pso
    training_data_dict = prune_train_data_dict_for_reinf_learn(training_data_dict, no_val=pso_args['no_validation'])
else:
    print('there is no pretrained network with that name.')
    print(backprop_nets_dir + pretrained_network_name + '/training_data_dict.p')
    sys.exit()

# Start training
network = Feed_Forward_Neural_Network(network_name, training_data_dict, 
                                      real_observations=real_observations, nr_processes=nr_processes, 
                                      start_from_pretrained_net=True, pretrained_net_name=pretrained_network_name, 
                                      pso_args=pso_args)
network.train_pso(nr_iterations, std_penalty=std_penalty, verbatim=verbatim, draw_figures=draw_figs, 
                  loss_dict=loss_dict, plots=plots)

















