import os
from os.path import expanduser
home_dir = expanduser("~")
module_path = home_dir + '/code/modules/'
import sys
sys.path.append(module_path)
bp_network_dir = home_dir + '/trained_networks/backprop_trained/'
import numpy as np
import pickle
from loading_datasets import load_galfiles
from data_processing import *
from keras.callbacks import EarlyStopping
from model_setup import standard_network

### General parameters
verbose = False
run_on_cpu = True

tot_nr_points = 'all' # how many examples will be used for training+validation+testing, 'all' or a number
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1
batch_size = 4e4
norm = {'input': 'zero_mean_unit_std',
        'output': 'none'} # 'none',   'zero_mean_unit_std',   'zero_to_one'
input_features = ['Halo_mass_peak', 'Scale_peak_mass', 'Halo_growth_rate', 'Halo_radius', 'Redshift']
output_features = ['Stellar_mass', 'SFR']
redshifts = [0,.1,.2,.5,1,2,3,4,6,8]
same_n_points_per_redshift = False

outputs_to_weigh = ['Stellar_mass'] # weigh by halo mass, that is
weigh_by_redshift = True

nr_epochs = 1e5 # more than enough. training will stop because of no validation improvement

early_stop_patience = 50
early_stop_monitor = 'val_loss'
early_stop_min_delta = 1e-16

validation_data = 'val' #'val' is normally used, use 'train' to check overfitting potential

### Network parameters
network_args = {        
    'nr_hidden_layers': 5,
    'nr_neurons_per_lay': 5,
    'input_features': input_features,
    'output_features': output_features,
    'activation_function': 'tanh', # 'tanh', 'leaky_relu'
    'output_activation': {'SFR': None, 'Stellar_mass': None},
    'reg_strength': 1e-20
}

if run_on_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# load the selected galaxyfile
galaxies, data_keys = load_galfiles(redshifts=redshifts, equal_numbers=same_n_points_per_redshift)

# prepare the training data
training_data_dict = divide_train_data(galaxies, data_keys, network_args, redshifts, outputs_to_weigh=outputs_to_weigh, 
                                       weigh_by_redshift=weigh_by_redshift, total_set_size=tot_nr_points, 
                                       train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, emerge_targets=True)
galaxies = None
training_data_dict = normalise_data(training_data_dict, norm)

model = standard_network(network_args['input_features'], network_args['output_features'], network_args['nr_neurons_per_lay'], 
                         network_args['nr_hidden_layers'], network_args['activation_function'], 
                         network_args['output_activation'], network_args['reg_strength'], clipvalue=.001)

earlystop = EarlyStopping(monitor=early_stop_monitor, min_delta=early_stop_min_delta, patience=early_stop_patience, \
                          verbose=verbose, mode='auto')
callbacks_list = [earlystop]

history = model.fit(x = training_data_dict['input_train_dict'], y = training_data_dict['output_train_dict'], 
                    validation_data = (training_data_dict['input_'+validation_data+'_dict'], 
                    training_data_dict['output_'+validation_data+'_dict'], training_data_dict['val_weights']), 
                    epochs=int(nr_epochs), batch_size=int(batch_size), callbacks=callbacks_list,
                    sample_weight=training_data_dict['train_weights'], verbose=verbose)

norm_scores = model.evaluate(x=training_data_dict['input_test_dict'], y=training_data_dict['output_test_dict'],
                             sample_weight=training_data_dict['test_weights'], verbose=verbose)

### Save the model
if len(redshifts) == 10:
    redshift_string = 'All'
else:
    redshift_string = '-'.join(['{:02.0f}'.format(red*10) for red in redshifts])
if tot_nr_points == 'all':
    nr_points_string = 'all-points'
else:
    nr_points_string = '{:.1e}points'.format(tot_nr_points)
network_name = '{:d}x{:d}_{}_redshifts{}_train-test-val{:03.0f}-{:03.0f}-{:03.0f}_{}_{}_to_{}_test_score{:.2e}'.format(
    network_args['nr_hidden_layers'], network_args['nr_neurons_per_lay'], nr_points_string, redshift_string, train_frac*100, 
    val_frac*100, test_frac*100, network_args['activation_function'], '-'.join(network_args['input_features']), 
    '-'.join(network_args['output_features']), norm_scores[0]
)

os.makedirs(os.path.dirname(bp_network_dir + network_name + '/model.h5'), exist_ok=True)

model.save(bp_network_dir + network_name + '/model.h5')
pickle.dump(training_data_dict, open(bp_network_dir + network_name + '/training_data_dict.p', 'wb'))
# save the position in weight space for the pso algorithm to use as starting point
model_weights = model.get_weights()
position = []
for weight_matrix in model_weights:
    position.extend(np.ndarray.flatten(weight_matrix))
position = np.array(position)

pickle.dump(position, open(bp_network_dir + network_name + '/best_position.p', 'wb'))
pickle.dump(data_keys, open(bp_network_dir + network_name + '/data_keys.p', 'wb'))

print('Saved model. Name: ', network_name)

















