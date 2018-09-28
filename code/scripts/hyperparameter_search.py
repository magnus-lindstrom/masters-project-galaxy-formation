import numpy as np
import os
from os.path import expanduser
home_dir = expanduser("~")
module_path = home_dir + '/code/modules/'
import sys
sys.path.append(module_path)
import json
import datetime
from loading_datasets import *
from data_processing import *
from keras.callbacks import EarlyStopping, TerminateOnNaN
from model_setup import *


### single process, GPU version
keep_writing_to_old_file = True
run_on_cpu = True

folder_name = 'code/backpropagation_code/hyperparameter_searches/'
file_name = '2018-09-24_network_size' # file to keep working on if keep_writing_to_old_file=True

nr_steps = 1e5 # 1e5 in full run

validation_data = 'val' #'val' is normally used, use 'train' to check overfitting potential

nr_folds = 3
tot_nr_points = 'all' # how many examples will be used for training+validation+testing, 'all' or a number
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

input_features = ['Halo_mass_peak', 'Scale_peak_mass', 'Halo_growth_rate', 'Halo_radius', 'Redshift']
output_features = ['Stellar_mass', 'SFR']
outputs_to_weigh = ['Stellar_mass']
redshifts = [0,.1,.2,.5,1,2,3,4,6,8]
same_n_points_per_redshift = False
weigh_by_redshift = True

norm = {'input': 'zero_mean_unit_std',
        'output': 'none'} # 'none',   'zero_mean_unit_std',   'zero_to_one'

nr_layers = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20]
neurons_per_layer = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20]
activation_functions = ['tanh']

loss_function = 'mse'
output_activation = {'SFR': None, 'Stellar_mass': None}
reg_strengths = np.array([1e-20]) # np.power(10, np.random.uniform(low=-20, high=-10, size=5)) 

batch_sizes = np.array([4e4]) # np.power(10, np.random.uniform(low=4, high=5, size=5))
if isinstance(tot_nr_points, float):
    nr_epochs = nr_steps * batch_sizes / (tot_nr_points*train_frac)
else:
    nr_epochs = nr_steps * batch_sizes / (1.29e6*train_frac)
early_stop_patience = [30]
early_stop_min_delta = 1e-20

if run_on_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

progress_file = folder_name + 'progress.txt'

verb = 0 # prints progress to stdout

parameter_dictionary = {
    'input_features': input_features,
    'output_features': output_features,
    'outputs_to_weigh': outputs_to_weigh,
    'loss_function': loss_function,
    'batch_sizes': batch_sizes.tolist(),
    'nr_epochs': nr_epochs.tolist(),
    'tot_nr_points': [tot_nr_points],
    'reg_strengths': reg_strengths.tolist(),
    'nr_folds': nr_folds,
    'activation_functions': activation_functions,
    'neurons_per_layer': neurons_per_layer,
    'nr_hidden_layers': nr_layers,
    'output_activation_function': 'none',
    'description': 'Each parameter setting is represented by one list containing five objects: [parameters, test_score, train_history, '+\
    'val_history, param_id]. The first one is ' + \
    'the parameters of the model. The second one is the test score of the final model'+\
    ' ' +\
    'The third one is the training' +\
    'loss history and the fourth one is the validation loss history. '+\
    'The fifth one is an id for the parameter combinations. If there are several runs for the same combs then they will have the same id.'
}
tot_nr_runs = len(nr_layers) * len(activation_functions) * \
                len(batch_sizes) * len(reg_strengths) * nr_folds

run_counter = 0 # to keep track of how many combinations I've gone through

if keep_writing_to_old_file:
    ### Load a result
    if os.path.isfile(folder_name + file_name + '.json'):
        with open(folder_name + file_name + '.json', 'r') as ff:
            old_results_list = json.load(ff)
        ff.close()
        results_list = old_results_list.copy()
    else:
        old_results_list = []
        results_list = [parameter_dictionary]
else:
    results_list = [parameter_dictionary]

with open(progress_file, 'w+') as f:
    
    date_string_proper = datetime.datetime.now().strftime("%H:%M, %Y-%m-%d")
    f.write('Benchmark done on input parameters at ' + date_string_proper + '\n\n')
    f.flush()
    
    # load the selected galaxyfile, be sure that the autoloaded galaxyfile is the one that you want to use!!
    galaxies, data_keys = load_galfiles(redshifts=redshifts, equal_numbers=same_n_points_per_redshift)

    for i_act_fun, act_fun in enumerate(activation_functions):
        for i_reg_strength, reg_strength in enumerate(reg_strengths):
            for i_neur_per_lay, neur_per_lay in enumerate(neurons_per_layer):
#                 for i_nr_lay, nr_lay in enumerate(nr_layers):
                for i_batch_size, batch_size in enumerate(batch_sizes):

                    ### Network parameters
                    network_args = {        
                        'nr_hidden_layers': nr_layers[i_neur_per_lay],
                        'nr_neurons_per_lay': neur_per_lay,
                        'input_features': input_features,
                        'output_features': output_features,
                        'activation_function': act_fun,
                        'output_activation': output_activation,
                        'reg_strength': reg_strength
                    }

                    earlystop = EarlyStopping(monitor='val_loss', min_delta=early_stop_min_delta, 
                                              patience=early_stop_patience[i_batch_size], verbose=verb, mode='auto')
                    nan_termination = TerminateOnNaN()
                    callbacks_list = [earlystop, nan_termination]

                    train_histories = []
                    val_histories = []
                    scores = []
                    
                    if keep_writing_to_old_file:
                        comb_tried = False
                        for params in [item[0] for item in old_results_list[1:]]:
                            
                            if params['nr_lay'] == network_args['nr_hidden_layers']:
                                comb_tried = True
                                run_counter += nr_folds
                                
                    if (not comb_tried) or (not keep_writing_to_old_file):
                        date_string_proper = datetime.datetime.now().strftime("%H:%M, %Y-%m-%d")
                        f.write(date_string_proper + '  Starting combination\n\n')
                        f.flush()
                        for fold in range(nr_folds):

                            date_string_proper = datetime.datetime.now().strftime("%H:%M, %Y-%m-%d")
                            run_counter += 1
                            f.write(date_string_proper + '        Run nr %d/%d. \n\n' % (run_counter, tot_nr_runs))
                            f.flush()

                            ### Prepare the training data

                            # prepare the training data
                            training_data_dict = divide_train_data(galaxies, data_keys, network_args, redshifts, 
                                                                   outputs_to_weigh=outputs_to_weigh, 
                                                                   weigh_by_redshift=weigh_by_redshift, 
                                                                   total_set_size=tot_nr_points, 
                                                                   train_frac=train_frac, val_frac=val_frac, 
                                                                   test_frac=test_frac, emerge_targets=True)
                            # galaxies = None
                            training_data_dict = normalise_data(training_data_dict, norm)

                            model = standard_network(network_args['input_features'], network_args['output_features'], 
                                                     network_args['nr_neurons_per_lay'], network_args['nr_hidden_layers'], 
                                                     network_args['activation_function'], 
                                                     network_args['output_activation'], network_args['reg_strength'], 
                                                     clipvalue=.001)

                            # Fit the model                        
                            history = model.fit(
                                x = training_data_dict['input_train_dict'], y = training_data_dict['output_train_dict'],
                                validation_data = (training_data_dict['input_'+validation_data+'_dict'], 
                                                   training_data_dict['output_'+validation_data+'_dict'], 
                                                   training_data_dict['val_weights']), 
                                epochs=int(nr_epochs[i_batch_size]), batch_size=int(batch_size), callbacks=callbacks_list,
                                sample_weight=training_data_dict['train_weights'], verbose=verb
                            )

                            test_scores = model.evaluate(x=training_data_dict['input_test_dict'], 
                                                         y=training_data_dict['output_test_dict'],
                                                         sample_weight=training_data_dict['test_weights'], verbose=verb)
                            scores.append(test_scores[0]) # take total test result

                            if 'loss' in history.history:
                                train_histories.append(history.history['loss'])
                            if 'val_loss' in history.history:                        
                                val_histories.append(history.history['val_loss'])

                        best_score = np.amin(scores)
                        mean_score = np.mean(scores)
                        score_std = np.std(scores)
                        total_score = [mean_score, score_std, best_score]
                        parameters = {'nr_lay': nr_layers[i_neur_per_lay], 'neur_per_lay': neur_per_lay, 
                                     'act_fun': act_fun, 'batch_size': batch_size, 'reg_strength': reg_strength}
                        results_list.append([parameters, total_score, train_histories, val_histories])
                        
                        with open(folder_name + file_name + '.json', 'w+') as ff:
                            json.dump(results_list, ff)
                        ff.close()
                        
                        date_string_proper = datetime.datetime.now().strftime("%H:%M, %Y-%m-%d")
                        f.write(date_string_proper + '  Saved combination scores to file\n\n')
                        f.flush()
                        
            
    date_string_proper = datetime.datetime.now().strftime("%H:%M, %Y-%m-%d")
    f.write('Benchmark completed at ' + date_string_proper + '\n')
    f.flush()