import pandas as pd
import numpy as np
from keras import backend as K
import tensorflow as tf


def load_galfile(galfile_directory='/home/magnus/code/non_network_notebooks/test_galcat_w_log_densities_and_log_sfr_3e5.h5'):
    # '/scratch/data/galcats/P200/galaxies.Z01.h5'
    galfile = pd.read_hdf(galfile_directory)
    galaxies = galfile.as_matrix()
    gal_header = galfile.keys().tolist()

    ### Remove data points with halo mass below 10.5
    galaxies = galaxies[galaxies[:,6] > 10.5, :]
    
    data_keys = {'X_pos': 0, 'Y_pos': 1, 'Z_pos': 2, 'X_vel': 3, 'Y_vel': 4, 'Z_vel': 5, 'Halo_mass': 6, 
             'Stellar_mass': 7, 'SFR': 8, 'Intra_cluster_mass': 9, 'Halo_mass_peak': 10, 'Stellar_mass_obs': 11, 
             'SFR_obs': 12, 'Halo_radius': 13, 'Concentration': 14, 'Halo_spin': 15, 'Scale_peak_mass': 16, 
             'Scale_half_mass': 17, 'Scale_last_MajM': 18, 'Type': 19, 'Environmental_density': 20}
    unit_dict = {'X_pos': '', 'Y_pos': '', 'Z_pos': '', 'X_vel': '', 'Y_vel': '', 
             'Z_vel': '', 'Halo_mass': 'log($M_{G}/M_{S}$)', 'Stellar_mass': 'log($M_{G}/M_{S}$)', 'SFR': '', 
             'Intra_cluster_mass': '', 'Halo_mass_peak': 'log($M_{G}/M_{S}$)', 
             'Stellar_mass_obs': '', 'SFR_obs': '', 'Halo_radius': '', 
             'Concentration': '', 'Halo_spin': '', 'Scale_peak_mass': 'a', 
             'Scale_half_mass': 'a', 'Scale_last_MajM': 'a', 'Type': '', 
             'Environmental_density': 'log($M_{G}/M_{S}/Mpc^3$)'}
    
    return galaxies, data_keys, unit_dict


def divide_train_data(galaxies, data_keys, input_features, output_features, total_set_size, train_size, val_size, test_size):
    
    n_data_points = galaxies.shape[0]
    subset_indices = np.random.choice(n_data_points, total_set_size, replace=False)
    train_indices = subset_indices[: int(train_size)]
    val_indices = subset_indices[int(train_size) : int(train_size+val_size)]
    test_indices = subset_indices[int(train_size+val_size) :]

    x_train = np.zeros((len(train_indices), len(input_features)))
    x_val = np.zeros((len(val_indices), len(input_features)))
    x_test = np.zeros((len(test_indices), len(input_features)))
    y_train = np.zeros((len(train_indices), len(output_features)))
    y_val = np.zeros((len(val_indices), len(output_features)))
    y_test = np.zeros((len(test_indices), len(output_features)))
    x_data_keys = {}
    y_data_keys = {}
    
    for i in range(len(input_features)):
        x_train[:,i] = galaxies[train_indices, data_keys[input_features[i]]]
        x_val[:,i] = galaxies[val_indices, data_keys[input_features[i]]]
        x_test[:,i] = galaxies[test_indices, data_keys[input_features[i]]]

        x_data_keys[input_features[i]] = i

    for i in range(len(output_features)):
        y_train[:,i] = galaxies[train_indices, data_keys[output_features[i]]]
        y_val[:,i] = galaxies[val_indices, data_keys[output_features[i]]]
        y_test[:,i] = galaxies[test_indices, data_keys[output_features[i]]]
        
        y_data_keys[output_features[i]] = i

    training_data_dict = {
        'output_features': output_features,
        'input_features': input_features,
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'x_data_keys': x_data_keys,
        'y_data_keys': y_data_keys,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'original_data': galaxies,
        'original_data_keys': data_keys
    }
    
    return training_data_dict


def normalise_data(training_data_dict, norm):
    
    x_train = training_data_dict['x_train']
    x_val = training_data_dict['x_val']
    x_test = training_data_dict['x_test']
    y_train = training_data_dict['y_train']
    y_val = training_data_dict['y_val']
    y_test = training_data_dict['y_test']
    
    if norm == 'none':
        
        training_data_dict['norm'] = norm

    elif norm == 'zero_mean_unit_std':

        x_data_means = np.mean(x_train, 0)
        x_data_stds = np.std(x_train, 0)

        x_train_norm = (x_train - x_data_means) / x_data_stds
        x_val_norm = (x_val - x_data_means) / x_data_stds
        x_test_norm = (x_test - x_data_means) / x_data_stds
            
        y_data_means = np.mean(y_train, 0)
        y_data_stds = np.std(y_train, 0)

        y_train_norm = (y_train - y_data_means) / y_data_stds
        y_val_norm = (y_val - y_data_means) / y_data_stds
        y_test_norm = (y_test - y_data_means) / y_data_stds
            
        training_data_dict['norm'] = norm
            
        training_data_dict['x_data_means'] = x_data_means
        training_data_dict['x_data_stds'] = x_data_stds
        training_data_dict['y_data_means'] = y_data_means
        training_data_dict['y_data_stds'] = y_data_stds
        
        training_data_dict['x_train_norm'] = x_train_norm
        training_data_dict['x_val_norm'] = x_val_norm
        training_data_dict['x_test_norm'] = x_test_norm
        training_data_dict['y_train_norm'] = y_train_norm
        training_data_dict['y_val_norm'] = y_val_norm
        training_data_dict['y_test_norm'] = y_test_norm

    elif norm == 'zero_to_one':

        x_data_max = np.max(x_train, 0)
        x_data_min = np.min(x_train, 0)

        x_train_norm = (x_train - x_data_min) / (x_data_max - x_data_min)
        x_val_norm = (x_val - x_data_min) / (x_data_max - x_data_min)
        x_test_norm = (x_test - x_data_min) / (x_data_max - x_data_min)

        y_data_max = np.max(y_train, 0)
        y_data_min = np.min(y_train, 0)

        y_train_norm = (y_train - y_data_min) / (y_data_max - y_data_min)
        y_val_norm = (y_val - y_data_min) / (y_data_max - y_data_min)
        y_test_norm = (y_test - y_data_min) / (y_data_max - y_data_min)
            
        training_data_dict['norm'] = norm
            
        training_data_dict['x_data_max'] = x_data_max
        training_data_dict['x_data_min'] = x_data_min
        training_data_dict['y_data_max'] = y_data_max
        training_data_dict['y_data_min'] = y_data_min
        
        training_data_dict['x_train_norm'] = x_train_norm
        training_data_dict['x_val_norm'] = x_val_norm
        training_data_dict['x_test_norm'] = x_test_norm
        training_data_dict['y_train_norm'] = y_train_norm
        training_data_dict['y_val_norm'] = y_val_norm
        training_data_dict['y_test_norm'] = y_test_norm
       
    else:
        print('Incorrect norm provided: ', norm)        
    
    return training_data_dict


def predict_points(model, training_data_dict, mode):
    ### Predicts galaxy features in their original units
    
    if training_data_dict['norm'] == 'none':
        additional_ending = mode
    else:
        additional_ending = mode + '_norm'
    
    predicted_points = model.predict(training_data_dict['x_'+additional_ending])

    if training_data_dict['norm'] == 'zero_mean_unit_std':
        
        predicted_points = np.multiply(predicted_points, training_data_dict['y_data_stds']) + \
                                        training_data_dict['y_data_means']

    elif training_data_dict['norm'] == 'zero_to_one':
        
        predicted_points = predicted_points * (training_data_dict['y_data_max'] - 
                               training_data_dict['y_data_min']) + training_data_dict['y_data_min']

    return predicted_points
















