import pandas as pd
import numpy as np
from keras import backend as K
import tensorflow as tf
import os.path


def load_galfiles(redshifts , with_densities=False, equal_numbers=False, with_growth=True):

    if with_growth:
        galfile_directory = '/home/magnus/data/galcats_nonzero_sfr_no_density_with_growth_rate_no_lastMajM/'
    elif with_densities:
        galfile_directory = '/home/magnus/data/galcats_nonzero_sfr_with_density/'
    else:
        galfile_directory = '/home/magnus/data/galcats_nonzero_sfr_no_density/'
    
    galaxy_list = []
    gal_header = None
    for redshift in redshifts:

        galfile_path = galfile_directory + 'galaxies.Z{:02.0f}.h5'.format(redshift*10)
        if os.path.isfile(galfile_path):
            galfile = pd.read_hdf(galfile_path)
            gals = galfile.values
            gal_header = galfile.keys().tolist()
            
            redshift_column = redshift * np.ones((np.shape(gals)[0],1))
            gals = np.concatenate((gals, redshift_column), axis=1)
            
            # Scramble the order of the galaxies, since they are somewhat ordered to begin with
            inds = np.random.permutation(np.shape(gals)[0])
            gals = gals[inds, :]
            
            galaxy_list.append(gals)
            
        else:
            print('file not found for redshift {:02.0f} in path \'{}\''.format(redshift, galfile_path))
    gal_header.append('Redshift')
            
    if galaxy_list:
       
        if equal_numbers:
            min_nr = 1e100
            for gals in galaxy_list:
                if np.shape(gals)[0] < min_nr:
                    min_nr = np.shape(gals)[0]
            galaxies = None
            for gals in galaxy_list:
                gals = gals[:min_nr, :]
                if galaxies is not None:
                    galaxies = np.concatenate((galaxies, gals), axis=0)
                else:
                    galaxies = gals
        else:
            galaxies = None
            for gals in galaxy_list:
                if galaxies is not None:
                    galaxies = np.concatenate((galaxies, gals), axis=0)
                else:
                    galaxies = gals

        # Scramble the order of the galaxies, since the redshifts are still in order
        inds = np.random.permutation(np.shape(galaxies)[0])
        galaxies = galaxies[inds, :]
        
        data_keys = {}
        for col_nr, key in enumerate(gal_header):
            data_keys[key] = col_nr

        unit_dict = {'X_pos': '', 'Y_pos': '', 'Z_pos': '', 'X_vel': '', 'Y_vel': '', 
                 'Z_vel': '', 'Halo_mass': 'M_{H}/M_{\odot}', 'Stellar_mass': r'm_{\ast}/M_{\odot}',
                 'SFR': 'M_{\odot}yr^{-1}', 
                 'Intra_cluster_mass': '', 'Halo_mass_peak': 'M_{G}/M_{\odot}', 
                 'Stellar_mass_obs': '', 'SFR_obs': '', 'Halo_radius': '', 
                 'Concentration': '', 'Halo_spin': '', 'Scale_peak_mass': 'a', 
                 'Scale_half_mass': 'a', 'Scale_last_MajM': 'a', 'Type': '',
                 'Environmental_density': 'log($M_{G}/M_{S}/Mpc^3$)', 'Redshift': 'z'}
        
        ### Remove data points with halo mass below 10.5
        galaxies = galaxies[galaxies[:,data_keys['Halo_mass']] > 10.5, :]
        
        return galaxies, data_keys, unit_dict
    else:
        print('No files with the specified redshifts found.')
        return 

def load_single_galfile(redshift, with_densities=True, with_growth=True):
    
    if with_growth:
        galfile_directory = '/home/magnus/data/galcats_nonzero_sfr_no_density_with_growth_rate_no_lastMajM/'
    elif with_densities:
        galfile_directory = '/home/magnus/data/galcats_nonzero_sfr_with_density/'
    else:
        galfile_directory = '/home/magnus/data/galcats_nonzero_sfr_no_density/'
    
    galfile_path = galfile_directory + 'galaxies.Z{:02d}.h5'.format(redshift)
    galfile = pd.read_hdf(galfile_path)
    galaxies = galfile.values
    gal_header = galfile.keys().tolist()
    
    data_keys = {}
    for col_nr, key in enumerate(gal_header):
        data_keys[key] = col_nr
            
    unit_dict = {'X_pos': '', 'Y_pos': '', 'Z_pos': '', 'X_vel': '', 'Y_vel': '', 
                 'Z_vel': '', 'Halo_mass': 'M_{H}/M_{\odot}', 'Stellar_mass': r'm_{\ast}/M_{\odot}',
                 'SFR': 'M_{\odot}yr^{-1}', 
                 'Intra_cluster_mass': '', 'Halo_mass_peak': 'M_{G}/M_{\odot}', 
                 'Stellar_mass_obs': '', 'SFR_obs': '', 'Halo_radius': '', 
                 'Concentration': '', 'Halo_spin': '', 'Scale_peak_mass': 'a', 
                 'Scale_half_mass': 'a', 'Scale_last_MajM': 'a', 'Type': '',
                 'Environmental_density': 'log($M_{G}/M_{S}/Mpc^3$)', 'Redshift': 'z'}
    
    print('shape before modification: ',np.shape(galaxies))
    ### Remove data points with halo mass below 10.5
    galaxies = galaxies[galaxies[:,data_keys['Halo_mass']] > 10.5, :]
    
    # Scramble the order of the galaxies, since they may be somewhat ordered to begin with
    inds = np.random.permutation(np.shape(galaxies)[0])
    galaxies = galaxies[inds, :]
    redshift_column = redshift * np.ones((np.shape(galaxies)[0],1))
    galaxies = np.concatenate((galaxies, redshift_column), axis=1)
    
    print('shape after removing small galaxies and adding redshift: ',np.shape(galaxies))
    
    if np.shape(galaxies)[1] == 22:
        data_keys['Environmental_density'] = 20
        unit_dict['Environmental_density'] = 'log($M_{G}/M_{S}/Mpc^3$)'
        data_keys['Redshift'] = 21
        unit_dict['Redshift'] = 'z'
    else:
        data_keys['Redshift'] = 20
        unit_dict['Redshift'] = 'z'
        
    return galaxies, data_keys, unit_dict


def divide_train_data(galaxies, data_keys, input_features, output_features, redshifts, total_set_size, train_size=0, val_size=0,
                      test_size=0, k_fold_cv=False, tot_cv_folds=0, cv_fold_nr=0):
    
    n_data_points = galaxies.shape[0]
    
    if k_fold_cv:
        subset_indices = np.random.choice(n_data_points, total_set_size, replace=False)
        fold_size = int(total_set_size / tot_cv_folds)
        val_indices = subset_indices[cv_fold_nr*fold_size : (cv_fold_nr+1)*fold_size]
        train_indices = np.concatenate((subset_indices[:cv_fold_nr*fold_size], subset_indices[(cv_fold_nr+1)*fold_size:]))
        test_indices = [0]
        
    else:
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
        'original_data_keys': data_keys,
        'redshifts': redshifts
    }
    
    return training_data_dict


def normalise_data(training_data_dict, norm):
    
    training_data_dict['norm'] = norm
    
    x_train = training_data_dict['x_train']
    x_val = training_data_dict['x_val']
    x_test = training_data_dict['x_test']
    y_train = training_data_dict['y_train']
    y_val = training_data_dict['y_val']
    y_test = training_data_dict['y_test']
    
    input_train_dict = {}
    input_val_dict = {}
    input_test_dict = {}
    
    output_train_dict = {}
    output_val_dict = {}
    output_test_dict = {}
    
    if norm['input'] == 'none':
        
        input_train_dict['main_input'] = x_train
        input_val_dict['main_input'] = x_val
        input_test_dict['main_input'] = x_test
        
    elif norm['input'] == 'zero_mean_unit_std':

        x_data_means = np.mean(x_train, 0)
        x_data_stds = np.std(x_train, 0)

        x_train_norm = (x_train - x_data_means) / x_data_stds
        x_val_norm = (x_val - x_data_means) / x_data_stds
        x_test_norm = (x_test - x_data_means) / x_data_stds
        
        input_train_dict['main_input'] = x_train_norm
        input_val_dict['main_input'] = x_val_norm
        input_test_dict['main_input'] = x_test_norm
        
        training_data_dict['x_data_means'] = x_data_means
        training_data_dict['x_data_stds'] = x_data_stds
        
    elif norm['input'] == 'zero_to_one':

        x_data_max = np.max(x_train, 0)
        x_data_min = np.min(x_train, 0)

        x_train_norm = (x_train - x_data_min) / (x_data_max - x_data_min)
        x_val_norm = (x_val - x_data_min) / (x_data_max - x_data_min)
        x_test_norm = (x_test - x_data_min) / (x_data_max - x_data_min)
        
        input_train_dict['main_input'] = x_train_norm
        input_val_dict['main_input'] = x_val_norm
        input_test_dict['main_input'] = x_test_norm
        
        training_data_dict['x_data_max'] = x_data_max
        training_data_dict['x_data_min'] = x_data_min
        
    else:
        print('Incorrect norm provided: ', norm)  
    
    if 'Redshift' in training_data_dict['input_features']:
        input_train_dict['main_input'][:, training_data_dict['x_data_keys']['Redshift']] = x_train[:, 
                                                            training_data_dict['x_data_keys']['Redshift']] / 100
        input_val_dict['main_input'][:, training_data_dict['x_data_keys']['Redshift']] = x_val[:, 
                                                            training_data_dict['x_data_keys']['Redshift']] / 100
        input_test_dict['main_input'][:, training_data_dict['x_data_keys']['Redshift']] = x_test[:, 
                                                            training_data_dict['x_data_keys']['Redshift']] / 100
        
    if norm['output'] == 'none':
        
        for i_feat, feat in enumerate(training_data_dict['output_features']):
            output_train_dict[feat] = y_train[:, i_feat]
            output_val_dict[feat] = y_val[:, i_feat]
            output_test_dict[feat] = y_test[:, i_feat]
    
    elif norm['output'] == 'zero_mean_unit_std':
            
        y_data_means = np.mean(y_train, 0)
        y_data_stds = np.std(y_train, 0)

        y_train_norm = (y_train - y_data_means) / y_data_stds
        y_val_norm = (y_val - y_data_means) / y_data_stds
        y_test_norm = (y_test - y_data_means) / y_data_stds
            
        for i_feat, feat in enumerate(training_data_dict['output_features']):
            output_train_dict[feat] = y_train_norm[:, i_feat]
            output_val_dict[feat] = y_val_norm[:, i_feat]
            output_test_dict[feat] = y_test_norm[:, i_feat]
                        
        training_data_dict['y_data_means'] = y_data_means
        training_data_dict['y_data_stds'] = y_data_stds
        

    elif norm['output'] == 'zero_to_one':
        
        y_data_max = np.max(y_train, 0)
        y_data_min = np.min(y_train, 0)

        y_train_norm = (y_train - y_data_min) / (y_data_max - y_data_min)
        y_val_norm = (y_val - y_data_min) / (y_data_max - y_data_min)
        y_test_norm = (y_test - y_data_min) / (y_data_max - y_data_min)
        
        for i_feat, feat in enumerate(training_data_dict['output_features']):
            output_train_dict[feat] = y_train_norm[:, i_feat]
            output_val_dict[feat] = y_val_norm[:, i_feat]
            output_test_dict[feat] = y_test_norm[:, i_feat]
            
        training_data_dict['y_data_max'] = y_data_max
        training_data_dict['y_data_min'] = y_data_min
       
    else:
        print('Incorrect norm provided: ', norm)    
        
    training_data_dict['input_train_dict'] = input_train_dict
    training_data_dict['input_val_dict'] = input_val_dict
    training_data_dict['input_test_dict'] = input_test_dict
    training_data_dict['output_train_dict'] = output_train_dict
    training_data_dict['output_val_dict'] = output_val_dict
    training_data_dict['output_test_dict'] = output_test_dict
    
    
    return training_data_dict


def get_test_score(model, training_data_dict, norm):
    ### Get the MSE for the test predictions in the original units of the dataset
    
    predicted_points = predict_points(training_data_dict, data_type = 'test')

    ### Get mse for the real predictions
    
    n_points, n_outputs = np.shape(predicted_points)
    x_minus_y = predicted_points - training_data_dict['y_test']

    feature_scores = np.sum(np.power(x_minus_y, 2), 0) / n_points
    test_score = np.sum(feature_scores) / n_outputs
    
    return test_score


def predict_points(model, training_data_dict, data_type='test'):

    if data_type == 'train':
        predicted_norm_points = model.predict(training_data_dict['input_train_dict'])
    elif data_type == 'val':
        predicted_norm_points = model.predict(training_data_dict['input_val_dict'])
    elif data_type == 'test':
        predicted_norm_points = model.predict(training_data_dict['input_test_dict'])
    else:
        print('Please enter a valid data type (\'train\', \'val\' or \'test\')')
        
    if len(training_data_dict['output_features']) == 1:

        if training_data_dict['norm']['output'] == 'zero_mean_unit_std':
            for i in range(len(training_data_dict['output_features'])):
                predicted_points = predicted_norm_points * training_data_dict['y_data_stds'] + \
                                       training_data_dict['y_data_means']

        elif training_data_dict['norm']['output'] == 'zero_to_one':
            for i in range(len(training_data_dict['output_features'])):
                predicted_points = predicted_norm_points * (training_data_dict['y_data_max'] - \
                                       training_data_dict['y_data_min']) + training_data_dict['y_data_min']

        elif training_data_dict['norm']['output'] == 'none':
            predicted_points = predicted_norm_points
        
        
    else:
        
        predicted_points = []
        if training_data_dict['norm']['output'] == 'zero_mean_unit_std':
            for i in range(len(training_data_dict['output_features'])):
                predicted_points.append(predicted_norm_points[i] * training_data_dict['y_data_stds'][i] + 
                                       training_data_dict['y_data_means'][i])

        elif training_data_dict['norm']['output'] == 'zero_to_one':
            for i in range(len(training_data_dict['output_features'])):
                predicted_points.append(predicted_norm_points[i] * (training_data_dict['y_data_max'][i] - 
                                       training_data_dict['y_data_min'][i]) + training_data_dict['y_data_min'][i])

        elif training_data_dict['norm']['output'] == 'none':
            predicted_points = predicted_norm_points
            
        predicted_points = np.asarray(predicted_points)
        predicted_points = np.squeeze(predicted_points, axis = -1)
        predicted_points = np.transpose(predicted_points)  
        
        
    return predicted_points

def get_weights(training_data_dict, output_features, outputs_to_weigh, weigh_by_redshift=False):
    
    unique_redshifts = training_data_dict['redshifts']
    train_w_tmp = np.zeros(len(training_data_dict['train_indices']))
    val_w_tmp = np.zeros(len(training_data_dict['val_indices']))
    test_w_tmp = np.zeros(len(training_data_dict['test_indices']))

    # make the heavier halos more important in every redshift
    for redshift in unique_redshifts:
        relevant_train_inds = training_data_dict['original_data'][training_data_dict['train_indices'], 
                                                              training_data_dict['original_data_keys']['Redshift']] == redshift
        relevant_val_inds = training_data_dict['original_data'][training_data_dict['val_indices'], 
                                                              training_data_dict['original_data_keys']['Redshift']] == redshift
        relevant_test_inds = training_data_dict['original_data'][training_data_dict['test_indices'], 
                                                              training_data_dict['original_data_keys']['Redshift']] == redshift

        train_masses = training_data_dict['original_data'][training_data_dict['train_indices'],
                                                           training_data_dict['original_data_keys']['Halo_mass']]
        train_w_redshift = train_masses[relevant_train_inds]
        train_w_redshift = np.power(10, train_w_redshift)
        train_w_redshift = train_w_redshift / np.sum(train_w_redshift) 
        
        val_masses = training_data_dict['original_data'][training_data_dict['val_indices'],
                                                           training_data_dict['original_data_keys']['Halo_mass']]
        val_w_redshift = val_masses[relevant_val_inds]
        val_w_redshift = np.power(10, val_w_redshift)
        val_w_redshift = val_w_redshift / np.sum(val_w_redshift) 
        
        test_masses = training_data_dict['original_data'][training_data_dict['test_indices'],
                                                           training_data_dict['original_data_keys']['Halo_mass']]
        test_w_redshift = test_masses[relevant_test_inds]
        test_w_redshift = np.power(10, test_w_redshift)
        test_w_redshift = test_w_redshift / np.sum(test_w_redshift) 
        
        train_w_tmp[relevant_train_inds] = train_w_redshift
        val_w_tmp[relevant_val_inds] = val_w_redshift
        test_w_tmp[relevant_test_inds] = test_w_redshift
    
    train_w_tmp = train_w_tmp / np.sum(train_w_tmp) 
    val_w_tmp = val_w_tmp / np.sum(val_w_tmp) 
    test_w_tmp = test_w_tmp / np.sum(test_w_tmp) 
    
    # check that weights sum up to one
    sums = np.array([np.sum(train_w_tmp), np.sum(val_w_tmp), np.sum(test_w_tmp)])
    errs = np.abs(1 - sums)
    too_high = errs > 1e-6
    if any(too_high):
        print('The weights were not normalised properly. Sums should be one: {}, {}, {}'.format(sums[0], sums[1], sums[2]))
        return
    
    train_weights = {}
    val_weights = {}
    test_weights = {}

    for output in output_features:
        if output in outputs_to_weigh:
            train_weights[output] = train_w_tmp
            val_weights[output] = val_w_tmp
            test_weights[output] = test_w_tmp
        else:
            train_weights[output] = np.ones(int(len(training_data_dict['train_indices']))) / len(training_data_dict['train_indices'])
            val_weights[output] = np.ones(int(len(training_data_dict['val_indices']))) / len(training_data_dict['val_indices'])
            test_weights[output] = np.ones(int(len(training_data_dict['test_indices']))) / len(training_data_dict['test_indices'])
            
    if weigh_by_redshift:
        train_redshifts = training_data_dict['input_train_dict']['main_input'][:, training_data_dict['x_data_keys']['Redshift']]
        val_redshifts = training_data_dict['input_val_dict']['main_input'][:, training_data_dict['x_data_keys']['Redshift']]
        test_redshifts = training_data_dict['input_test_dict']['main_input'][:, training_data_dict['x_data_keys']['Redshift']]
        
        train_unique_redshifts = np.unique(train_redshifts)
        val_unique_redshifts = np.unique(val_redshifts)
        test_unique_redshifts = np.unique(test_redshifts)
        
        train_redshift_weights = np.zeros(len(train_redshifts))
        val_redshift_weights = np.zeros(len(val_redshifts))
        test_redshift_weights = np.zeros(len(test_redshifts))
        for redshift in train_unique_redshifts:
            weight = 1 / (len(train_unique_redshifts) * np.sum(train_redshifts == redshift))
            train_redshift_weights[train_redshifts == redshift] = weight
        for redshift in val_unique_redshifts:
            weight = 1 / (len(val_unique_redshifts) * np.sum(val_redshifts == redshift))
            val_redshift_weights[val_redshifts == redshift] = weight
        for redshift in test_unique_redshifts:
            weight = 1 / (len(test_unique_redshifts) * np.sum(test_redshifts == redshift))
            test_redshift_weights[test_redshifts == redshift] = weight
        
        for output in output_features:
            train_weights[output] = (train_weights[output] + train_redshift_weights) / 2
            val_weights[output] = (val_weights[output] + val_redshift_weights) / 2
            test_weights[output] = (test_weights[output] + test_redshift_weights) / 2
              
    # check that weights sum up to one
    sums = np.array([np.sum(train_weights['Stellar_mass']), np.sum(train_weights['SFR']), np.sum(val_weights['Stellar_mass']),
                     np.sum(val_weights['SFR']), np.sum(test_weights['Stellar_mass']), np.sum(test_weights['SFR'])])
    errs = np.abs(1 - sums)
    too_high = errs > 1e-6
    if any(too_high):
        print('The weights were not normalised properly. Sums should be one: {}, {}, {}, {}, {}, {}'.format(sums[0], sums[1], sums[2],
                                                                                                     sums[3], sums[4], sums[5]))
        return
            
    return [train_weights, val_weights, test_weights]


def get_weights_old(training_data_dict, output_features, outputs_to_weigh, weigh_by_redshift=False):
    train_w_tmp = training_data_dict['original_data'][training_data_dict['train_indices'], 
                                        training_data_dict['original_data_keys']['Halo_mass']]
    train_w_tmp = np.power(10, train_w_tmp)
    train_w_tmp = train_w_tmp / np.sum(train_w_tmp)
    val_w_tmp = training_data_dict['original_data'][training_data_dict['val_indices'], 
                                        training_data_dict['original_data_keys']['Halo_mass']]
    val_w_tmp = np.power(10, val_w_tmp)
    val_w_tmp = val_w_tmp / np.sum(val_w_tmp)
    train_weights = {}
    val_weights = {}

    for output in output_features:
        if output in outputs_to_weigh:
            train_weights[output] = train_w_tmp
            val_weights[output] = val_w_tmp
        else:
            train_weights[output] = np.ones(int(len(training_data_dict['train_indices']))) / len(training_data_dict['train_indices'])
            val_weights[output] = np.ones(int(len(training_data_dict['val_indices']))) / len(training_data_dict['val_indices'])
            
    if weigh_by_redshift:
        train_redshifts = training_data_dict['input_train_dict']['main_input'][:, training_data_dict['x_data_keys']['Redshift']]
        val_redshifts = training_data_dict['input_val_dict']['main_input'][:, training_data_dict['x_data_keys']['Redshift']]
        
        train_unique_redshifts = np.unique(train_redshifts)
        val_unique_redshifts = np.unique(val_redshifts)
        
        train_redshift_weights = np.zeros(len(train_redshifts))
        val_redshift_weights = np.zeros(len(val_redshifts))
        for redshift in train_unique_redshifts:
            weight = 1 / (len(train_unique_redshifts) * np.sum(train_redshifts == redshift))
            train_redshift_weights[train_redshifts == redshift] = weight
        for redshift in val_unique_redshifts:
            weight = 1 / (len(val_unique_redshifts) * np.sum(val_redshifts == redshift))
            val_redshift_weights[val_redshifts == redshift] = weight
        
        for output in output_features:
            train_weights[output] = (train_weights[output] + train_redshift_weights) / 2
            val_weights[output] = (val_weights[output] + val_redshift_weights) / 2

            
    return [train_weights, val_weights]













