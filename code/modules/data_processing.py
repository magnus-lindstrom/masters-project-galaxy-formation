import pandas as pd
import numpy as np
import os.path
import json
import sys
from scipy.stats import binned_statistic


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

        ### Remove data points with halo mass below 10.5
        galaxies = galaxies[galaxies[:,data_keys['Halo_mass']] > 10.5, :]
        
        return galaxies, data_keys
    else:
        print('No files with the specified redshifts found.')
        return 

def load_single_galfile(redshift, with_densities=False, with_growth=True):
    
    if with_growth:
        galfile_directory = '/home/magnus/data/galcats_nonzero_sfr_no_density_with_growth_rate_no_lastMajM/'
    elif with_densities:
        galfile_directory = '/home/magnus/data/galcats_nonzero_sfr_with_density/'
    else:
        galfile_directory = '/home/magnus/data/galcats_nonzero_sfr_no_density/'
    
    galfile_path = galfile_directory + 'galaxies.Z{:02d}.h5'.format(redshift*10)
    galfile = pd.read_hdf(galfile_path)
    galaxies = galfile.values
    gal_header = galfile.keys().tolist()
    
    data_keys = {}
    for col_nr, key in enumerate(gal_header):
        data_keys[key] = col_nr
    
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
        
    return galaxies, data_keys


def get_unit_dict():
    
    unit_dict = {'X_pos': '', 'Y_pos': '', 'Z_pos': '', 'X_vel': '', 'Y_vel': '', 
                 'Z_vel': '', 'Halo_mass': 'M_{H}/M_{\odot}', 'Stellar_mass': r'm_{\ast}/M_{\odot}',
                 'SFR': 'M_{\odot}yr^{-1}', 'SSFR': 'yr^{-1}', 'SMF': '\Phi / Mpc^{-3} dex^{-1}', 'FQ': 'f_q',
                 'Intra_cluster_mass': '', 'Halo_mass_peak': 'M_{G}/M_{\odot}', 
                 'Stellar_mass_obs': '', 'SFR_obs': '', 'Halo_radius': '', 
                 'Concentration': '', 'Halo_spin': '', 'Scale_peak_mass': 'a', 
                 'Scale_half_mass': 'a', 'Scale_last_MajM': 'a', 'Type': '',
                 'Environmental_density': 'log($M_{G}/M_{S}/Mpc^3$)', 'Redshift': 'z'}
    
    return unit_dict

def divide_train_data(galaxies, data_keys, input_features, output_features, redshifts, weigh_by_redshift=0, outputs_to_weigh=0, 
                      total_set_size=0, train_size=0, val_size=0,
                      test_size=0, k_fold_cv=False, tot_cv_folds=0, cv_fold_nr=0, pso=False, use_emerge_targets=False):
    
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
    
    data_redshifts = {}
    
    for i in range(len(input_features)):
        x_train[:,i] = galaxies[train_indices, data_keys[input_features[i]]]
        x_val[:,i] = galaxies[val_indices, data_keys[input_features[i]]]
        x_test[:,i] = galaxies[test_indices, data_keys[input_features[i]]]

    data_redshifts['train_data'] = galaxies[train_indices, data_keys['Redshift']]
    data_redshifts['val_data'] = galaxies[val_indices, data_keys['Redshift']]
    data_redshifts['test_data'] = galaxies[test_indices, data_keys['Redshift']]
    
    original_nr_data_points_by_redshift = []
    
    for redshift in redshifts:
        original_nr_data_points_by_redshift.append(np.sum(galaxies[:, data_keys['Redshift']] == redshift))
    
    training_data_dict = {
        'output_features': output_features,
        'input_features': input_features,
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'original_halo_masses_train': galaxies[train_indices, data_keys['Halo_mass']],
        'original_halo_masses_val': galaxies[val_indices, data_keys['Halo_mass']],
        'original_halo_masses_test': galaxies[test_indices, data_keys['Halo_mass']],
        'original_nr_data_points_by_redshift': original_nr_data_points_by_redshift,
        'unique_redshifts': redshifts,
        'data_redshifts': data_redshifts
    }
    
    if use_emerge_targets:
        
        y_train = np.zeros((len(train_indices), len(output_features)))
        y_val = np.zeros((len(val_indices), len(output_features)))
        y_test = np.zeros((len(test_indices), len(output_features)))
        
        for i in range(len(output_features)):
            y_train[:,i] = galaxies[train_indices, data_keys[output_features[i]]]
            y_val[:,i] = galaxies[val_indices, data_keys[output_features[i]]]
            y_test[:,i] = galaxies[test_indices, data_keys[output_features[i]]]
            
        training_data_dict['y_train'] = y_train
        training_data_dict['y_val'] = y_val
        training_data_dict['y_test'] = y_test
        
        train_weights, val_weights, test_weights = get_weights(training_data_dict, output_features, outputs_to_weigh, 
                                                               weigh_by_redshift, pso=pso)
        training_data_dict['train_weights'] = train_weights
        training_data_dict['val_weights'] = val_weights
        training_data_dict['test_weights'] = test_weights
        
    else:
                
        # store the relevant SSFR data
        ssfr_directory = '/home/magnus/data/mock_data/ssfr/'
        ssfr_data = {}
        for redshift in training_data_dict['unique_redshifts']:
            
            file_name = 'galaxies.Z{:02.0f}'.format(redshift*10)
            with open(ssfr_directory + file_name + '.json', 'r') as f:
                ssfr = json.load(f)
            
            parameter_dict = ssfr.pop(0)
            bin_widths = parameter_dict['bin_widths']
            bin_edges = parameter_dict['bin_edges']
            
            bin_centers = [item[0] for item in ssfr]
            mean_ssfr = [item[1] for item in ssfr]
            errors = [item[2] for item in ssfr]
            
            redshift_data = {
                'bin_centers': np.array(bin_centers),
                'bin_widths': np.array(bin_widths),
                'bin_edges': np.array(bin_edges),
                'ssfr': np.array(mean_ssfr),
                'errors': np.array(errors)
            }
                
            ssfr_data['{:.1f}'.format(redshift)] = redshift_data
            
        training_data_dict['ssfr_data'] = ssfr_data
        
        # store the relevant SMF data
        smf_directory = '/home/magnus/data/mock_data/stellar_mass_functions/'
        smf_data = {}
        
        for redshift in training_data_dict['unique_redshifts']:
            
            file_name = 'galaxies.Z{:02.0f}'.format(redshift*10)
            with open(smf_directory + file_name + '.json', 'r') as f:
                smf_list = json.load(f)
                
            parameter_dict = smf_list.pop(0)
            bin_widths = parameter_dict['bin_widths']
            bin_edges = parameter_dict['bin_edges']
            
            bin_centers = [item[0] for item in smf_list]
            smf = [item[1] for item in smf_list]
            errors = [item[2] for item in smf_list]
            
            redshift_data = {
                'bin_centers': np.array(bin_centers),
                'bin_widths': np.array(bin_widths),
                'bin_edges': np.array(bin_edges),
                'smf': np.array(smf),
                'errors': np.array(errors)
            }
            
            smf_data['{:.1f}'.format(redshift)] = redshift_data
            
        training_data_dict['smf_data'] = smf_data
        
        # store the relevant FQ data
        fq_directory = '/home/magnus/data/mock_data/fq/'
        fq_data = {}
        
        for redshift in training_data_dict['unique_redshifts']:
            
            file_name = 'galaxies.Z{:02.0f}'.format(redshift*10)
            with open(fq_directory + file_name + '.json', 'r') as f:
                fq_list = json.load(f)
                
            parameter_dict = fq_list.pop(0)
            bin_widths = parameter_dict['bin_widths']
            bin_edges = parameter_dict['bin_edges']
            
            bin_centers = [item[0] for item in fq_list]
            fq = [item[1] for item in fq_list]
            errors = [item[2] for item in fq_list]
            
            redshift_data = {
                'bin_centers': np.array(bin_centers),
                'bin_widths': np.array(bin_widths),
                'bin_edges': np.array(bin_edges),
                'fq': np.array(fq),
                'errors': np.array(errors)
            }
            
            fq_data['{:.1f}'.format(redshift)] = redshift_data
            
        training_data_dict['fq_data'] = fq_data
        
        # store the relevant SHM data
        shm_directory = '/home/magnus/data/mock_data/stellar_halo_mass_relations/'
        shm_data = {}
        
        for redshift in training_data_dict['unique_redshifts']:
            
            file_name = 'galaxies.Z{:02.0f}'.format(redshift*10)
            with open(shm_directory + file_name + '.json', 'r') as f:
                shm_list = json.load(f)
                
            parameter_dict = shm_list.pop(0)
            bin_widths = parameter_dict['bin_widths']
            bin_edges = parameter_dict['bin_edges']
            
            bin_centers = [item[0] for item in shm_list]
            shm = [item[1] for item in shm_list]
            errors = [item[2] for item in shm_list]
            
            redshift_data = {
                'bin_centers': np.array(bin_centers),
                'bin_widths': np.array(bin_widths),
                'bin_edges': np.array(bin_edges),
                'shm': np.array(shm),
                'errors': np.array(errors)
            }
            
            shm_data['{:.1f}'.format(redshift)] = redshift_data
            
        training_data_dict['shm_data'] = shm_data
    
    return training_data_dict


def convert_units(data, norm, back_to_original=False, conv_values=None):
    
    if back_to_original:
        
        if norm == 'none':
            return data
        elif norm == 'zero_mean_unit_std':
            if conv_values is None:
                print('provide conversion values')
                return
            else:
                data_means = conv_values['data_means']
                data_stds = conv_values['data_stds']
                            
            data_orig = data * data_stds + data_means
            return data_orig
        
        elif norm == 'zero_to_one':
            if conv_values is None:
                print('provide conversion values')
                return
            else:
                data_max = conv_values['data_max']
                data_min = conv_values['data_min']
                
            data_orig = data * (data_max - data_min) + data_min
            
            return data_orig
        else:
            print('invalid norm provided: {}'.format(norm))
            return
    
    else:
        
        if norm == 'none':
            return data
        elif norm == 'zero_mean_unit_std':
            
            if conv_values is None:
                data_means = np.mean(data, 0)
                data_stds = np.std(data, 0)
                conv_values = {
                    'data_means': data_means,
                    'data_stds': data_stds
                }
#                 print('data: ',np.shape(data))
#                 print('data_means: ',np.shape(data_means))
#                 print('data_stds: ',np.shape(data_stds))
                data_norm = (data - data_means) / data_stds
                return [data_norm, conv_values]
            else:
                data_means = conv_values['data_means']
                data_stds = conv_values['data_stds']
#                 print('data: ',np.shape(data))
#                 print('data_means: ',np.shape(data_means))
#                 print('data_stds: ',np.shape(data_stds))
                data_norm = (data - data_means) / data_stds
                return data_norm
                    
        elif norm == 'zero_to_one':
            if conv_values is None:
                data_max = np.max(data, 0)
                data_min = np.min(data, 0)
                conv_values = {
                    'data_max': data_max,
                    'data_min': data_min
                }
                data_norm = (data - data_min) / (data_max - data_min)
                return [data_norm, conv_values]
            else:
                data_max = conv_values['data_max']
                data_min = conv_values['data_min']
                data_norm = (data - data_min) / (data_max - data_min)
                return data_norm
            
        else:
            print('invalid norm provided: {}'.format(norm))
            return

def normalise_data(training_data_dict, norm, pso=False):
    
    training_data_dict['norm'] = norm
    
    x_train = training_data_dict['x_train']
    x_val = training_data_dict['x_val']
    x_test = training_data_dict['x_test']

    input_train_dict = {}
    input_val_dict = {}
    input_test_dict = {}
    
    # convert units based on the train data only
    x_train_norm, conv_values_input = convert_units(x_train, norm['input'])  
    x_val_norm = convert_units(x_val, norm['input'], conv_values=conv_values_input)
    x_test_norm = convert_units(x_test, norm['input'], conv_values=conv_values_input)
    
    input_train_dict['main_input'] = x_train_norm
    input_val_dict['main_input'] = x_val_norm
    input_test_dict['main_input'] = x_test_norm
    
    training_data_dict['input_train_dict'] = input_train_dict
    training_data_dict['input_val_dict'] = input_val_dict
    training_data_dict['input_test_dict'] = input_test_dict

    training_data_dict['conv_values_input'] = conv_values_input
        
    del training_data_dict["x_train"]
    del training_data_dict["x_val"]
    del training_data_dict["x_test"]
    
    if 'y_train' in training_data_dict:
        
        y_train = training_data_dict['y_train']
        y_val = training_data_dict['y_val']
        y_test = training_data_dict['y_test']

        output_train_dict = {}
        output_val_dict = {}
        output_test_dict = {}

        # convert units based on the train data only
        y_train_norm, conv_values_output = convert_units(y_train, norm['output'])  
        y_val_norm = convert_units(y_val, norm['output'], conv_values=conv_values_output)
        y_test_norm = convert_units(y_test, norm['output'], conv_values=conv_values_output)
        
        training_data_dict['conv_values_output'] = conv_values_output
    
        if pso:

            training_data_dict['y_train'] = y_train_norm
            training_data_dict['y_val'] = y_val_norm
            training_data_dict['y_test'] = y_test_norm

        if not pso:       # if trained with backpropagation

            for i_feat, feat in enumerate(training_data_dict['output_features']):
                output_train_dict[feat] = y_train_norm[:, i_feat]
                output_val_dict[feat] = y_val_norm[:, i_feat]
                output_test_dict[feat] = y_test_norm[:, i_feat]

          #  todo: have to keep the real halo weights and the true output in training data dict....

            del training_data_dict["y_train"]
            del training_data_dict["y_val"]
            del training_data_dict["y_test"]

            training_data_dict['output_train_dict'] = output_train_dict
            training_data_dict['output_val_dict'] = output_val_dict
            training_data_dict['output_test_dict'] = output_test_dict

    training_data_dict['norm'] = norm
        
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


def predict_points(model, training_data_dict, original_units=True, as_lists=False, data_type='test'):

    predicted_norm_points = model.predict(training_data_dict['input_{}_dict'.format(data_type)])
    
    if type(predicted_norm_points) is list:
        predicted_norm_points = np.asarray(predicted_norm_points)
        predicted_norm_points = np.squeeze(predicted_norm_points, axis = -1)
        predicted_norm_points = np.transpose(predicted_norm_points)
#     print(np.shape(predicted_norm_points))
        
#     if len(training_data_dict['output_features']) == 1:
        
    if original_units:
        predicted_points = convert_units(predicted_norm_points, training_data_dict['norm']['output'], 
                                         back_to_original=True, conv_values=training_data_dict['conv_values_output'])
    else:
        predicted_points = predicted_norm_points

#     else:

#         for i in range(len(training_data_dict['output_features'])):
#             pred_feat_points = predicted_norm_points[i]
#             convert_units(predicted_norm_points[i], training_data_dict['norm']['output'], 
#                                          back_to_original=original_units, conv_values=training_data_dict['conv_values'])
            
#             predicted_points.append(predicted_norm_points[i] * training_data_dict['y_data_stds'][i] + 
#                                    training_data_dict['y_data_means'][i])

            
#         if not as_lists:
#             predicted_points = np.asarray(predicted_points)
#             predicted_points = np.squeeze(predicted_points, axis = -1)
#             predicted_points = np.transpose(predicted_points)          
        
    return predicted_points
    

def get_weights(training_data_dict, output_features, outputs_to_weigh, weigh_by_redshift=False, pso=False):
    
    unique_redshifts = training_data_dict['unique_redshifts']
    train_w_tmp = np.zeros(len(training_data_dict['train_indices']))
    val_w_tmp = np.zeros(len(training_data_dict['val_indices']))
    test_w_tmp = np.zeros(len(training_data_dict['test_indices']))
    
    halo_mass_index = training_data_dict['input_features'].index('Halo_mass')

    # make the heavier halos more important in every redshift
    for redshift in unique_redshifts:
        relevant_train_inds = training_data_dict['data_redshifts']['train_data'] == redshift
        relevant_val_inds = training_data_dict['data_redshifts']['val_data'] == redshift
        relevant_test_inds = training_data_dict['data_redshifts']['test_data'] == redshift

#         if pso:
        train_masses = training_data_dict['x_train'][:, halo_mass_index]
        val_masses = training_data_dict['x_val'][:, halo_mass_index]
        test_masses = training_data_dict['x_test'][:, halo_mass_index]
     
#         else:
#             train_masses = training_data_dict['input_train_dict']['main_input'][:, halo_mass_index]
#             val_masses = training_data_dict['input_val_dict']['main_input'][:, halo_mass_index]
#             test_masses = training_data_dict['input_test_dict']['main_input'][:, halo_mass_index]
                
        train_w_redshift = train_masses[relevant_train_inds]
        train_w_redshift = np.power(10, train_w_redshift)
        train_w_redshift = train_w_redshift / np.sum(train_w_redshift) 

        val_w_redshift = val_masses[relevant_val_inds]
        val_w_redshift = np.power(10, val_w_redshift)
        val_w_redshift = val_w_redshift / np.sum(val_w_redshift) 
        
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
        train_redshifts = training_data_dict['data_redshifts']['train_data']
        val_redshifts = training_data_dict['data_redshifts']['val_data']
        test_redshifts = training_data_dict['data_redshifts']['test_data']
        
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
    sums = np.array([[np.sum(train_weights[key]), np.sum(val_weights[key]), np.sum(test_weights[key])] for key in output_features])
#     sums = np.array([np.sum(train_weights['Stellar_mass']), np.sum(train_weights['SFR']), np.sum(val_weights['Stellar_mass']),
#                      np.sum(val_weights['SFR']), np.sum(test_weights['Stellar_mass']), np.sum(test_weights['SFR'])])

    errs = np.squeeze(np.abs(1 - sums))
    too_high = np.any(errs > 1e-6)
    if too_high:
        print('The weights were not normalised properly. Sums should be one: {}, {}, {}, {}, {}, {}'.format(sums[0], sums[1], sums[2],
                                                                                                     sums[3], sums[4], sums[5]))
        return
    
    if pso:
        train_weights_tmp = np.zeros(np.shape(training_data_dict['y_train']))
        val_weights_tmp = np.zeros(np.shape(training_data_dict['y_val']))
        test_weights_tmp = np.zeros(np.shape(training_data_dict['y_test']))
        
        for i_key, key in enumerate(train_weights.keys()):
            train_weights_tmp[:, i_key] = train_weights[key]
            val_weights_tmp[:, i_key] = val_weights[key]
            test_weights_tmp[:, i_key] = test_weights[key]
        
        train_weights = train_weights_tmp
        val_weights = val_weights_tmp
        test_weights = test_weights_tmp
                    
    return [train_weights, val_weights, test_weights]


def get_weights_old(training_data_dict, output_features, outputs_to_weigh, weigh_by_redshift=False):
    
    unique_redshifts = training_data_dict['unique_redshifts']
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
        print('The weights were not normalised properly. Sums should be one: {}, {}, {}, {}, {}, {}'.format(sums[0], sums[1], 
                                                                                                            sums[2], sums[3], 
                                                                                                            sums[4], sums[5]))
        return
            
    return [train_weights, val_weights, test_weights]


def binned_loss(training_data_dict, binning_feat, bin_feat, bin_feat_name, data_type, loss_dict):
    
    loss = 0
    dist_outside = 0
#     nr_empty_bins = np.zeros(len(training_data_dict['unique_redshifts']))
    tot_nr_points = 0

    for i_red, redshift in enumerate(training_data_dict['unique_redshifts']):

        relevant_inds = training_data_dict['data_redshifts']['{}_data'.format(data_type)] == redshift

        bin_edges = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges']

#         nr_points_outside_binning_feat_range = \
#             np.sum(binning_feat[relevant_inds] < 
#                   training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][0]) \
#           + np.sum(binning_feat[relevant_inds] > 
#                   training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][-1])
        tot_nr_points += len(binning_feat[relevant_inds])

        # sum up distances outside the accepted range
        inds_below = binning_feat[relevant_inds] < \
            training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][0]
        inds_above = binning_feat[relevant_inds] > \
            training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][-1]
        dist_outside += np.sum(training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][0] \
                                 - binning_feat[relevant_inds][inds_below])
        dist_outside += np.sum(binning_feat[relevant_inds][inds_above] \
                                 - training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][-1])

        true_bin_feat_dist = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)][bin_feat_name]
        errors = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['errors']

        n_bins = len(bin_edges)-1
        bin_means, bin_edges, bin_numbers = binned_statistic(binning_feat[relevant_inds], bin_feat[relevant_inds], 
                                           bins=bin_edges, statistic='mean')
    
        if bin_feat_name == 'smf':
            bin_counts = [np.sum(bin_numbers == i) for i in range(1, n_bins+1)]
            bin_counts = [float('nan') if count == 0 else count for count in bin_counts]
            bin_counts = np.array(bin_counts, dtype=np.float)
            
            bin_widths = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_widths']
            pred_bin_feat_dist = bin_counts / 200**3 / bin_widths
            
            # since we're only using a subset of the original data points, compensate for this
            pred_bin_feat_dist *= training_data_dict['original_nr_data_points_by_redshift'][i_red] \
                                / len(training_data_dict['{}_indices'.format(data_type)])
            
            non_nan_indeces = np.invert(np.isnan(pred_bin_feat_dist))
            pred_bin_feat_dist[non_nan_indeces] = np.log10(pred_bin_feat_dist[non_nan_indeces])
    
        elif bin_feat_name == 'fq':
            
            scale_factor = 1 / (1 + redshift)

            h_0 = 67.81 / (3.09e19) # 1/s
            h_0 = h_0 * 60 * 24 * 365 # 1/yr
            h_r = h_0 * np.sqrt(1e-3*scale_factor**(-4) + 0.308*scale_factor**(-3) + 0*scale_factor**(-2) + 0.692)
            ssfr_cutoff = 0.3*h_r
            log_ssfr_cutoff = np.log10(ssfr_cutoff)
            
            pred_bin_feat_dist = np.zeros(n_bins)
            for bin_num in range(1, n_bins+1):

                if len(bin_feat[bin_numbers == bin_num]) != 0:
                    fq = np.sum(bin_feat[bin_numbers == bin_num] < log_ssfr_cutoff) / len(bin_feat[bin_numbers == bin_num])
                else:
                    fq = float('nan')

                pred_bin_feat_dist[bin_num-1] = fq

            ### TODO: check if the following line makes sense. Do we ever have zeros? Not just NaNs?
            zero_indeces = [True if (frac == 0) else False for frac in pred_bin_feat_dist]
            pred_bin_feat_dist[zero_indeces] = 1e-5

            non_nan_indeces = np.invert(np.isnan(pred_bin_feat_dist))
            
        else:    
            pred_bin_feat_dist = bin_means
            non_nan_indeces = np.invert(np.isnan(pred_bin_feat_dist))

#         nr_empty_bins[i_red] = np.sum(np.invert(non_nan_indeces))

        if (np.sum(non_nan_indeces) > 0 and np.sum(non_nan_indeces)/n_bins > loss_dict['min_filled_bin_frac']) \
            or bin_feat_name == 'shm':
        
            loss += np.sum(np.power(true_bin_feat_dist[non_nan_indeces] - pred_bin_feat_dist[non_nan_indeces], 2) \
                            / errors[non_nan_indeces]) / n_bins
        elif np.sum(non_nan_indeces) > 0 and np.sum(non_nan_indeces)/n_bins < loss_dict['min_filled_bin_frac']:
            loss += (np.sum(np.power(true_bin_feat_dist[non_nan_indeces] - pred_bin_feat_dist[non_nan_indeces], 2) \
                            / errors[non_nan_indeces]) / n_bins) + np.sum(np.invert(non_nan_indeces))
        else:
            loss += 1000
         
        
    # Get the dist outside per redshift measured
    dist_outside /= len(training_data_dict['unique_redshifts'])
    
    if loss_dict != None:
    
        dist_outside_punish = loss_dict['dist_outside_punish']
        
        if dist_outside_punish == 'exp':
        
            xi = loss_dict['exp_factor']
            loss*= np.exp(xi * dist_outside/tot_nr_points)

        elif dift_outside_punish == 'lin':
            slope = loss_dict['lin_slope']
            redshift_score *= (1 + slope*frac_outside)
            
        else:
            print('do not recognise the dist_outside_punish')

    #                     theta = 10
    #                     redshift_score *= (1 + np.exp((frac_outside - 0.1) * theta))

    return [loss, dist_outside]
    
    
def binned_dist_func(training_data_dict, binning_feat, bin_feat, bin_feat_name, data_type, full_range):
            
    pred_bin_feat_dist = []
    true_bin_feat_dist = []
    redshifts = []
    pred_bin_centers = []
    obs_bin_centers = []

    acceptable_binning_feat_intervals = []

    for i_red, redshift in enumerate(training_data_dict['unique_redshifts']):

        relevant_inds = training_data_dict['data_redshifts']['{}_data'.format(data_type)] == redshift

        if full_range:
            bin_widths = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_widths']

            min_stellar_mass = np.amin(binning_feat)
            max_stellar_mass = np.amax(binning_feat)
            min_bin_edge = np.floor(min_stellar_mass * 1/bin_widths)*bin_widths
            max_bin_edge = np.ceil(max_stellar_mass * 1/bin_widths)*bin_widths

            # make sure that the predicted range is wider than the observed range
            if min_bin_edge < training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][0]:

                if max_bin_edge > training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][-1]:
                    bin_edges = np.arange(min_bin_edge, max_bin_edge + bin_widths, bin_widths)
                else:
                    bin_edges = np.arange(min_bin_edge, training_data_dict['{}_data'.format(bin_feat_name)]
                                               ['{:.1f}'.format(redshift)]['bin_edges'][-1] + bin_widths, 
                                               bin_widths)
            else:

                if max_bin_edge > training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][-1]:
                    bin_edges = \
                        np.arange(training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][0], 
                                  max_bin_edge + bin_widths, bin_widths)
                else:
                    bin_edges = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges']

        else:
            bin_edges = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges']

        true_bin_feat_dist_redshift = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)][bin_feat_name]

        n_bins = len(bin_edges)-1
        bin_means, bin_edges, bin_numbers = binned_statistic(binning_feat[relevant_inds], bin_feat[relevant_inds], 
                                           bins=bin_edges, statistic='mean')
    #                 bin_stats_stds = binned_statistic(binning_feat[relevant_inds], ssfr_log[relevant_inds], 
    #                                                   bins=bin_edges, statistic=np.std)
        if bin_feat_name == 'smf':
            bin_counts = [np.sum(bin_numbers == i) for i in range(1, n_bins+1)]
            bin_counts = [float('nan') if count == 0 else count for count in bin_counts]
            bin_counts = np.array(bin_counts, dtype=np.float)
            
            bin_widths = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_widths']
            pred_bin_feat_dist_redshift = bin_counts / 200**3 / bin_widths
            
            # since we're only using a subset of the original data points, compensate for this
            pred_bin_feat_dist_redshift *= training_data_dict['original_nr_data_points_by_redshift'][i_red] \
                                / len(training_data_dict['{}_indices'.format(data_type)])
            
            non_nan_indeces = np.invert(np.isnan(pred_bin_feat_dist_redshift))
            pred_bin_feat_dist_redshift[non_nan_indeces] = np.log10(pred_bin_feat_dist_redshift[non_nan_indeces])
    
        elif bin_feat_name == 'fq':
            
            scale_factor = 1 / (1 + redshift)

            h_0 = 67.81 / (3.09e19) # 1/s
            h_0 = h_0 * 60 * 24 * 365 # 1/yr
            h_r = h_0 * np.sqrt(1e-3*scale_factor**(-4) + 0.308*scale_factor**(-3) + 0*scale_factor**(-2) + 0.692)
            ssfr_cutoff = 0.3*h_r
            log_ssfr_cutoff = np.log10(ssfr_cutoff)
            
            pred_bin_feat_dist_redshift = np.zeros(n_bins)
            for bin_num in range(1, n_bins+1):

                if len(bin_feat[bin_numbers == bin_num]) != 0:
                    fq = np.sum(bin_feat[bin_numbers == bin_num] < log_ssfr_cutoff) / len(bin_feat[bin_numbers == bin_num])
                else:
                    fq = float('nan')

                pred_bin_feat_dist_redshift[bin_num-1] = fq

            ### TODO: check if the following line makes sense. Do we ever have zeros? Not just NaNs?
            zero_indeces = [True if (frac == 0) else False for frac in pred_bin_feat_dist_redshift]
            pred_bin_feat_dist_redshift[zero_indeces] = 1e-5

            non_nan_indeces = np.invert(np.isnan(pred_bin_feat_dist_redshift))
            
        else:    
            pred_bin_feat_dist_redshift = bin_means
            non_nan_indeces = np.invert(np.isnan(pred_bin_feat_dist_redshift))

        pred_bin_feat_dist.append(pred_bin_feat_dist_redshift.copy())
        true_bin_feat_dist.append(true_bin_feat_dist_redshift.copy())
        redshifts.append(redshift)
        pred_bin_centers.append([(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
        obs_bin_centers.append(training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_centers'])
        acceptable_binning_feat_intervals.append(
            [training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][0], 
            training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][-1]]
        )

    return [pred_bin_feat_dist, true_bin_feat_dist, pred_bin_centers, obs_bin_centers, redshifts, acceptable_binning_feat_intervals]


def loss_func_obs_stats(model, training_data_dict, loss_dict, real_obs=True, data_type='train'):
    
    np.seterr(over='raise', divide='raise')
    
    if real_obs:
        pass
    
    else:
        
        y_pred = predict_points(model, training_data_dict, original_units=False, as_lists=False, data_type=data_type)
        
        sfr_index = training_data_dict['output_features'].index('SFR')
        stellar_mass_index = training_data_dict['output_features'].index('Stellar_mass')

        predicted_sfr_log = y_pred[:, sfr_index]
        predicted_sfr_log[predicted_sfr_log < -15] = -15
        predicted_sfr_log[predicted_sfr_log > 15] = 15
        predicted_sfr = np.power(10, predicted_sfr_log)

        predicted_stellar_mass_log = y_pred[:, stellar_mass_index]
        predicted_stellar_mass_log[predicted_stellar_mass_log < -15] = -15
        predicted_stellar_mass_log[predicted_stellar_mass_log > 15] = 15
        predicted_stellar_mass = np.power(10, predicted_stellar_mass_log)

        try:
            ssfr = np.divide(predicted_sfr, predicted_stellar_mass)
        except:
            print(np.dtype(predicted_sfr[0]), np.dtype(predicted_stellar_mass[0]))
            print('predicted_sfr: ',predicted_sfr)
            print('predicted_stellar_mass: ', predicted_stellar_mass)
            sys.exit('overflow error while dividing')

        try:
            ssfr_log = np.log10(ssfr)
        except:
            print(np.dtype(ssfr[0]))
            print('ssfr: ',ssfr)
            sys.exit('divide by zero error while taking log')
             
        loss = 0
        dist_outside_tot = 0

        ############### mean SSFR ###############

        loss_ssfr, dist_outside = \
            binned_loss(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'ssfr', data_type, loss_dict)
        loss += loss_dict['ssfr_weight'] * loss_ssfr
        dist_outside_tot += dist_outside
#         nr_empty_bins =+ nr_empty_bins_ssfr

        ############### SMF ###############  
        
        loss_smf, dist_outside = \
            binned_loss(training_data_dict, predicted_stellar_mass_log, predicted_stellar_mass_log, 'smf', data_type, loss_dict)
        loss += loss_dict['smf_weight'] * loss_smf
        dist_outside_tot += dist_outside
#         nr_empty_bins =+ nr_empty_bins_smf

        ############### FQ ###############

        loss_fq, dist_outside = \
            binned_loss(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'fq', data_type, loss_dict)
        loss += loss_dict['fq_weight'] * loss_fq
        dist_outside_tot += dist_outside
#         nr_empty_bins =+ nr_empty_bins_fq

        ############### SHM ###############

        loss_shm, dist_outside = \
            binned_loss(training_data_dict, training_data_dict['original_halo_masses_{}'.format(data_type)], 
                                                               predicted_stellar_mass_log, 'shm', data_type, loss_dict)
        loss += loss_dict['shm_weight'] * loss_shm
        dist_outside_tot += dist_outside
#         nr_empty_bins =+ nr_empty_bins_fq

        loss /= loss_dict['ssfr_weight'] + loss_dict['smf_weight'] + loss_dict['fq_weight'] + loss_dict['shm_weight']
    
        return loss


def plots_obs_stats(model, training_data_dict, real_obs=True, data_type='train', full_range=False):
    
    
    if real_obs:
        pass
    
    else:
        
        y_pred = predict_points(model, training_data_dict, original_units=False, as_lists=False, data_type=data_type)
        
        sfr_index = training_data_dict['output_features'].index('SFR')
        stellar_mass_index = training_data_dict['output_features'].index('Stellar_mass')

        predicted_sfr_log = y_pred[:, sfr_index]
        predicted_sfr_log[predicted_sfr_log < -15] = -15
        predicted_sfr_log[predicted_sfr_log > 15] = 15
        predicted_sfr = np.power(10, predicted_sfr_log)

        predicted_stellar_mass_log = y_pred[:, stellar_mass_index]
        predicted_stellar_mass_log[predicted_stellar_mass_log < -15] = -15
        predicted_stellar_mass_log[predicted_stellar_mass_log > 15] = 15
        predicted_stellar_mass = np.power(10, predicted_stellar_mass_log)

        try:
            ssfr = np.divide(predicted_sfr, predicted_stellar_mass)
        except:
            print(np.dtype(predicted_sfr[0]), np.dtype(predicted_stellar_mass[0]))
            print('predicted_sfr: ',predicted_sfr)
            print('predicted_stellar_mass: ', predicted_stellar_mass)
            sys.exit('overflow error while dividing')

        try:
            ssfr_log = np.log10(ssfr)
        except:
            print(np.dtype(ssfr[0]))
            print('ssfr: ',ssfr)
            sys.exit('divide by zero error while taking log')

#         predicted_stellar_masses_redshift = []
#         acceptable_interval_redshift = []

        nr_empty_bins_redshift = np.zeros(len(training_data_dict['unique_redshifts']), dtype='int')
        frac_outside_redshift = np.zeros(len(training_data_dict['unique_redshifts']))

      
        ############### mean SSFR ###############

        pred_ssfr, true_ssfr, pred_bin_centers_ssfr, obs_bin_centers_ssfr, redshifts_ssfr, obs_mass_interval_ssfr = \
            binned_dist_func(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'ssfr', data_type, full_range)

     
        ############### SMF ###############  

        pred_smf, true_smf, pred_bin_centers_smf, obs_bin_centers_smf, redshifts_smf, obs_mass_interval_smf = \
            binned_dist_func(training_data_dict, predicted_stellar_mass_log, predicted_stellar_mass_log, 'smf', data_type, 
                             full_range)

        ############### FQ ###############

        pred_fq, true_fq, pred_bin_centers_fq, obs_bin_centers_fq, redshifts_fq, obs_mass_interval_fq = \
            binned_dist_func(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'fq', data_type, full_range)
        
        ############### SHM ###############

        pred_shm, true_shm, pred_bin_centers_shm, obs_bin_centers_shm, redshifts_shm, obs_mass_interval_shm = \
            binned_dist_func(training_data_dict, training_data_dict['original_halo_masses_{}'.format(data_type)], 
                             predicted_stellar_mass_log, 'shm', data_type, full_range)      
        
        return {
            'ssfr': [pred_ssfr, true_ssfr, pred_bin_centers_ssfr, obs_bin_centers_ssfr, redshifts_ssfr, obs_mass_interval_ssfr],
            'smf': [pred_smf, true_smf, pred_bin_centers_smf, obs_bin_centers_smf, redshifts_smf, obs_mass_interval_smf],
            'fq': [pred_fq, true_fq, pred_bin_centers_fq, obs_bin_centers_fq, redshifts_fq, obs_mass_interval_fq],
            'shm': [pred_shm, true_shm, pred_bin_centers_shm, obs_bin_centers_shm, redshifts_shm, obs_mass_interval_shm]
#             'predicted_stellar_masses_redshift': predicted_stellar_masses_redshift,
#             'nr_empty_bins_redshift': nr_empty_bins_redshift,
#             'fraction_of_points_outside_redshift': frac_outside_redshift,
#             'acceptable_interval_redshift': acceptable_interval_redshift
        }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    