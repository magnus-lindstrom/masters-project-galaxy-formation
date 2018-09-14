import pandas as pd
import numpy as np
import json
import sys
import h5py
import io
            

def get_unit_dict():
    
    unit_dict = {'X_pos': '', 'Y_pos': '', 'Z_pos': '', 'X_vel': '', 'Y_vel': '', 
                 'Z_vel': '', 'Halo_mass': 'M_{H}/M_{\odot}', 'Stellar_mass': r'm_{\ast}/M_{\odot}',
                 'SFR': 'M_{\odot}yr^{-1}', 'SSFR': 'yr^{-1}', 'SMF': '\Phi / Mpc^{-3} dex^{-1}', 'FQ': 'f_q',
                 'Intra_cluster_mass': '', 'Halo_mass_peak': 'M_{G}/M_{\odot}', 
                 'Stellar_mass_obs': '', 'SFR_obs': '', 'Halo_radius': '', 
                 'Concentration': '', 'Halo_spin': '', 'Scale_peak_mass': 'a', 
                 'Scale_half_mass': 'a', 'Scale_last_MajM': 'a', 'Type': '',
                 'Environmental_density': 'log($M_{G}/M_{S}/Mpc^3$)', 'Redshift': 'z', 
                 'CSFRD': r'\rho_{\ast} / M_{\odot} yr^{-1} Mpc^{-3}'
                }
    
    return unit_dict


def redshift_from_scale(scale_factor):
    
    if type(scale_factor) is list:
        redshift = (1/np.array(scale_factor) - 1).tolist()
    else:
        redshift = 1/scale_factor - 1
    return redshift
    
    
def scale_from_redshift(redshift):
    
    if type(redshift) is list:
        scale_factor = (1/(1+np.array(redshift))).tolist()
    else:
        scale_factor = 1/(1+redshift)
    return scale_factor
    

def divide_train_data(galaxies, data_keys, network_args, redshifts, weigh_by_redshift=0, outputs_to_weigh=0, 
                      total_set_size=0, train_frac=0, val_frac=0, test_frac=0, pso=False, emerge_targets=False, 
                      real_observations=False, mock_observations=False, h_0=.6781):
    
    n_data_points = galaxies.shape[0]

    if total_set_size == 'all':
        total_set_size = n_data_points
    else:
        total_set_size = int(total_set_size)
    train_size = int(train_frac * total_set_size)
    val_size = int(val_frac * total_set_size)
    test_size = int(test_frac * total_set_size)
    
    subset_indices = np.random.choice(n_data_points, total_set_size, replace=False)
    train_indices = subset_indices[: int(train_size)]
    val_indices = subset_indices[int(train_size) : int(train_size+val_size)]
    test_indices = subset_indices[int(train_size+val_size) :]

    x_train = np.zeros((len(train_indices), len(network_args['input_features'])))
    x_val = np.zeros((len(val_indices), len(network_args['input_features'])))
    x_test = np.zeros((len(test_indices), len(network_args['input_features'])))
    
    data_redshifts = {}
    
    for i in range(len(network_args['input_features'])):
        x_train[:,i] = galaxies[train_indices, data_keys[network_args['input_features'][i]]]
        x_val[:,i] = galaxies[val_indices, data_keys[network_args['input_features'][i]]]
        x_test[:,i] = galaxies[test_indices, data_keys[network_args['input_features'][i]]]

    data_redshifts['train_data'] = galaxies[train_indices, data_keys['Redshift']]
    data_redshifts['val_data'] = galaxies[val_indices, data_keys['Redshift']]
    data_redshifts['test_data'] = galaxies[test_indices, data_keys['Redshift']]
    
    original_nr_data_points_by_redshift = []
    train_frac_of_tot_by_redshift = []
    val_frac_of_tot_by_redshift = []
    test_frac_of_tot_by_redshift = []
    
    for redshift in redshifts:
        original_nr_data_points_by_redshift.append(np.sum(galaxies[:, data_keys['Redshift']] == redshift))
        train_frac_of_tot_by_redshift.append(
            np.sum(galaxies[train_indices, data_keys['Redshift']] == redshift) / original_nr_data_points_by_redshift[-1]
        )
        val_frac_of_tot_by_redshift.append(
            np.sum(galaxies[val_indices, data_keys['Redshift']] == redshift) / original_nr_data_points_by_redshift[-1]
        )
        test_frac_of_tot_by_redshift.append(
            np.sum(galaxies[test_indices, data_keys['Redshift']] == redshift) / original_nr_data_points_by_redshift[-1]
        )
        
    training_data_dict = {
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'train_coordinates': np.vstack((galaxies[train_indices, data_keys['X_pos']] * h_0, # coordinates are given in Mpc/h since
                                        galaxies[train_indices, data_keys['Y_pos']] * h_0, # halotools require this unit
                                        galaxies[train_indices, data_keys['Z_pos']] * h_0)).T,
        'val_coordinates': np.vstack((galaxies[val_indices, data_keys['X_pos']] * h_0, 
                                      galaxies[val_indices, data_keys['Y_pos']] * h_0,
                                      galaxies[val_indices, data_keys['Z_pos']] * h_0)).T,
        'test_coordinates': np.vstack((galaxies[test_indices, data_keys['X_pos']] * h_0, 
                                       galaxies[test_indices, data_keys['Y_pos']] * h_0,
                                       galaxies[test_indices, data_keys['Z_pos']] * h_0)).T,
        'original_halo_masses_train': galaxies[train_indices, data_keys['Halo_mass']],
        'original_halo_masses_val': galaxies[val_indices, data_keys['Halo_mass']],
        'original_halo_masses_test': galaxies[test_indices, data_keys['Halo_mass']],
        'original_nr_data_points_by_redshift': original_nr_data_points_by_redshift,
        'train_frac_of_tot_by_redshift': train_frac_of_tot_by_redshift,
        'val_frac_of_tot_by_redshift': val_frac_of_tot_by_redshift,
        'test_frac_of_tot_by_redshift': test_frac_of_tot_by_redshift,
        'unique_redshifts': redshifts,
        'data_redshifts': data_redshifts,
        'network_args': network_args
    }
    
    if emerge_targets:
        
        y_train = np.zeros((len(train_indices), len(network_args['output_features'])))
        y_val = np.zeros((len(val_indices), len(network_args['output_features'])))
        y_test = np.zeros((len(test_indices), len(network_args['output_features'])))
        
        for i in range(len(network_args['output_features'])):
            y_train[:,i] = galaxies[train_indices, data_keys[network_args['output_features'][i]]]
            y_val[:,i] = galaxies[val_indices, data_keys[network_args['output_features'][i]]]
            y_test[:,i] = galaxies[test_indices, data_keys[network_args['output_features'][i]]]
            
        training_data_dict['y_train'] = y_train
        training_data_dict['y_val'] = y_val
        training_data_dict['y_test'] = y_test
        
        train_weights, val_weights, test_weights = get_weights(training_data_dict, network_args['output_features'], outputs_to_weigh, 
                                                               train_frac, val_frac, test_frac, weigh_by_redshift=weigh_by_redshift, 
                                                               pso=pso)
        training_data_dict['train_weights'] = train_weights
        training_data_dict['val_weights'] = val_weights
        training_data_dict['test_weights'] = test_weights
        
    del training_data_dict["original_halo_masses_train"]
    del training_data_dict["original_halo_masses_val"]
    del training_data_dict["original_halo_masses_test"]
    
    if real_observations:
        
        training_data_dict = add_obs_data(training_data_dict, h_0, real_obs=True)
        
    if mock_observations:
        
        training_data_dict = add_obs_data(training_data_dict, h_0, mock_obs=True)
    
    return training_data_dict


def convert_units(data, norm, back_to_original=False, conv_values=None):
    
    np.seterr(invalid='raise')
    
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
                try:
                    data_stds = np.std(data, 0)
                except:
                    print('Could not get stds from data: ', data[:4, :])
                    sys.exit()
                conv_values = {
                    'data_means': data_means,
                    'data_stds': data_stds
                }
#                 print('data: ',np.shape(data))
#                 print('data_means: ',np.shape(data_means))
#                 print('data_stds: ',np.shape(data_stds))
                try:
                    data_norm = (data - data_means) / data_stds
                except:
                    print('Tried subtracting invalid means: ', data_means)
                    sys.exit()
                    
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
    
    if 'y_train' in training_data_dict: # i.e.: if emerge_targets == True
        
        y_train = training_data_dict['y_train']
        y_val = training_data_dict['y_val']
        y_test = training_data_dict['y_test']

        output_train_dict = {}
        output_val_dict = {}
        output_test_dict = {}

        # convert units based on the train data only
        if norm['output'] != 'none':
            y_train_norm, conv_values_output = convert_units(y_train, norm['output'])
            y_val_norm = convert_units(y_val, norm['output'], conv_values=conv_values_output)
            y_test_norm = convert_units(y_test, norm['output'], conv_values=conv_values_output)
            training_data_dict['conv_values_output'] = conv_values_output
        else:
            y_train_norm = y_train
            y_val_norm = y_val
            y_test_norm = y_test
        
        if pso:

            training_data_dict['y_train'] = y_train_norm
            training_data_dict['y_val'] = y_val_norm
            training_data_dict['y_test'] = y_test_norm

        if not pso:       # if trained with backpropagation

            for i_feat, feat in enumerate(training_data_dict['network_args']['output_features']):
                output_train_dict[feat] = y_train_norm[:, i_feat]
                output_val_dict[feat] = y_val_norm[:, i_feat]
                output_test_dict[feat] = y_test_norm[:, i_feat]

            del training_data_dict["y_train"]
            del training_data_dict["y_val"]
            del training_data_dict["y_test"]

            training_data_dict['output_train_dict'] = output_train_dict
            training_data_dict['output_val_dict'] = output_val_dict
            training_data_dict['output_test_dict'] = output_test_dict

    training_data_dict['norm'] = norm
        
    return training_data_dict


def prune_train_data_dict_for_reinf_learn(training_data_dict, no_val=False):
    
    del training_data_dict['output_train_dict']
    del training_data_dict['output_val_dict']
    del training_data_dict['output_test_dict']
    
    if no_val:
        training_data_dict['train_coordinates'] = np.vstack((training_data_dict['train_coordinates'], 
                                                            training_data_dict['val_coordinates'], 
                                                            training_data_dict['test_coordinates']))
        del training_data_dict['val_coordinates']
        del training_data_dict['test_coordinates']

        training_data_dict['input_train_dict']['main_input'] = np.vstack((training_data_dict['input_train_dict']['main_input'], 
                                                                          training_data_dict['input_val_dict']['main_input'], 
                                                                          training_data_dict['input_test_dict']['main_input']))
        del training_data_dict['input_val_dict']
        del training_data_dict['input_test_dict']

        training_data_dict['data_redshifts']['train_data'] = np.concatenate((training_data_dict['data_redshifts']['train_data'], 
                                                                             training_data_dict['data_redshifts']['val_data'], 
                                                                             training_data_dict['data_redshifts']['test_data']))
        del training_data_dict['data_redshifts']['val_data']
        del training_data_dict['data_redshifts']['test_data']

        training_data_dict['train_frac_of_tot_by_redshift'] = (
            training_data_dict['train_frac_of_tot_by_redshift'] 
            + training_data_dict['val_frac_of_tot_by_redshift']
            + training_data_dict['test_frac_of_tot_by_redshift']
        )
        del training_data_dict['val_frac_of_tot_by_redshift']
        del training_data_dict['test_frac_of_tot_by_redshift']
    
    return training_data_dict


def get_test_score(model, training_data_dict, norm):
    ### Get the MSE for the test predictions in the original units of the dataset###
    
    predicted_points = predict_points(training_data_dict, data_type = 'test')

    # Get mse for the real predictions
    
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
        
    if original_units and 'conv_values_output' in training_data_dict:
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
    

def get_weights(training_data_dict, output_features, outputs_to_weigh, train_frac, val_frac, test_frac, weigh_by_redshift=False, 
                pso=False):
    """Returns the weights needed for training with backpropagation."""
    
    unique_redshifts = training_data_dict['unique_redshifts']
    
    data_types = ['train', 'val', 'test']
    fracs = np.array([train_frac, val_frac, test_frac])
    final_weights = []
    
    for i_data_type, data_type in enumerate(data_types):
        if fracs[i_data_type] > 0:
            
            weights_tmp = np.zeros(len(training_data_dict['original_halo_masses_{}'.format(data_type)]))
            # make the heavier halos more important in every redshift
            for redshift in unique_redshifts:
                relevant_inds = training_data_dict['data_redshifts']['{}_data'.format(data_type)] == redshift
                halo_masses = training_data_dict['original_halo_masses_{}'.format(data_type)]
                
                masses_redshift = halo_masses[relevant_inds]
                masses_redshift = np.power(10, masses_redshift)
                weights_redshift = masses_redshift / np.sum(masses_redshift)
                
                weights_tmp[relevant_inds] = weights_redshift
                
            weights_tmp = weights_tmp / np.sum(weights_tmp)
    
    
            # check that weights sum up to one
            summ = np.sum(weights_tmp)
            err = np.abs(1 - summ)
            too_high = err > 1e-6
            if too_high:
                print('The {} weights were not normalised properly for mass. Sum should be: {:.2e}'.format(data_type, summ))
                return
    
            weights = {}

            for output in output_features:
                if output in outputs_to_weigh:
                    weights[output] = weights_tmp
                else:
                    weights[output] = np.ones(int(len(training_data_dict['original_halo_masses_{}'.format(data_type)]))) \
                                                    / len(training_data_dict['original_halo_masses_{}'.format(data_type)])
            
            if weigh_by_redshift:
                data_redshifts = training_data_dict['data_redshifts']['{}_data'.format(data_type)]
                unique_redshifts = training_data_dict['unique_redshifts']
                redshift_weights = np.zeros(len(data_redshifts))
                
                for redshift in unique_redshifts:
                    redshift_weight = 1 / (len(unique_redshifts) * np.sum(data_redshifts == redshift))
                    redshift_weights[data_redshifts == redshift] = redshift_weight
                    
                for output in output_features:
                    weights[output] = (weights[output] + redshift_weights) / 2
                    
                # check that weights sum up to one
                summ = np.array([np.sum(weights[output]) for output in output_features])
                err = np.abs(1 - summ)
                too_high = err > 1e-6
                if any(too_high):
                    print('The {} weights were not normalised properly for redshift. Sums should be 1 but are: {:.2e} {:.2e}'.format(
                        data_type, summ[0], summ[1]
                    ))
                    return
            
    
            if pso:
                weights_tmp = np.zeros(np.shape(training_data_dict['y_train']))

                for i_output, output in enumerate(weights.keys()):
                    weights_tmp[:, i_output] = weights[output]

                weights = weights_tmp
                
            final_weights.append(weights)
        else:
            final_weights.append(None)
                
            

    return final_weights


def get_weights_old(training_data_dict, output_features, outputs_to_weigh, weigh_by_redshift=False, pso=False):
    
    """ Returns the weights needed for training with backpropagation. Weighs by halo mass by default, possibility to weigh by redshift as well. Specify pso=True if the the pso will be used to find the best parameter combinations.
    """
    
    unique_redshifts = training_data_dict['unique_redshifts']
    train_w_tmp = np.zeros(len(training_data_dict['original_halo_masses_train']))
    val_w_tmp = np.zeros(len(training_data_dict['original_halo_masses_val']))
    test_w_tmp = np.zeros(len(training_data_dict['original_halo_masses_test']))
    
    # make the heavier halos more important in every redshift
    for redshift in unique_redshifts:
        relevant_train_inds = training_data_dict['data_redshifts']['train_data'] == redshift
        relevant_val_inds = training_data_dict['data_redshifts']['val_data'] == redshift
        relevant_test_inds = training_data_dict['data_redshifts']['test_data'] == redshift

        train_masses = training_data_dict['original_halo_masses_train']
        val_masses = training_data_dict['original_halo_masses_val']
        test_masses = training_data_dict['original_halo_masses_test']
                
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
            train_weights[output] = np.ones(int(len(training_data_dict['original_halo_masses_train']))) \
                                            / len(training_data_dict['original_halo_masses_train'])
            val_weights[output] = np.ones(int(len(training_data_dict['original_halo_masses_val']))) \
                                            / len(training_data_dict['original_halo_masses_val'])
            test_weights[output] = np.ones(int(len(training_data_dict['original_halo_masses_test']))) \
                                            / len(training_data_dict['original_halo_masses_test'])
            
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


def chi_squared_loss(predictions, targets, errors):
    
    np.seterr(over='raise')
    try:
        loss = np.power(predictions - targets, 2) / np.power(errors, 2)        
        loss = np.sum(loss) / len(predictions)
    except:
        loss = 1e400
    return loss



    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    