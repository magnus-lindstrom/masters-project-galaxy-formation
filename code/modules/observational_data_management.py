import h5py
import numpy as np
import data_processing
import sys
from scipy.interpolate import SmoothBivariateSpline, interp1d
from halotools.mock_observables import wp as projected_corr_func
from numpy.polynomial.polynomial import polyval
from scipy.stats import binned_statistic
import warnings
from contextlib import contextmanager, redirect_stdout
import os.path


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
            
def get_max_resolvable_stellar_mass(redshifts, bin_width, box_size, min_count=10, degree=6, n_points=1000):
    """
    Returns the maximum resolvable stellar mass. Above this mass, constraints can not be enforced
    
    Arguments
    redshifts -- The redshifts of the snapshots that the network is predicting
    bin_width -- The bin width in stellar mass for the stellar mass function
    box_size -- The side length of the box containing the galaxies (assumed to be cubic)
    
    Keyword arguments
    min_count -- The minimum number of galaxies that is to be expected in a box for the loss function to 'care about' it 
                 (default 10)
    degree -- The degree of the polynomial used to fit the observational constraints in order to get the mass corresponding 
              to the minimum resolvable abundance (default 6)
    n_points -- The number of points of the approximated polynomial, affects precision of the resolvable stellar mass
    """
    warnings.simplefilter('ignore', np.RankWarning) # the fit always throws a warning, but it is ok for all snapshots we have so far
    
    filename = '/home/magnus/data/observational_data/all_data.h5'
    file = h5py.File(filename, 'r')
    universe_0 = file['Universe_0']
    smf = universe_0['SMF']
    data = smf['Data']
    sets = smf['Sets']

    min_abundance = np.log10(min_count / box_size**3 / bin_width)
    
    max_masses = []

    for redshift in redshifts:
        stellar_masses = []
        abundances = []
        errors = []

        for i_key, key in enumerate(list(data)):
            min_redshift = list(sets)[i_key][2]
            max_redshift = list(sets)[i_key][3]
            if redshift >= min_redshift and redshift <= max_redshift:
                stellar_masses.extend([list(point)[0] for point in list(data[key])])
                abundances.extend([list(point)[1] for point in list(data[key])])
                errors.extend([list(point)[2] for point in list(data[key])])

        weights = 1 / np.array(errors)
        x = np.linspace(np.min(stellar_masses), np.max(stellar_masses), n_points) # contains masses
        p3 = np.polyfit(stellar_masses, abundances, degree, w=weights)
        max_mass_ind = np.argmin(np.absolute(polyval(x, np.flip(p3, axis=0)) - min_abundance))
        max_mass = x[max_mass_ind] + bin_width
        max_masses.append(max_mass)
        
    return max_masses


def add_obs_data(training_data_dict, loss_dict, h_0, real_obs=False, mock_observations=False, validation_fraction=0, box_size=200):
    
    if real_obs:
        
        data_path = '/home/magnus/data/observational_data/all_data.h5'
        file = h5py.File(data_path, 'r')
        universe_0 = file['Universe_0']
        
        max_redshift = np.max(training_data_dict['unique_redshifts'])
        min_redshift = np.min(training_data_dict['unique_redshifts'])
        min_scalefactor = 1 / (1 + max_redshift)
        max_scalefactor = 1 / (1 + min_redshift)
        
        # Get the max resolvable stellar mass for each redshift
        max_res_masses = get_max_resolvable_stellar_mass(training_data_dict['unique_redshifts'], 
                                                         loss_dict['stellar_mass_bin_width'], box_size)
        
        # Get the SMF object from the universe
        smf = universe_0['SMF']
        data = smf['Data']
        sets = smf['Sets']
        data_keys = list(data.keys())
        
        real_smf_data = {'train': None, 'val': None}
        real_smf_data['train'] = {
            'surveys': [],
            'scale_factor': [],
            'scale_factor_range': [], # the scale factor interval covered by a data point
            'binning_feat': [],
            'smf': [],
            'errors': []
        }
        if validation_fraction > 0:
            real_smf_data['val'] = {
                'surveys': [],
                'scale_factor': [],
                'scale_factor_range': [], # the scale factor interval covered by a data point
                'binning_feat': [],
                'smf': [],
                'errors': []
            }

        for i_key, key in enumerate(list(data)):
            data_scalefactor = data[key][0][3]
            if data_scalefactor > min_scalefactor and data_scalefactor < max_scalefactor:
                real_smf_data['train']['surveys'].append(key)
                real_smf_data['train']['scale_factor'].extend([list(point)[3] for point in list(data[key])])
                real_smf_data['train']['scale_factor_range'].extend([[1 / (1 + list(sets)[i_key][3]), 
                                                                       1 / (1 + list(sets)[i_key][2])]] * len(list(data[key])))        
                real_smf_data['train']['binning_feat'].extend([list(point)[0] for point in list(data[key])])
                real_smf_data['train']['smf'].extend([list(point)[1] for point in list(data[key])])
                real_smf_data['train']['errors'].extend([list(point)[2] for point in list(data[key])])
                
        # prune away points corresponding to statistically insignificant stellar masses
        for i_redshift, redshift in enumerate(training_data_dict['unique_redshifts']):
            points_to_delete = []
            for i_point in range(len(real_smf_data['train']['scale_factor_range'])):
                if (
                    redshift >= data_processing.redshift_from_scale(real_smf_data['train']['scale_factor_range'][i_point][1])
                    and
                    redshift <= data_processing.redshift_from_scale(real_smf_data['train']['scale_factor_range'][i_point][0])
                ):
                    if real_smf_data['train']['binning_feat'][i_point] > max_res_masses[i_redshift]:
                        points_to_delete.append(i_point)
                        
            for i_point in reversed(points_to_delete):
                del real_smf_data['train']['scale_factor'][i_point]
                del real_smf_data['train']['scale_factor_range'][i_point]
                del real_smf_data['train']['binning_feat'][i_point]
                del real_smf_data['train']['smf'][i_point]
                del real_smf_data['train']['errors'][i_point]
                
        surveys_covering_redshifts = []
        for redshift in training_data_dict['unique_redshifts']:
            redshift_surveys = []
            for sett in list(sets):
                if redshift >= sett[2] and redshift <= sett[3]:
                    redshift_surveys.append(sett[-1])
            surveys_covering_redshifts.append(redshift_surveys)
        real_smf_data['surveys_covering_redshifts'] = surveys_covering_redshifts
                
        training_data_dict['real_smf_data'] = real_smf_data
                
        # Get the sSFR object from the universe
        ssfr = universe_0['SSFR']
        data = ssfr['Data']
        sets = ssfr['Sets']
        data_keys = list(data.keys())
        
        real_ssfr_data = {'train': None, 'val': None}
        real_ssfr_data['train'] = {
            'surveys': [],
            'scale_factor': [],
            'binning_feat': [],
            'ssfr': [],
            'errors': []
        }
        if validation_fraction > 0:
            real_ssfr_data['val'] = {
                'surveys': [],
                'scale_factor': [],
                'binning_feat': [],
                'ssfr': [],
                'errors': []
            }
        
        for i_key, key in enumerate(list(data)):
            for data_point in list(data[key]):
                redshift = data_point[0]
                scale_factor = 1 / (1 + redshift)

                if scale_factor >= min_scalefactor and scale_factor <= max_scalefactor:                    
                    if list(sets)[i_key][-1] not in real_ssfr_data['train']['surveys']:
                        real_ssfr_data['train']['surveys'].append(list(sets)[i_key][-1])
                    real_ssfr_data['train']['scale_factor'].append(scale_factor)
                    real_ssfr_data['train']['binning_feat'].append(data_point[3])
                    real_ssfr_data['train']['ssfr'].append(data_point[1])
                    real_ssfr_data['train']['errors'].append(data_point[2])
                    
        # prune away points corresponding to statistically insignificant stellar masses
        unique_scale_factors = data_processing.scale_from_redshift(training_data_dict['unique_redshifts'])
        points_to_delete = []
        for i_point in range(len(real_ssfr_data['train']['scale_factor'])):
#             print(np.array(unique_scale_factors))
#             print(real_ssfr_data['train']['scale_factor'][i_point])
#             print(np.array(unique_scale_factors) - real_ssfr_data['train']['scale_factor'][i_point])
            closest_redshift_index = np.argmin(np.absolute(np.array(unique_scale_factors) 
                                                           - real_ssfr_data['train']['scale_factor'][i_point]))
            if real_ssfr_data['train']['binning_feat'][i_point] >= max_res_masses[closest_redshift_index]:
                points_to_delete.append(i_point)
                                    
        for i_point in reversed(points_to_delete):
            del real_ssfr_data['train']['scale_factor'][i_point]
            del real_ssfr_data['train']['binning_feat'][i_point]
            del real_ssfr_data['train']['ssfr'][i_point]
            del real_ssfr_data['train']['errors'][i_point]
        
        training_data_dict['real_ssfr_data'] = real_ssfr_data

                    
        # Get the FQ object from the universe
        fq = universe_0['FQ']
        data = fq['Data']
        sets = fq['Sets']
        data_keys = list(data.keys())
        
        real_fq_data = {'train': None, 'val': None}
        real_fq_data['train'] = {
            'surveys': [],
            'scale_factor': [],
            'scale_factor_range': [],
            'binning_feat': [],
            'fq': [],
            'errors': []
        }
        if validation_fraction > 0:
            real_fq_data['val'] = {
                'surveys': [],
                'scale_factor': [],
                'scale_factor_range': [],
                'binning_feat': [],
                'fq': [],
                'errors': []
            }

        for i_key, key in enumerate(list(data)):
            scale_factor = data[key][0][3]

            if scale_factor >= min_scalefactor and scale_factor <= max_scalefactor:
                
                real_fq_data['train']['surveys'].append(list(sets)[i_key][-1])
                real_fq_data['train']['scale_factor'].extend([list(point)[3] for point in list(data[key])])
                real_fq_data['train']['scale_factor_range'].extend([[1 / (1 + list(sets)[i_key][3]), 1 / (1 + list(sets)[i_key][2])]] 
                                                           * len(list(data[key])))
                real_fq_data['train']['binning_feat'].extend([list(point)[0] for point in list(data[key])])
                real_fq_data['train']['fq'].extend([list(point)[1] for point in list(data[key])])
                real_fq_data['train']['errors'].extend([list(point)[2] for point in list(data[key])])
                
        # prune away points corresponding to statistically insignificant stellar masses
        for i_redshift, redshift in enumerate(training_data_dict['unique_redshifts']):
            points_to_delete = []
            for i_point in range(len(real_fq_data['train']['scale_factor_range'])):
                if (
                    redshift >= data_processing.redshift_from_scale(real_fq_data['train']['scale_factor_range'][i_point][1])
                    and
                    redshift <= data_processing.redshift_from_scale(real_fq_data['train']['scale_factor_range'][i_point][0])
                ):
                    if real_fq_data['train']['binning_feat'][i_point] > max_res_masses[i_redshift]:
                        points_to_delete.append(i_point)
                        
            for i_point in reversed(points_to_delete):
                del real_fq_data['train']['scale_factor'][i_point]
                del real_fq_data['train']['scale_factor_range'][i_point]
                del real_fq_data['train']['binning_feat'][i_point]
                del real_fq_data['train']['fq'][i_point]
                del real_fq_data['train']['errors'][i_point]
                
        surveys_covering_redshifts = []
        for redshift in training_data_dict['unique_redshifts']:
            redshift_surveys = []
            for sett in list(sets):
                if redshift >= sett[2] and redshift <= sett[3]:
                    redshift_surveys.append(sett[-1])
            surveys_covering_redshifts.append(redshift_surveys)
        real_fq_data['surveys_covering_redshifts'] = surveys_covering_redshifts
        
        training_data_dict['real_fq_data'] = real_fq_data
        
        # Get the CSFRD object from the universe
        csfrd = universe_0['CSFRD']
        data = csfrd['Data']
        sets = csfrd['Sets']
        data_keys = list(data.keys())

        real_csfrd_data = {'train': None, 'val': None}
        real_csfrd_data['train'] = {
            'surveys': [],
            'scale_factor': [],
            'csfrd': [],
            'errors': []
        }
        if validation_fraction > 0:
            real_csfrd_data['val'] = {
                'surveys': [],
                'scale_factor': [],
                'csfrd': [],
                'errors': []
            }

        for i_key, key in enumerate(list(data)):

            for data_point in list(data[key]):
                redshift = data_point[0]
                scale_factor = 1 / (1 + redshift)

                if scale_factor >= min_scalefactor and scale_factor <= max_scalefactor:
                    if key not in real_csfrd_data['train']['surveys']:
                        real_csfrd_data['train']['surveys'].append(key)
                    real_csfrd_data['train']['scale_factor'].append(scale_factor) # todo: place the surveys on the same level as train and val
                    real_csfrd_data['train']['csfrd'].append(data_point[1])
                    real_csfrd_data['train']['errors'].append(data_point[2])
                    
        training_data_dict['real_csfrd_data'] = real_csfrd_data
        
        # Get the clustering object from the universe
        clustering = universe_0['Clustering']
        data = clustering['Data']
        sets = clustering['Sets']
        data_keys = list(data.keys())
            
        real_clustering_data = {'train': None, 'val': None}
        real_clustering_data['train'] = {
            'surveys': [],
            'redshift': 0.08, # the only data we have is at redshift 0.08
            'stellar_mass_bin_edges': [],
            'pi_max': None,
            'rp_bin_edges': None,
            'wp': [],
            'errors': []
        }
        if validation_fraction > 0:
            real_clustering_data['val'] = {
                'surveys': [],
                'redshift': 0.08, # the only data we have is at redshift 0.08
                'stellar_mass_bin_edges': [],
                'pi_max': None,
                'rp_bin_edges': None,
                'wp': [],
                'errors': []
            }
        clust_bin_mids = None # will temporarily contain the midpoints of the rp bins
        for i_key, key in enumerate(list(data)):

            if list(sets)[i_key][-1] not in real_clustering_data['train']['surveys']:
                real_clustering_data['train']['surveys'].append(list(sets)[i_key][-1])
            if i_key == 0:
                real_clustering_data['train']['stellar_mass_bin_edges'].extend([list(sets)[i_key][2], list(sets)[i_key][3]])
            else:
                real_clustering_data['train']['stellar_mass_bin_edges'].append(list(sets)[i_key][3])
            if real_clustering_data['train']['pi_max'] is None:
                real_clustering_data['train']['pi_max'] = list(sets)[i_key][-2] * h_0 # units are now in Mpc/h
            if clust_bin_mids is None:
                clust_bin_mids = np.array([list(point)[0] for point in list(data[key])]) * h_0 # units are now in Mpc/h
            real_clustering_data['train']['wp'].append([list(point)[1] * h_0 for point in list(data[key])]) # units are now in Mpc/h
            real_clustering_data['train']['errors'].append([list(point)[2] * h_0 for point in list(data[key])]) # Mpc/h

        clust_bin_mids = np.log10(clust_bin_mids) # units are now in log(Mpc/h)

        real_clustering_data['train']['rp_bin_edges'] = (clust_bin_mids[:-1] + clust_bin_mids[1:]) / 2
        real_clustering_data['train']['rp_bin_edges'] = np.insert(
            real_clustering_data['train']['rp_bin_edges'], 
            0, 
            clust_bin_mids[0] - (real_clustering_data['train']['rp_bin_edges'][0] - clust_bin_mids[0])
        )
        real_clustering_data['train']['rp_bin_edges'] = np.append(
            real_clustering_data['train']['rp_bin_edges'], 
            clust_bin_mids[-1] + (clust_bin_mids[-1] - real_clustering_data['train']['rp_bin_edges'][-1])
        )
        real_clustering_data['train']['rp_bin_edges'] = np.power(10, real_clustering_data['train']['rp_bin_edges']) # back to Mpc/h
        
        # if the right bin edge corresponds to a mass higher than the max allowed, take away the bin
        redshift_01_index = training_data_dict['unique_redshifts'].index(0.1)
        for stellar_mass in reversed(real_clustering_data['train']['stellar_mass_bin_edges']):
        
            if stellar_mass > max_res_masses[redshift_01_index]:
                del real_clustering_data['train']['stellar_mass_bin_edges'][-1]
                del real_clustering_data['train']['wp'][-1]
                del real_clustering_data['train']['errors'][-1]
         
        training_data_dict['real_clustering_data'] = real_clustering_data
        
    if mock_observations:
        
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


def csfrd_loss(training_data_dict, sfr, loss_dict, data_type, box_side=200, interp='cubic'):
    
    csfrd = []
    
    for i_redshift, redshift in enumerate(training_data_dict['unique_redshifts']):
        tot_redshift_sfr = np.sum(sfr[training_data_dict['data_redshifts']['{}_data'.format(data_type)] == redshift])
        tot_redshift_sfr /= training_data_dict['{}_frac_of_tot_by_redshift'.format(data_type)][i_redshift]
        log_csfrd_redshift = np.log10(np.sum(tot_redshift_sfr) / box_side**3)
        csfrd.append(log_csfrd_redshift)
        
    scale_factors = 1 / (1 + np.array(list(reversed(training_data_dict['unique_redshifts'])))) # a now increasing
    csfrd.reverse() # to match the corresponding snapshots
    
    spline_func = interp1d(scale_factors, csfrd, kind=interp)
    pred_observations = spline_func(training_data_dict['real_csfrd_data'][data_type]['scale_factor'])
    
    loss = data_processing.chi_squared_loss(pred_observations, training_data_dict['real_csfrd_data'][data_type]['csfrd'], 
                                training_data_dict['real_csfrd_data'][data_type]['errors'])
#     print('size of csfrd loss: ', loss)

    return loss


def csfrd_distributions(training_data_dict, sfr, loss_dict, data_type, box_side=200, interp='cubic'):
    
    csfrd = []
    
    for i_redshift, redshift in enumerate(training_data_dict['unique_redshifts']):
        tot_redshift_sfr = np.sum(sfr[training_data_dict['data_redshifts']['{}_data'.format(data_type)] == redshift])
        tot_redshift_sfr /= training_data_dict['{}_frac_of_tot_by_redshift'.format(data_type)][i_redshift]
        log_csfrd_redshift = np.log10(np.sum(tot_redshift_sfr) / box_side**3)
        csfrd.append(log_csfrd_redshift)
        
    snapshot_scale_factors = 1 / (1 + np.array(list(reversed(training_data_dict['unique_redshifts'])))) # 'a' now increasing
    csfrd.reverse() # to match the corresponding snapshots
    
    spline_func = interp1d(snapshot_scale_factors, csfrd, kind=interp)
    
    pred_bin_centers_csfrd = np.linspace(np.min(snapshot_scale_factors),np.max(snapshot_scale_factors),1000)
    pred_csfrd = spline_func(pred_bin_centers_csfrd)
    
    true_csfrd = training_data_dict['real_csfrd_data'][data_type]['csfrd']
    obs_bin_centers_csfrd = training_data_dict['real_csfrd_data'][data_type]['scale_factor']
    obs_errors_csfrd = training_data_dict['real_csfrd_data'][data_type]['errors']
    
    return [pred_csfrd, true_csfrd, pred_bin_centers_csfrd, obs_bin_centers_csfrd, obs_errors_csfrd]


def clustering_loss(training_data_dict, stellar_masses, loss_dict, data_type, h_0=0.6781):
    ### Returns the loss wrt the projected 2p correlation function. h_0 is assumed to be 0.6781 and the box side length 200Mpc.###
    
    compared_redshift = 0.1
    
    
    cluster_data = training_data_dict['real_clustering_data'][data_type]
    redshift_index = training_data_dict['unique_redshifts'].index(compared_redshift)
    redshift_indeces = training_data_dict['data_redshifts']['{}_data'.format(data_type)] == compared_redshift
    stellar_masses = stellar_masses[redshift_indeces]
    
    bin_means, bin_edges, bin_numbers = binned_statistic(stellar_masses, stellar_masses, 
                                                         bins=cluster_data['stellar_mass_bin_edges'], statistic='mean')
#     print(bin_numbers)

    loss = 0
    for bin_num in range(1, len(bin_means) + 1):
        
        mass_indeces = bin_numbers == bin_num
        
        if np.any(mass_indeces):
#             print(
#                 'training_data_dict[\'{}_coordinates\'.format(data_type)][redshift_indeces, :][mass_indeces, :]: '.format(data_type),
#                 training_data_dict['{}_coordinates'.format(data_type)][redshift_indeces, :][mass_indeces, :]
#             )
#             print('cluster_data[\'rp_bin_edges\']: ', cluster_data['rp_bin_edges'])
#             print('cluster_data[\'pi_max\']: ', cluster_data['pi_max'])
#             print('200*h_0: ', 200*h_0)
            pred_wp = projected_corr_func(
                training_data_dict['{}_coordinates'.format(data_type)][redshift_indeces, :][mass_indeces, :], 
                cluster_data['rp_bin_edges'][:-1], 
                cluster_data['pi_max'], 
                period=200*h_0
            )

#             print('true wp: ', cluster_data['wp'][bin_num - 1][:-1])
#             print('pred wp: ', pred_wp)
            bin_loss = data_processing.chi_squared_loss(
                pred_wp, cluster_data['wp'][bin_num - 1][:-1], cluster_data['errors'][bin_num - 1][-1]
            )
#             print('size of wp loss: ', bin_loss)
            loss += bin_loss
        else:
            loss += 1e400 # arbitrary big number if the stellar mass bin happens to be empty (it shouldn't be)
        
    loss /= len(bin_means)
    return loss


def clustering_distribution(training_data_dict, stellar_masses, loss_dict, data_type, h_0=0.6781):
    ### Returns the predicted and the observed 2p correlation function. h_0 is assumed to be 0.6781 and the box side length 200Mpc. Compares the observational data at 0.08z with simulated data at 0.1z.###
    
    compared_redshift = 0.1
    
    cluster_data = training_data_dict['real_clustering_data'][data_type]
    redshift_index = training_data_dict['unique_redshifts'].index(compared_redshift)
    redshift_indeces = training_data_dict['data_redshifts']['{}_data'.format(data_type)] == compared_redshift
    stellar_masses = stellar_masses[redshift_indeces]
    
    bin_means, bin_edges, bin_numbers = binned_statistic(stellar_masses, stellar_masses, 
                                                         bins=cluster_data['stellar_mass_bin_edges'], statistic='mean')

    pred_wp = []
    true_wp = []
    bin_mids = ((cluster_data['rp_bin_edges'][1:] + cluster_data['rp_bin_edges'][:-1]) / 2)[:-1]
    obs_errors = []
    
    
    for bin_num in range(1, len(bin_means) + 1):
        
        mass_indeces = bin_numbers == bin_num
        
        if np.any(mass_indeces):
            pred_wp_bin = projected_corr_func(
                training_data_dict['{}_coordinates'.format(data_type)][redshift_indeces, :][mass_indeces, :], 
                cluster_data['rp_bin_edges'][:-1], 
                cluster_data['pi_max'], 
                period=200*h_0
            )

        else:
            print('no stars in bin ', bin_edges[bin_num-1], '-', bin_edges[bin_num])
            pred_wp_bin = []

        true_wp.append(cluster_data['wp'][bin_num-1][:-1])
        pred_wp.append(pred_wp_bin)
        obs_errors.append(cluster_data['errors'][bin_num-1][:-1])
        
    return [pred_wp, true_wp, bin_mids, obs_errors, cluster_data['stellar_mass_bin_edges']]


def binned_loss(training_data_dict, binning_feat, bin_feat, bin_feat_name, data_type, loss_dict, real_obs):

    """
    function returning the average loss/obs_data_points for fq, smf and ssfr. 
    
    keys:
        binning feat -- the feature that the data will be binned according to, happens to be stellar mass in all three function.
        bin feat -- log(ssfr) for bin_feat_name = ssfr or fq and log(stellar_mass) for smf
        bin_feat_name -- the name of the statistical measure that you want to produce
        data_type -- 'train' or 'val' 
    """
    
    if real_obs:
        
        # if one wants to train on only a few redshifts. not really used anymore (didn't work out great)
        if data_type != 'train' or loss_dict['nr_redshifts_per_eval'] == 'all':
            nr_redshifts = len(training_data_dict['unique_redshifts'])
        else:
            nr_redshifts = loss_dict['nr_redshifts_per_eval']
        evaluated_redshift_indeces = np.random.choice(len(training_data_dict['unique_redshifts']), nr_redshifts, replace=False)
        evaluated_redshifts = np.array(training_data_dict['unique_redshifts'])[evaluated_redshift_indeces]
        
        scale_factor_of_pred_points = []
        binning_feat_values_pred_points = []
        pred_bin_feat = []
        
        for i, (i_red, redshift) in enumerate(zip(evaluated_redshift_indeces, evaluated_redshifts)):

            relevant_inds = training_data_dict['data_redshifts']['{}_data'.format(data_type)] == redshift
                        
            bin_means, bin_edges, bin_numbers = binned_statistic(binning_feat[relevant_inds], bin_feat[relevant_inds], 
                                                                 bins=loss_dict['stellar_mass_bins'], statistic='mean')
            bin_mids = np.array([(bin_edges[i+1] - bin_edges[i])/2 for i in range(len(bin_means))])

            if bin_feat_name == 'smf':
                bin_counts = [np.sum(bin_numbers == i) for i in range(1, len(bin_means)+1)]
                bin_counts = np.array(bin_counts, dtype=np.float)
                nonzero_inds = np.nonzero(bin_counts)
                bin_counts = bin_counts[nonzero_inds]
                bin_mids = bin_mids[nonzero_inds]

                bin_widths = bin_edges[1] - bin_edges[0]
                pred_bin_feat_dist = bin_counts / 200**3 / bin_widths

                # since we might only be using a subset of the original data points, compensate for this
                pred_bin_feat_dist /= training_data_dict['{}_frac_of_tot_by_redshift'.format(data_type)][i]

                pred_bin_feat_dist = np.log10(pred_bin_feat_dist)

            elif bin_feat_name == 'fq':

                scale_factor = 1 / (1 + redshift)

                H_0 = 67.81 / (3.09e19) # 1/s
                H_0 = H_0 * 60 * 60 * 24 * 365 # 1/yr
                h_r = H_0 * np.sqrt(1e-3*scale_factor**(-4) + 0.308*scale_factor**(-3) + 0*scale_factor**(-2) + 0.692)
                ssfr_cutoff = 0.3*h_r
                log_ssfr_cutoff = np.log10(ssfr_cutoff)

                pred_bin_feat_dist = []
                nonzero_inds = []
                for bin_num in range(1, len(bin_means)+1):

                    if len(bin_feat[relevant_inds][bin_numbers == bin_num]) != 0:
                        fq = np.sum(bin_feat[relevant_inds][bin_numbers == bin_num] < log_ssfr_cutoff) \
                            / len(bin_feat[relevant_inds][bin_numbers == bin_num])
                        pred_bin_feat_dist.append(fq)
                        nonzero_inds.append(bin_num-1)
                bin_mids = bin_mids[nonzero_inds]

            else:    
                pred_bin_feat_dist = bin_means[np.invert(np.isnan(bin_means))]
                
                bin_mids = bin_mids[np.invert(np.isnan(bin_means))]

            scale_factor_of_pred_points.extend([1 / (1 + redshift)] * len(pred_bin_feat_dist))
            binning_feat_values_pred_points.extend(bin_mids)
            pred_bin_feat.extend(pred_bin_feat_dist)
        
        # Create the spline function based on predictions
        if len(pred_bin_feat) > 16:
#             print('bin_mids: ', len(bin_mids))
#             print('scale_factor_of_pred_points: ', len(scale_factor_of_pred_points))
#             print('pred_bin_feat: ', len(pred_bin_feat))
            with warnings.catch_warnings():
                with suppress_stdout():
                    warnings.simplefilter("ignore")
                    spline = SmoothBivariateSpline(binning_feat_values_pred_points, scale_factor_of_pred_points, pred_bin_feat)
        #             print('spline: ', spline)
                    pred_observations = spline.ev(
                        training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['binning_feat'],
                        training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['scale_factor']
                    )

            loss = data_processing.chi_squared_loss(pred_observations, 
                                    training_data_dict['real_{}_data'.format(bin_feat_name)][data_type][bin_feat_name], 
                                    training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['errors'])
#             print('size of {} loss: '.format(bin_feat_name), loss)

        else:
            loss = 1e100

        return loss
        
        
    else:
    
        loss = 0
        dist_outside = 0
    #     nr_empty_bins = np.zeros(len(training_data_dict['unique_redshifts']))
        tot_nr_points = 0
        frac_of_non_covered_interval = 0

        if data_type != 'train' or loss_dict['nr_redshifts_per_eval'] == 'all':
            nr_redshifts = len(training_data_dict['unique_redshifts'])
        else:
            nr_redshifts = loss_dict['nr_redshifts_per_eval']
        evaluated_redshift_indeces = np.random.choice(len(training_data_dict['unique_redshifts']), nr_redshifts, replace=False)
        evaluated_redshifts = np.array(training_data_dict['unique_redshifts'])[evaluated_redshift_indeces]
        frac_outside = np.zeros(len(evaluated_redshifts))


        for i, (i_red, redshift) in enumerate(zip(evaluated_redshift_indeces, evaluated_redshifts)):

            relevant_inds = training_data_dict['data_redshifts']['{}_data'.format(data_type)] == redshift

            bin_edges = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges']

            nr_points_outside_binning_feat_range = \
                np.sum(binning_feat[relevant_inds] < 
                      training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][0]) \
              + np.sum(binning_feat[relevant_inds] > 
                      training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][-1])
            nr_points_redshift = len(binning_feat[relevant_inds])
            tot_nr_points += nr_points_redshift

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
                pred_bin_feat_dist /= training_data_dict['{}_frac_of_tot_by_redshift'.format(data_type)][i]

                non_nan_indeces = np.invert(np.isnan(pred_bin_feat_dist))
                pred_bin_feat_dist[non_nan_indeces] = np.log10(pred_bin_feat_dist[non_nan_indeces])

            elif bin_feat_name == 'fq':

                scale_factor = 1 / (1 + redshift)

                H_0 = 67.81 / (3.09e19) # 1/s
                H_0 = H_0 * 60 * 60 * 24 * 365 # 1/yr
                h_r = H_0 * np.sqrt(1e-3*scale_factor**(-4) + 0.308*scale_factor**(-3) + 0*scale_factor**(-2) + 0.692)
                ssfr_cutoff = 0.3*h_r
                log_ssfr_cutoff = np.log10(ssfr_cutoff)

                pred_bin_feat_dist = np.zeros(n_bins)
                for bin_num in range(1, n_bins+1):

                    if len(bin_feat[relevant_inds][bin_numbers == bin_num]) != 0:
                        fq = np.sum(bin_feat[relevant_inds][bin_numbers == bin_num] < log_ssfr_cutoff) \
                            / len(bin_feat[relevant_inds][bin_numbers == bin_num])
                    else:
                        fq = float('nan')

                    pred_bin_feat_dist[bin_num-1] = fq

                non_nan_indeces = np.invert(np.isnan(pred_bin_feat_dist))

            else:    
                pred_bin_feat_dist = bin_means
                non_nan_indeces = np.invert(np.isnan(pred_bin_feat_dist))

    #         nr_empty_bins[i_red] = np.sum(np.invert(non_nan_indeces))
            frac_outside[i] = nr_points_outside_binning_feat_range / nr_points_redshift

            if (np.sum(non_nan_indeces) > 0 and np.sum(non_nan_indeces)/n_bins > loss_dict['min_filled_bin_frac']):

                loss += data_processing.chi_squared_loss(true_bin_feat_dist[non_nan_indeces], pred_bin_feat_dist[non_nan_indeces], 
                                    training_data_dict['real_{}_data'.format(bin_feat_name)]['errors'])
    #             if bin_feat_name == 'shm':
    #                 print('true_bin_feat_dist[non_nan_indeces]: ', true_bin_feat_dist[non_nan_indeces])
    #                 print('pred_bin_feat_dist[non_nan_indeces]: ', pred_bin_feat_dist[non_nan_indeces])
            elif np.sum(non_nan_indeces) > 0 and np.sum(non_nan_indeces)/n_bins < loss_dict['min_filled_bin_frac']:
                loss += data_processing.chi_squared_loss(true_bin_feat_dist[non_nan_indeces], pred_bin_feat_dist[non_nan_indeces], 
                                    training_data_dict['real_{}_data'.format(bin_feat_name)]['errors']) \
                         + np.sum(np.invert(non_nan_indeces))
            else:
                loss += 1000

            if bin_feat_name != 'smf':
                bin_widths = training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_widths']
                bin_interval_length = bin_edges[-1] - bin_edges[0] - bin_widths # the wanted range of predictions
                pred_interval_length = np.max(bin_feat[relevant_inds]) - np.min(bin_feat[relevant_inds])
                if pred_interval_length < bin_interval_length:
                    frac_of_non_covered_interval += 1 - pred_interval_length / bin_interval_length

        # Get the dist outside per redshift measured
        dist_outside /= nr_redshifts
        frac_of_non_covered_interval /= nr_redshifts
        loss /= nr_redshifts

        if loss_dict != None:

            dist_outside_punish = loss_dict['dist_outside_punish']
            no_coverage_punish = loss_dict['no_coverage_punish']

            if no_coverage_punish == 'exp':
                theta = loss_dict['no_coverage_factor']
                loss*= np.exp(theta * frac_of_non_covered_interval)
            elif no_coverage_punish == 'none':
                pass
            else:
                print('do not recognise the no_coverage_punish')

            if dist_outside_punish == 'exp':
                xi = loss_dict['dist_outside_factor']
                loss*= np.exp(xi * dist_outside/tot_nr_points)
            elif dist_outside_punish == 'lin':
                slope = loss_dict['dist_outside_factor']
                redshift_score *= (1 + slope*frac_outside)
            elif dist_outside_punish == 'none':
                pass
            else:
                print('do not recognise the dist_outside_punish')

        #                     theta = 10
        #                     redshift_score *= (1 + np.exp((frac_outside - 0.1) * theta))

        return [loss, dist_outside]
    
    
def get_ssfr_fq_smf_splines(training_data_dict, binning_feat, bin_feat, bin_feat_name, data_type, loss_dict, real_obs, 
                            grid_points=100):
    todo: implement the function that receives the output from this function (spline_plots which is in turn called on by get_ssfr_smf_fq_surface_plot)
    if real_obs:

#         # if one wants to train on only a few redshifts. not really used anymore (didn't work out great)
#         if data_type != 'train' or loss_dict['nr_redshifts_per_eval'] == 'all':
#             nr_redshifts = len(training_data_dict['unique_redshifts'])
#         else:
#             nr_redshifts = loss_dict['nr_redshifts_per_eval']
#         evaluated_redshift_indeces = np.random.choice(len(training_data_dict['unique_redshifts']), nr_redshifts, replace=False)
#         evaluated_redshifts = np.array(training_data_dict['unique_redshifts'])[evaluated_redshift_indeces]

        scale_factor_of_pred_points = []
        binning_feat_values_pred_points = []
        pred_bin_feat = []

        for i, (i_red, redshift) in enumerate(zip(evaluated_redshift_indeces, evaluated_redshifts)):

            relevant_inds = training_data_dict['data_redshifts']['{}_data'.format(data_type)] == redshift

            bin_means, bin_edges, bin_numbers = binned_statistic(binning_feat[relevant_inds], bin_feat[relevant_inds], 
                                                                 bins=loss_dict['stellar_mass_bins'], statistic='mean')
            bin_mids = np.array([(bin_edges[i+1] - bin_edges[i])/2 for i in range(len(bin_means))])

            if bin_feat_name == 'smf':
                bin_counts = [np.sum(bin_numbers == i) for i in range(1, len(bin_means)+1)]
                bin_counts = np.array(bin_counts, dtype=np.float)
                nonzero_inds = np.nonzero(bin_counts)
                bin_counts = bin_counts[nonzero_inds]
                bin_mids = bin_mids[nonzero_inds]

                bin_widths = bin_edges[1] - bin_edges[0]
                pred_bin_feat_dist = bin_counts / 200**3 / bin_widths

                # since we might only be using a subset of the original data points, compensate for this
                pred_bin_feat_dist /= training_data_dict['{}_frac_of_tot_by_redshift'.format(data_type)][i]

                pred_bin_feat_dist = np.log10(pred_bin_feat_dist)

            elif bin_feat_name == 'fq':

                scale_factor = 1 / (1 + redshift)

                H_0 = 67.81 / (3.09e19) # 1/s
                H_0 = H_0 * 60 * 60 * 24 * 365 # 1/yr
                h_r = H_0 * np.sqrt(1e-3*scale_factor**(-4) + 0.308*scale_factor**(-3) + 0*scale_factor**(-2) + 0.692)
                ssfr_cutoff = 0.3*h_r
                log_ssfr_cutoff = np.log10(ssfr_cutoff)

                pred_bin_feat_dist = []
                nonzero_inds = []
                for bin_num in range(1, len(bin_means)+1):

                    if len(bin_feat[relevant_inds][bin_numbers == bin_num]) != 0:
                        fq = np.sum(bin_feat[relevant_inds][bin_numbers == bin_num] < log_ssfr_cutoff) \
                            / len(bin_feat[relevant_inds][bin_numbers == bin_num])
                        pred_bin_feat_dist.append(fq)
                        nonzero_inds.append(bin_num-1)
                bin_mids = bin_mids[nonzero_inds]

            else:    
                pred_bin_feat_dist = bin_means[np.invert(np.isnan(bin_means))]

                bin_mids = bin_mids[np.invert(np.isnan(bin_means))]

            scale_factor_of_pred_points.extend([1 / (1 + redshift)] * len(pred_bin_feat_dist))
            binning_feat_values_pred_points.extend(bin_mids)
            pred_bin_feat.extend(pred_bin_feat_dist)

        # Create the spline function based on predictions
        if len(pred_bin_feat) > 16:
    #             print('bin_mids: ', len(bin_mids))
    #             print('scale_factor_of_pred_points: ', len(scale_factor_of_pred_points))
    #             print('pred_bin_feat: ', len(pred_bin_feat))
            with warnings.catch_warnings():
                with suppress_stdout():
                    warnings.simplefilter("ignore")
                    spline = SmoothBivariateSpline(binning_feat_values_pred_points, scale_factor_of_pred_points, pred_bin_feat)
        #             print('spline: ', spline)
                    pred_observations = spline.ev(
                        training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['binning_feat'],
                        training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['scale_factor']
                    )

        # make the grid 
        min_pred_scale_factor = np.min(scale_factor_of_pred_points)
        max_pred_scale_factor = np.max(scale_factor_of_pred_points)
        min_pred_stellar_mass = np.min(binning_feat_values_pred_points)
        max_pred_stellar_mass = np.max(binning_feat_values_pred_points)
        
        scale_factor_vals = []
        stellar_mass_vals = []
        for i in np.linspace(min_pred_scale_factor, max_pred_scale_factor, num=grid_points):
            for j in np.linspace(min_pred_stellar_mass, max_pred_stellar_mass, num=grid_points):
                scale_factor_vals.append(i)
                stellar_mass_vals.append(j)
        pred_vals = spline.ev(x_points, y_points)

        return [scale_factor_vals, stellar_mass_vals, pred_vals, obs_vals, 
                training_data_dict['real_{}_data'.format(bin_feat_name)][data_type][bin_feat_name]]
    
    else: # use mock observations
        pass
    
    
def binned_dist_func(training_data_dict, binning_feat, bin_feat, bin_feat_name, data_type, full_range, real_obs, loss_dict=None):
            
    pred_bin_feat_dists = []
    true_bin_feat_dists = []
    obs_errors = []
    redshifts = []
    pred_bin_centers = []
    obs_bin_centers = []

    acceptable_binning_feat_intervals = []
    
    frac_outside = np.zeros(len(training_data_dict['unique_redshifts']))
    
    if real_obs:
        
        for i_red, redshift in enumerate(training_data_dict['unique_redshifts']):

            relevant_inds = training_data_dict['data_redshifts']['{}_data'.format(data_type)] == redshift
                        
            bin_means, bin_edges, bin_numbers = binned_statistic(binning_feat[relevant_inds], bin_feat[relevant_inds], 
                                               bins=loss_dict['stellar_mass_bins'], statistic='mean')
            bin_mids = np.array([(bin_edges[i+1] + bin_edges[i])/2 for i in range(len(bin_means))])
#             print('bin edges: ', bin_edges)
#             print(bin_feat_name, 'bin mids: ', bin_mids)
#             print('predicted stellar masses: ', binning_feat[relevant_inds])
#             print('number of bins: ', loss_dict['nr_bins_real_obs'])

            if bin_feat_name == 'smf':
                bin_counts = [np.sum(bin_numbers == i) for i in range(1, len(bin_means)+1)]
                bin_counts = np.array(bin_counts, dtype=np.float)
                nonzero_inds = np.nonzero(bin_counts)
                bin_counts = bin_counts[nonzero_inds]
                bin_mids = bin_mids[nonzero_inds]

                bin_widths = bin_edges[1] - bin_edges[0]
                pred_bin_feat_dist = bin_counts / 200**3 / bin_widths

                # since we might only be using a subset of the original data points, compensate for this
                pred_bin_feat_dist /= training_data_dict['{}_frac_of_tot_by_redshift'.format(data_type)][i_red]

                pred_bin_feat_dist = np.log10(pred_bin_feat_dist)

            elif bin_feat_name == 'fq':

                scale_factor = 1 / (1 + redshift)

                H_0 = 67.81 / (3.09e19) # 1/s
                H_0 = H_0 * 60 * 60 * 24 * 365 # 1/yr
                h_r = H_0 * np.sqrt(1e-3*scale_factor**(-4) + 0.308*scale_factor**(-3) + 0*scale_factor**(-2) + 0.692)
                ssfr_cutoff = 0.3*h_r
                log_ssfr_cutoff = np.log10(ssfr_cutoff)

                pred_bin_feat_dist = []
                nonzero_inds = []
                for bin_num in range(1, len(bin_means)+1):

                    if len(bin_feat[relevant_inds][bin_numbers == bin_num]) != 0:
                        fq = np.sum(bin_feat[relevant_inds][bin_numbers == bin_num] < log_ssfr_cutoff) \
                            / len(bin_feat[relevant_inds][bin_numbers == bin_num])
                        pred_bin_feat_dist.append(fq)
                        nonzero_inds.append(bin_num-1)
                bin_mids = bin_mids[nonzero_inds]

            else:    
                pred_bin_feat_dist = bin_means[np.invert(np.isnan(bin_means))]
                
                bin_mids = bin_mids[np.invert(np.isnan(bin_means))]
            
            redshift_bin_mids = []
            redshift_bin_feat_values = []
            redshift_errors = []
            for i_point in range(len(training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['binning_feat'])):
                scale_factor = np.round(1/(1+redshift), decimals=2)
                if ( # check if the scale factor of the snapshot is within the covered scale factors from the survey
                    bin_feat_name != 'ssfr'
                    and
                    scale_factor 
                    >=
                    training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['scale_factor_range'][i_point][0]
                    and
                    scale_factor 
                    <=
                    training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['scale_factor_range'][i_point][1]
                ):
                    redshift_bin_mids.append(
                        training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['binning_feat'][i_point]
                    )
                    redshift_bin_feat_values.append(
                        training_data_dict['real_{}_data'.format(bin_feat_name)][data_type][bin_feat_name][i_point]
                    )
                    redshift_errors.append(training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['errors'][i_point])
                elif ( # ssfr has to be matched exactly
                training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['scale_factor'][i_point] 
                    == scale_factor
                ):
                    redshift_bin_mids.append(
                        training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['binning_feat'][i_point]
                    )
                    redshift_bin_feat_values.append(
                        training_data_dict['real_{}_data'.format(bin_feat_name)][data_type][bin_feat_name][i_point]
                    )
                    redshift_errors.append(training_data_dict['real_{}_data'.format(bin_feat_name)][data_type]['errors'][i_point])
#                     print('point added to {} obs data. lens:'.format(bin_feat_name), redshift_bin_mids[-1], 
#                           redshift_bin_feat_values[-1], redshift_errors[-1])
            pred_bin_centers.append(bin_mids)
            obs_bin_centers.append(redshift_bin_mids)
            
            pred_bin_feat_dists.append(pred_bin_feat_dist)
            true_bin_feat_dists.append(redshift_bin_feat_values)
            
            redshifts.append(redshift)
            obs_errors.append(redshift_errors)
            
#         print('in data_processing')
#         print('true_bin_feat_dists: ', true_bin_feat_dists)
#         print('obs_bin_centers: ', obs_bin_centers)
#         print('obs_errors: ', obs_errors)

        return [pred_bin_feat_dists, true_bin_feat_dists, pred_bin_centers, obs_bin_centers, obs_errors, redshifts]
        
    else: # using mock observations

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
                pred_bin_feat_dist_redshift /= training_data_dict['{}_frac_of_tot_by_redshift'.format(data_type)][i_red]

                non_nan_indeces = np.invert(np.isnan(pred_bin_feat_dist_redshift))
                pred_bin_feat_dist_redshift[non_nan_indeces] = np.log10(pred_bin_feat_dist_redshift[non_nan_indeces])

            elif bin_feat_name == 'fq':

                scale_factor = 1 / (1 + redshift)

                H_0 = 67.81 / (3.09e19) # 1/s
                H_0 = H_0 * 60 * 60 * 24 * 365 # 1/yr
                h_r = H_0 * np.sqrt(1e-3*scale_factor**(-4) + 0.308*scale_factor**(-3) + 0*scale_factor**(-2) + 0.692)
                ssfr_cutoff = 0.3*h_r
                log_ssfr_cutoff = np.log10(ssfr_cutoff)

                pred_bin_feat_dist_redshift = np.zeros(n_bins)
                for bin_num in range(1, n_bins+1):

                    if len(bin_feat[relevant_inds][bin_numbers == bin_num]) != 0:
                        fq = np.sum(bin_feat[relevant_inds][bin_numbers == bin_num] < log_ssfr_cutoff) \
                            / len(bin_feat[relevant_inds][bin_numbers == bin_num])
                    else:
                        fq = float('nan')

                    pred_bin_feat_dist_redshift[bin_num-1] = fq


            else:    
                pred_bin_feat_dist_redshift = bin_means

            pred_bin_feat_dist.append(pred_bin_feat_dist_redshift.copy())
            true_bin_feat_dist.append(true_bin_feat_dist_redshift.copy())
            redshifts.append(redshift)
            pred_bin_centers.append([(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
            obs_bin_centers.append(training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_centers'])
            acceptable_binning_feat_intervals.append(
                [training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][0], 
                training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][-1]]
            )

            nr_points_outside_binning_feat_range = \
                np.sum(binning_feat[relevant_inds] < 
                      training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][0]) \
              + np.sum(binning_feat[relevant_inds] > 
                      training_data_dict['{}_data'.format(bin_feat_name)]['{:.1f}'.format(redshift)]['bin_edges'][-1])
            nr_points_redshift = len(binning_feat[relevant_inds])
            frac_outside[i_red] = nr_points_outside_binning_feat_range / nr_points_redshift

        return [pred_bin_feat_dist, true_bin_feat_dist, pred_bin_centers, obs_bin_centers, redshifts, 
                acceptable_binning_feat_intervals, frac_outside]


def loss_func_obs_stats(model, training_data_dict, loss_dict, real_obs=True, data_type='train', batch_size='all_points'):
    
    np.seterr(over='raise', divide='raise')
    
    if real_obs:
        y_pred = data_processing.predict_points(model, training_data_dict, original_units=False, as_lists=False, data_type=data_type)
            
        sfr_index = training_data_dict['network_args']['output_features'].index('SFR')
        stellar_mass_index = training_data_dict['network_args']['output_features'].index('Stellar_mass')

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

        if loss_dict['ssfr_weight'] > 0:
            loss_ssfr = \
                binned_loss(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'ssfr', data_type, loss_dict, True) 
            loss += loss_dict['ssfr_weight'] * loss_ssfr

        ############### SMF ###############  
        
        if loss_dict['smf_weight'] > 0:
            loss_smf = \
                binned_loss(training_data_dict, predicted_stellar_mass_log, predicted_stellar_mass_log, 'smf', data_type, loss_dict,
                            True)
            loss += loss_dict['smf_weight'] * loss_smf

        ############### FQ ###############

        if loss_dict['fq_weight'] > 0:
            loss_fq = \
                binned_loss(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'fq', data_type, loss_dict, True)
            loss += loss_dict['fq_weight'] * loss_fq
            
        ############### CSFRD ###############

        if loss_dict['csfrd_weight'] > 0:
            loss_csfrd = csfrd_loss(
                training_data_dict, predicted_sfr, loss_dict, data_type
            )
            loss += loss_dict['csfrd_weight'] * loss_csfrd
            
        ############### Clustering ###############

        if loss_dict['clustering_weight'] > 0:
            loss_clustering = clustering_loss(
                training_data_dict, predicted_stellar_mass_log, loss_dict, data_type
            )
            loss += loss_dict['clustering_weight'] * loss_clustering
            
        loss /= (loss_dict['ssfr_weight'] + loss_dict['smf_weight'] + loss_dict['fq_weight'] + loss_dict['clustering_weight']
                 + loss_dict['csfrd_weight'])
    
        return loss
                                            
    else:
        
#         if batch_size == 'all_points':
        y_pred = data_processing.predict_points(model, training_data_dict, original_units=False, as_lists=False, data_type=data_type)
#         else
            
        sfr_index = training_data_dict['network_args']['output_features'].index('SFR')
        stellar_mass_index = training_data_dict['network_args']['output_features'].index('Stellar_mass')

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

        if loss_dict['ssfr_weight'] > 0:
            loss_ssfr, dist_outside = \
                binned_loss(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'ssfr', data_type, loss_dict, False)
            loss += loss_dict['ssfr_weight'] * loss_ssfr
            dist_outside_tot += dist_outside
#         nr_empty_bins =+ nr_empty_bins_ssfr

        ############### SMF ###############  
        
        if loss_dict['smf_weight'] > 0:
            loss_smf, dist_outside = \
                binned_loss(training_data_dict, predicted_stellar_mass_log, predicted_stellar_mass_log, 'smf', data_type, loss_dict, 
                            False)
            loss += loss_dict['smf_weight'] * loss_smf
            dist_outside_tot += dist_outside
#         nr_empty_bins =+ nr_empty_bins_smf

        ############### FQ ###############

        if loss_dict['fq_weight'] > 0:
            loss_fq, dist_outside = \
                binned_loss(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'fq', data_type, loss_dict, False)
            loss += loss_dict['fq_weight'] * loss_fq
            dist_outside_tot += dist_outside
#         nr_empty_bins =+ nr_empty_bins_fq

        ############### SHM ###############

        if loss_dict['shm_weight'] > 0:
            loss_shm, dist_outside = \
                binned_loss(training_data_dict, training_data_dict['original_halo_masses_{}'.format(data_type)], 
                                                                   predicted_stellar_mass_log, 'shm', data_type, loss_dict, False)
            loss += loss_dict['shm_weight'] * loss_shm
            dist_outside_tot += dist_outside
#         nr_empty_bins =+ nr_empty_bins_fq

        loss /= (loss_dict['ssfr_weight'] + loss_dict['smf_weight'] + loss_dict['fq_weight'] + loss_dict['shm_weight'])
    
        return loss
    
    
def spline_plots(model, training_data_dict, real_obs=True, csfrd_only=False, clustering_only=False, data_type='train', 
                 full_range=False, loss_dict=None):
    
    if real_obs:
        
        
        
        
        
        
    else:
        pass


def plots_obs_stats(model, training_data_dict, real_obs=True, csfrd_only=False, clustering_only=False, data_type='train', 
                    full_range=False, loss_dict=None):
    
    
    if real_obs:
        
        y_pred = data_processing.predict_points(model, training_data_dict, original_units=True, as_lists=False, data_type=data_type)
        
        sfr_index = training_data_dict['network_args']['output_features'].index('SFR')
        stellar_mass_index = training_data_dict['network_args']['output_features'].index('Stellar_mass')

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

        ############### CSFRD ###############
        if csfrd_only:
            pred_csfrd, true_csfrd, pred_bin_centers_csfrd, obs_bin_centers_csfrd, obs_errors_csfrd = \
                csfrd_distributions(training_data_dict, predicted_sfr, loss_dict, data_type)
            return {
            'csfrd': [pred_csfrd, true_csfrd, pred_bin_centers_csfrd, obs_bin_centers_csfrd, obs_errors_csfrd]
            }

        ############### Clustering ###############                       
        elif clustering_only:
            pred_wp, true_wp, rp_bin_mids, obs_errors_wp, mass_bin_edges_wp = \
                clustering_distribution(training_data_dict, predicted_stellar_mass_log, loss_dict, data_type)
            return {'clustering': [pred_wp, true_wp, rp_bin_mids, obs_errors_wp, mass_bin_edges_wp]}
                 
        else: # get the rest of the plots in the same figure
            ############### mean SSFR ###############

            pred_ssfr, true_ssfr, pred_bin_centers_ssfr, obs_bin_centers_ssfr, obs_errors_ssfr, redshifts_ssfr = \
                binned_dist_func(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'ssfr', data_type, full_range, real_obs,
                                 loss_dict)


            ############### SMF ###############  

            pred_smf, true_smf, pred_bin_centers_smf, obs_bin_centers_smf, obs_errors_smf, redshifts_smf = \
                binned_dist_func(training_data_dict, predicted_stellar_mass_log, predicted_stellar_mass_log, 'smf', data_type, 
                                 full_range, real_obs, loss_dict)

            ############### FQ ###############

            pred_fq, true_fq, pred_bin_centers_fq, obs_bin_centers_fq, obs_errors_fq, redshifts_fq = \
                binned_dist_func(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'fq', data_type, full_range, real_obs,
                                 loss_dict)



            return {
                'ssfr': [pred_ssfr, true_ssfr, pred_bin_centers_ssfr, obs_bin_centers_ssfr, obs_errors_ssfr, redshifts_ssfr],
                'smf': [pred_smf, true_smf, pred_bin_centers_smf, obs_bin_centers_smf, obs_errors_smf, redshifts_smf],
                'fq': [pred_fq, true_fq, pred_bin_centers_fq, obs_bin_centers_fq, obs_errors_fq, redshifts_fq]
            }
    
    else:
        
        y_pred = data_processing.predict_points(model, training_data_dict, original_units=True, as_lists=False, data_type=data_type)
        
        sfr_index = training_data_dict['network_args']['output_features'].index('SFR')
        stellar_mass_index = training_data_dict['network_args']['output_features'].index('Stellar_mass')

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

        (pred_ssfr, true_ssfr, pred_bin_centers_ssfr, obs_bin_centers_ssfr, redshifts_ssfr, obs_mass_interval_ssfr, 
        frac_outside_ssfr) = \
            binned_dist_func(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'ssfr', data_type, full_range)

     
        ############### SMF ###############  

        pred_smf, true_smf, pred_bin_centers_smf, obs_bin_centers_smf, redshifts_smf, obs_mass_interval_smf, frac_outside_smf = \
            binned_dist_func(training_data_dict, predicted_stellar_mass_log, predicted_stellar_mass_log, 'smf', data_type, 
                             full_range)

        ############### FQ ###############

        pred_fq, true_fq, pred_bin_centers_fq, obs_bin_centers_fq, redshifts_fq, obs_mass_interval_fq, frac_outside_fq = \
            binned_dist_func(training_data_dict, predicted_stellar_mass_log, ssfr_log, 'fq', data_type, full_range)
        
        ############### SHM ###############

        pred_shm, true_shm, pred_bin_centers_shm, obs_bin_centers_shm, redshifts_shm, obs_mass_interval_shm, frac_outside_shm = \
            binned_dist_func(training_data_dict, training_data_dict['original_halo_masses_{}'.format(data_type)], 
                             predicted_stellar_mass_log, 'shm', data_type, full_range)      
        
        return {
            'ssfr': [pred_ssfr, true_ssfr, pred_bin_centers_ssfr, obs_bin_centers_ssfr, redshifts_ssfr, obs_mass_interval_ssfr, 
                     frac_outside_ssfr],
            'smf': [pred_smf, true_smf, pred_bin_centers_smf, obs_bin_centers_smf, redshifts_smf, obs_mass_interval_smf, 
                     frac_outside_smf],
            'fq': [pred_fq, true_fq, pred_bin_centers_fq, obs_bin_centers_fq, redshifts_fq, obs_mass_interval_fq, 
                     frac_outside_fq],
            'shm': [pred_shm, true_shm, pred_bin_centers_shm, obs_bin_centers_shm, redshifts_shm, obs_mass_interval_shm, 
                     frac_outside_shm]
#             'predicted_stellar_masses_redshift': predicted_stellar_masses_redshift,
#             'nr_empty_bins_redshift': nr_empty_bins_redshift,
#             'fraction_of_points_outside_redshift': frac_outside_redshift,
#             'acceptable_interval_redshift': acceptable_interval_redshift
        }