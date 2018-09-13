import os
import sys
from os.path import expanduser
home_dir = expanduser("~")
bp_network_dir = home_dir + '/trained_networks/backprop_trained/'
import numpy as np
import time
import datetime
import multiprocessing as mp
import queue
import pickle
from model_setup import *
from data_processing import get_weights, predict_points, loss_func_obs_stats
from sklearn.metrics import mean_squared_error
from distance_metrics import minkowski_distance
from plotting import get_real_obs_plot, get_smf_ssfr_fq_plot_mock_obs


class Feed_Forward_Neural_Network():
    
    def __init__(self, nr_hidden_layers, nr_neurons_per_lay, input_features, output_features, 
                 activation_function, output_activation, reg_strength, network_name):
        
        self.nr_hidden_layers = nr_hidden_layers
        self.nr_neurons_per_lay = nr_neurons_per_lay
        self.input_features = input_features
        self.output_features = output_features
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.reg_strength = reg_strength
        self.name = network_name
                
    def setup_pso(self, pso_param_dict={}, reinf_learning=True, real_observations=True, nr_processes=30, 
                  start_from_pretrained_net=False, pretrained_net_name=None):
        
        self.pso_swarm = PSO_Swarm(self, self.nr_hidden_layers, self.nr_neurons_per_lay, self.input_features, 
                                   self.output_features, self.activation_function, self.output_activation,
                                   pso_param_dict, self.reg_strength, reinf_learning, real_observations, nr_processes,
                                   start_from_pretrained_net, pretrained_net_name)
        
    def train_pso(self, nr_iterations, training_data_dict, speed_check=False, std_penalty=False, verbatim=False, 
                  draw_figures=True, loss_dict=None):
        
        self.pso_swarm.train_network(nr_iterations, training_data_dict, std_penalty, speed_check, verbatim, draw_figures, 
                                     loss_dict)
        

class PSO_Swarm(Feed_Forward_Neural_Network):
    
    def __init__(self, parent, nr_hidden_layers, nr_neurons_per_lay, input_features, output_features, 
                 activation_function, output_activation, pso_param_dict, reg_strength, reinf_learning, real_observations,
                 nr_processes, start_from_pretrained_net, pretrained_net_name):
        self.pso_param_dict = {
            'nr_particles': 40,
            'xMin': -10,
            'xMax': 10,
            'alpha': 1,
            'delta_t': 1,
            'c1': 2,
            'c2': 2,
            'inertia_weight_start': 1.4,
            'inertia_weight_min': 0.3,
            'exploration_iters': 1000,
            'min_std_tol': 0.01,
            'patience': 300,
            'patience_parameter': 'val',
            'restart_check_interval': 100
        }
    
        if pso_param_dict is not None:
            for key in pso_param_dict:
                if key in self.pso_param_dict:
                    self.pso_param_dict[key] = pso_param_dict[key]
                else:
                    print('\'%s\ is not a valid key. Choose between:' % (key), self.pso_param_dict.keys())
                    break
                    
        self.parent = parent
        self.reinf_learning = reinf_learning
        self.train_on_real_obs = real_observations
        if real_observations:
            self.obs_type = 'real_observations'
        else:
            self.obs_type = 'mock_observations'
            
        self.start_from_pretrained_net = start_from_pretrained_net
        if self.start_from_pretrained_net:
            path = '{}/trained_networks/backprop_and_pso_trained/{}/{}'.format(home_dir, self.obs_type, self.parent.name)
            already_exists = True
            while already_exists:
                if os.path.exists(path):
                    path += '_new'
                else:
                    already_exists = False
                self.model_path = path + '/'
        else:
            path = '{}/trained_networks/pso_trained/{}/{}'.format(home_dir, self.obs_type, self.parent.name)
            already_exists = True
            while already_exists:
                if os.path.exists(path):
                    path += '_new'
                else:
                    already_exists = False
                self.model_path = path + '/'
        self.pretrained_net_name = pretrained_net_name
            
        self.nr_processes = nr_processes
        self.nr_hidden_layers = nr_hidden_layers
        self.nr_neurons_per_lay = nr_neurons_per_lay
        self.activation_function = activation_function
        self.input_features = input_features
        self.output_features = output_features
                
        self.nr_variables, self.weight_shapes = standard_network_get_nr_variables_weight_shapes(input_features, output_features, 
                                                                                                nr_neurons_per_lay, nr_hidden_layers)
        
        self.network_args = [input_features, output_features, nr_neurons_per_lay, nr_hidden_layers, 
                             activation_function, output_activation, reg_strength]
        
        self.best_weights_train = None
        self.best_weights_val = None
        
        self.vMax = (self.pso_param_dict['xMax'] - self.pso_param_dict['xMin']) / self.pso_param_dict['delta_t']
                
    def train_network(self, nr_iterations, training_data_dict, std_penalty, speed_check, verbatim, draw_figs,
                      loss_dict):
        
        self.training_data_dict = training_data_dict
        
        self.nr_iterations_trained = nr_iterations
        self.std_penalty = std_penalty
        self.verbatim = verbatim
        
        self.draw_figs = draw_figs
        
        os.makedirs(os.path.dirname(self.model_path + 'progress.txt'), exist_ok=True)
        with open(self.model_path + 'progress.txt', 'w+') as f:
            
            self.inp_queue = mp.Queue()
            self.results_queue = mp.Queue()
            self.print_dist_queue = mp.Queue()
            if draw_figs:
                self.figure_drawer_queue = mp.Queue()
            
            process_list = []
            for i in range(self.nr_processes):
                process = mp.Process(target=particle_evaluator, args=(self.inp_queue, self.results_queue, training_data_dict, 
                                                                      self.reinf_learning, self.train_on_real_obs, 
                                                                      self.network_args, self.weight_shapes, loss_dict))
                process_list.append(process)
                
            distance_process = mp.Process(target=save_distances, args=(self.print_dist_queue, self.model_path))
            process_list.append(distance_process)
            
            if draw_figs:
                figure_drawer_process = mp.Process(target=figure_drawer, args=(self.figure_drawer_queue, self.model_path, 
                                                                               self.weight_shapes, self.network_args, 
                                                                               training_data_dict, self.train_on_real_obs,
                                                                               loss_dict))
                process_list.append(figure_drawer_process)
            
            for process in process_list:
                process.start()
            
            should_start_fresh = True
            while should_start_fresh:
                should_start_fresh = False

                self.inertia_weight_reduction = np.exp(np.log(self.pso_param_dict['inertia_weight_min'] 
                                                         / self.pso_param_dict['inertia_weight_start'])
                                                         / (self.pso_param_dict['exploration_iters'] / nr_iterations)
                                                         / nr_iterations)
                self.inertia_weight = self.pso_param_dict['inertia_weight_start']
                
                self.progress = 0
                
                self.swarm_best_score_train = float('Inf')
                self.swarm_best_score_val = float('Inf')
                
                self.swarm_best_distance_moved_p_point_one = []
                self.swarm_best_distance_moved_p_one = []
                self.swarm_best_distance_moved_p_two = []
                self.swarm_best_distance_moved_p_inf = []
                                                
                self.validation_score_history = []
                self.training_score_history = []
                
                self.iterations_of_swarm_val_best = []
                self.iterations_of_swarm_train_best = []
                
                self.avg_speed_before_history = []
                self.avg_speed_after_history = []
                    
                self.time_since_train_improvement = 0
                self.time_since_val_improvement = 0
                
                self.initialise_positions_velocities()

                glob_start = time.time()
                for iteration in range(nr_iterations):
                    
                    self.time_since_train_improvement += 1
                    self.time_since_val_improvement += 1
                    self.progress = iteration / nr_iterations

                    should_start_fresh, training_is_done, end_train_message = self.check_progress(f, glob_start, iteration)
                    if should_start_fresh or training_is_done:
                        break
                        
                    if int(iteration/10) == iteration/10:
                        self.print_dist_queue.put([self.positions, iteration])

                    for i_particle in range(self.pso_param_dict['nr_particles']):
                        self.inp_queue.put([self.positions[i_particle], i_particle, 'train'])

                    particle_scores = np.zeros(self.pso_param_dict['nr_particles'])
                    for i_particle in range(self.pso_param_dict['nr_particles']):
                        
                        score, particle_nr = self.results_queue.get()
                        particle_scores[particle_nr] = score
                    
                    self.update_loss_stats(particle_scores, f, iteration)

                    self.update_swarm(speed_check, f)

                    self.update_inertia_weight(iteration, f)
                    
            end = time.time()
            if end_train_message is None:
                end_train_message = 'Training ran through all {:d} iterations without premature stopping.'.format(nr_iterations)
                
            if self.verbatim:
                print('{}, Training complete. {}'.format(datetime.datetime.now().strftime("%H:%M:%S"), end_train_message))
            f.write('{}, Training complete. {}'.format(datetime.datetime.now().strftime("%H:%M:%S"), end_train_message))
            f.flush()
            
            for process in process_list:
                process.terminate()
                process.join()
            
    def update_loss_stats(self, results, f, iteration):
    
        for i_particle, result in enumerate(results):
            
            is_particle_best = result < self.particle_train_best_scores[i_particle]
            
            if is_particle_best:
                self.particle_train_best_scores[i_particle] = result
                self.particle_train_best_pos[i_particle] = self.positions[i_particle]
                
            is_swarm_best_train = result < self.swarm_best_score_train
            
            if is_swarm_best_train:
                
                self.swarm_best_score_train = result
                self.training_score_history.append(result)
                self.iterations_of_swarm_train_best.append(iteration)
                
                self.swarm_best_distance_moved_p_point_one.append(
                    minkowski_distance([self.positions[i_particle], self.swarm_best_pos_train], p=.1)
                )
                self.swarm_best_distance_moved_p_one.append(
                    minkowski_distance([self.positions[i_particle], self.swarm_best_pos_train], p=1)
                )
                self.swarm_best_distance_moved_p_two.append(
                    minkowski_distance([self.positions[i_particle], self.swarm_best_pos_train], p=2)
                )
                self.swarm_best_distance_moved_p_inf = [np.max(np.absolute(self.positions[i_particle] - self.swarm_best_pos_train))]
                distance_dict = {
                    'swarm_best_distance_moved_p_point_one': self.swarm_best_distance_moved_p_point_one,
                    'swarm_best_distance_moved_p_one': self.swarm_best_distance_moved_p_one,
                    'swarm_best_distance_moved_p_two': self.swarm_best_distance_moved_p_two,
                    'swarm_best_distance_moved_p_inf': self.swarm_best_distance_moved_p_inf
                }
                os.makedirs(os.path.dirname(self.model_path + 'swarm_best_distance_moved.p'), exist_ok=True)
                pickle.dump(distance_dict, open(self.model_path + 'swarm_best_distance_moved.p', 'wb'))
                
                self.swarm_best_pos_train = self.positions[i_particle]
                self.time_since_train_improvement = 0
                
                self.inp_queue.put([self.positions[i_particle], i_particle, 
                                    'save {:d} training {}'.format(iteration, self.model_path)])
                out = self.results_queue.get()

                if out != 'save_successful':
                    print('out: ', out)
                    print('model could not be saved')
                
                # see if the result is also the highest val result so far
                self.inp_queue.put([self.positions[i_particle], i_particle, 'val'])
                val_score, stds = self.results_queue.get()
                
                is_swarm_best_val = val_score < self.swarm_best_score_val
                
                if is_swarm_best_val:
                    
                    self.best_val_stds = stds
                    
                    self.swarm_best_pos_val = self.positions[i_particle]
                    self.validation_score_history.append(val_score)
                    self.iterations_of_swarm_val_best.append(iteration)
                    self.swarm_best_score_val = val_score
                    
                    self.time_since_val_improvement = 0
                    
                    if self.draw_figs:
                        self.figure_drawer_queue.put([self.swarm_best_pos_val, iteration])
                    
                    # save the model
                    self.inp_queue.put([self.positions[i_particle], i_particle, 
                                        'save {:d} validation {}'.format(iteration, self.model_path)])
                    out = self.results_queue.get()

                    if out != 'save_successful':
                        print('out: ', out)
                        print('model could not be saved')
                                            
                if self.verbatim:
                    print('{}  Iteration {:4d}, particle {:2d}, new swarm best. Train: {:.3e}, Val: {:.3e}'.format(
                          datetime.datetime.now().strftime("%H:%M:%S"), iteration, i_particle, result, val_score))
                f.write('{}  Iteration {:4d}, particle {:2d}, new swarm best. Train: {:.3e}, Val: {:.3e}\n'.format(
                      datetime.datetime.now().strftime("%H:%M:%S"), iteration, i_particle, result, val_score))
                f.flush()
                
                score_dict = {
                    'iterations_train_best': self.iterations_of_swarm_train_best,
                    'iterations_val_best': self.iterations_of_swarm_val_best,
                    'train_score_history': self.training_score_history,
                    'val_score_history': self.validation_score_history
                }
                pickle.dump(score_dict, open(self.model_path + 'score_history.p', 'wb'))
    
    def check_progress(self, f, glob_start, iteration):
        
        should_start_fresh = False
        training_is_done = False
        end_train_message = None
        
        if self.pso_param_dict['patience_parameter'] == 'val':
            if self.time_since_val_improvement > self.pso_param_dict['patience']:
                training_is_done = True
                end_train_message = 'Early stopping in iteration {}. Max patience for val loss improvement reached ({})'.format(
                                     iteration, self.pso_param_dict['patience'])
        elif self.pso_param_dict['patience_parameter'] == 'train':
            if self.time_since_train_improvement > self.pso_param_dict['patience']:
                training_is_done = True
                end_train_message = 'Early stopping in iteration {}. Max patience for train loss improvement reached ({})'.format(
                                     iteration, self.pso_param_dict['patience'])
        
        if (int(iteration/self.pso_param_dict['restart_check_interval']) == iteration/self.pso_param_dict['restart_check_interval']) \
            and (iteration > 0):
            
            should_start_fresh = np.any(self.best_val_stds < self.pso_param_dict['min_std_tol'])
            if should_start_fresh:
                if self.verbatim:
                    print('Restarting training because of too low predicted feature variance in validation set ({:.2e}).\n'.format(
                          np.min(self.best_val_stds)))
                f.write('Restarting training because of too low predicted feature variance in validation set ({:.2e}).\n'.format(
                      np.min(self.best_val_stds)))
                return [should_start_fresh, training_is_done, end_train_message]

            progress_end = time.time()
            elapsed_so_far = int((progress_end - glob_start) / 60)
            time_remaining = int(elapsed_so_far / iteration * (self.nr_iterations_trained - iteration))

            if self.verbatim:
                print('\n{}, Iteration {:d}\n'.format(datetime.datetime.now().strftime("%H:%M:%S"), iteration))
            f.write('\n{}      '.format(datetime.datetime.now().strftime("%H:%M:%S")))
            f.write('Iterations tried: {:d}/{:d}     '.format(iteration, self.nr_iterations_trained))
            f.write('Elapsed time: {:d}min     '.format(elapsed_so_far))
            f.write('Time remaining: {:d}min.\n'.format(time_remaining))
            f.flush()
            
        return [should_start_fresh, training_is_done, end_train_message]
        
    def update_inertia_weight(self, iteration, f):
        
        isExploring = (self.inertia_weight > self.pso_param_dict['inertia_weight_min'])
        if isExploring:
            self.inertia_weight = self.inertia_weight * self.inertia_weight_reduction
            isExploring = (self.inertia_weight > self.pso_param_dict['inertia_weight_min'])
            if not isExploring:
                if self.verbatim:
                    print('SWITCH TO EPLOIT! Iteration %d/%d.' % (iteration, self.nr_iterations_trained))
                f.write('SWITCH TO EPLOIT! Iteration %d/%d.\n' % (iteration, self.nr_iterations_trained))
                f.flush()
        
    def update_swarm(self, speed_check, f):
        
#         self.speeds_before = []
#         self.speeds_after = []
        
        for i_particle in range(self.pso_param_dict['nr_particles']):
            self.update_particle(i_particle)
        
#         avg_speed_before = np.mean(self.speeds_before)
#         avg_speed_after = np.mean(self.speeds_after)
#         self.avg_speed_before_history.append(avg_speed_before)
#         self.avg_speed_after_history.append(avg_speed_after)
        
        if speed_check:
            print('Average speed of the particles before normalization is: ', avg_speed_before)
            print('Average speed of the particles after normalization is: ', avg_speed_after)
            f.write('Average speed of the particles before normalization is: %.2f' % (avg_speed_before))
            f.write('Average speed of the particles after normalization is: %.2f' % (avg_speed_after))
            f.flush()
    
    def initialise_positions_velocities(self):
                    
        self.positions = []
        self.velocities = []
        self.particle_train_best_scores = []
        self.particle_train_best_pos = []
        
        for i_particle in range(self.pso_param_dict['nr_particles']):
            r1 = np.random.uniform(size=(self.nr_variables))
            r2 = np.random.uniform(size=(self.nr_variables))

            position = self.pso_param_dict['xMin'] + r1 * (self.pso_param_dict['xMax'] - 
                                    self.pso_param_dict['xMin'])
            velocity = self.pso_param_dict['alpha']/self.pso_param_dict['delta_t'] * \
                            ((self.pso_param_dict['xMin'] - self.pso_param_dict['xMax'])/2 + r2 * 
                             (self.pso_param_dict['xMax'] - self.pso_param_dict['xMin']))

            starting_score = 1e20
            
            self.positions.append(position)
            self.velocities.append(velocity)
            self.particle_train_best_scores.append(starting_score)
            self.particle_train_best_pos.append(position)
            
        if self.start_from_pretrained_net:
            if os.path.exists(bp_network_dir + self.pretrained_net_name + '/best_position.p'):
                best_pos = pickle.load(open(bp_network_dir + self.pretrained_net_name + '/best_position.p', 'rb'))
                self.positions[0] = best_pos
            else:
                print('In subprocess:', os.getpid(),', no pretrained network exists with that name.')
                print(bp_network_dir + self.pretrained_net_name + '/best_position.p')
                sys.exit()
        
        self.swarm_best_pos_train = self.positions[0]
        self.swarm_best_pos_val = self.positions[0]
   
    def update_particle(self, i_particle):

        q = np.random.uniform()
        r = np.random.uniform()
        
        particle_best_difference = self.particle_train_best_pos[i_particle] - self.positions[i_particle]
        swarm_best_difference = self.swarm_best_pos_train - self.positions[i_particle]

        self.velocities[i_particle] = self.inertia_weight * self.velocities[i_particle] + self.pso_param_dict['c1'] * q * \
                                      particle_best_difference / self.pso_param_dict['delta_t'] + \
                                      self.pso_param_dict['c2'] * r * swarm_best_difference / \
                                      self.pso_param_dict['delta_t']
        
        # now limit velocity to vMax
        absolute_velocity_before_normalization = np.sqrt(np.sum(np.power(self.velocities[i_particle], 2)))
        is_too_fast = absolute_velocity_before_normalization > self.vMax
        if is_too_fast:
            #self.parent.too_fast_count += 1
            self.velocities[i_particle] = self.velocities[i_particle] * self.vMax / absolute_velocity_before_normalization
            
        absolute_velocity_after_normalization = np.sqrt(np.sum(np.power(self.velocities[i_particle], 2)))

#         self.parent.speeds_before.append(absolute_velocity_before_normalization)
#         self.parent.speeds_after.append(absolute_velocity_after_normalization)
            
        self.positions[i_particle] = self.positions[i_particle] + self.velocities[i_particle] * self.pso_param_dict['delta_t']
        
        
def particle_evaluator(inp_queue, results_queue, training_data_dict, reinf_learning, train_on_real_obs, network_args, 
                       weight_shapes, loss_dict):

    input_features, output_features, nr_neurons_per_lay, nr_hidden_layers, \
                    activation_function, output_activation, reg_strength = network_args
    model = standard_network(input_features, output_features, nr_neurons_per_lay, nr_hidden_layers, 
                             activation_function, output_activation, reg_strength)
    
    keep_evaluating = True
    while keep_evaluating:
        position, particle_nr, string = inp_queue.get()
        
        if position is None:
            keep_evaluating = False
            print('particle evaluator stopped')
        else:

            weight_mat_list = get_weights(position, weight_shapes)
            model.set_weights(weight_mat_list)

            str_list = string.split(' ')            
            
            if len(str_list) == 1:
                data_type = str_list[0]
                score = evaluate_model(model, training_data_dict, reinf_learning, train_on_real_obs, data_type=data_type,
                                       loss_dict=loss_dict)
                if data_type == 'train':
                    results_queue.put([score, particle_nr])
                elif data_type == 'val':
                    y_pred = predict_points(model, training_data_dict, original_units=False, data_type=data_type)
                    stds = np.std(y_pred, axis=0)
                    results_queue.put([score, stds])
                
            elif len(str_list) == 4:
                iteration = str_list[1]
                mode_of_hs = str_list[2]
                model_path = str_list[3]
                
                directory = '{}{}_best/'.format(model_path, mode_of_hs)
                
                dir_exists = os.path.exists(directory)
                if not dir_exists:
                    os.makedirs(os.path.dirname(directory + 'iter_{}.h5'.format(iteration)), exist_ok=True)
                    
                    for the_file in os.listdir(directory):
                        file_path = os.path.join(directory, the_file)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            
                    pickle.dump(training_data_dict, open(model_path + 'training_data_dict.p', 'wb'))
            
                model.save(directory + 'iter_{}.h5'.format(iteration))
                
                results_queue.put('save_successful')
                
            else:
                print(string)
                print(str_list)
                print('wrong string sent')
                break
                

def get_weights(position, weight_shapes): # get the weight list corresponding to the position in parameter space

    weight_mat_list = [] # will contain the weight matrices 
    weight_counter = 0 # to help assign weights and biases to their correct matrix

    for mat_shape in weight_shapes:
        nr_weights = np.prod(mat_shape)
        weights = position[weight_counter : weight_counter+nr_weights]
        mat = np.reshape(weights, mat_shape)
        weight_mat_list.append(mat)
        weight_counter += nr_weights
        
    return weight_mat_list

def evaluate_model(model, training_data_dict, reinf_learning, train_on_real_obs, data_type, loss_dict):
    
    if reinf_learning:
            
        score = loss_func_obs_stats(model, training_data_dict, loss_dict, real_obs=train_on_real_obs, data_type=data_type)
            
    else:

        y_pred = predict_points(model, training_data_dict, original_units=False, as_lists=False, data_type=data_type)
    #     diff = y_pred - self.training_data_dict['y_'+data_type]
    #     squares = np.power(diff, 2) / np.shape(y_pred[0])
    #     weighted_squares = squares * self.training_data_dict[data_type+'_weights'] / 
#             np.sum(self.training_data_dict[data_type+'_weights'])
    #     score = np.sum(self.training_data_dict[data_type+'_weights'])

        score = mean_squared_error(training_data_dict['y_'+data_type], y_pred, training_data_dict[data_type+'_weights'])
    #     print('score diff: {:.2e}'.format(score2 - score))

    return score

def save_distances(queue, model_path):
        
    keep_evaluating = True
    
    while keep_evaluating:
        positions, iteration = queue.get()
        
        p_point_one = minkowski_distance(positions, p=.1)
        p_one = minkowski_distance(positions, p=1)
        p_two = minkowski_distance(positions, p=2)

        distance_dict = {
            'p_point_one': p_point_one,
            'p_one': p_one,
            'p_two': p_two
        }

        file_path = '{}interparticle_distances/iter_{:d}.p'.format(model_path, iteration)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        pickle.dump(distance_dict, open(file_path, 'wb'))

def figure_drawer(queue, model_path, weight_shapes, network_args, training_data_dict, real_obs, loss_dict):
    
    input_features, output_features, nr_neurons_per_lay, nr_hidden_layers, \
                    activation_function, output_activation, reg_strength = network_args
    model = standard_network(input_features, output_features, nr_neurons_per_lay, nr_hidden_layers, 
                             activation_function, output_activation, reg_strength)
    
    keep_evaluating = True
    
    while keep_evaluating:
        position, iteration = queue.get()
        
        weight_mat_list = get_weights(position, weight_shapes)
        model.set_weights(weight_mat_list)
        
        if real_obs: # first plot contains only csfrd, which is not redshift specific
            title = 'Iteration {}, best validation weights, validation data points shown'.format(iteration)
            fig_csfrd_file_path = '{}figures_validation_weights/val_data/all_losses/csfrd/iteration_{}.png'.format(
                model_path, iteration
            )
            get_real_obs_plot(model, training_data_dict, csfrd_plot=True, title=title, data_type='val', 
                              save=True, file_path=fig_csfrd_file_path, running_from_script=True, loss_dict=loss_dict)
            
        if real_obs: # second plot is of the projected correlation function, also not redshift specific
            title = 'Iteration {}, best validation weights, validation data points shown'.format(iteration)
            fig_wp_file_path = '{}figures_validation_weights/val_data/all_losses/wp/iteration_{}.png'.format(
                model_path, iteration
            )
            get_real_obs_plot(model, training_data_dict, clustering_plot=True, title=title, data_type='val', 
                              save=True, file_path=fig_wp_file_path, running_from_script=True, loss_dict=loss_dict)
            
        for redshift in training_data_dict['unique_redshifts']:
        
            title = 'Redshift {:.1f}, iteration {}, best validation weights, validation data points shown'.format(redshift, iteration)
            fig_file_path = '{}figures_validation_weights/val_data/all_losses/Z{:02.0f}/iteration_{}.png'.format(
                model_path, redshift*10, iteration
            )
            if real_obs:
                # second plot contains redshift specific quantities
                get_real_obs_plot(model, training_data_dict, redshift=redshift, title=title, data_type='val', 
                                  save=True, file_path=fig_file_path, running_from_script=True, loss_dict=loss_dict)
            else:
                get_smf_ssfr_fq_plot_mock_obs(model, training_data_dict, redshift=redshift, title=title, data_type='val', 
                                              full_range=True, save=True, file_path=fig_file_path, running_from_script=True)
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        









