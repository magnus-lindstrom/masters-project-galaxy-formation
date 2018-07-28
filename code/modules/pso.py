import numpy as np
import time
import datetime
from keras.models import Sequential
from keras.layers import Dense
from model_setup import *
from data_processing import get_weights, predict_points, loss_func_obs_stats
from sklearn.metrics import mean_squared_error

class Feed_Forward_Neural_Network():
    
    def __init__(self, nr_hidden_layers, nr_neurons_per_lay, input_features, output_features, 
                 activation_function, output_activation, reg_strength):
        
        self.nr_hidden_layers = nr_hidden_layers
        self.nr_neurons_per_lay = nr_neurons_per_lay
        self.input_features = input_features
        self.output_features = output_features
        self.activation_function = activation_function
        self.reg_strength = reg_strength
        
        self.model = standard_network(input_features, output_features, nr_neurons_per_lay, nr_hidden_layers, activation_function, 
                                      output_activation, reg_strength)
        
    def setup_pso(self, pso_param_dict={}, reinf_learning=True, real_observations=True):
        
        self.pso_swarm = PSO_Swarm(self, self.nr_hidden_layers, self.nr_neurons_per_lay, self.input_features, 
                                   self.output_features, self.activation_function, pso_param_dict, self.reg_strength, 
                                   reinf_learning, real_observations)
        
    def train_pso(self, nr_iterations, training_data_dict, speed_check=False, std_penalty=False, verbatim=False):
        
        self.pso_swarm.train_network(nr_iterations, training_data_dict,
                                     std_penalty, speed_check, verbatim)
        

class PSO_Swarm(Feed_Forward_Neural_Network):
    
    def __init__(self, parent, nr_hidden_layers, nr_neurons_per_lay, input_features, output_features, 
                 activation_function, pso_param_dict, reg_strength, reinf_learning, real_observations):
        self.pso_param_dict = {
            'nr_particles': 40,
            'xMin': -10,
            'xMax': 10,
            'alpha': 1,
            'deltaT': 1,
            'c1': 2,
            'c2': 2,
            'inertiaWeightStart': 1.4,
            'inertiaWeightMin': 0.3,
            'explorationFraction': 0.8,
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
        self.nr_variables = self.parent.model.count_params()
        self.nr_hidden_layers = nr_hidden_layers
        self.nr_neurons_per_lay = nr_neurons_per_lay
        self.activation_function = activation_function
        self.input_features = input_features
        self.output_features = output_features
        self.reg_strength = reg_strength
        
        weights = self.parent.model.get_weights()
        self.weight_shapes = []
        
        for mat in weights:
            self.weight_shapes.append(np.shape(mat))
        
        self.best_weights_train = None
        self.best_weights_val = None
        
        self.inertia_weight = self.pso_param_dict['inertiaWeightStart']
        self.vMax = (self.pso_param_dict['xMax']-self.pso_param_dict['xMin']) / self.pso_param_dict['deltaT']
                
        self.initialise_swarm()
        
    def train_network(self, nr_iterations, training_data_dict, std_penalty, speed_check, verbatim):
        
        self.training_data_dict = training_data_dict
        
        self.nr_iterations_trained = nr_iterations
        self.std_penalty = std_penalty
        self.verbatim = verbatim
        
        with open('progress.txt', 'w+') as f:

            # make sure the output isn't just the same
            should_start_fresh = True
            while should_start_fresh:
                should_start_fresh = False

                inertia_weight_reduction = np.exp(np.log(self.pso_param_dict['inertiaWeightMin'] 
                                                         / self.pso_param_dict['inertiaWeightStart'])
                                                         / self.pso_param_dict['explorationFraction'] 
                                                         / nr_iterations)
                inertia_weight = self.pso_param_dict['inertiaWeightStart']

                self.progress = 0
                
                self.validation_score_history = []
                self.training_score_history = []
                
                self.avg_speed_before_history = []
                self.avg_speed_after_history = []
                    
                self.time_since_train_improvement = 0
                self.time_since_val_improvement = 0
                
                self.initialise_swarm()

                glob_start = time.time()
                for iteration in range(nr_iterations):
                    
                    self.time_since_train_improvement += 1
                    self.time_since_val_improvement += 1
                    self.progress = iteration / nr_iterations

                    should_start_fresh, training_is_done, end_train_message = self.check_progress(f, glob_start, iteration)
                    if should_start_fresh or training_is_done:
                        break

                    for i_particle, particle in enumerate(self.particle_list):
                        
                        train_score = particle.evaluate_particle('train')
                        
                        is_swarm_best_train = (train_score < self.swarm_best_train)
                        
                        if is_swarm_best_train:
                            
                            self.update_best_weights(particle, train_score, f, iteration, i_particle)
                            
                    self.update_swarm(speed_check, f)
                    
                    inertia_weight = self.update_inertia_weight(inertia_weight, inertia_weight_reduction, 
                                                                iteration, f)
                    
            end = time.time()
            if end_train_message is None:
                end_train_message = 'Training ran through all {:d} iterations without premature stopping.'.format(nr_iterations)
                
            if self.verbatim:
                print('{}, Training complete. {}'.format(datetime.datetime.now().strftime("%H:%M:%S"), end_train_message))
            f.write('{}, Training complete. {}'.format(datetime.datetime.now().strftime("%H:%M:%S"), end_train_message))
            f.flush()
            
    def update_best_weights(self, particle, train_score, f, iteration, i_particle):
        
        self.best_weights_train = particle.get_weights()
                        
        self.time_since_train_improvement = 0
        self.swarm_best_train = train_score
        self.swarm_best_position = particle.position

        val_score = particle.evaluate_particle('val')
        is_swarm_best_val = (val_score < self.swarm_best_val)
        if is_swarm_best_val: # only update best weights after val highscore
            
            self.best_weights_val = particle.get_weights()
            self.swarm_best_val = val_score
            self.time_since_val_improvement = 0

        self.validation_score_history.append(val_score)
        self.training_score_history.append(train_score)
        
        if self.verbatim:
            print('{}  Iteration {:4d}, particle {:2d}, new swarm best. Train: {:.3e}, Val: {:.3e}'.format(
                  datetime.datetime.now().strftime("%H:%M:%S"), iteration, i_particle, train_score, val_score))
        f.write('{}  Iteration {:4d}, particle {:2d}, new swarm best. Train: {:.3e}, Val: {:.3e}\n'.format(
              datetime.datetime.now().strftime("%H:%M:%S"), iteration, i_particle, train_score, val_score))
        f.flush()
    
    
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
            # see if network has run into a local minima   

            self.set_best_weights('val')
            y_pred = predict_points(self.parent.model, self.training_data_dict, original_units=False, mode='val')

            stds = np.std(y_pred, axis=0)
#             print('\nStandard deviations of predicted parameters (validation set): ', stds)
            should_start_fresh = np.any(stds < self.pso_param_dict['min_std_tol'])
            if should_start_fresh:
                if self.verbatim:
                    print('Restarting training because of too low predicted feature variance in validation set ({:.2e}).\n'.format(
                          np.min(stds)))
                f.write('Restarting training because of too low predicted feature variance in validation set ({:.2e}).\n'.format(
                      np.min(stds)))
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
        
        
    def update_inertia_weight(self, inertia_weight, inertia_weight_reduction, iteration, f):
        
        isExploring = (inertia_weight > self.pso_param_dict['inertiaWeightMin'])
        if isExploring:
            inertia_weight = inertia_weight * inertia_weight_reduction
            isExploring = (inertia_weight > self.pso_param_dict['inertiaWeightMin'])
            if not isExploring:
                if self.verbatim:
                    print('SWITCH TO EPLOIT! Iteration %d/%d.' % (iteration, self.nr_iterations_trained))
                f.write('SWITCH TO EPLOIT! Iteration %d/%d.\n' % (iteration, self.nr_iterations_trained))
                f.flush()
        return inertia_weight
        
    def predict_output_old(self, mode):
        
        if self.training_data_dict['norm'] == 'none':
            y_pred = self.parent.model.predict(self.training_data_dict['x_'+mode])
        else:
            y_pred = self.parent.model.predict(self.training_data_dict['x_'+mode+'_norm'])

        return y_pred
    
    def initialise_swarm(self):
        
        self.particle_list = []
        
        for i in range(self.pso_param_dict['nr_particles']):
            
            particle = PSO_Particle(self)
            self.particle_list.append(particle)
            
        self.swarm_best_train = 1e20
        self.swarm_best_val = 1e20
        self.swarm_best_position = self.particle_list[0].best_position  # arbitrarily take the first position
        self.best_particle_nr = 0
        
    def update_swarm(self, speed_check, f):
        
        self.speeds_before = []
        self.speeds_after = []
        #self.term_one = []
        #self.term_two = []
        #self.too_fast_count = 0
        #self.mean_particle_best_difference = []
        #self.mean_swarm_best_difference = []
        
        #q = np.random.uniform(size = self.nr_variables)
        #r = np.random.uniform(size = self.nr_variables)
        
        for particle in self.particle_list:
            particle.update_particle()
            
        #print('term 1: ', np.mean(self.term_one))
        #print('term 2:', np.mean(self.term_two)) 
        #print('%d/%d particles were too fast.' % (self.too_fast_count, self.pso_param_dict['nr_particles']))
        #print('mean particle best diff: %.2f'% (np.mean(self.mean_particle_best_difference)))
        #print('mean swarm best diff: %.2f'% (np.mean(self.mean_swarm_best_difference)))
        #print('q: ', np.mean(q))
        #print('r: ', np.mean(r))
        avg_speed_before = np.mean(self.speeds_before)
        avg_speed_after = np.mean(self.speeds_after)
        self.avg_speed_before_history.append(avg_speed_before)
        self.avg_speed_after_history.append(avg_speed_after)
        
        if speed_check:
            print('Average speed of the particles before normalization is: ', avg_speed_before)
            print('Average speed of the particles after normalization is: ', avg_speed_after)
            f.write('Average speed of the particles before normalization is: %.2f' % (avg_speed_before))
            f.write('Average speed of the particles after normalization is: %.2f' % (avg_speed_after))
            f.flush()
            
    def set_best_weights(self, mode):
        if mode == 'train':
            self.parent.model.set_weights(self.best_weights_train)
        elif mode == 'val':
            self.parent.model.set_weights(self.best_weights_val)
        
    def evaluate_model(self, mode, weights=None):
            
        y_pred = predict_points(self.parent.model, self.training_data_dict, original_units=False, as_lists=False, mode=mode)
        
        if self.reinf_learning:
            
            score = loss_func_obs_stats(self.parent.model, self.training_data_dict, real_obs=self.train_on_real_obs, mode=mode)
            
        else:

            diff = y_pred - self.training_data_dict['y_'+mode]
            squares = np.power(diff, 2) / np.shape(y_pred[0])
            weighted_squares = squares * self.training_data_dict[mode+'_weights'] / np.sum(self.training_data_dict[mode+'_weights'])

            score = np.sum(weighted_squares)

    #         score = mean_squared_error(self.training_data_dict['y_'+mode], y_pred, self.training_data_dict[mode+'_weights'])

    #         print('score diff: {:.2e}'.format(score2 - score))

            if weights is None:
                weights = np.array([])
                weight_mat_list = self.parent.model.get_weights()
                for weight_mat in weight_mat_list:
                    weights = np.concatenate((weights, np.ndarray.flatten(weight_mat)))

            score = score + np.sum(np.power(weights, 2) * self.reg_strength)
            
#         print(score)
            
        return score
    
    def evaluate_model_old(self, mode):
            
        y_pred = predict_points(self.parent.model, self.training_data_dict, original_units=False, as_lists=False, mode=mode)
        n_points = np.shape(y_pred[0])
        
        score = 0
        for i_output, output in enumerate(self.output_features):
            diff = y_pred[:, i_output] - self.training_data_dict['output_'+mode+'_dict'][output]
            squares = np.power(diff, 2) / n_points
            weighted_squares = squares * self.data_weights[mode][output] / np.sum(self.data_weights[mode][output])
            feature_score = np.sum(squares)  # an error here
            score += np.sum(feature_score)
            
        return score
            
    def evaluate_model_oldest(self, mode):
    
        if self.training_data_dict['norm'] == 'none':
            additional_ending = mode
        else:
            additional_ending = mode + '_norm'
            
        y_pred = self.parent.model.predict(self.training_data_dict['x_'+additional_ending])
    
        if self.loss_function == 'mse':
            n_points, n_outputs = np.shape(y_pred)
            diff = y_pred - self.training_data_dict['y_'+additional_ending]
            square = np.power(diff, 2)
            feature_scores = np.sum(square, 0) / n_points
            score = np.sum(feature_scores)
            
        elif self.loss_function == 'halo_mass_weighted_loss':
            
            end_weights = self.real_halo_masses[mode] / np.sum(self.real_halo_masses[mode])
            weights = np.ones(self.real_halo_masses[mode]) 
            n_points, n_outputs = np.shape(y_pred)
            diff = y_pred - self.training_data_dict['y_'+additional_ending]
            square = np.power(diff, 2)
            transp_square = np.transpose(square)
            weighted_transp_square = np.multiply(transp_square, weights)
            weighted_square = np.transpose(weighted_transp_square)
            feature_scores = np.sum(weighted_square, 0) / n_points
            score = np.sum(feature_scores)
            
        if self.std_penalty:
            pred_stds = np.std(y_pred, axis=0)
            true_stds = np.std(self.training_data_dict['y_'+additional_ending], axis=0)
            std_ratio = np.sum(true_stds / pred_stds)
            if std_ratio > 10:
                encourage_factor = std_ratio / 10
                score = score * encourage_factor
            
        return score
        
class PSO_Particle(PSO_Swarm):
        
    def __init__(self, parent):
        
        self.parent = parent
            
        r1 = np.random.uniform(size=(self.parent.nr_variables))
        r2 = np.random.uniform(size=(self.parent.nr_variables))

        self.position = self.parent.pso_param_dict['xMin'] + r1 * (self.parent.pso_param_dict['xMax'] - 
                                self.parent.pso_param_dict['xMin'])
        self.velocity = self.parent.pso_param_dict['alpha']/self.parent.pso_param_dict['deltaT'] * \
                        ((self.parent.pso_param_dict['xMin'] - self.parent.pso_param_dict['xMax'])/2 + r2 * 
                         (self.parent.pso_param_dict['xMax'] - self.parent.pso_param_dict['xMin']))
        
        self.best_score = 1e20
        self.best_position = self.position
        
    def evaluate_particle(self, mode):
        
        weight_mat_list = self.get_weights()
        self.parent.parent.model.set_weights(weight_mat_list)
        
        score = self.parent.evaluate_model(mode, weights=self.position)
                
        if mode == 'train' and score < self.best_score:
            self.best_score = score
            self.best_position = self.position
            
        return score
            
    def get_weights(self): # get the weight list corresponding to the position in parameter space
        
        weight_mat_list = [] # will contain the weight matrices 
        weight_counter = 0 # to help assign weights and biases to their correct matrix
        
        for mat_shape in self.parent.weight_shapes:
            nr_weights = np.prod(mat_shape)
            weights = self.position[weight_counter : weight_counter+nr_weights]
            mat = np.reshape(weights, mat_shape)
            weight_mat_list.append(mat)
            weight_counter += nr_weights
        
        return weight_mat_list
        
    def get_weights_old(self): # get the weight list corresponding to the position in parameter space
        
        weight_matrix_list = [] # will contain the weight matrices 
        bias_list = []   # will contain the biases

        weight_counter = 0 # to help assign weights and biases to their correct matrix

        ### Extract weight matrices
        input_dim = len(self.parent.input_features)
        output_dim = len(self.parent.output_features)
        weight_matrix = np.zeros((input_dim, self.parent.nr_neurons_per_lay)) 
        for i in range(input_dim):  
            weight_matrix[i,:] = self.position[weight_counter : weight_counter+self.parent.nr_neurons_per_lay]
            weight_counter += self.parent.nr_neurons_per_lay
        weight_matrix_list.append(weight_matrix)

        
        for iLayer in range(self.parent.nr_hidden_layers-1):
            weight_matrix = np.zeros((self.parent.nr_neurons_per_lay, self.parent.nr_neurons_per_lay))
            for iNeuron in range(self.parent.nr_neurons_per_lay):

                weight_matrix[iNeuron,:] = self.position[weight_counter : weight_counter+self.parent.nr_neurons_per_lay]
                weight_counter += self.parent.nr_neurons_per_lay

            weight_matrix_list.append(weight_matrix)

        weight_matrix = np.zeros((self.parent.nr_neurons_per_lay, output_dim))
        for i in range(self.parent.nr_neurons_per_lay):  
            weight_matrix[i,:] = self.position[weight_counter:weight_counter+output_dim]
            weight_counter += output_dim

        weight_matrix_list.append(weight_matrix)

        ### Extract bias vectors
        for iLayer in range(self.parent.nr_hidden_layers):

            bias_vector = self.position[weight_counter : weight_counter+self.parent.nr_neurons_per_lay]
            weight_counter += self.parent.nr_neurons_per_lay

            bias_list.append(bias_vector)

        bias_vector = np.zeros(output_dim)
        bias_vector = self.position[weight_counter : weight_counter+output_dim] # for the output layer
        bias_list.append(bias_vector)

        weight_counter += output_dim
        
        weight_list = [weight_matrix_list, bias_list]

        #print(weight_counter == len(self.position))  # a check if the number of variables is correct
        
        return weight_list

    def update_particle(self):

        q = np.random.uniform()#size = self.parent.nr_variables)
        r = np.random.uniform()#size = self.parent.nr_variables)
        #print(q)
        #print(r)
        particle_best_difference = self.best_position - self.position
        swarm_best_difference = self.parent.swarm_best_position - self.position
        
        #self.parent.mean_particle_best_difference.append(np.mean(np.abs(particle_best_difference)))
        #self.parent.mean_swarm_best_difference.append(np.mean(np.abs(swarm_best_difference)))

        self.velocity = self.parent.inertia_weight * self.velocity + self.parent.pso_param_dict['c1'] * q * \
                        particle_best_difference / self.parent.pso_param_dict['deltaT'] + \
                        self.parent.pso_param_dict['c2'] * r * swarm_best_difference / \
                        self.parent.pso_param_dict['deltaT']
                    
        #self.parent.term_one.append(np.mean(np.abs(self.parent.pso_param_dict['c1'] * q * \
        #                particle_best_difference / self.parent.pso_param_dict['deltaT'])))
        #self.parent.term_two.append(np.mean(np.abs(self.parent.pso_param_dict['c2'] * r * swarm_best_difference / \
        #                self.parent.pso_param_dict['deltaT'])))

        # now limit velocity to vMax
        absolute_velocity_before_normalization = np.sqrt(np.sum(np.power(self.velocity, 2)))
        is_too_fast = absolute_velocity_before_normalization > self.parent.vMax
        if is_too_fast:
            #self.parent.too_fast_count += 1
            self.velocity = self.velocity * self.parent.vMax / absolute_velocity_before_normalization
            
        absolute_velocity_after_normalization = np.sqrt(np.sum(np.power(self.velocity, 2)))

        self.parent.speeds_before.append(absolute_velocity_before_normalization)
        self.parent.speeds_after.append(absolute_velocity_after_normalization)
            
        self.position = self.position + self.velocity * self.parent.pso_param_dict['deltaT']
        
        
