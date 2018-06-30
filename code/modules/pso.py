import numpy as np
import time
import datetime
from keras.models import Sequential
from keras.layers import Dense
from model_setup import *

class Feed_Forward_Neural_Network():
    
    def __init__(self, nr_hidden_layers, nr_neurons_per_lay, input_features, output_features, 
                 activation_function, output_activation, reg_strength, outputs_to_weigh):
        
        self.nr_hidden_layers = nr_hidden_layers
        self.nr_neurons_per_lay = nr_neurons_per_lay
        self.input_features = input_features
        self.output_features = output_features
        self.activation_function = activation_function
        
        self.model = standard_network(input_features, output_features, neurons_per_layer, nr_layers, activation_function, 
                                      output_activation, reg_strength)
        
        self.train_weights, self.val_weights, self.test_weights = get_weights(training_data_dict, output_features, outputs_to_weigh, 
                                                                              weigh_by_redshift=False)
        
    def setup_pso(self, pso_param_dict={}):
        
        self.pso_swarm = PSO_Swarm(self, self.nr_hidden_layers, self.nr_neurons_per_lay, self.input_features, 
                                   self.output_features, self.activation_function, pso_param_dict=pso_param_dict)
        
    def train_pso(self, nr_iterations, training_data_dict, loss_func_dict, nr_iters_before_restart_check=None, 
                  speed_check=False, std_penalty=False):
        
        self.pso_swarm.train_network(nr_iterations, training_data_dict, loss_func_dict, nr_iters_before_restart_check, speed_check,
                                     std_penalty)
        

class PSO_Swarm(Feed_Forward_Neural_Network):
    
    def __init__(self, parent, nr_hidden_layers, nr_neurons_per_lay, input_features, output_features, 
                 activation_function, pso_param_dict=None):
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
            'min_std_tol': 0.01
        }
    
        if pso_param_dict is not None:
            for key in pso_param_dict:
                if key in self.pso_param_dict:
                    self.pso_param_dict[key] = pso_param_dict[key]
                else:
                    print('\'%s\ is not a valid key. Choose between:' % (key), self.pso_param_dict.keys())
                    break
        
        self.parent = parent
        self.nr_variables = self.parent.model.count_params()
        self.nr_hidden_layers = nr_hidden_layers
        self.nr_neurons_per_lay = nr_neurons_per_lay
        self.activation_function = activation_function
        self.input_features = input_features
        self.output_features = output_features
        
        weights = model.get_weights()
        self.weight_shapes = []
        
        for mat in weights:
            self.weight_shapes.append(np.shape(mat))
        
        self.best_weights_train = None
        self.best_weights_val = None
        
        self.inertia_weight = self.pso_param_dict['inertiaWeightStart']
        self.vMax = (self.pso_param_dict['xMax']-self.pso_param_dict['xMin']) / self.pso_param_dict['deltaT']
        
        self.set_up_model()
        
        self.initialise_swarm()
        
    def train_network(self, nr_iterations, training_data_dict, loss_function, nr_iters_before_restart_check, speed_check,
                     std_penalty):
        
        self.training_data_dict = training_data_dict
        self.loss_function = loss_function
        self.nr_iterations_trained = nr_iterations
        self.std_penalty = std_penalty
        
        with open('progress.txt', 'w+') as f:

            # make sure the output isn't just the same
            shouldStartFresh = True
            while shouldStartFresh:
                shouldStartFresh = 0

                inertia_weight_reduction = np.exp(np.log(self.pso_param_dict['inertiaWeightMin'] 
                                                         / self.pso_param_dict['inertiaWeightStart'])
                                                         / self.pso_param_dict['explorationFraction'] 
                                                         / nr_iterations)
                inertia_weight = self.pso_param_dict['inertiaWeightStart']

                self.progress = 0
                
                self.validationScoreHistory = []
                self.trainingScoreHistory = []
                
                self.avg_speed_before_history = []
                self.avg_speed_after_history = []
                    
                time_since_swarm_best = 0
                
                self.initialise_swarm()

                glob_start = time.time()
                for iteration in range(nr_iterations):
                    
                    self.progress = iteration / nr_iterations

                    if (int(iteration/10) == iteration/10) and (iteration > 0):
                        # see if network has run into a local minima   
                        if nr_iters_before_restart_check is not None:
                            if (iteration - time_since_swarm_best) > nr_iters_before_restart_check:
                                self.set_weights(self.best_weights_val)
                                y_pred = self.predict_output('val')

                                stds = np.std(y_pred, axis=0)
                                print('standard deviations of predicted parameters (validation set): ', stds)
                                shouldStartFresh = np.any(stds < self.pso_param_dict['min_std_tol'])
                                if shouldStartFresh:
                                    break

                        progress_end = time.time()
                        elapsed_so_far = (progress_end - glob_start) / 60
                        time_remaining = elapsed_so_far / iteration * (self.nr_iterations_trained - iteration)

                        print('%s, Iteration %d' % (datetime.datetime.now().strftime("%H:%M:%S"), iteration))
                        f.write('%s      ' % (datetime.datetime.now().strftime("%H:%M:%S")))
                        f.write('Iterations tried: %d/%d     ' % (iteration, self.nr_iterations_trained))
                        f.write('Elapsed time: %dmin     ' % (elapsed_so_far))
                        f.write('Time remaining: %dmin.\n' % (time_remaining))
                        f.flush()

                    for i_particle, particle in enumerate(self.particle_list):
                        
                        train_score = particle.evaluate_particle('train')

                        is_swarm_best_train = (train_score < self.swarm_best_train)
                        
                        if is_swarm_best_train:
                            
                            self.best_weights_train = particle.get_weights()
                        
                            lastTimeSwarmBest = iteration
                            self.swarm_best_train = train_score
                            self.swarm_best_position = particle.position
                            
                            val_score = particle.evaluate_particle('val')
                            is_swarm_best_val = (val_score < self.swarm_best_val)
                            if is_swarm_best_val: # only update best weights after val highscore
                                self.best_weights_val = particle.get_weights()
                            
                            self.validationScoreHistory.append(val_score)
                            self.trainingScoreHistory.append(train_score)

                            print('Iteration %d, particle %d, new swarm best. Train: %.3e, Val: %.3e' % (iteration, 
                                                            iParticle, train_score, val_score))
                            f.write('Iteration %d, particle %d, new swarm best. Train: %.3e, Val: %.3e\n' % (iteration, 
                                                            iParticle, train_score, val_score))
                            f.flush()


                    self.update_swarm(speed_check, f)
                    
                    inertia_weight = self.update_inertia_weight(inertia_weight, inertia_weight_reduction, 
                                                                iteration, f)
                    
                    

            end = time.time()
            print('%s, Training complete.' % (datetime.datetime.now().strftime("%H:%M:%S")))
            f.write('%s, Training complete.' % (datetime.datetime.now().strftime("%H:%M:%S")))
            f.flush()
        
    def update_inertia_weight(self, inertia_weight, inertia_weight_reduction, iteration, f):
        
        isExploring = (inertia_weight > self.pso_param_dict['inertiaWeightMin'])
        if isExploring:
            inertia_weight = inertia_weight * inertia_weight_reduction
            isExploring = (inertia_weight > self.pso_param_dict['inertiaWeightMin'])
            if not isExploring:
                print('SWITCH TO EPLOIT! Iteration %d/%d.' % (iteration, self.nr_iterations_trained))
                f.write('SWITCH TO EPLOIT! Iteration %d/%d.\n' % (iteration, self.nr_iterations_trained))
                f.flush()
        return inertia_weight
        
    def predict_output(self, mode):
        
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
            weightMatrixList = self.best_weights_train[0]
            biasList = self.best_weights_train[1]
        elif mode == 'val':
            weightMatrixList = self.best_weights_val[0]
            biasList = self.best_weights_val[1]
            
        for i in range(len(weightMatrixList)):
            self.parent.model.layers[i].set_weights([weightMatrixList[i], biasList[i]])
            
            
    def evaluate_model(self, mode):
    
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
        
        score = self.parent.evaluate_model(mode)
        
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
        
        
